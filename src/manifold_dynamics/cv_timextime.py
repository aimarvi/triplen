from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

import manifold_dynamics.session_raster_extraction as sre
import manifold_dynamics.spike_response_stats as srs
import manifold_dynamics.tuning_utils as tut


@dataclass(frozen=True)
class CvConfig:
    """Configuration for the cross-validated timextime replication."""

    alpha_responsive: float = 0.05
    onset_ms: int = 50
    baseline_ms: tuple[int, int] = (-50, 0)
    response_ms: tuple[int, int] = (50, 220)
    tstart: int = 100
    tend: int = 400
    k_step: int = 5
    k_max: int = 200
    l2_step: int = 5
    smooth_sigma: float = 1.0


def _roi_label_from_uid(roi_uid: str) -> str:
    """
    Convert uid format to the timextime-style ROI label.

    Args:
        roi_uid (str):
            UID from `roi-uid.csv`, expected format
            ``SesIdx.RoiIndex.AREALABEL.Categoty``.

    Returns:
        (str):
            roi_label (str):
            ROI label in ``AREALABEL_RoiIndex_Categoty`` format.
    """
    parts = roi_uid.split(".")
    if len(parts) != 4:
        raise ValueError(f"Unexpected roi_uid format: {roi_uid}")

    roi_index = int(parts[1])
    area_label = parts[2]
    category = parts[3]
    return f"{area_label}_{roi_index}_{category}"


def _split_repeats_odd_even(
    raster_4d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split repeats axis into odd/even partitions.

    Args:
        raster_4d (np.ndarray):
            Raster array with shape ``(units, time, images, reps)``.

    Returns:
        (tuple[np.ndarray, np.ndarray]):
            split_a (np.ndarray):
                Odd-repeat partition with shape ``(units, time, images, reps_a)``.
            split_b (np.ndarray):
                Even-repeat partition with shape ``(units, time, images, reps_b)``.
    """
    if raster_4d.ndim != 4:
        raise ValueError(
            f"Expected 4D raster (units,time,images,reps), got {raster_4d.shape}"
        )

    split_a = raster_4d[:, :, :, 0::2]
    split_b = raster_4d[:, :, :, 1::2]
    return split_a, split_b


def _responsive_mask(split_4d: np.ndarray, config: CvConfig) -> np.ndarray:
    """
    Return mask of units responsive in a split.

    Args:
        split_4d (np.ndarray):
            Split raster with shape ``(units,time,images,reps)``.
        config (CvConfig):
            Analysis settings.

    Returns:
        (np.ndarray):
            mask_units (np.ndarray):
                Boolean mask over units.
    """
    pvals = srs.is_responsive(
        X=split_4d,
        onset=config.onset_ms,
        baseline_win=config.baseline_ms,
        stim_win=[(50, 120)],
        # stim_win=((50, 120), (120, 240)),
    )
    return np.any(pvals < config.alpha_responsive, axis=1)


def _mean_over_repeats(split_4d: np.ndarray) -> np.ndarray:
    """
    Average repeats for a split.

    Args:
        split_4d (np.ndarray):
            Split raster, shape ``(units,time,images,reps)``.

    Returns:
        (np.ndarray):
            split_3d (np.ndarray):
                Shape ``(units,time,images)``.
    """
    return np.nanmean(split_4d, axis=3)


def _image_order(split_3d: np.ndarray, config: CvConfig) -> np.ndarray:
    """
    Rank images by baseline-subtracted response magnitude.

    Args:
        split_3d (np.ndarray):
            Response tensor with shape ``(units,time,images)``.
        config (CvConfig):
            Analysis settings.

    Returns:
        (np.ndarray):
            order_indices (np.ndarray):
                Descending rank order for image indices.
    """
    base = slice(config.onset_ms + config.baseline_ms[0], config.onset_ms + config.baseline_ms[1])
    resp = slice(config.onset_ms + config.response_ms[0], config.onset_ms + config.response_ms[1])

    scores = np.nanmean(split_3d[:, resp, :], axis=(0, 1)) - np.nanmean(
        split_3d[:, base, :], axis=(0, 1)
    )
    order = np.argsort(scores)[::-1]
    return order


def _time_time_tuning_rdm(split_3d: np.ndarray, image_idx: np.ndarray, config: CvConfig) -> np.ndarray:
    """
    Build time-by-time tuning RDM using per-time image RDM vectors.

    Args:
        split_3d (np.ndarray):
            Tensor with shape ``(units,time,images)``.
        image_idx (np.ndarray):
            1D image index array.
        config (CvConfig):
            Analysis settings.

    Returns:
        (np.ndarray):
            R_time_time (np.ndarray):
                Square time-by-time representational dissimilarity matrix.
    """
    X = split_3d[:, config.tstart : config.tend, :][:, :, image_idx]

    if X.shape[2] < 2:
        raise ValueError("Need at least 2 images to form an image RDM.")

    rdv_by_time = np.array(
        [pdist(X[:, t, :].T, metric="correlation") for t in range(X.shape[1])]
    )
    valid_pair_mask = np.isfinite(rdv_by_time).all(axis=0)
    if int(np.sum(valid_pair_mask)) < 2:
        raise ValueError(
            "Insufficient finite image-pair distances across time for time-time RDM."
        )
    rdv_by_time = rdv_by_time[:, valid_pair_mask]
    rdv_rank = np.apply_along_axis(rankdata, 1, rdv_by_time)
    valid_time_mask = np.isfinite(rdv_rank).all(axis=1) & (np.nanstd(rdv_rank, axis=1) > 0)
    if int(np.sum(valid_time_mask)) < 2:
        raise ValueError("Insufficient non-constant timepoints for time-time RDM.")
    rdv_rank = rdv_rank[valid_time_mask]
    return squareform(pdist(rdv_rank, metric="correlation"))


def _estimate_top_k(split_3d_train: np.ndarray, config: CvConfig) -> int:
    """
    Estimate the local manifold scale using the existing l2-norm heuristic.

    Args:
        split_3d_train (np.ndarray):
            Training split with shape ``(units,time,images)``.
        config (CvConfig):
            Analysis settings.

    Returns:
        (int):
            top_k (int):
                Chosen manifold scale.
    """
    order = _image_order(split_3d=split_3d_train, config=config)
    n_images = split_3d_train.shape[2]
    sizes = [k for k in range(config.k_step, min(config.k_max, n_images) + 1, config.k_step)]
    if not sizes:
        raise ValueError("No valid top-k sizes available for this ROI.")

    rdms = []
    for k in sizes:
        idx = order[:k]
        R = _time_time_tuning_rdm(split_3d=split_3d_train, image_idx=idx, config=config)
        rdms.append(R)

    tri = np.triu_indices_from(rdms[0], k=1)
    l2_values = []
    for i in range(len(rdms)):
        i0 = i
        i1 = min(i + config.l2_step, len(rdms))
        chunk = np.array([rdms[j][tri] for j in range(i0, i1)])
        mean_chunk = np.nanmean(chunk, axis=0)
        l2_values.append(np.sqrt(np.nansum(mean_chunk**2)))

    l2_values = np.asarray(l2_values)
    l2_smooth = gaussian_filter1d(l2_values, sigma=config.smooth_sigma)
    idx_min = int(np.nanargmin(l2_smooth))
    return int(sizes[idx_min])


def _safe_ed2(R: np.ndarray) -> float:
    """
    Compute ED2 with explicit finite-data validation.

    Args:
        R (np.ndarray):
            Distance matrix.

    Returns:
        (float):
            ed_value (float):
                Effective dimensionality.
    """
    if not np.isfinite(R).all():
        raise ValueError("RDM contains non-finite values; ED2 is undefined.")
    return float(tut.ED2(R))


def run_roi_cv(
    roi_uid: str,
    *,
    top_k: int | None = None,
    config: CvConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Run cross-validated timextime analysis for one ROI UID.

    Args:
        roi_uid (str):
            ROI uid from `roi-uid.csv`.
        top_k (int | None):
            If provided, forces this manifold scale. If ``None``,
            top-k is estimated from each training split.
        config (CvConfig | None):
            Analysis configuration. If ``None``, defaults are used.

    Returns:
        (tuple[pd.DataFrame, dict[str, np.ndarray]]):
            df_summary (pd.DataFrame):
                Two-row table (one row per CV direction) with ED and compression metrics.
            artifacts (dict[str, np.ndarray]):
                Stored time-by-time RDMs keyed by fold and condition.
    """
    roi_label = _roi_label_from_uid(roi_uid=roi_uid)
    return run_combined_roi_cv(
        roi_uids=[roi_uid],
        roi_label=roi_label,
        top_k=top_k,
        config=config,
    )


def run_combined_roi_cv(
    roi_uids: list[str],
    *,
    roi_label: str,
    top_k: int | None = None,
    config: CvConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Run cross-validated timextime analysis for one ROI label, combining sessions.

    Args:
        roi_uids (list[str]):
            Session-specific UIDs that share the same ROI label.
        roi_label (str):
            Common ROI label (e.g., ``Unknown_19_F``).
        top_k (int | None):
            If provided, forces this manifold scale. If ``None``,
            top-k is estimated from each training split.
        config (CvConfig | None):
            Analysis configuration. If ``None``, defaults are used.

    Returns:
        (tuple[pd.DataFrame, dict[str, np.ndarray]]):
            df_summary (pd.DataFrame):
                Two-row table (one row per CV direction) with ED and compression metrics.
            artifacts (dict[str, np.ndarray]):
                Stored time-by-time RDMs keyed by fold and condition.
    """
    cfg = CvConfig() if config is None else config

    rasters_by_uid = {}
    for roi_uid in roi_uids:
        rasters_by_uid[roi_uid] = sre.extract_session_raster(roi_uid=roi_uid, verbose=False)

    folds = [("A_to_B", 0), ("B_to_A", 1)]

    summary_rows: list[dict[str, object]] = []
    artifacts: dict[str, np.ndarray] = {}

    for fold_name, fold_index in folds:
        train_3d_parts = []
        test_3d_parts = []
        n_units_responsive_total = 0

        for roi_uid, raster in rasters_by_uid.items():
            split_a, split_b = _split_repeats_odd_even(raster_4d=raster)
            if fold_index == 0:
                train_4d, test_4d = split_a, split_b
            else:
                train_4d, test_4d = split_b, split_a

            train_mask = _responsive_mask(split_4d=train_4d, config=cfg)
            n_responsive = int(np.sum(train_mask))
            if n_responsive == 0:
                continue

            train_3d_uid = _mean_over_repeats(split_4d=train_4d)[train_mask]
            test_3d_uid = _mean_over_repeats(split_4d=test_4d)[train_mask]

            train_3d_parts.append(train_3d_uid)
            test_3d_parts.append(test_3d_uid)
            n_units_responsive_total += n_responsive

        if n_units_responsive_total < 2:
            raise ValueError(
                f"{roi_label} {fold_name}: fewer than 2 responsive units across combined sessions."
            )

        train_3d = np.concatenate(train_3d_parts, axis=0)
        test_3d = np.concatenate(test_3d_parts, axis=0)

        fold_top_k = int(top_k) if top_k is not None else _estimate_top_k(train_3d, cfg)
        order_train = _image_order(split_3d=train_3d, config=cfg)
        idx_top_k = order_train[:fold_top_k]
        idx_global = np.arange(test_3d.shape[2])
        idx_local = np.arange(1000, test_3d.shape[2])

        if idx_local.size < 2:
            raise ValueError(f"{roi_uid} {fold_name}: fewer than 2 localizer images.")

        R_top_k = _time_time_tuning_rdm(split_3d=test_3d, image_idx=idx_top_k, config=cfg)
        R_global = _time_time_tuning_rdm(split_3d=test_3d, image_idx=idx_global, config=cfg)
        R_local = _time_time_tuning_rdm(split_3d=test_3d, image_idx=idx_local, config=cfg)

        ED_top_k = _safe_ed2(R_top_k)
        ED_global = _safe_ed2(R_global)
        ED_local = _safe_ed2(R_local)

        pct_change_top_k = ((ED_top_k - ED_global) / ED_global) * 100.0
        pct_change_local = ((ED_local - ED_global) / ED_global) * 100.0
        compression_top_k = 1.0 - (ED_top_k / ED_global)
        compression_local = 1.0 - (ED_local / ED_global)

        summary_rows.append(
            {
                "roi_uid": "|".join(roi_uids),
                "roi": roi_label,
                "fold": fold_name,
                "n_units_responsive_train": n_units_responsive_total,
                "top_k": fold_top_k,
                "ED_topk": ED_top_k,
                "ED_localizer": ED_local,
                "ED_global": ED_global,
                "percent_change_topk_vs_global": pct_change_top_k,
                "percent_change_localizer_vs_global": pct_change_local,
                "compression_change_topk_vs_global": compression_top_k,
                "compression_change_localizer_vs_global": compression_local,
            }
        )

        artifacts[f"{fold_name}_R_topk"] = R_top_k
        artifacts[f"{fold_name}_R_global"] = R_global
        artifacts[f"{fold_name}_R_localizer"] = R_local

    df_summary = pd.DataFrame(summary_rows)
    return df_summary, artifacts


def run_many_rois_cv(
    roi_uids: list[str],
    *,
    dir_output: str,
    top_k: int | None = None,
    config: CvConfig | None = None,
    save_artifacts: bool = True,
) -> pd.DataFrame:
    """
    Run cross-validated timextime replication over multiple ROIs.

    Args:
        roi_uids (list[str]):
            ROI UID strings.
        dir_output (str):
            Output directory for summary and optional RDM artifacts.
        top_k (int | None):
            Fixed top-k if provided.
        config (CvConfig | None):
            Analysis settings.
        save_artifacts (bool):
            Whether to save fold RDM artifacts.

    Returns:
        (pd.DataFrame):
            df_all (pd.DataFrame):
                Concatenated ROI-fold summary table.
    """
    os.makedirs(dir_output, exist_ok=True)
    rows_all = []

    for roi_uid in roi_uids:
        print(f"[CV] Processing {roi_uid} ...")
        df_roi, artifacts = run_roi_cv(roi_uid=roi_uid, top_k=top_k, config=config)
        rows_all.append(df_roi)

        if save_artifacts:
            filepath_npz = os.path.join(dir_output, f"{roi_uid}_cv_rdms.npz")
            np.savez_compressed(filepath_npz, **artifacts)

    df_all = pd.concat(rows_all, ignore_index=True)
    filepath_parquet = os.path.join(dir_output, "cv_tuning_ed_summary.parquet")
    filepath_csv = os.path.join(dir_output, "cv_tuning_ed_summary.csv")
    df_all.to_parquet(filepath_parquet, index=False)
    df_all.to_csv(filepath_csv, index=False)
    return df_all
