from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

import manifold_dynamics.paths as pth
import manifold_dynamics.spike_response_stats as srs
import visionlab_utils.storage as vst


def roi_uids_for_label(roi_label: str) -> list[str]:
    """
    Resolve all session UIDs for an ROI label.

    UID format in roi-uid.csv: SesIdx.RoiIndex.AREALABEL.Categoty
    ROI label format used here: AREALABEL_RoiIndex_Categoty
    """
    uid_csv = os.path.join(pth.OTHERS, "roi-uid.csv")
    f = vst.fetch(uid_csv)
    df_uid = pd.read_csv(f)

    uids = []
    for uid in df_uid["uid"].astype(str):
        parts = uid.split(".")
        if len(parts) != 4:
            continue
        label = f"{parts[2]}_{int(parts[1])}_{parts[3]}"
        if label == roi_label:
            uids.append(uid)
    return uids


def load_session_raster_npy(uid: str) -> np.ndarray:
    """Load precomputed raw raster tensor from S3 as (units, time, images, repeats)."""
    npy_path = os.path.join(pth.PROCESSED, "single-session-raster", f"{uid}.npy")
    f = vst.fetch(npy_path)
    return np.load(f)


def bin_raster_to_20ms_psth(raster_4d: np.ndarray) -> np.ndarray:
    """
    Convert raw raster to 20 ms binned PSTH per trial, preserving time length.

    Input/Output shape: (units, 450, 1072, reps)
    """
    if raster_4d.ndim != 4:
        raise ValueError(f"Expected raster shape (units,time,images,reps), got {raster_4d.shape}")
    return uniform_filter1d(
        raster_4d.astype(np.float32, copy=False),
        size=20,
        axis=1,
        mode="nearest",
    )


def cached_fullrep_responsive_mask(uid: str, raster_4d: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute/load per-unit p-values on all repeats, then return responsive mask.

    P-values are cached locally at:
      <cache_dir>/pvalues/full_reps/<roi_uid>.npy
    """
    pval_dir = vst.get_cache_dir("pvalues/full_reps")
    pval_path = Path(pval_dir) / f"{uid}.npy"

    if pval_path.exists():
        pvals = np.load(pval_path)
    else:
        pvals = srs.is_responsive(X=raster_4d, roi_uid=uid, test_type="paired").squeeze()
        pval_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(pval_path, pvals)

    pvals = np.asarray(pvals).squeeze()
    if pvals.ndim != 1 or pvals.shape[0] != raster_4d.shape[0]:
        pvals = srs.is_responsive(X=raster_4d, roi_uid=uid, test_type="paired").squeeze()
        np.save(pval_path, pvals)

    return np.isfinite(pvals) & (pvals < alpha)
