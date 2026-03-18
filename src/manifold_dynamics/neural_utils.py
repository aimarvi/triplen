from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
    uid_csv = Path(pth.OTHERS) / "roi-uid.csv"
    df_uid = pd.read_csv(vst.fetch(str(uid_csv)))

    uids = []
    for uid in df_uid["uid"].astype(str):
        parts = uid.split(".")
        if len(parts) != 4:
            continue
        label = f"{parts[2]}_{int(parts[1])}_{parts[3]}"
        if label == roi_label:
            uids.append(uid)
    return uids

def load_cached_session_raster(uid):
    """
    Load precomputed session raster from storage cache.

    Args:
        uid (str): Session ROI UID (e.g., ``18.19.Unknown.F``).

    Returns:
        np.ndarray: Raster tensor of shape (units, time, images, repeats).
    """
    npy_path = f"{pth.PROCESSED}/single-session-raster/{uid}.npy"
    f = vst.fetch(npy_path)
    return np.load(f)


def bin_to_psth(raster_4d, bin_size_ms=20):
    """
    Convert raw raster to a binned PSTH per trial.

    This preserves the time axis length, so output shape remains:
      (units, 450, 1072, repeats)

    Args:
        raster_4d (np.ndarray): Input raster of shape (units, time, images, repeats).
        bin_size_ms (int): Temporal bin size in ms. Assumes 1 ms native bins.
    """
    if raster_4d.ndim != 4:
        raise ValueError(f"Expected raster shape (units,time,images,repeats), got {raster_4d.shape}")
    if int(bin_size_ms) < 1:
        raise ValueError(f"bin_size_ms must be >= 1, got {bin_size_ms}")
    return uniform_filter1d(
        raster_4d.astype(np.float32, copy=False),
        size=int(bin_size_ms),
        axis=1,
        mode="nearest",
    )


def responsive_mask(uid, raster_4d, alpha=0.05):
    """
    Compute/load per-unit p-values on all repeats and return a responsive-unit mask.

    Cached file path:
      <cache_dir>/pvalues/full_reps/<uid>.npy
    """
    pval_dir = Path(vst.get_cache_dir("pvalues/full_reps"))
    pval_path = pval_dir / f"{uid}.npy"

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


def significant_trial_raster(roi_uid: str, alpha: float = 0.05, bin_size_ms: int = 20) -> np.ndarray:
    """
    Return trial-level raster data for one ROI target after significance filtering.

    Accepted ROI target formats:
        - 4-part UID: ``SesIdx.RoiIndex.AREALABEL.Categoty``
        - 3-part ROI: ``RoiIndex.AREALABEL.Categoty``

    For 3-part ROI targets, all matching sessions are combined across the unit axis.
    Sessions with fewer trials are padded with ``NaN`` along the repeat axis before
    concatenation so the final output is a single dense array.

    Args:
        roi_uid: ROI target string in 3-part or 4-part format.
        alpha: P-value threshold for responsive-unit filtering.
        bin_size_ms: PSTH bin width in ms applied before significance filtering.

    Returns:
        Array with shape ``(units, time, images, trials)``.
    """
    uid_csv_local = vst.fetch(f"{pth.OTHERS}/roi-uid.csv")
    df_uid = pd.read_csv(uid_csv_local)

    target_parts = roi_uid.split(".")
    if len(target_parts) not in (3, 4):
        raise ValueError(
            "roi_uid must be either 4-part UID (SesIdx.RoiIndex.AREALABEL.Categoty) "
            "or 3-part ROI (RoiIndex.AREALABEL.Categoty)."
        )

    if len(target_parts) == 4:
        roi_uids = [roi_uid]
    else:
        roi_index = int(target_parts[0])
        area_label = target_parts[1]
        category = target_parts[2]
        roi_uids = []
        for uid in df_uid["uid"].astype(str):
            parts = uid.split(".")
            if len(parts) != 4:
                continue
            if int(parts[1]) == roi_index and parts[2] == area_label and parts[3] == category:
                roi_uids.append(uid)
        roi_uids = sorted(roi_uids, key=lambda x: int(x.split(".")[0]))

    if len(roi_uids) == 0:
        raise ValueError(f"No matching ROI UIDs found for target: {roi_uid}")

    raster_parts = []
    max_trials = 0
    for uid in roi_uids:
        raster_4d = load_cached_session_raster(uid)
        raster_4d = bin_to_psth(raster_4d, bin_size_ms=bin_size_ms)
        responsive = responsive_mask(uid=uid, raster_4d=raster_4d, alpha=alpha)
        if int(np.sum(responsive)) == 0:
            continue
        raster_sig = raster_4d[responsive]
        raster_parts.append(raster_sig)
        max_trials = max(max_trials, raster_sig.shape[3])

    if len(raster_parts) == 0:
        raise ValueError(f"No responsive units found for target: {roi_uid}")

    padded_parts = []
    for raster_sig in raster_parts:
        if raster_sig.shape[3] == max_trials:
            padded_parts.append(raster_sig)
            continue

        padded = np.full(
            (raster_sig.shape[0], raster_sig.shape[1], raster_sig.shape[2], max_trials),
            np.nan,
            dtype=raster_sig.dtype,
        )
        padded[:, :, :, : raster_sig.shape[3]] = raster_sig
        padded_parts.append(padded)

    return np.concatenate(padded_parts, axis=0)

def compute_noise_ceiling(data_in):
    """
    Compute the noise ceiling signal-to-noise ratio (SNR) and percentage noise ceiling for each unit.
    
    Parameters:
    ----------
    data_in : np.ndarray
        A 3D array of shape (units/voxels, conditions, trials), representing the data for which to compute 
        the noise ceiling. Each unit requires more than 1 trial for each condition.

    Returns:
    -------
    noiseceiling : np.ndarray
        The noise ceiling for each unit, expressed as a percentage.
    ncsnr : np.ndarray
        The noise ceiling signal-to-noise ratio (SNR) for each unit.
    signalvar : np.ndarray
        The signal variance for each unit.
    noisevar : np.ndarray
        The noise variance for each unit.
    """
    # noisevar: mean variance across trials for each unit
    noisevar = np.nanmean(np.std(data_in, axis=2, ddof=1) ** 2, axis=1)

    # datavar: variance of the trial means across conditions for each unit
    datavar = np.nanstd(np.nanmean(data_in, axis=2), axis=1, ddof=1) ** 2

    # signalvar: signal variance, obtained by subtracting noise variance from data variance
    signalvar = np.maximum(datavar - noisevar / data_in.shape[2], 0)  # Ensure non-negative variance

    # ncsnr: signal-to-noise ratio (SNR) for each unit
    ncsnr = np.sqrt(signalvar) / np.sqrt(noisevar)

    # noiseceiling: percentage noise ceiling based on SNR
    noiseceiling = 100 * (ncsnr ** 2 / (ncsnr ** 2 + 1 / data_in.shape[2]))

    return noiseceiling, ncsnr, signalvar, noisevar

def stack_ragged_firing_rates(data_in, period="early"):
    """
    Convert ragged per-image trial responses into a dense padded tensor.

    Args:
        data_in (pd.DataFrame): Table with a column containing ragged trial arrays.
        period (str): Column name (`pre`, `early`, `late`, etc.).

    Returns:
        np.ndarray: Array with shape (units, images, max_trials), padded with NaN.
    """
    in_period = list(data_in[period])
    num_units = len(in_period)
    num_images = len(in_period[0])

    # maximum number of reps for a single image
    max_reps = max(
        len(arr) if hasattr(arr, "__len__") else 0
        for unit in in_period
        for arr in unit)

    stacked = np.full((num_units, num_images, max_reps), np.nan, dtype=float)
    for unit_i, unit in enumerate(in_period):
        for img in range(num_images):
            arr = np.array(unit[img])
            reps_here = len(arr)
            if reps_here > 0:
                stacked[unit_i, img, :reps_here] = arr
                
    return stacked


def extract_unit_timecourse(row, start=None, end=None):
    """
    Return a unit-level 1D timecourse from row metadata.

    Uses `avg_psth` when available; otherwise derives it from `img_psth`.

    Args:
        row (pd.Series): Row with `avg_psth` and/or `img_psth`.
        start (int | None): Inclusive time index start.
        end (int | None): Exclusive time index end.

    Returns:
        np.ndarray: 1D timecourse slice of shape (T,).
    """
    avg = row["avg_psth"]
    if avg is None or (isinstance(avg, float) and np.isnan(avg)):
        A = np.asarray(row["img_psth"])  # (time, images)
        if A.ndim != 2:
            raise ValueError("img_psth must be 2D (time x images)")
        avg = A.mean(axis=1)
    avg = np.asarray(avg)
    if avg.ndim != 1:
        raise ValueError("avg_psth must be 1D (time,)")
    # take all values if start/end is not specified
    if start is None:
        start = 0
        end = len(avg)
    if len(avg) < end:
        raise ValueError(f"avg_psth length {len(avg)} < required end index {end}")
    return avg[start:end]  # (T,)


def plot_stimulus_image(idx, ax=None):
    """
    Plot the NSD1000_LOC image corresponding to the requested index.

    Args:
        idx (int): Image index in [0, 1071].
        ax (matplotlib.axes.Axes | None): Axis to draw on.

    Returns:
        matplotlib.axes.Axes: Axis containing the rendered image.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1)

    if idx < 1000:
        fname = f"{idx + 1:04d}.bmp"
    else:
        fname = f"MFOB{idx - 999:03d}.bmp"  # 1000 -> MFOB001, 1071 -> MFOB072

    fpath = pth._join_path(pth.IMAGEDIR, fname)
    local_fpath = vst.fetch(fpath)
    if local_fpath.exists():
        img = mpimg.imread(local_fpath)
        ax.imshow(img, cmap="gray")
        ax.set_title("")
        ax.axis("off")
    else:
        ax.text(0.5, 0.5, "missing", ha="center", va="center")
        ax.axis("off")

    return ax


#  Original name `derag_fr`.
def derag_fr(data_in, period="early"):
    """Backward-compatible wrapper for `stack_ragged_firing_rates`."""
    return stack_ragged_firing_rates(data_in, period=period)


#  Original name `get_unit_timecourse`.
def get_unit_timecourse(row, start=None, end=None):
    """Backward-compatible wrapper for `extract_unit_timecourse`."""
    return extract_unit_timecourse(row, start=start, end=end)


#  Original name `load_image`.
def load_image(idx, ax=None):
    """Backward-compatible wrapper for `plot_stimulus_image`."""
    return plot_stimulus_image(idx, ax=ax)
