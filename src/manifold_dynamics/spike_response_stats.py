import numpy as np
from scipy.stats import mannwhitneyu, ttest_rel

ONSET_TIME = 50
RESP_WIN_MS = (50, 220)
BASE_WIN_MS = (-50, 0)
IMG_SLICE = slice(1000, None)


def _ms_to_slice(onset, t_ms, bin_ms=1):
    # t_ms is (start_ms, end_ms) relative to onset; end is exclusive
    t0 = onset + int(np.round(t_ms[0] / bin_ms))
    t1 = onset + int(np.round(t_ms[1] / bin_ms))
    return slice(t0, t1)

def is_responsive(
    X,
    roi_uid,
    test_type="paired",
    onset=ONSET_TIME,
    bin_ms=1,
    baseline_win=BASE_WIN_MS,
    stim_win=(RESP_WIN_MS,),
    image_slice=IMG_SLICE,
):
    """
    Compute per-unit p-values for baseline vs response windows after trial averaging.

    Args:
        X (array):
            Raster with shape ``(units, time, images, reps)`` or
            ``(units, time, images)``.
        roi_uid (str):
            ROI/session UID for bookkeeping and validation context.
        test_type (str):
            ``"paired"`` uses paired t-test across images.
            ``"unpaired"`` uses Mann-Whitney U across images.
        onset (int):
            Onset time in ms.
        bin_ms (int):
            Bin size in ms.
        baseline_win (tuple):
            Baseline window relative to onset (start_ms, end_ms).
        stim_win (tuple):
            One or more response windows relative to onset.
        image_slice (slice):
            Image subset used for p-value testing. Defaults to ``slice(1000, None)``.

    Returns:
        np.ndarray:
            P-value matrix with shape ``(units, len(stim_win))``.
    """
    if not isinstance(roi_uid, str) or not roi_uid:
        raise ValueError(f"roi_uid must be a non-empty string. Got: {roi_uid!r}")

    X = np.asarray(X)
    if X.ndim == 4:
        Xavg = np.nanmean(X, axis=3)  # average across trials/repeats
    elif X.ndim == 3:
        Xavg = X
    else:
        raise ValueError(
            f"{roi_uid}: expected X with shape (units,time,images[,reps]), got {X.shape}"
        )

    test_type = str(test_type).lower()
    if test_type not in {"paired", "unpaired"}:
        raise ValueError(
            f"{roi_uid}: test_type must be 'paired' or 'unpaired', got {test_type!r}"
        )

    base_sl = _ms_to_slice(onset, baseline_win, bin_ms=bin_ms)
    post_sls = [_ms_to_slice(onset, w, bin_ms=bin_ms) for w in stim_win]

    # Standard behavior: only evaluate localizer images [1000:].
    Xavg = Xavg[:, :, image_slice]
    n_images = Xavg.shape[2]
    if n_images != 72:
        raise ValueError(
            f"{roi_uid}: expected 72 images from X[..., 1000:], got {n_images}"
        )

    n_units = Xavg.shape[0]
    pvals = np.full((n_units, len(post_sls)), np.nan, dtype=float)

    for u in range(n_units):
        base = np.nanmean(Xavg[u, base_sl, :], axis=0)
        for j, sl in enumerate(post_sls):
            resp = np.nanmean(Xavg[u, sl, :], axis=0)
            m = np.isfinite(base) & np.isfinite(resp)
            if int(np.sum(m)) < 2:
                continue
            if test_type == "paired":
                pvals[u, j] = ttest_rel(resp[m], base[m]).pvalue
            else:
                pvals[u, j] = mannwhitneyu(base[m], resp[m], alternative="two-sided").pvalue

    return pvals
