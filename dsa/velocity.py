import numpy as np
import scipy.stats as ss
from scipy.spatial.distance import squareform
from scipy.signal import savgol_filter
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

# ------------------------------
# Helpers
# ------------------------------

def _upper_tri_vec(M):
    """Vectorize the upper triangle (excluding diagonal) of a symmetric matrix."""
    return squareform(M, checks=False)

def _corr_rdm(pop_mat, rank=True):
    """
    Image×image RDM from population responses at one time bin.
    pop_mat: (n_images, n_units)
    Returns condensed vector (len = n_images*(n_images-1)/2)
    """
    if rank:
        pop_mat = ss.rankdata(pop_mat, axis=1)  # rank across units per image (Spearman)
    # Corr across images (rows) over units (columns)
    C = np.corrcoef(pop_mat)
    np.fill_diagonal(C, 1.0)
    RDM = 1.0 - C  # correlation distance
    return _upper_tri_vec(RDM)

def _split_half_reliability(pop_mat, n_splits=20, rank=True, rng=None):
    """
    Split-half reliability of the RDM at one time bin by randomly splitting units.
    pop_mat: (n_images, n_units)
    Returns: mean Spearman correlation between half-RDMs, Spearman-Brown corrected.
    """
    rng = np.random.default_rng() if rng is None else rng
    n_units = pop_mat.shape[1]
    if n_units < 10:  # not enough units to split
        return np.nan
    vals = []
    for _ in range(n_splits):
        perm = rng.permutation(n_units)
        A = pop_mat[:, perm[:n_units//2]]
        B = pop_mat[:, perm[n_units//2:]]
        rdmA = _corr_rdm(A, rank=rank)
        rdmB = _corr_rdm(B, rank=rank)
        r = ss.spearmanr(rdmA, rdmB, nan_policy='omit').correlation
        vals.append(r)
    r_bar = np.nanmean(vals)
    # Spearman-Brown prophecy formula to approximate full-set reliability
    if np.isfinite(r_bar):
        return (2*r_bar) / (1 + r_bar)
    return np.nan

def _rdm_time_series(X_roi, img_idx, rank=True, use_repeats=False):
    """
    Compute RDM(t) across time for a set of images.
    X_roi: (n_units, n_time, n_images[, n_reps])
    """
    if X_roi.ndim == 4 and use_repeats:
        # cross-validated RDM via crossnobis (simple split across reps)
        # For simplicity we do split-half across reps; fall back to Spearman if reps<2
        n_units, n_time, n_images, n_reps = X_roi.shape
        assert n_reps >= 2, "Need at least 2 repeats for cross-validation."
        half1 = X_roi[:, :, img_idx, :n_reps//2].mean(-1)  # (u,t,i)
        half2 = X_roi[:, :, img_idx, n_reps//2:].mean(-1)
        # whiten within halves (optional): z-score units across images, per time
        RDMs = []
        for t in range(n_time):
            A = half1[:, t, :].T  # (i,u)
            B = half2[:, t, :].T  # (i,u)
            A = (A - A.mean(0)) / (A.std(0) + 1e-8)
            B = (B - B.mean(0)) / (B.std(0) + 1e-8)
            # crossnobis estimate: pairwise squared Mahalanobis with cross-validated dot-products
            # Here, use simple cross-validated Euclidean surrogate (dot product cross halves):
            G = A @ B.T / A.shape[1]  # cross-cov across images
            # turn cross-cov into dissimilarity: D_ij = G_ii + G_jj - 2*G_ij
            diag = np.diag(G)
            D = diag[:, None] + diag[None, :] - 2*G
            np.fill_diagonal(D, 0.0)
            RDMs.append(_upper_tri_vec(D))
        RDMs = np.asarray(RDMs)
        reliab = None  # cross-validated, so skip ceiling correction
    else:
        # Spearman correlation distance RDMs from single-try or averaged data
        n_units, n_time, n_images = X_roi.shape[:3]
        RDMs, reliab = [], []
        for t in range(n_time):
            # (images, units)
            M = X_roi[:, t, :][:, img_idx].T
            # z-score units across images for Pearson (if rank=False). For Spearman we'll rank.
            if not rank:
                M = (M - M.mean(0)) / (M.std(0) + 1e-8)
            RDMs.append(_corr_rdm(M, rank=rank))
            # split-half reliability per time bin (ceiling for attenuation correction)
            reliab.append(_split_half_reliability(M, rank=rank))
        RDMs = np.asarray(RDMs)
        print(f'vRDM shape: {RDMs.shape}')
        reliab = np.asarray(reliab)
    return RDMs, reliab

def rdm_velocity(RDMs, delta=1, attenuation=None):
    """
    Velocity v(t) = 1 - corr(vec(RDM_t), vec(RDM_{t+delta})).
    Optionally attenuation-correct using reliability at t and t+delta.
    """
    T = RDMs.shape[0]
    v = np.full(T-delta, np.nan)
    for t in range(T-delta):
        a, b = RDMs[t], RDMs[t+delta]
        rho = ss.pearsonr(a, b)[0]  # similarity of geometry
        if attenuation is not None and np.isfinite(attenuation[t]) and np.isfinite(attenuation[t+delta]):
            denom = np.sqrt(max(attenuation[t], 1e-6) * max(attenuation[t+delta], 1e-6))
            rho = np.clip(rho / denom, -1.0, 1.0)  # attenuation-corrected
        v[t] = 1.0 - rho
    return v

def circular_shift_null(X_roi, img_idx, n_shuff=500, delta=1, rank=True, use_repeats=False, rng=None):
    """
    Null that preserves each image's temporal autocorrelation but destroys across-image alignment:
    randomly circularly shift the time axis **independently per image**, recompute RDM(t), then velocity.
    """
    rng = np.random.default_rng() if rng is None else rng
    n_units, n_time, n_images = X_roi.shape[:3]
    V = []
    for s in range(n_shuff):
        Xs = X_roi.copy()
        for j, img in enumerate(img_idx):
            shift = int(rng.integers(0, n_time))
            Xs[:, :, img] = np.roll(Xs[:, :, img], shift=shift, axis=1)
        RDMs_s, _ = _rdm_time_series(Xs, img_idx, rank=rank, use_repeats=use_repeats)
        V.append(rdm_velocity(RDMs_s, delta=delta))
    return np.asarray(V)  # (n_shuff, T-delta)

def time_permutation_null(RDMs, n_shuff=1000, delta=1, rng=None):
    """
    Simple null by permuting time indices of RDMs (destroys continuity).
    """
    rng = np.random.default_rng() if rng is None else rng
    T = RDMs.shape[0]
    V = []
    for _ in range(n_shuff):
        perm = rng.permutation(T)
        R = RDMs[perm]
        V.append(rdm_velocity(R, delta=delta))
    return np.asarray(V)

def maxT_threshold(Vnull, alpha=0.05):
    """
    Family-wise error control across time: take the 1-alpha quantile of the null distribution
    of the **maximum** velocity across time per shuffle.
    """
    max_per_shuffle = np.nanmax(Vnull, axis=1)
    thr = np.quantile(max_per_shuffle, 1 - alpha)
    return thr

# ------------------------------
# Main analysis
# ------------------------------

def run_velocity_analysis(
    X, img_sets, rank=True, use_repeats=False, delta=1, bin_ms=1, smooth_plot=True,
    n_shuff_perm=1000, n_shuff_shift=500, alpha=0.05, do_structure_fn=True
):
    """
    X: (n_units, n_time, n_images[, n_reps])
    img_sets: dict name -> np.array of image indices
    """
    print(X.shape)
    n_time = X.shape[1]
    time_axis_ms = np.arange(n_time) * bin_ms

    results = {}
    for label, idxs in img_sets.items():
        # --- RDM time series + reliability ---
        RDMs, reliab = _rdm_time_series(X, idxs, rank=rank, use_repeats=use_repeats)

        # --- Velocity (attenuation-corrected if reliability available) ---
        att = None if reliab is None else reliab
        vel = rdm_velocity(RDMs, delta=delta, attenuation=att)

        # --- Nulls ---
        Vperm = time_permutation_null(RDMs, n_shuff=n_shuff_perm, delta=delta)
        Vshift = circular_shift_null(X, idxs, n_shuff=n_shuff_shift, delta=delta, rank=rank, use_repeats=use_repeats)

        # --- Significance (max-T) ---
        thr_perm = maxT_threshold(Vperm, alpha=alpha)
        thr_shift = maxT_threshold(Vshift, alpha=alpha)

        # Optional FDR across time (complementary to max-T)
        # p-values vs permutation null
        pv = np.mean(Vperm >= vel[None, :], axis=0)
        sig_mask = multipletests(pv, alpha=alpha, method='fdr_bh')[0]

        # --- Structure function (optional): decorrelation vs lag ---
        D_tau = None
        if do_structure_fn:
            max_tau = min(12, n_time-1)  # e.g., up to ~240 ms if bin_ms=20
            D_tau = []
            for tau in range(1, max_tau+1):
                sims = []
                for t in range(n_time - tau):
                    r = ss.pearsonr(RDMs[t], RDMs[t+tau])[0]
                    sims.append(1 - r)
                D_tau.append(np.nanmedian(sims))
            D_tau = np.asarray(D_tau)

        # --- Plot ---
        fig, ax = plt.subplots(1, 1, figsize=(7, 3))
        print(vel.shape)
        print(vel)
        ax.plot(time_axis_ms[:len(vel)], vel, label='Velocity (1 - corr)', alpha=0.7)
        if smooth_plot:
            w = 9 if len(vel) >= 9 else (len(vel)//2)*2 + 1
            vel_smooth = savgol_filter(vel, window_length=w, polyorder=2, mode='interp')
            ax.plot(time_axis_ms[:len(vel)], vel_smooth, linewidth=2, label='Savitzky–Golay')
        ax.axhline(thr_perm, color='r', linestyle='--', linewidth=1, label='max-T perm thr')
        ax.axhline(thr_shift, color='orange', linestyle='--', linewidth=1, label='max-T shift thr')
        if reliab is not None:
            ax2 = ax.twinx()
            ax2.plot(time_axis_ms, reliab, color='gray', alpha=0.3, label='RDM reliability')
            ax2.set_ylabel('Reliability')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Velocity')
        ax.set_title(label)
        ax.legend(frameon=False, loc='upper right')
        plt.tight_layout()
        plt.show()

        results[label] = dict(
            vel=vel, vel_smooth=vel_smooth if smooth_plot else None,
            Vperm=Vperm, Vshift=Vshift, thr_perm=thr_perm, thr_shift=thr_shift,
            pvals=pv, sig_mask=sig_mask, reliab=reliab, D_tau=D_tau
        )
    return results
