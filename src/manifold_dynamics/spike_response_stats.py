import numpy as np
from scipy.stats import mannwhitneyu

def _ms_to_slice(onset, t_ms, bin_ms=1):
    # t_ms is (start_ms, end_ms) relative to onset; end is exclusive
    t0 = onset + int(np.round(t_ms[0] / bin_ms))
    t1 = onset + int(np.round(t_ms[1] / bin_ms))
    return slice(t0, t1)

def _trial_rates(X, t_sl, bin_ms=1):
    '''
    X: [units, time, images, reps] (spike counts per bin)
    returns firing rate of each unit for all trials within a given time window
    '''
    # window spike counts per trial: sum over time bins in window
    win_counts = np.nansum(X[:, t_sl, :, :], axis=1)  # [units, images, reps]
    dur_s = (t_sl.stop - t_sl.start) * (bin_ms / 1000.0)
    win_rates = win_counts / max(dur_s, 1e-12)

    # flatten images x reps
    unit_firing_rate = []
    for u in range(X.shape[0]):
        v = win_rates[u].reshape(-1)
        unit_firing_rate.append(v[np.isfinite(v)])
    return unit_firing_rate

def is_responsive(X, onset=50, bin_ms=1, baseline_win=(-25, 30), stim_win=((50, 120), (120, 240))):
    '''
    significance test for above-baseline firing 
    
    Args:
        X (array): raster (units, time, images, reps)
        onset (int): onset time, in milliseconds
        bin_ms (int): size of a single time-bin
        baseline_win (tuple): window start and end times, in milliseconds and relative to image onset
        stim_win (tuple): same as above, but for stim-on periods

    Returns: 
        pvals (array): array of p-values for significance test
    '''
    # compute baseline and stim windows 
    base_sl = _ms_to_slice(onset, baseline_win, bin_ms=bin_ms)
    post_sls = [_ms_to_slice(onset, w, bin_ms=bin_ms) for w in stim_win]

    unit_baselines = _trial_rates(X, base_sl, bin_ms=bin_ms)

    # pre-made array of p-values
    pvals = np.full((X.shape[0], len(post_sls)), np.nan, dtype=float)
    for j, sl in enumerate(post_sls):
        unit_responses = _trial_rates(X, sl, bin_ms=bin_ms)
        for u in range(X.shape[0]):
            a = unit_baselines[u]
            b = unit_responses[u]

            pvals[u, j] = mannwhitneyu(a, b, alternative='two-sided').pvalue

    return pvals
