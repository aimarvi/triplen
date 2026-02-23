import numpy as np
import pandas as pd
from tqdm import tqdm

from gsn.perform_gsn import perform_gsn

def session_gsn(
    out,
    *,
    win=1,
    step=None,
    overlap=True,
    scaling=1e3,
    time_offset=0,
    wantverbose=False,
):
    '''
    run gsn in sliding windows over a 4d raster:
      out: (units, time, images, reps) with spike counts (or rates)

    args:
        win (int): window length in bins
        step (int|None): stride in bins; if None, chosen from overlap
        overlap (bool): if True, default step=1; else default step=win
        scaling (float): multiply Xavg by this before gsn
        time_offset (int): add to reported time (useful if aligning to onset)
        wantverbose (bool): passed to perform_gsn

    returns:
        cov_df: long df with one row per time x type and cov vector + summary
        ncsnr_df: long df with one row per time and ncsnr vector + summary
    '''
    if out.ndim != 4:
        raise ValueError(f'out must be 4d (units,time,images,reps); got shape {out.shape}')

    if step is None:
        step = 1 if overlap else win

    if win <= 0 or step <= 0:
        raise ValueError('win and step must be positive integers')

    if out.shape[1] < win:
        raise ValueError(f'time dim {out.shape[1]} < win {win}')

    cov_rows = []
    ncsnr_rows = []

    # precompute upper triangle indices once (sigcov/noisecov are same shape each time)
    # note: gsn returns (units x units) covariances
    triu = None

    for t0 in tqdm(range(0, out.shape[1] - win + 1, step), leave=False):
        Xw = out[:, t0:t0 + win, :, :]            # (units, win, images, reps)
        Xavg = np.nanmean(Xw, axis=1)             # (units, images, reps)
        Xavg = Xavg * scaling

        results = perform_gsn(Xavg, {'wantverbose': wantverbose})
        sigcov = results['cSb']
        noisecov = results['cNb']
        ncsnr = results['ncsnr']

        if triu is None:
            triu = np.triu_indices_from(sigcov, k=1)

        t_report = t0 + time_offset

        sig_vec = sigcov[triu]
        noi_vec = noisecov[triu]

        cov_rows.append({
            'time': t_report,
            't0': t0,
            't1': t0 + win,
            'win': win,
            'step': step,
            'type': 'signal',
            'covariance': sig_vec,
            'mean_abs_covariance': np.nanmean(np.abs(sig_vec)),
        })
        cov_rows.append({
            'time': t_report,
            't0': t0,
            't1': t0 + win,
            'win': win,
            'step': step,
            'type': 'noise',
            'covariance': noi_vec,
            'mean_abs_covariance': np.nanmean(np.abs(noi_vec)),
        })

        ncsnr_rows.append({
            'time': t_report,
            't0': t0,
            't1': t0 + win,
            'win': win,
            'step': step,
            'ncsnr': ncsnr,
            'mean_abs_ncsnr': np.nanmean(np.abs(ncsnr)),
        })

    cov_df = pd.DataFrame(cov_rows)
    ncsnr_df = pd.DataFrame(ncsnr_rows)

    return cov_df, ncsnr_df
