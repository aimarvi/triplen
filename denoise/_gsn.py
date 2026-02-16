import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import pearsonr

from gsn.perform_gsn import perform_gsn

from manifold_dynamics.raw_raster import process_session
import manifold_dynamics.process_data as prcs

DATADIR = './../../datasets/NNN/'
SAVEDIR = './../../../buckets/manifold-dynamics/denoise'
if not os.path.exists(SAVEDIR):
    os.makedirs(SAVEDIR)
uid_sheet = pd.read_csv(os.path.join(DATADIR, 'roi-uid.csv'))
unique_rois = uid_sheet['uid'].unique()

for roi_uid in tqdm(unique_rois):
    session_num = roi_uid.split('.')[0]
    out = process_session(roi_uid) # shape is (units, 450, 1072, reps)
    print(f'Session {session_num} raster obtained!\n\nPerforming GSN...')
    
    ####### PERFORM GSN ON THE SINGLE ROI/SESSION
    cov_rows = []
    ncsnr_rows = []
    step = 1
    scaling = 1e3
    for t in range(0, out.shape[1] - step, step):
        Xw = out[:, slice(t, t + step), :, :]
        Xavg = np.mean(Xw, axis=1)
        Xavg = Xavg * scaling
    
        results = perform_gsn(Xavg, {'wantverbose': False})
        sigcov = results['cSb']
        noisecov = results['cNb']
        ncsnr = results['ncsnr']
    
        triu = np.triu_indices_from(sigcov, k=1)
    
        cov_rows.append({'time': t, 'type': 'signal', 'covariance': sigcov[triu]})
        cov_rows.append({'time': t, 'type': 'noise', 'covariance': noisecov[triu]})
        ncsnr_rows.append({'time': t, 'ncsnr': ncsnr})
    
    cov_df = pd.DataFrame(cov_rows)
    cov_df['mean_covariance'] = cov_df['covariance'].apply(lambda x: np.nanmean(np.abs(x)))
    ncsnr_df = pd.DataFrame(ncsnr_rows)
    ncsnr_df['mean_ncsnr'] = ncsnr_df['ncsnr'].apply(lambda x: np.nanmean(np.abs(x)))
    
    #### PLOT ###
    customp = ['red', 'black']# sns.color_palette('Dark2')
    fig,ax = plt.subplots(1,1, figsize=(5,3))
    sns.scatterplot(cov_df, x='time', y='mean_covariance', hue='type', 
                    marker='o', palette=customp, alpha=.75, ax=ax)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cov.')
    ax.legend(title='')
    sns.despine(fig=fig, trim=True, offset=5)
    outpath = os.path.join(SAVEDIR, f'session-{session_num}-gsn.png')
    plt.savefig(outpath, dpi=300, transparent=True, bbox_inches='tight')
    
    fig,ax = plt.subplots(1,1, figsize=(5,3))
    sns.lineplot(ncsnr_df, x='time', y='mean_ncsnr', color='gray', ax=ax)
    ax.set_xlabel('Time')
    ax.set_ylabel('NCSNR')
    sns.despine(fig=fig, trim=True, offset=5)
    outpath = os.path.join(SAVEDIR, f'session-{session_num}-ncsnr.png')
    plt.savefig(outpath, dpi=300, transparent=True, bbox_inches='tight')
    print(f'Saved figures for session {session_num}!')
