import os, fsspec, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import manifold_dynamics.session_raster_extraction as sre
from manifold_dynamics.session_gsn import session_gsn
import manifold_dynamics.paths as pth

fs = fsspec.filesystem("s3")

# outdir = os.path.join(pth.SAVEDIR, 'gsn')
outdir = './'
uid_sheet = pd.read_csv(os.path.join(pth.OTHERS, 'roi-uid.csv'))
unique_rois = uid_sheet['uid'].unique()

roi_uid = unique_rois[0]
inpath = os.path.join(pth.PROCESSED, 'single-session-raster', f'{roi_uid}.npy')

### time how long it takes to load via fsspec
size_bytes = fs.size(inpath)
print('starting to load data...')
t0 = time.perf_counter()
with fs.open(inpath, 'rb') as f:
    out = np.load(f, allow_pickle=False)
dt = time.perf_counter() - t0
print(f'size: {size_bytes/1e9:.2f} GB')
print(f'time: {dt:.2f} sec')
print(f'throughput: {(size_bytes/1e6)/dt:.2f} MB/s')
### end time

cov_df, ncsnr_df = session_gsn(out, win=1)

#### PLOT ###
customp = ['red', 'black']# sns.color_palette('Dark2')
fig,ax = plt.subplots(1,1, figsize=(5,3))
sns.scatterplot(cov_df, x='time', y='mean_abs_covariance', hue='type', 
                marker='o', palette=customp, alpha=.75, ax=ax)
ax.set_xlabel('Time')
ax.set_ylabel('Cov.')
ax.legend(title='')
sns.despine(fig=fig, trim=True, offset=5)

# save
outpath = os.path.join(outdir, f'{roi_uid}_covariance.png')
plt.savefig(outdir, dpi=300, transparent=True, bbox_inches='tight')

fig,ax = plt.subplots(1,1, figsize=(5,3))
sns.lineplot(ncsnr_df, x='time', y='mean_abs_ncsnr', color='gray', ax=ax)
ax.set_xlabel('Time')
ax.set_ylabel('NCSNR')
sns.despine(fig=fig, trim=True, offset=5)

# save
outpath = os.path.join(outdir, f'{roi_uid}_ncsnr.png')
plt.savefig(outpath, dpi=300, transparent=True, bbox_inches='tight')
