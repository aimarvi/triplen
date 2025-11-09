import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import imageio.v2 as iio

'''
same as _plot.py, but computes multiple ROIs and plots them all together (TOP & SHUFF)
also forgot to mention in other script:
    --> data is filtered for p_val <0.05
'''

def rdm_sequence(dat, ROI, mode='top', step=10, k_max=200, metric='correlation',
                 onset=50, resp=(50,220), base=(-50,0), random_state=0):
    rng = np.random.default_rng(random_state)
    ONSET = onset
    RESP = slice(ONSET + resp[0], ONSET + resp[1])
    BASE = slice(ONSET + base[0], ONSET + base[1])

    sig = dat[dat['p_value'] < 0.05]
    df = sig[sig['roi'] == ROI]
    if len(df) == 0:
        raise ValueError(f"No data for ROI {ROI}")
    X = np.stack(df['img_psth'].to_numpy())          # (units, time, images)

    scores = np.nanmean(X[:, RESP, :], axis=(0,1)) - np.nanmean(X[:, BASE, :], axis=(0,1))
    order = np.argsort(scores)[::-1] if mode == 'top' else rng.permutation(scores.size)

    # choose the image-set bins to calculate RDMs
    # sizes = [k for k in range(step, min(k_max, X.shape[2]) + 1, step)]
    sizes = [k for k in range(1, 2*step)] + [k for k in range(2*step, min(k_max, X.shape[2])+1, step)]
    rdms = []
    for k in sizes:
        idx = order[:k]
        Xavg = np.nanmean(X[:, :, idx], axis=0)      # (time, k)
        R = squareform(pdist(Xavg, metric=metric))   # (time, time)
        rdms.append(R)
    return sizes, rdms

def rdm_to_image(R, vmin, vmax, dpi=120, cmap='magma'):
    fig, ax = plt.subplots(figsize=(3,3), dpi=dpi)
    sns.heatmap(R, ax=ax, cmap=sns.color_palette(cmap, as_cmap=True),
                vmin=vmin, vmax=vmax, square=True, cbar=False,
                xticklabels=False, yticklabels=False)
    ax.set_xlabel(""); ax.set_ylabel(""); fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(img)

def build_grid_gif(dat, ROI_LIST, step=10, k_max=200, metric='correlation',
                   out_path='rdm_grid.gif', random_state=0):
    # compute sequences for all ROIs (top & shuffle)
    seqs_top, seqs_shuf = {}, {}
    for roi in tqdm(ROI_LIST):
        sizes, rdms_top = rdm_sequence(dat, roi, mode='top', step=step, k_max=k_max,
                                       metric=metric, random_state=random_state)
        _,     rdms_shf = rdm_sequence(dat, roi, mode='shuffle', step=step, k_max=k_max,
                                       metric=metric, random_state=random_state)
        seqs_top[roi], seqs_shuf[roi] = rdms_top, rdms_shf

    # global color limits for consistency
    vmax = np.nanmax([np.nanmax(R) for v in (*seqs_top.values(), *seqs_shuf.values()) for R in v])
    vmin = 0.0

    T = len(next(iter(seqs_top.values())))  # number of frames (same step schedule)
    frames = []
    for t in range(T):
        top_tiles  = [rdm_to_image(seqs_top[roi][t],  vmin, vmax)  for roi in ROI_LIST]
        shuf_tiles = [rdm_to_image(seqs_shuf[roi][t], vmin, vmax)  for roi in ROI_LIST]
    
        W, H = top_tiles[0].size
        row_top  = Image.new('RGB', (W*len(ROI_LIST), H))
        row_shuf = Image.new('RGB', (W*len(ROI_LIST), H))
        for j, img in enumerate(top_tiles):  row_top.paste(img,  (j*W, 0))
        for j, img in enumerate(shuf_tiles): row_shuf.paste(img, (j*W, 0))
    
        grid = Image.new('RGB', (W*len(ROI_LIST), H*2 + 40), color=(255,255,255))
        grid.paste(row_top, (0, 20))
        grid.paste(row_shuf, (0, H+20))
        draw = ImageDraw.Draw(grid)
    
        # ROI names
        for j, roi in enumerate(ROI_LIST):
            xmid = j*W + W//2
            draw.text((xmid, 10), roi, fill="black", anchor='mt', font_size=18)
    
        # image IDs (the subset size used)
        used_k = sizes[t]
        draw.text((grid.width//2, 2*H+30),
                  f"Images used: top {used_k}",  anchor='mb', fill="black", font_size=18)
    
        frames.append(grid)

    iio.mimsave(out_path, frames, duration=2, loop=0)
    return out_path

# ============ USAGE ================
DATA_DIR = '../../datasets/NNN/face_roi_data.pkl'
dat = pd.read_pickle(DATA_DIR)
ROI_LIST = ['MF1_8_F']
# ROI_LIST = ['AO5_25_O',
#  'Unknown_6_O',
#  'PITP4_10_O',
#  'Unknown_4_O',
#  'MO1s1_4_O',]

label = 'test4'
SAVE_PATH = f'/Users/aim/Desktop/HVRD/workspace/dynamics/gifs/{label}_ramp.gif'
out = build_grid_gif(dat, ROI_LIST, step=20, k_max=750, metric='correlation', out_path=SAVE_PATH)
print("Saved:", out)
