import os, sys
if './..' not in sys.path:
    sys.path.insert(0, './..')

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as iio

from tqdm import tqdm

import utils_txt as tut
from _grid import rdm_to_image, make_colorbar_image
font = ImageFont.load_default(size=32)

def rdm_grid(dat, ROI_LIST, WIN=120, STEP=5, START=0, END=450, 
    metric='correlation', out_path='rdm_grid.gif', 
    random_state=0):

    seqs_top = {}
    for roi in tqdm(ROI_LIST):
        roi_Rs = []
        windows = [(t, t + WIN) for t in range(START, END-WIN, STEP)]
        for t0, t1 in windows:
            R, Xrdv = tut.time_avg_rdm(dat, roi=roi, images='localizer', window=(t0, t1))
            np.fill_diagonal(R, np.nan)
            roi_Rs.append(R)
        roi_Rs = np.array(roi_Rs)
        seqs_top[roi] = roi_Rs
    vmax = np.nanmax([np.nanmax(R) for v in seqs_top.values() for R in v])
    vmin = np.nanmin([np.nanmin(R) for v in seqs_top.values() for R in v])

    T = len(next(iter(seqs_top.values())))
    frames = []

    print(vmax, vmin, T)

    cmap = 'PiYG'
    first_top  = rdm_to_image(next(iter(seqs_top.values()))[0], vmin, vmax)
    H = first_top.size[1]
    cb_img = make_colorbar_image(vmin, vmax, cmap=cmap, height=H+20)

    for t in range(T):
        top_tiles = [rdm_to_image(seqs_top[roi][t], vmin, vmax, cmap=cmap)
                     for roi in ROI_LIST]
    
        W, H = top_tiles[0].size
        row_top = Image.new('RGB', (W * len(ROI_LIST), H))
    
        for j, img in enumerate(top_tiles):
            row_top.paste(img, (j * W, 0))
    
        # add vertical space for titles
        title_h = 10
        grid = Image.new('RGB', (W * len(ROI_LIST), H + title_h), color=(255, 255, 255))
        grid.paste(row_top, (0, title_h))
    
        draw = ImageDraw.Draw(grid)
    
        # ---- frame-level title (centered) ----
        frame_title = f'time {t}'   # or whatever you want
        draw.text(
            (grid.width // 2, 10),
            frame_title,
            fill=(0, 0, 0),
            anchor='ma',   # middle / above
            font=font,
        )
    
        # # ---- per-ROI labels ----
        # for j, roi in enumerate(ROI_LIST):
        #     xmid = j * W + W // 2
        #     draw.text((xmid, title_h - 5), roi, fill=(0, 0, 0), anchor='ma')
    
        # colorbar + canvas
        # cb_w, cb_h = cb_img.size
        # canvas = Image.new('RGB', (grid.width + cb_w + 20, grid.height), color=(255, 255, 255))
        # canvas.paste(grid, (0, 0))
        # y0 = (canvas.height - cb_h) // 2
        # canvas.paste(cb_img, (grid.width + 10, y0))

        # frames.append(canvas)
        frames.append(grid)
    
    iio.mimsave(out_path, frames, duration=2, loop=0)

    return out_path

if __name__ == "__main__":
    # ---- Paths ----
    CAT = "face"
    DATA_DIR = "./../../datasets/NNN/"
    SAVE_DIR = "./../../../buckets/manifold-dynamics/time-averaged/"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # ---- Load data ----
    dat = pd.read_pickle(os.path.join(DATA_DIR, f"{CAT}_roi_data.pkl"))
    print(f"Unique {CAT} ROIs: {list(dat['roi'].unique())}")

    # ---- Choose ROI(s) ----
    ROI_LIST = ["Unknown_19_F"]  # or a list: ["Unknown_19_F", "MF1_7_F"]
    out = os.path.join(SAVE_DIR, f"test.gif")
    out_path = rdm_grid(dat, ROI_LIST, out_path=out)
    print("Saved:", out_path)

