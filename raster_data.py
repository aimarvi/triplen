import os
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.io

import utils as utils

datadir = '../datasets/NNN/'
fnames = utils.fnames(datadir)
parcels = pd.read_excel('./Demo/exclude_area.xls')

cols = ['session', 'monkey', 'roi', 'pre', 'early', 'late']
window_names = ['pre', 'early', 'late']
windows = [
    (25, 80),    # -25 to 30 ms (indices 25 to 80 inclusive, Python slice is exclusive at end)
    (100, 170),  # 50 to 120 ms
    (170, 240)   # 120 to 240 ms
]

rows_batch = []
total_units = 0
batch_size_files = 10
batch_num = 0


for i, pair in tqdm(enumerate(fnames), total=len(fnames)):
    gus_fname = os.path.join(datadir, pair[0])
    proc_fname = os.path.join(datadir, pair[1])
    m = re.match(r'Processed_ses(\d+)_(\d{6})_M(\d+)_(\d+)\.mat', os.path.basename(proc_fname))
    if i == 28: # this file has inconsistent unit numbers
        print(f'skipping {proc_fname}...')
        continue
    if not m:
        print(f"Could not parse {proc_fname}")
        continue
    try:
        gus_data = utils.load_mat(gus_fname)
        proc_data = scipy.io.loadmat(proc_fname)
        
        session_num = int(m.group(1))
        monkey = int(m.group(3))
        
        num_units = len(proc_data['UnitType'][0])
        this_parcel = parcels[parcels['SesIdx']==session_num]
        
        
        raster = gus_data['GoodUnitStrc']['Raster']; assert len(raster) == num_units  # list, len=num_units; each: (450, n_trials)
        unit_pos = proc_data['pos'].squeeze(); assert unit_pos.shape[0] == num_units
        
        trial_idx = gus_data['meta_data']['trial_valid_idx'].squeeze()
        trial_idx = trial_idx[trial_idx!=0]

        num_imgs = len(np.unique(trial_idx))
        counts = [np.sum(trial_idx == img) for img in range(1, num_imgs+1)]
        max_reps = np.max(counts)

        for unit_idx in range(num_units):
            unit_raster = raster[unit_idx]  # (450, n_trials)
            upos = unit_pos[unit_idx]
            label = None
            for _, row in this_parcel.iterrows():
                y1 = row['y1']
                y2 = row['y2']
                if (upos > y1) and (upos < y2):
                    label = '_'.join((str(row['AREALABEL']), str(row['RoiIndex']), str(row['Categoty'])))
                    break
                    
            img_firing = {
                'session': session_num,
                'monkey': monkey,
                'roi': label,
            }
            for (start, end), name in zip(windows, window_names):
                # For each image: mean firing rate for this window, across reps
                vals = np.full((num_imgs, max_reps), np.nan, dtype=float)
                for img in range(1, num_imgs+1):
                    trial_idxs = np.where(trial_idx == img)[0]
                    mean_val = np.full((max_reps), np.nan, dtype=float)
                    if len(trial_idxs) > 0:
                        data = unit_raster[start:end+1, trial_idxs]  # (window_length, n_reps)
                        avg_fr = np.nanmean(data, axis=0) # mean over time, not reps; shape (n_reps,)
                        mean_val[:len(avg_fr)] = avg_fr  
                    vals[img-1] = mean_val
                img_firing[f"{name}"] = vals  # This is a list of length num_imgs per window
            rows_batch.append(img_firing)
            
        # # --- INTERMEDIATE SAVE ---
        # if (i+1) % batch_size_files == 0 or (i+1) == len(fnames):
        #     filename = f'../datasets/NNN/only_raster_data_batch{batch_num:03d}.pkl'
        #     pd.DataFrame(rows_batch, columns=cols).to_pickle(filename)
        #     print(f"Saved {len(rows_batch)} units to {filename} at file {i+1}")
        #     rows_batch = []  # clear for the next batch
        #     batch_num += 1

    except AssertionError as e:
        print(f"Assertion failed for {proc_fname or gus_fname}: {e}")
        continue
    except Exception as e:
        print(f"Error processing {proc_fname or gus_fname}: {e}")
        continue
          
print(f"successfully loaded all units\ntotal units: {total_units}")

df = pd.DataFrame(rows_batch, columns=cols)
df.to_pickle('../datasets/NNN/fr_data.pkl')