import os
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.io

import utils as utils

datadir = '../../datasets/NNN/'
SAVE_DIR = '../../datasets/NNN/'
fnames = utils.fnames(datadir)
parcels = pd.read_excel('../demo/exclude_area.xls')

cols = ['session', 'monkey', 'roi', 'raster']

rows_batch = []
total_units = 0
batch_size_files = 10
batch_num = 0

REFDIR = '../../datasets/NNN/unit_data_full.pkl'
ref = pd.read_pickle(REFDIR)
ref = ref[~(ref['session'] == 29)]
skip_count = 0
total_units = 0

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
            ref_df = ref.iloc[total_units]
            total_units += 1
            print(ref_df['roi'])
            if (ref_df['p_value'] > 0.5) or (ref_df['roi'] is None) or ('F' not in ref_df['roi'].split('_')[-1]):
                skip_count += 1
                # print(f'Skipping file {proc_fname}')
                continue
            print('not skipped')
            unit_raster = raster[unit_idx]  # (450, n_trials)
            time_points = unit_raster.shape[0]
            upos = unit_pos[unit_idx]
            label = None
            for _, row in this_parcel.iterrows():
                y1 = row['y1']
                y2 = row['y2']
                if (upos > y1) and (upos < y2):
                    label = '_'.join((str(row['AREALABEL']), str(row['RoiIndex']), str(row['Categoty'])))
                    break
                    
            # For each image: mean firing rate for this window, across reps
            vals = np.full((time_points, num_imgs, max_reps), np.nan, dtype=float) # size should be (450, 1072, 7))
            for img in range(1, num_imgs+1):
                trial_idxs = np.where(trial_idx == img)[0]
                nan_raster = np.full((time_points, max_reps), np.nan, dtype=float)
                if len(trial_idxs) > 0:
                    # all raster data for a single image (multiple reps)
                    data = unit_raster[:, trial_idxs] # data is shape (450, n_reps)
                    nan_raster[:, :data.shape[1]] = data
                        
                vals[:, img-1, :] = nan_raster

            img_firing = {
                'session': session_num,
                'monkey': monkey,
                'roi': label,
                'raster': vals,
            }
            rows_batch.append(img_firing)
            
        # --- INTERMEDIATE SAVE ---
        if (len(rows_batch)>0) and ((i+1) % batch_size_files == 0 or (i+1) == len(fnames)):
            filename = os.path.join(SAVE_DIR, f'raw_raster_data_batch{batch_num:03d}.pkl')
            pd.DataFrame(rows_batch, columns=cols).to_pickle(filename)
            print(f"Saved {len(rows_batch)} units to {filename} at file {i+1}")
            print(f'Skipped {skip_count} units')
            rows_batch = []  # clear for the next batch
            batch_num += 1

    except AssertionError as e:
        print(f"Assertion failed for {proc_fname or gus_fname}: {e}")
        continue
    except Exception as e:
        print(f"Error processing {proc_fname or gus_fname}: {e}")
        continue
          
print(f"successfully loaded all units\ntotal units: {total_units}")

df = pd.DataFrame(rows_batch, columns=cols)
df.to_pickle(os.path.join(SAVE_DIR, 'trial_raster_data.pkl'))
