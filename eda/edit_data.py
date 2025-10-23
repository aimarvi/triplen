import os
import re
from tqdm import tqdm  # optional for progress bar
import numpy as np
import pandas as pd
import scipy.io

import utils as utils

df = pd.read_pickle('../datasets/NNN/all_unit_data.pkl')

datadir = '../datasets/NNN/'
fnames = utils.fnames(datadir)

# add the column name for data you want to add
new_col = ['img_raster'] 
cols = ['session', 'monkey'] + new_col
new_df = pd.DataFrame(columns=cols)

total_units = 0
for i, pair in tqdm(enumerate(fnames)):
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
        proc_data = scipy.io.loadmat(proc_fname)
        gus_data = utils.load_mat(gus_fname)
        
        session_num = int(m.group(1))
        monkey = int(m.group(3))
        unit_types = proc_data['UnitType'][0]
        num_units = len(proc_data['UnitType'][0])
        
        # data to add goes here:
        raster = gus_data['GoodUnitStrc']['Raster']; assert len(raster) == num_units
        trial_idx = gus_data['meta_data']['trial_valid_idx'].squeeze()
        trial_idx = trial_idx[trial_idx!=0]
        
        for unit_idx in range(num_units):
            unit_raster = raster[unit_idx]; assert(unit_raster.shape[0] == 450)# (time_points, 7434)                               
            df_raster = pd.DataFrame(unit_raster.T)  # shape: (trials, timebins)
            df_raster['img'] = trial_idx
            img_avg = df_raster.groupby('img').mean().T.reindex(range(1,1073), axis=1).values  # (450, 1072)
                                            
            new_df.loc[len(new_df)] = {
                'session': session_num,
                'monkey': monkey,
                new_col[0]: img_avg
            }
        total_units += num_units

    except AssertionError as e:
        print(f"Assertion failed for {proc_fname or gus_fname}: {e}")
        continue
    except Exception as e:
        print(f"Error processing {proc_fname or gus_fname}: {e}")
        continue

df[new_col[0]] = new_df[new_col[0]]
df.to_pickle('../datasets/NNN/all_raster_data.pkl')
df
