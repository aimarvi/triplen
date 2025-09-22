import os
import re
from tqdm import tqdm  # optional for progress bar
import numpy as np
import pandas as pd
import scipy.io

import utils as utils

df = pd.read_pickle('../datasets/NNN/all_unit_data')

datadir = '../datasets/NNN/'
fnames = utils.fnames(datadir)

# add the column name for data you want to add
new_col = ['...'] 
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
        
        session_num = int(m.group(1))
        monkey = int(m.group(3))
        unit_types = proc_data['UnitType'][0]
        num_units = len(proc_data['UnitType'][0])

        # data to add goes here:
        new_dat = proc_data['snrmax'].T.squeeze(); assert new_dat.shape[0] == num_units
        
        for unit_idx in range(num_units):
            df.loc[len(df)] = {
                'session': session_num,
                'monkey': monkey,
                new_col: snr_max[unit_idx]
            }
        total_units += num_units

    except AssertionError as e:
        print(f"Assertion failed for {proc_fname or gus_fname}: {e}")
        continue
    except Exception as e:
        print(f"Error processing {proc_fname or gus_fname}: {e}")
        continue

df[new_col] = new_df[new_col]
# df.to_pickle('../datasets/NNN/all_unit_data.pkl')
