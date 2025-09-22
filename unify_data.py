import os
import re
from tqdm import tqdm  # optional for progress bar
import numpy as np
import pandas as pd
import scipy.io

import utils as utils

datadir = '../datasets/NNN/'
fnames = utils.fnames(datadir)

cols = ['session', 'monkey', 'unit_type', 'avg_psth', 'img_psth', 'avg_firing_rate', 'snr']
df = pd.DataFrame(columns=cols)

total_units = 0
for i, pair in tqdm(enumerate(fnames)):
    gus_fname = os.path.join(datadir, pair[0])
    proc_fname = os.path.join(datadir, pair[1])
    m = re.match(r'Processed_ses(\d+)_(\d{6})_M(\d+)_(\d+)\.mat', os.path.basename(proc_fname))
    if not m:
        print(f"Could not parse {proc_fname}")
        continue
    try:
        gus_data = utils.load_mat(gus_fname)
        proc_data = scipy.io.loadmat(proc_fname)
        
        session_num = int(m.group(1))
        monkey = int(m.group(3))
        unit_types = proc_data['UnitType'][0]
        num_units = len(proc_data['UnitType'][0])

        snr = proc_data['snr'].T.squeeze(); assert snr.shape[0] == num_units
        img_psth_arr = np.stack(gus_data['GoodUnitStrc']['response_matrix_img']); assert img_psth_arr.shape[0] == num_units
        avg_psth_arr = proc_data['mean_psth']; assert avg_psth_arr.shape[0] == num_units
        avg_fr_arr = proc_data['response_basic']; assert avg_fr_arr.shape[0] == num_units
            
        for unit_idx in range(num_units):
            df.loc[len(df)] = {
                'session': session_num,
                'monkey': monkey,
                'unit_type': unit_types[unit_idx],
                'avg_psth': avg_psth_arr[unit_idx],
                'img_psth': img_psth_arr[unit_idx],
                'avg_firing_rate': avg_fr_arr[unit_idx],
                'snr': snr[unit_idx]
            }
        total_units += num_units

    except AssertionError as e:
        print(f"Assertion failed for {proc_fname or gus_fname}: {e}")
        continue
    except Exception as e:
        print(f"Error processing {proc_fname or gus_fname}: {e}")
        continue

print(f"successfully loaded all units\ntotal units: {total_units}")
df.to_pickle('../datasets/NNN/all_unit_data.pkl')
