import os
import re
import h5py
import numpy as np
import pandas as pd
import scipy.io

from tqdm import tqdm

import utils

datadir = '../datasets/NNN/'
fnames = utils.fnames(datadir)

cols = ['session', 'monkey', 'unit_types', 'avg_psth', 'img_psth', 'avg_firing_rate', 'snr']
df = pd.DataFrame(columns=cols)

total_units = 0
for i, pair in tqdm(enumerate(fnames)):
    gus_fname = os.path.join(datadir, pair[0])
    proc_fname = os.path.join(datadir, pair[1])
    m = re.match(r'Processed_ses(\d+)_(\d{6})_M(\d+)_(\d+)\.mat', proc_fname.split('/')[-1])
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
        img_psth = np.stack(gus_data['GoodUnitStrc']['response_matrix_img']); assert img_psth.shape[0] == num_units
        avg_psth = proc_data['mean_psth']; assert avg_psth.shape[0] == num_units
        avg_fr = proc_data['response_basic']; assert avg_fr.shape[0] == num_units
            
        total_units += num_units

        df.loc[len(df)] = {
            'session': session_num,
            'monkey': monkey,
            'unit_types': unit_types,
            'avg_psth': avg_psth,
            'img_psth': img_psth,
            'avg_firing_rate': avg_fr,
            'snr': snr
        }
    except AssertionError as e:
        print(f"Assertion failed for {proc_fname or gus_fname}: {e}")
        continue
    except Exception as e:
        print(f"Error processing {proc_fname or gus_fname}: {e}")
        continue

print(f'successfully loaded all data\ntotal units: {total_units}')
df.to_pickle('../datasets/NNN/all_data.pkl')
