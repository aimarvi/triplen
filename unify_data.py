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

cols = ['session', 'monkey', 'unit_type', 'avg_psth', 'img_psth', 'avg_firing_rate', 'snr', 'snr_max', 'F_SI', 'B_SI', 'O_SI', 'raster', 'roi']

rows_batch = []
total_units = 0
batch_size_files = 20
batch_num = 0


for i, pair in tqdm(enumerate(fnames), total=len(fnames)):
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
        this_parcel = parcels[parcels['SesIdx']==session_num]

        snr = proc_data['snr'].T.squeeze(); assert snr.shape[0] == num_units
        snr_max = proc_data['snrmax'].T.squeeze(); assert snr_max.shape[0] == num_units
        img_psth_arr = np.stack(gus_data['GoodUnitStrc']['response_matrix_img']); assert img_psth_arr.shape[0] == num_units
        avg_psth_arr = proc_data['mean_psth']; assert avg_psth_arr.shape[0] == num_units
        avg_fr_arr = proc_data['response_basic']; assert avg_fr_arr.shape[0] == num_units
        unit_pos = proc_data['pos'].squeeze(); assert unit_pos.shape[0] == num_units
        
        bsi = proc_data['B_SI'].T.squeeze(); assert bsi.shape[0] == num_units
        osi = proc_data['O_SI'].T.squeeze(); assert osi.shape[0] == num_units
        fsi = proc_data['F_SI'].T.squeeze(); assert fsi.shape[0] == num_units
        
        raster = gus_data['GoodUnitStrc']['Raster']; assert len(raster) == num_units
        trial_idx = gus_data['meta_data']['trial_valid_idx'].squeeze()
        trial_idx = trial_idx[trial_idx!=0]
            
        for unit_idx in range(num_units):
            unit_raster = raster[unit_idx]; assert(unit_raster.shape[0] == 450)
            df_raster = pd.DataFrame(unit_raster.T)
            df_raster['img'] = trial_idx
            img_avg = df_raster.groupby('img').mean().T.reindex(range(1,1073), axis=1).values
            
            upos = unit_pos[unit_idx]
            label = None
            for _, row in this_parcel.iterrows():
                y1 = row['y1']
                y2 = row['y2']
                if (upos > y1) and (upos < y2):
                    label = '_'.join((str(row['AREALABEL']), str(row['RoiIndex']), str(row['Categoty'])))
                    break
            
            rows_batch.append({
                'session': session_num,
                'monkey': monkey,
                'unit_type': unit_types[unit_idx],
                'avg_psth': avg_psth_arr[unit_idx],
                'img_psth': img_psth_arr[unit_idx],
                'avg_firing_rate': avg_fr_arr[unit_idx],
                'snr': snr[unit_idx],
                'snr_max': snr_max[unit_idx],
                'F_SI': fsi[unit_idx],
                'B_SI': bsi[unit_idx],
                'O_SI': osi[unit_idx],
                'raster': img_avg,
                'roi': label,
            })
        total_units += num_units
        
        # --- INTERMEDIATE SAVE ---
        if (i+1) % batch_size_files == 0 or (i+1) == len(fnames):
            filename = f'../datasets/NNN/all_raster_data_batch{batch_num:03d}.pkl'
            pd.DataFrame(rows_batch, columns=cols).to_pickle(filename)
            print(f"Saved {len(rows_batch)} units to {filename} at file {i+1}")
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
df.to_pickle('../datasets/NNN/all_raster_data.pkl')