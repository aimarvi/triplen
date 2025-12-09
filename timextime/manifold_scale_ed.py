import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

import tixti_utils as tut

# some CONFIG parameters
RAND = 0
RESP = (50,220)
BASE = (-50,0)
ONSET = 50
RESP = slice(ONSET + RESP[0], ONSET + RESP[1])
BASE = slice(ONSET + BASE[0], ONSET + BASE[1])

DATA_DIR = '../../datasets/NNN/'
CATEGORY = 'face'

dat = pd.read_pickle(os.path.join(DATA_DIR, (f'{CATEGORY}_roi_data.pkl')))
print(f'Unique {CATEGORY} ROIs: {list(dat['roi'].unique())}')

# load in optimal manifold scales (see l2_norm.py)
with open(os.path.join(DATA_DIR, f'{CATEGORY}_mins.pkl'), 'rb') as f:
    mins = pickle.load(f)

all_rows = []
for ROI, SC in mins.items():
    scale = SC[0]
    print(f'================= starting ED analysis for ROI {ROI} with manifold scale {scale} ====================')

    t0, window = 100, 200
    common_kwargs = dict(dat=dat, roi=ROI, tstart=t0, tend=t0 + window)

    # method, mode, scale, n_bootstraps
    configs = [
        ('local',  'top',   scale,  1),
        ('global', 'top',   1072,   1),
        ('shuf',   'shuff', scale,  500),
    ]

    for method, mode, scl, n_boot in configs:
        if method == 'shuf':
            Rss = []
            for i in tqdm(range(n_boot), desc=f'{ROI} shuf', leave=False):
                R, _ = tut.static_rdm(mode=mode, scale=scl,
                                      random_state=i, **common_kwargs)
                Rss.append(R)
            try:
                eds = [tut.ED2(R) for R in Rss]
            except Exception:
                eds = [np.nan] * len(Rss)
                print(f'eigenvalues do not converge for ROI: {ROI}')

            for ed in eds:
                all_rows.append({
                    'ROI': ROI,
                    'Method': method,
                    'Scale': scl,
                    'ED': ed,
                })
        else:
            R, _ = tut.static_rdm(mode=mode, scale=scl, **common_kwargs)
            ed = tut.ED2(R)
            all_rows.append({
                'ROI': ROI,
                'Method': method,
                'Scale': scl,
                'ED': ed,
            })

df = pd.DataFrame(all_rows)
print(df)

SAVE_DIR = '../../datasets/NNN/'
df.to_pickle(os.path.join(SAVE_DIR, f'{CATEGORY}_ed.pkl'))
