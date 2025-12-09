import os
import pandas as pd
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
STEP = 1 # choose 1 so you can change it later on

# load in ROI data for a single category
# reduces wait time since full dataset is so large
dat = pd.read_pickle(os.path.join(DATA_DIR, f'{CATEGORY}_roi_data.pkl'))
print(f'Unique {CATEGORY} ROIs: {list(dat['roi'].unique())}')

ROI_LIST = list(dat['roi'].unique())
ROI_LIST = ['MF1_7_F']

# build cache with all 4 fields per roi
cache = {}
for _roi in ROI_LIST:
    cache[_roi] = {}  # init once per roi
    for MODE in ['top', 'shuff']:
        sizes, rdms = tut.geo_rdm(dat, roi=_roi, mode=MODE, step=STEP)
        cache[_roi][f'sizes_{MODE}'] = sizes
        cache[_roi][f'{MODE}_rdms'] = rdms

# flatten into a dataframe
df = pd.DataFrame([
    {'ROI': roi, **vals}
    for roi, vals in cache.items()
])
