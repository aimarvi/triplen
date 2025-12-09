import os
import pandas as pd

DATA_DIR = '../../datasets/NNN/'
CATEGORY = 'face'

# load in ROI data for a single category
# reduces wait time since full dataset is so large
dat = pd.read_pickle(os.path.join(DATA_DIR, f'{CATEGORY}_roi_data.pkl'))
print(f'Unique {CATEGORY} ROIs: {list(dat['roi'].unique())}')

ROI_LIST = list(dat['roi'].unique())


