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
STEP = 1 # choose 1 so you can change it later on. see L2_STEP later on

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

# flatten into a dataframe to save out
df = pd.DataFrame([
    {'ROI': roi, **vals}
    for roi, vals in cache.items()
])

# step size of manifold scale used to calculate the L2 norm between two Time x Time RDMs
# L2_STEP >= STEP
L2_STEP = 5
cols = ['ROI', 'Scale', 'Derivative', 'Mode']
diffs = pd.DataFrame(columns=cols)
for _roi in ROI_LIST:
    roi_dict = cache[_roi]
    sizes = roi_dict['sizes_top']
   
    for mode in ['top', 'shuff']:
        rdms = roi_dict[f'{mode}_rdms']
        triu = np.triu_indices_from(rdms[0], k=1)

        ## single time point RDM, or average over previous time step chunk
        # R0 = rdms[step][triu]
        R0 = np.mean(np.array([rdm[triu] for rdm in rdms[0:step]]), axis=0) #######################################
        for k in np.arange(1*step, len(rdms), step):
            prev = R0
            ## same as above
            # R0 = rdms[t-1][triu]
            R0 = np.mean(np.array([rdm[triu] for rdm in rdms[k:k+step]]), axis=0) ######################################
    
            ## difference metric for scale K
            ## this is L2
            diff = np.sqrt(np.sum((R0)**2))
    
            diffs.loc[len(diffs)] = {'ROI': _roi, 'Scale': sizes[k-1], 'Derivative': diff, 'Mode': mode}
