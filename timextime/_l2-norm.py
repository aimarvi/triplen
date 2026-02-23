import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

import manifold_dynamics.tuning_utils as tut

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
   
    # the 'shuff' mode should really be bootstrapped...
    # but 'shuff' is not used to calculate the optimal k, so here it is just for visualization
    for mode in ['top', 'shuff']:
        rdms = roi_dict[f'{mode}_rdms']
        triu = np.triu_indices_from(rdms[0], k=1)

        ## single time point RDM, or average over previous time step chunk
        # R0 = rdms[step][triu]
        R0 = np.mean(np.array([rdm[triu] for rdm in rdms[0:L2_STEP]]), axis=0) #######################################
        for k in np.arange(1*L2_STEP, len(rdms), L2_STEP):
            prev = R0
            ## same as above
            # R0 = rdms[t-1][triu]
            R0 = np.mean(np.array([rdm[triu] for rdm in rdms[k:k+L2_STEP]]), axis=0) ######################################
    
            ## difference metric for scale K
            ## this is L2
            diff = np.sqrt(np.sum((R0)**2))
    
            diffs.loc[len(diffs)] = {'ROI': _roi, 'Scale': sizes[k-1], 'Derivative': diff, 'Mode': mode}

# apply smoothing function (not necessary)
diffs["diff_smooth"] = diffs["Derivative"].groupby(diffs["Mode"]).transform(
    lambda v: gaussian_filter1d(v, sigma=1)
)

# collect optimal K for each ROI
mins = {}
for r in ROI_LIST:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    d = diffs[diffs['ROI'] == r]

    # main lineplot
    sns.lineplot(data=d, x='Scale', y='Derivative', hue='Mode', alpha=0.5, ax=ax)
    sns.lineplot(data=d, x='Scale', y='diff_smooth', hue='Mode', ax=ax)

    # add red dot + legend label for each Mode separately
    labels  = list(ax.get_legend_handles_labels()[1])

    for i, mode in enumerate(['top', 'shuff']):
        dm = d[(d['Mode'] == mode)]
        idx_min = dm['diff_smooth'].idxmin()
        if np.isnan(idx_min):
            continue
        x_min   = dm.loc[idx_min, 'Scale']
        y_min   = dm.loc[idx_min, 'diff_smooth']

        # draw red dot
        mins[r] = (x_min, y_min)
        h = ax.scatter(x_min, y_min, color='red', s=60, zorder=5)

        # label for legend
        labels[i] = f'{mode} min @ {int(x_min)}'

    # PLOT RESULTS
    ax.legend(ax.get_legend_handles_labels()[0], labels, frameon=False)
    ax.set_title(r)
    plt.tight_layout()
    plt.show()

# save optimal K values for each ROI
SAVE_DIR = os.path.join(DATA_DIR, f'{CATEGORY}_mins.pkl')
with open(SAVE_DIR, 'wb') as f:
    pickle.dump(mins, f)
