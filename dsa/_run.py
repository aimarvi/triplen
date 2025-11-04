import velocity as v
import os
import pandas as pd
import numpy as np

DATA_DIR = '../../datasets/NNN/'
dat = pd.read_pickle(os.path.join(DATA_DIR, ('face_roi_data.pkl')))

ROI = 'MF1_8_F'
PVAL = 0.05

roi_dat = dat[(dat['roi']==ROI) & (dat['p_value']<PVAL)].reset_index(drop=True)
# units, time points, images
X = np.stack(roi_dat['img_psth']) 

img_sets = {'all images': np.arange(1000,1072), 
           'all faces': np.arange(1000,1024),
           'monkey faces':  np.concatenate([np.arange(1000,1006), np.arange(1009,1016)]),
           'human faces': np.concatenate([np.arange(1006,1009), np.arange(1016,1025)]),
           'all nonfaces': np.arange(1025,1072),
            'all objects': np.setdiff1d(np.arange(1000, 1072), np.concatenate([np.arange(1000,1024), np.arange(1025,1031), np.arange(1043,1049), np.arange(1051,1062)])),
           'monkey bodies': np.concatenate([np.arange(1026,1031), np.arange(1043,1049)]),
            'animal bodies': np.concatenate([np.arange(1026,1031), np.arange(1043,1049), np.arange(1051,1062)]),
           }

# choose 20 ms bins if that matches your pre-processing
out = v.run_velocity_analysis(
    X, img_sets,
    rank=True,          # Spearman RDM
    use_repeats=False,  # set True if X has per-trial repeats and you want cross-validation
    delta=1,            # 1 bin step (e.g., 20 ms)
    bin_ms=1,
    smooth_plot=True,
    n_shuff_perm=10,
    n_shuff_shift=5,
    alpha=0.05
)

