import os, sys, re
from datetime import datetime

import numpy as np
import pandas as pd

import manifold_dynamics.utils_standard as sut

DATADIR = './../../../datasets/NNN'

def process_session(session_number):
    '''
    Return the raw-est form of the data
    Data matrix of (units x time x stimuli x trials) for a single session
    
    '''
    if (session_number < 1) or (session_number > 90):
        print(f'===================== Could not process session {session_number} ==========================\nPlease provide a valid session number (1 - 90).')

    # all_fnames is sorted by session number, 0 indexed
    all_fnames = sut.fnames(DATADIR)
    session_fnames = all_fnames[session_number-1]
    goodunit, processed = session_fnames[0], session_fnames[1]

    goodunit_data = sut.load_mat(os.path.join(DATADIR, goodunit))
    raster_raw = goodunit_data['GoodUnitStrc']['Raster']

    # all valid trials
    trial_idx = goodunit_data['meta_data']['trial_valid_idx'].squeeze()
    trial_idx = trial_idx[trial_idx!=0]

    # find the maximum number of times an image was repeated
    num_imgs = len(np.unique(trial_idx))
    counts = [np.sum(trial_idx == img) for img in range(1, num_imgs+1)]
    max_reps = np.max(counts)

    units = []
    for uid, unit in enumerate(raster_raw):
        time_points = unit.shape[0]  # (450, n_trials)
        vals = np.full((time_points, num_imgs, max_reps), np.nan, dtype=float)
    
        # create the time x rep matrix for each image
        for img in range(1, num_imgs + 1):
            trial_idxs = np.where(trial_idx == img)[0]
            nan_img_raster = np.full((time_points, max_reps), np.nan, dtype=float)
    
            if len(trial_idxs) > 0:
                # all raster data for a single image (multiple reps)
                data = unit[:, trial_idxs]  # (450, n_reps)
                nan_img_raster[:, :data.shape[1]] = data
    
            vals[:, img - 1, :] = nan_img_raster
        # end for
        units.append(vals)
    # end for
    raster = np.array(units)

    return raster

process_session(9)
