import os, sys, re
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import io
from tqdm import tqdm

import manifold_dynamics.utils_standard as sut
import manifold_dynamics.PATHS as PTH

def process_session(roi_uid, verbose=False):
    '''
    Return the raw-est form of the data
    Data matrix of (units x time x stimuli x trials) for a single session
    
    '''
    parsed = roi_uid.split('.')
    session_number = int(parsed[0])
    roi = parsed[2]
    category = parsed[3]

    if not (1 <= session_number <= 90):
        raise ValueError(
            f'Could not process session {session_number}. '
            'Please provide a valid session number (1–90).'
        )
    # all_fnames is sorted by session number, 0 indexed
    all_fnames = sut.fnames(PTH.RAW, PTH.PROCESSED)
    session_fnames = all_fnames[session_number-1]
    goodunit, processed = session_fnames[0], session_fnames[1]

    # load in the relevant data
    goodunit_data = sut.load_mat(os.path.join(PTH.RAW, goodunit), fformat='v7.3', verbose=verbose)
    processed_data = sut.load_mat(os.path.join(PTH.PROCESSED, processed), fformat='v5', verbose=verbose)
    unique_id_data = pd.read_csv(os.path.join(PTH.OTHERS, 'roi-uid.csv'))

    raster_raw = goodunit_data['GoodUnitStrc']['Raster']
    unit_positions = processed_data['pos'].squeeze()
    row = unique_id_data.loc[unique_id_data['uid'] == roi_uid]
    y1 = row['y1'].iloc[0]; y2 = row['y2'].iloc[0]

    # all valid trials
    trial_idx = goodunit_data['meta_data']['trial_valid_idx'].squeeze()
    trial_idx = trial_idx[trial_idx!=0]

    # find the maximum number of times an image was repeated
    num_imgs = len(np.unique(trial_idx))
    counts = [np.sum(trial_idx == img) for img in range(1, num_imgs+1)]
    max_reps = np.max(counts)

    units = []
    for uid, unit in tqdm(enumerate(raster_raw), disable=not verbose):
        # only take units inside the pre-defined area
        this_position = unit_positions[uid]
        if (this_position <= y1) or (this_position >= y2):
            continue

        # pre-make the (time, image, reps) raster
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

if __name__ == '__main__':
    print('starting first session...')
    out = process_session('18.19.Unknown.F')
    print('18.19.Unknown.F DONE:', out.shape)

    print('starting second session...')
    out = process_session('20.19.Unknown.F')
    print('20.19.Unknown.F DONE:', out.shape)
