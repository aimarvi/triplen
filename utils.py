import re
import os
import h5py
import numpy as np


def fnames(datadir='/Users/aim/Desktop/HVRD/workspace/dynamics/datasets/NNN/'):
    gus_fnames = [f for f in os.listdir(datadir) if re.match(r'^GoodUnit.*\.mat$', f)]
    proc_fnames = [f for f in os.listdir(datadir) if re.match(r'^Processed.*\.mat$', f)]

    gus_keys = []
    for fn in gus_fnames:
        m = re.match(r'GoodUnit_(\d{6}).*_g(\d+)\.mat', fn)
        if m:
            code = m.group(1)
            idx  = m.group(2)
            gus_keys.append((code, idx))
        else:
            gus_keys.append(None)

    proc_map = {}
    for fn in proc_fnames:
        m = re.match(r'Processed_ses\d+_(\d{6})_.*_(\d+)\.mat', fn)
        if m:
            code = m.group(1)
            idx  = m.group(2)
            proc_map[(code, idx)] = fn

    # Helper function to extract ses number from proc_fname
    def extract_ses(proc_fname):
        m = re.search(r'Processed_ses(\d+)_', proc_fname)
        return int(m.group(1)) if m else -1

    # Build a sorted list of (goodunit_fname, proc_fname) pairs in order of ses
    paired_files = sorted(
        [
            (g, proc_map[k]) 
            for g, k in zip(gus_fnames, gus_keys) 
            if k in proc_map
        ], 
        key=lambda pair: extract_ses(pair[1])
    )

    return paired_files

def mat_struct_to_dict(f, obj, verbose=False):
    """
    Recursively convert HDF5 Matlab structure/group or dataset to nested dicts and numpy arrays.
    Automatically dereferences cell arrays and groups.
    """
    result = {}
    for key in obj.keys():
        item = obj[key]
        # If subgroup, recurse
        if isinstance(item, h5py.Group):
            result[key] = mat_struct_to_dict(f, item)
        # If dataset, check if cell array (object refs) or regular array
        elif isinstance(item, h5py.Dataset):
            # Cell arrays: h5py Datasets where dtype == object or ref
            if item.dtype == 'O' or h5py.check_dtype(ref=item.dtype) is not None:
                cell_data = []
                item = np.squeeze(item)  # optional; you can keep this or not
                for idx in np.ndindex(item.shape):
                    ref = item[idx]
                    if isinstance(ref, (np.ndarray,)):
                        ref = ref.item()
                    if isinstance(f[ref], h5py.Group):
                        arr = mat_struct_to_dict(f, f[ref])
                    else:
                        arr = f[ref][()]
                    cell_data.append(arr)
                try:
                    cell_data = np.array(cell_data, dtype=object).reshape(item.shape)
                except Exception as e:
                    if verbose:
                        print(f'{key} data is ragged or shape is weird: {e}. Returning as a list of arrays.')
                result[key] = cell_data
            else:
                # Standard numeric array (possibly needs squeezing)
                result[key] = item[()]
    return result

def load_mat(filename, verbose=False):
    """
    Load a Matlab -v7.3 (HDF5-format) .mat file, dereference cell arrays, and
    return a dict of numpy arrays and nested dicts (for group/structs).
    """
    with h5py.File(filename, 'r') as f:
        data = {}
        for key in f.keys():
            obj = f[key]
            if key.startswith('#'):
                continue
            if isinstance(obj, h5py.Group):
                data[key] = mat_struct_to_dict(f, obj, verbose)
            elif isinstance(obj, h5py.Dataset):
                # Top-level cell array or dataset
                if obj.dtype == 'O' or h5py.check_dtype(ref=obj.dtype) is not None:
                    cell_data = []
                    indices = np.ndindex(obj.shape)
                    for idx in indices:
                        ref = obj[idx]
                        if isinstance(ref, (np.ndarray,)):
                            ref = ref.item()
                        arr = f[ref][()]
                        cell_data.append(arr)
                    cell_data = np.array(cell_data, dtype=object).reshape(obj.shape)
                    data[key] = cell_data
                else:
                    data[key] = obj[()]
        if verbose:
            print(f'successfully loaded data with keys {data.keys()}')
        return data
    
def compute_noise_ceiling(data_in):
    """
    Compute the noise ceiling signal-to-noise ratio (SNR) and percentage noise ceiling for each unit.
    
    Parameters:
    ----------
    data_in : np.ndarray
        A 3D array of shape (units/voxels, conditions, trials), representing the data for which to compute 
        the noise ceiling. Each unit requires more than 1 trial for each condition.

    Returns:
    -------
    noiseceiling : np.ndarray
        The noise ceiling for each unit, expressed as a percentage.
    ncsnr : np.ndarray
        The noise ceiling signal-to-noise ratio (SNR) for each unit.
    signalvar : np.ndarray
        The signal variance for each unit.
    noisevar : np.ndarray
        The noise variance for each unit.
    """
    # noisevar: mean variance across trials for each unit
    noisevar = np.nanmean(np.std(data_in, axis=2, ddof=1) ** 2, axis=1)

    # datavar: variance of the trial means across conditions for each unit
    datavar = np.nanstd(np.nanmean(data_in, axis=2), axis=1, ddof=1) ** 2

    # signalvar: signal variance, obtained by subtracting noise variance from data variance
    signalvar = np.maximum(datavar - noisevar / data_in.shape[2], 0)  # Ensure non-negative variance

    # ncsnr: signal-to-noise ratio (SNR) for each unit
    ncsnr = np.sqrt(signalvar) / np.sqrt(noisevar)

    # noiseceiling: percentage noise ceiling based on SNR
    noiseceiling = 100 * (ncsnr ** 2 / (ncsnr ** 2 + 1 / data_in.shape[2]))

    return noiseceiling, ncsnr, signalvar, noisevar

def derag_fr(data_in, period='early'):
    """
    Return the per-trial firing rate data for all units in a specific time period
    
    Arguments:
    ---------
    data_in : pd.DataFrame
    period : str 
        'pre' --> -25 to 30 ms
        'early' --> 50 to 120 ms
        'late' --> 120 to 240 ms
                
    Returns:
    --------
    stacked : np.ndarray (num_units, images, trials)
        nan padded
    """
    # the array is still ragged
    in_period = list(data_in[period])
    num_units = len(in_period)
    num_images = len(in_period[0])

    # maximum number of reps for a single image
    max_reps = max(
        len(arr) if hasattr(arr, "__len__") else 0
        for unit in in_period
        for arr in unit)

    # pad with nan
    stacked = np.full((num_units, num_images, max_reps), np.nan, dtype=float)
    for unit_i, unit in enumerate(in_period):
        for img in range(num_images):
            arr = np.array(unit[img])
            reps_here = len(arr)
            if reps_here > 0:
                stacked[unit_i, img, :reps_here] = arr
                
    return stacked
    
def get_unit_timecourse(row, start=None, end=None):
    """
    Return the unit's avg PSTH within the analysis window.
    If avg_psth is missing, derive it by averaging img_psth across images.
    Ensures shape (T,) where T=end-start.
    """
    avg = row['avg_psth']
    if avg is None or (isinstance(avg, float) and np.isnan(avg)):
        A = np.asarray(row['img_psth'])  # (time, images)
        if A.ndim != 2:
            raise ValueError("img_psth must be 2D (time x images)")
        avg = A.mean(axis=1)
    avg = np.asarray(avg)
    if avg.ndim != 1:
        raise ValueError("avg_psth must be 1D (time,)")
    # take all values if start/end is not specified
    if start is None:
        start = 0
        end = len(avg)
    if len(avg) < end:
        raise ValueError(f"avg_psth length {len(avg)} < required end index {end}")
    return avg[start:end]  # (T,)
