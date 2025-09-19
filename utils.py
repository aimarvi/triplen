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