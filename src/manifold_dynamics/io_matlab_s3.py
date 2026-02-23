import re, os, h5py, fsspec, tempfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import io

import manifold_dynamics.paths as pth
fs = fsspec.filesystem("s3")

def fnames(rawdir=pth.RAW, processdir=pth.PROCESSED):
    '''
    Return filenames for raw and processed data for each session

    Args:
        rawdir (Path): aws/s3 path to GoodUnit data. 
        processdir (Path): aws/s3 path to Processed session data.
    '''
    gus_fnames = [f.split('/')[-1] for f in fs.ls(rawdir) if re.match(r'^GoodUnit_.*\.mat$', f.split('/')[-1])]
    proc_fnames = [f.split('/')[-1] for f in fs.ls(processdir) if re.match(r'^Processed.*\.mat$', f.split('/')[-1])]

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
            # print(f'converting mat struct to dict for {key}...')
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

def load_mat(filename, fformat='v7.3', verbose=False):
    """
    Load a Matlab -v7.3 (HDF5-format) .mat file, dereference cell arrays, and
    return a dict of numpy arrays and nested dicts (for group/structs).
    """
    fs, path = fsspec.core.url_to_fs(filename)

    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
        # print('temp file created')
        fs.get(path, tmp.name)
        with open(tmp.name, "rb") as fh:
            sig = fh.read(8)
        if fformat=='v7.3':  # v7.3
            if verbose: print("v7.3 (HDF5)")
            with h5py.File(tmp.name, 'r') as f:
                # print('opened with h5py')
                data = {}
                for key in f.keys():
                    obj = f[key]
                    if key.startswith('#'):
                        if verbose: print('skipping key...')
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
                            if verbose: print(f'no data found for key: {key}')
                if verbose:
                    print(f'successfully loaded data with keys {data.keys()}')
                return data
        elif fformat=='v5':
            if verbose: print("v5 (scipy.io.loadmat)")
            return io.loadmat(tmp.name, squeeze_me=True, struct_as_record=False)
        else:
            print('Unknown format, exiting...')
            return None
