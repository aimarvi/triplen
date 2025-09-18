import re
import os

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
