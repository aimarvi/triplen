import os
import re
import h5py
import numpy as np
import scipy.io

import utils

datadir = '../datasets/NNN/'
fnames = utils.fnames(datadir)

for pair in fnames:
    gus_fname = os.path.join(datadir, pair[0])
    proc_fname = os.path.join(datadir, pair[1])
    
    gus_data = utils.load_mat(gus_fname)
    proc_data = scipy.io.loadmat(proc_fname)
    
    print(proc_data.keys())
    print(gus_data.keys())
    break
