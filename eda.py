import os
import re
import h5py
import numpy as np

datadir = '../datasets/NNN/'
fnames = [f for f in os.listdir(datadir) if re.match(r'^GoodUnit.*\.mat$', f)]

session_num = 1

with h5py.File(os.path.join(datadir, fnames[session_num]), 'r') as f:
    # Choose your path as appropriate
    raster = f['GoodUnitStrc/Raster']
    
    # Dereference all cell elements into a list of np arrays
    raster_npy = [f[raster[i,0]][()] for i in range(raster.shape[0])]
    # Note: use raster[i] instead of raster[i, 0] if it's 1D

    raster_stacked = np.stack(raster_npy)

# shape is (376, 450, 5720) --> (units, time point, trial)
print(raster_stacked.shape)
