import os
import re
import h5py
import numpy as np

datadir = '../datasets/NNN/'
fnames = [f for f in os.listdir(datadir) if re.match(r'^GoodUnit.*\.mat$', f)]

session_num = 1

f = h5py.File(os.path.join(datadir, fnames[session_num]), 'r')
print(f['GoodUnitStrc'].keys())
f2 = f.get('GoodUnitStrc/Raster')

# the shape is correct but it is an h5py dataset (full of obj references??)
# how do i turn it into an numpy array?
print(f2.shape) 
