import os
import re
import h5py
import numpy as np

datadir = '../datasets/NNN/'


fnames = [f for f in os.listdir(datadir) if re.match(r'^GoodUnit.*\.mat$', f)]
print(fnames)
