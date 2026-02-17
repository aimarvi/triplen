import os
import numpy as np
from manifold_dynamics.raw_raster import process_session

DATADIR = './../../datasets/NNN/'
SAVEDIR = '/Users/aim/Downloads/'
roi_uid = '18.19.Unknown.F'

out = process_session(roi_uid)
out_path = os.path.join(SAVEDIR, f'{roi_uid}.npy')
np.save(out_path, out, allow_pickle=False)
print(f'Sucessfully saved raster for {roi_uid}')
