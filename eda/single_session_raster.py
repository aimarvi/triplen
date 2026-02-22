import os, sys, fsspec
import numpy as np
import pandas as pd

import manifold_dynamics.session_raster_extraction as sre
import manifold_dynamics.paths as pth

fs = fsspec.filesystem("s3")

# set some pathnames
csv_path = os.path.join(pth.OTHERS, sys.argv[1])
out_dir  = os.path.join(pth.PROCESSED, sys.argv[2])
uid_col  = sys.argv[3] if len(sys.argv) > 3 else "uid"

# run via sbatch array
task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

uids = pd.read_csv(csv_path)[uid_col].astype(str).tolist()
uid = uids[task_id]

# process the session and save to aws bucket
out_path = os.path.join(out_dir, f"{uid}.npy")
if not os.path.exists(out_path):
    out = sre.extract_session_raster(uid, verbose=True)
    print(f'\n\nSaving to {out_path}...')
    with fs.open(out_path, 'wb') as f:
        np.save(f, out)
    print(f'\n\nFinished saving!')
else:
    print(f'Skipping {uid}, already processed...')
