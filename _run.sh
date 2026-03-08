#!/bin/bash
#SBATCH -p shared
#SBATCH -c 1
#SBATCH --mem=50G
#SBATCH -t 05:00:00
#SBATCH -o errlog/denoise.%j.out

set -euo pipefail

cd /n/holylabs/LABS/konkle_lab/Users/amarvi/workspace/manifold-dynamics
ls -la

# run as a module (i think?)
# uv run python -m manifold_dynamics.session_raster_extraction

# mutli-function via sbatch array
# uv run python eda/single_session_raster.py roi-uid.csv single-session-raster
uv run python denoise/single_session_gsn.py 1 denoise False 

# run the script by abs. path 
# uv run python src/manifold_dynamics/session_raster_extraction.py

# cross-validation

idx="${SLURM_ARRAY_TASK_ID}"

target="$(
  uv run python -c "
import sys
import pandas as pd
import manifold_dynamics.paths as pth
import visionlab_utils.storage as vst

idx = int(sys.argv[1])

uid_csv_uri = f'{pth.OTHERS}/roi-uid.csv'
uid_csv_local = vst.fetch(uid_csv_uri)
df = pd.read_csv(uid_csv_local)

val = df.iloc[idx, 'uid']
print(str(val))
" "$idx"
)"

if [[ -z "${target}" ]]; then
  echo "no target for idx=${idx}" >&2
  exit 1
fi

uv run python crossval.py --target "$target"
