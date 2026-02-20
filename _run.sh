#!/bin/bash
#SBATCH -p shared
#SBATCH -c 1
#SBATCH --mem=50G
#SBATCH -t 05:00:00
#SBATCH -o errlog/raster.%j.out

set -euo pipefail

cd /n/holylabs/LABS/konkle_lab/Users/amarvi/workspace/manifold-dynamics
ls -la

# run as a module (i think?)
# uv run python -m manifold_dynamics.raw_raster

# mutli-function via sbatch array
uv run python eda/single_session_raster.py roi-uid.csv single-session-raster

# run the script by abs. path 
# uv run python src/manifold_dynamics/raw_raster.py
