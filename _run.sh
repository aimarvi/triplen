#!/bin/bash
#SBATCH -p shared
#SBATCH -c 1
#SBATCH --mem=100G
#SBATCH -t 1-00:00
#SBATCH -o errlog/raster.%j.out

set -euo pipefail

cd /n/holylabs/LABS/konkle_lab/Users/amarvi/workspace/manifold-dynamics
ls -la

# run as a module (i think?)
uv run python -m manifold_dynamics.raw_raster

# run the script by abs. path 
# uv run python src/manifold_dynamics/raw_raster.py
