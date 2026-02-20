#!/bin/bash
#SBATCH -p shared
#SBATCH -c 1
#SBATCH --mem=100G
#SBATCH -t 1-00:00
#SBATCH -o err.%j.out

set -euo pipefail

cd /n/holylabs/LABS/konkle_lab/Users/amarvi/workspace/manifold-dynamics
ls -la

# Option A: run as a module (preferred if raw_raster.py uses package imports)
uv run python -m manifold_dynamics.raw_raster

# Option B: run the script by path (if it’s a standalone script file)
# uv run python src/manifold_dynamics/raw_raster.py
