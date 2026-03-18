#!/bin/bash
#SBATCH -p shared
#SBATCH -c 1
#SBATCH --mem=100G
#SBATCH -t 10:00:00
#SBATCH -o errlog/centroid.%j.out

set -euo pipefail

cd /n/holylabs/LABS/konkle_lab/Users/amarvi/workspace/manifold-dynamics

# ========== raw rasters for individual sessions
# uv run python -m manifold_dynamics.session_raster_extraction
# uv run python eda/single_session_raster.py roi-uid.csv single-session-raster
# uv run python denoise/single_session_gsn.py 1 denoise False 
# uv run python src/manifold_dynamics/session_raster_extraction.py

# ========== ROI centroids in alexnet PC space
uv run python alexnet/roi_centroids.py
