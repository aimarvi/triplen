from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import manifold_dynamics.neural_utils as nu
import manifold_dynamics.paths as pth
import manifold_dynamics.tuning_utils as tut
import visionlab_utils.storage as vst


# ROI format:
# - 4-part UID: SesIdx.RoiIndex.AREALABEL.Categoty (e.g., 18.19.Unknown.F)
# - 3-part ROI: RoiIndex.AREALABEL.Categoty (e.g., 19.Unknown.F)
ROI_TARGET = "08.MF1.F" # "19.Unknown.F"
BIN_SIZE_MS = 20
ALPHA = 0.05
N_PCS = 20
RANDOM_STATE = 0
LOG_SCALE = True
VERBOSE = True

OUTPUT_PATH = Path.home() / "Downloads" / f"eigenspectra_{ROI_TARGET.replace('.', '_')}.png"


def vprint(msg: str) -> None:
    if VERBOSE:
        print(msg)


topk_local = vst.fetch(f"{pth.OTHERS}/topk_vals.pkl")
with open(topk_local, "rb") as f:
    topk_vals = pickle.load(f)

if ROI_TARGET not in topk_vals:
    raise ValueError(f"No top-k entry found for ROI: {ROI_TARGET}")

local_k = int(topk_vals[ROI_TARGET]["k"])
vprint(f"ROI target: {ROI_TARGET}")
vprint(f"Local k: {local_k}")

raster_4d = nu.significant_trial_raster(roi_uid=ROI_TARGET, alpha=ALPHA, bin_size_ms=BIN_SIZE_MS)
trial_avg = np.nanmean(raster_4d, axis=3)
vprint(f"Responsive trial raster shape: {raster_4d.shape}")
vprint(f"Trial-averaged PSTH shape: {trial_avg.shape}")

image_order = tut.rank_images_by_response(trial_avg)
rng = np.random.default_rng(RANDOM_STATE)

idx_top = np.asarray(image_order[:local_k], dtype=int)
idx_all = np.arange(trial_avg.shape[2], dtype=int)
idx_rand = rng.choice(trial_avg.shape[2], size=local_k, replace=False)

L_top = np.full((N_PCS, trial_avg.shape[1]), np.nan, dtype=np.float64)
L_all = np.full((N_PCS, trial_avg.shape[1]), np.nan, dtype=np.float64)
L_rand = np.full((N_PCS, trial_avg.shape[1]), np.nan, dtype=np.float64)

for t in range(trial_avg.shape[1]):
    M_top = trial_avg[:, t, idx_top]
    M_top = M_top - np.nanmean(M_top, axis=1, keepdims=True)
    s_top = np.linalg.svd(np.nan_to_num(M_top, nan=0.0), full_matrices=False, compute_uv=False)
    lam_top = (s_top ** 2) / max(M_top.shape[1] - 1, 1)
    L_top[:, t] = lam_top[:N_PCS]

    M_all = trial_avg[:, t, idx_all]
    M_all = M_all - np.nanmean(M_all, axis=1, keepdims=True)
    s_all = np.linalg.svd(np.nan_to_num(M_all, nan=0.0), full_matrices=False, compute_uv=False)
    lam_all = (s_all ** 2) / max(M_all.shape[1] - 1, 1)
    L_all[:, t] = lam_all[:N_PCS]

    M_rand = trial_avg[:, t, idx_rand]
    M_rand = M_rand - np.nanmean(M_rand, axis=1, keepdims=True)
    s_rand = np.linalg.svd(np.nan_to_num(M_rand, nan=0.0), full_matrices=False, compute_uv=False)
    lam_rand = (s_rand ** 2) / max(M_rand.shape[1] - 1, 1)
    L_rand[:, t] = lam_rand[:N_PCS]

Z_top = np.log1p(L_top) if LOG_SCALE else L_top
Z_all = np.log1p(L_all) if LOG_SCALE else L_all
Z_rand = np.log1p(L_rand) if LOG_SCALE else L_rand
shared_vmax = float(max(np.nanmax(Z_top), np.nanmax(Z_all), np.nanmax(Z_rand)))

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True, constrained_layout=True)

im0 = axes[0].imshow(
    Z_top,
    aspect="auto",
    origin="lower",
    interpolation="nearest",
    vmin=0.0,
    vmax=shared_vmax,
)
axes[0].set_title(f"Top-{local_k}")
axes[0].set_xlabel("time bin")
axes[0].set_ylabel("PC index")
axes[0].set_yticks(range(N_PCS))
axes[0].set_yticklabels([f"PC{i + 1}" for i in range(N_PCS)])

im1 = axes[1].imshow(
    Z_all,
    aspect="auto",
    origin="lower",
    interpolation="nearest",
    vmin=0.0,
    vmax=shared_vmax,
)
axes[1].set_title("All images")
axes[1].set_xlabel("time bin")

im2 = axes[2].imshow(
    Z_rand,
    aspect="auto",
    origin="lower",
    interpolation="nearest",
    vmin=0.0,
    vmax=shared_vmax,
)
axes[2].set_title(f"Random-{local_k}")
axes[2].set_xlabel("time bin")

fig.colorbar(
    im2,
    ax=axes,
    fraction=0.025,
    pad=0.02,
    label="log1p(λ)" if LOG_SCALE else "λ",
)

fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {OUTPUT_PATH}")
