from __future__ import annotations

import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import manifold_dynamics.model_utils as mut
import manifold_dynamics.neural_utils as nu
import manifold_dynamics.paths as pth
import manifold_dynamics.tuning_utils as tut
import visionlab_utils.storage as vst


# Ad-hoc configuration
ROI_TARGET = os.getenv("ROI_TARGET", "19.Unknown.F")
TOP_K = 30
ALPHA = 0.05
BIN_SIZE_MS = 20
TSTART = 100
TEND = 350
FEATURE_URI = f"{pth.SAVEDIR}/alexnet/alexnet_acts.pkl"
FEATURE_LAYERS = ["classifier.5"]  # fc7 post-ReLU
RANDOM_STATE = 0
VERBOSE = True
OUTPUT_PATH = Path(
    os.getenv(
        "OUTPUT_PATH",
        str(Path.home() / "Downloads" / f"neighbor_scales_{ROI_TARGET.replace('.', '_')}.png"),
    )
)


def vprint(msg: str) -> None:
    if VERBOSE:
        print(msg)


raster_4d = nu.significant_trial_raster(roi_uid=ROI_TARGET, alpha=ALPHA, bin_size_ms=BIN_SIZE_MS)
X = np.nanmean(raster_4d, axis=3)
vprint(f"Responsive trial raster shape (units, time, images, trials): {raster_4d.shape}")
vprint(f"Trial-averaged PSTH shape (units, time, images): {X.shape}")

image_order = tut.rank_images_by_response(X)
top_indices = image_order[:TOP_K]
vprint(f"Top-{TOP_K} seed image indices: {top_indices.tolist()}")

acts_local = vst.fetch(FEATURE_URI)
with open(acts_local, "rb") as f:
    acts = pickle.load(f)

feature_blocks = []
for layer in FEATURE_LAYERS:
    if layer not in acts:
        raise ValueError(f"Missing AlexNet feature layer: {layer}")
    arr = acts[layer]
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D activations for {layer}, got shape {arr.shape}")
    feature_blocks.append(arr)
    vprint(f"{layer}: feature shape={arr.shape}")

feature_matrix = np.concatenate(feature_blocks, axis=1)
if feature_matrix.shape[0] != X.shape[2]:
    raise ValueError(
        f"Feature/image mismatch: feature_matrix has {feature_matrix.shape[0]} images, "
        f"but PSTH has {X.shape[2]} images."
    )

rng = np.random.default_rng(RANDOM_STATE)
nearest_sets = mut.neighbor_sets(feature_matrix=feature_matrix, seed_indices=top_indices, k=TOP_K)
R_topk, _ = tut.tuning_rdm(
    X=X,
    indices=top_indices,
    tstart=TSTART,
    tend=TEND,
    metric="correlation",
)
ed_topk = float(tut.ED2(R_topk))

rows = []
for set_idx, (seed_idx, neighbor_indices) in enumerate(zip(top_indices, nearest_sets)):
    random_indices = rng.choice(X.shape[2], size=TOP_K, replace=False)

    R_neighbor, _ = tut.tuning_rdm(
        X=X,
        indices=neighbor_indices,
        tstart=TSTART,
        tend=TEND,
        metric="correlation",
    )
    R_random, _ = tut.tuning_rdm(
        X=X,
        indices=random_indices,
        tstart=TSTART,
        tend=TEND,
        metric="correlation",
    )

    ed_neighbor = float(tut.ED2(R_neighbor))
    ed_random = float(tut.ED2(R_random))

    rows.append(
        {
            "roi": ROI_TARGET,
            "set_idx": int(set_idx),
            "seed_image_idx": int(seed_idx),
            "neighbor_indices": neighbor_indices.tolist(),
            "random_indices": random_indices.tolist(),
            "ED_neighbor": ed_neighbor,
            "ED_random": ed_random,
            "delta_neighbor_minus_random": ed_neighbor - ed_random,
        }
    )

df_out = pd.DataFrame(rows)

print(df_out[["set_idx", "seed_image_idx", "ED_neighbor", "ED_random", "delta_neighbor_minus_random"]].to_string(index=False))
print(f"ED_topk:          {ed_topk:.6f}")
print(f"mean ED_neighbor: {df_out['ED_neighbor'].mean():.6f}")
print(f"mean ED_random:   {df_out['ED_random'].mean():.6f}")
print(f"mean delta:       {df_out['delta_neighbor_minus_random'].mean():.6f}")

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.boxplot(
    [np.array([ed_topk]), df_out["ED_neighbor"].to_numpy(), df_out["ED_random"].to_numpy()],
    positions=[1, 2, 3],
    widths=0.5,
    tick_labels=["top-k", "neighbor", "random"],
)
ax.set_ylabel("effective dimensionality")
ax.set_title(ROI_TARGET)
ax.scatter(1, ed_topk, color="black", zorder=3)
ax.text(1, ed_topk, f"{ed_topk:.2f}", ha="center", va="bottom")
fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {OUTPUT_PATH}")
