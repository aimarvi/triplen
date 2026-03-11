from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

import manifold_dynamics.model_utils as mut
import manifold_dynamics.neural_utils as nu
import manifold_dynamics.paths as pth
import manifold_dynamics.tuning_utils as tut
import visionlab_utils.storage as vst


# Ad-hoc configuration
ROI_TARGET = "19.Unknown.F"
TOP_K = 30
ALPHA = 0.05
BIN_SIZE_MS = 20
FEATURE_URI = f"{pth.SAVEDIR}/alexnet/alexnet_acts.pkl"
FEATURE_LAYERS = ["classifier.5"]  # fc7 post-ReLU
VERBOSE = True


def vprint(msg: str) -> None:
    if VERBOSE:
        print(msg)


raster_4d = nu.significant_trial_raster(roi_uid=ROI_TARGET, alpha=ALPHA, bin_size_ms=BIN_SIZE_MS)
X = np.nanmean(raster_4d, axis=3)
vprint(f"Responsive trial raster shape (units, time, images, trials): {raster_4d.shape}")
vprint(f"Trial-averaged PSTH shape (units, time, images): {X.shape}")

image_order = tut.rank_images_by_response(X)
top_indices = image_order[:TOP_K]
vprint(f"Top-{TOP_K} image indices: {top_indices.tolist()}")

acts_local = vst.fetch(FEATURE_URI)
with open(acts_local, "rb") as f:
    acts = pickle.load(f)

missing_layers = [layer for layer in FEATURE_LAYERS if layer not in acts]
if missing_layers:
    raise ValueError(f"Missing AlexNet feature layers: {missing_layers}")

feature_blocks = []
for layer in FEATURE_LAYERS:
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

nearest_sets = mut.neighbor_sets(feature_matrix=feature_matrix, seed_indices=top_indices, k=TOP_K)
neighbor_sets = []
for seed_idx, nearest in zip(top_indices, nearest_sets):
    neighbor_sets.append(
        {
            "seed_image_idx": int(seed_idx),
            "neighbor_image_indices": nearest.tolist(),
        }
    )

df_neighbors = pd.DataFrame(neighbor_sets)

print(df_neighbors.head().to_string(index=False))
print(f"neighbor set table shape: {df_neighbors.shape}")
