from __future__ import annotations

import pickle
from pathlib import Path

import fsspec
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import manifold_dynamics.neural_utils as nu
import manifold_dynamics.paths as pth
import manifold_dynamics.tuning_utils as tut
import visionlab_utils.storage as vst


# Use every ROI with a defined top-k value.
ROI_TARGETS = None
SELECTIVITIES_TO_PLOT = {"B", "F", "O"}
LAYER_KEY = "classifier.5"
ALPHA = 0.05
BIN_SIZE_MS = 20
SAVE = True
VERBOSE = True


def vprint(msg: str) -> None:
    """Print only when VERBOSE is enabled."""
    if VERBOSE:
        print(msg)


topk_local = vst.fetch(f"{pth.OTHERS}/topk_vals.pkl")
with open(topk_local, "rb") as f:
    topk_vals = pickle.load(f)

if ROI_TARGETS is None:
    ROI_TARGETS = sorted(
        target for target in topk_vals.keys() if target.split(".")[-1] in SELECTIVITIES_TO_PLOT
    )

feature_uri = f"{pth.SAVEDIR}/alexnet/alexnet_acts.pkl"
acts_local = vst.fetch(feature_uri)
with open(acts_local, "rb") as f:
    acts = pickle.load(f)

if LAYER_KEY not in acts:
    raise ValueError(f"Missing AlexNet feature layer: {LAYER_KEY}")

feature_matrix = acts[LAYER_KEY]
if hasattr(feature_matrix, "detach"):
    feature_matrix = feature_matrix.detach().cpu().numpy()
feature_matrix = np.asarray(feature_matrix)
if feature_matrix.ndim != 2:
    raise ValueError(f"Expected 2D feature matrix, got shape {feature_matrix.shape}")

pca = PCA(n_components=2)
Z = pca.fit_transform(feature_matrix)

fig, ax = plt.subplots(1, 1, figsize=(7, 6))
ax.scatter(Z[:, 0], Z[:, 1], s=10, alpha=0.12, c="black", label="all images")

selectivities = sorted({target.split(".")[-1] for target in ROI_TARGETS})
palette = plt.cm.tab10(np.linspace(0, 1, len(selectivities)))
color_map = {sel: palette[i] for i, sel in enumerate(selectivities)}
seen_labels = set()
rows = []
for target in ROI_TARGETS:
    if target not in topk_vals:
        raise ValueError(f"No top-k entry found for ROI: {target}")
    top_k = int(topk_vals[target]["k"])
    selectivity = target.split(".")[-1]
    color = color_map[selectivity]

    try: 
        raster_4d = nu.significant_trial_raster(
            roi_uid=target,
            alpha=ALPHA,
            bin_size_ms=BIN_SIZE_MS,
        )
    except Exception as e:
        print(f"Could not load data for {target}: {type(e).__name__}: {e}")
        continue
    raster_3d = np.nanmean(raster_4d, axis=3)
    image_order = tut.rank_images_by_response(raster_3d)
    idx_topk = np.asarray(image_order[:top_k], dtype=int)

    Z_topk = Z[idx_topk]
    centroid = np.nanmean(Z_topk, axis=0)

    ax.scatter(
        centroid[0],
        centroid[1],
        s=140,
        marker="X",
        color=color,
        edgecolor="white",
        linewidth=0.8,
        label=selectivity if selectivity not in seen_labels else None,
    )
    seen_labels.add(selectivity)
    ax.text(centroid[0], centroid[1], f" {target}", color=color, va="center", fontsize=7)

    rows.append(
        {
            "roi": target,
            "top_k": top_k,
            "selectivity": selectivity,
            "centroid_pc1": float(centroid[0]),
            "centroid_pc2": float(centroid[1]),
        }
    )
    vprint(f"{target}: k={top_k}, centroid=({centroid[0]:.4f}, {centroid[1]:.4f})")

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title(f"ROI centroids in {LAYER_KEY} space")
ax.legend(frameon=False, title="selectivity", loc="best")
fig.tight_layout()

if SAVE:
    df_out = rows
    s3_base = f"{pth.SAVEDIR}/alexnet/roi_centroids_{LAYER_KEY.replace('.', '_')}"
    with fsspec.open(f"{s3_base}.pkl", "wb") as f:
        pickle.dump(df_out, f)
    with fsspec.open(f"{s3_base}.png", "wb") as f:
        fig.savefig(f, format="png", dpi=300, bbox_inches="tight")

# Save to local
#      download_dir = Path.home() / "Downloads"
#      fig.savefig(
#          download_dir / f"roi_centroids_{LAYER_KEY.replace('.', '_')}.png",
#          dpi=300,
#          bbox_inches="tight",
#      )
