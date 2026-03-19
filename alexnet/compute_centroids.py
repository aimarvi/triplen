from __future__ import annotations

import argparse
import pickle

import fsspec
import numpy as np
from sklearn.decomposition import PCA

import manifold_dynamics.neural_utils as nu
import manifold_dynamics.paths as pth
import manifold_dynamics.tuning_utils as tut
import visionlab_utils.storage as vst


def main() -> None:
    """
    Compute and save one ROI centroid in AlexNet PCA space.

    The PCA is fit on all images for the requested AlexNet layer, then the
    centroid is computed from the ROI's top-k preferred images in that shared
    2D space.
    """
    parser = argparse.ArgumentParser(
        description="Compute one ROI centroid in AlexNet PCA space."
    )
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--layer-key", type=str, default="classifier.5")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bin-size-ms", type=int, default=20)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    def vprint(msg: str) -> None:
        if args.verbose:
            print(msg)

    topk_local = vst.fetch(f"{pth.OTHERS}/topk_vals.pkl")
    with open(topk_local, "rb") as f:
        topk_vals = pickle.load(f)

    if args.target not in topk_vals:
        raise ValueError(f"No top-k entry found for ROI: {args.target}")
    top_k = int(topk_vals[args.target]["k"])
    selectivity = args.target.split(".")[-1]

    embedding_uri = (
        f"{pth.SAVEDIR}/alexnet/pca_embedding_{args.layer_key.replace('.', '_')}.pkl"
    )
    try:
        embedding_local = vst.fetch(embedding_uri)
        with open(embedding_local, "rb") as f:
            embedding_payload = pickle.load(f)
        Z = np.asarray(embedding_payload["embedding"])
        vprint(f"Loaded PCA embedding from {embedding_uri}")
    except Exception:
        feature_uri = f"{pth.SAVEDIR}/alexnet/alexnet_acts.pkl"
        acts_local = vst.fetch(feature_uri)
        with open(acts_local, "rb") as f:
            acts = pickle.load(f)

        if args.layer_key not in acts:
            raise ValueError(f"Missing AlexNet feature layer: {args.layer_key}")

        feature_matrix = acts[args.layer_key]
        if hasattr(feature_matrix, "detach"):
            feature_matrix = feature_matrix.detach().cpu().numpy()
        feature_matrix = np.asarray(feature_matrix)
        if feature_matrix.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape {feature_matrix.shape}")

        pca = PCA(n_components=2)
        Z = pca.fit_transform(feature_matrix)
        embedding_payload = {
            "layer_key": args.layer_key,
            "embedding": Z,
        }
        if args.save:
            with fsspec.open(embedding_uri, "wb") as f:
                pickle.dump(embedding_payload, f)
            vprint(f"Saved PCA embedding to {embedding_uri}")

    raster_4d = nu.significant_trial_raster(
        roi_uid=args.target,
        alpha=args.alpha,
        bin_size_ms=args.bin_size_ms,
    )
    raster_3d = np.nanmean(raster_4d, axis=3)
    image_order = tut.rank_images_by_response(raster_3d)
    idx_topk = np.asarray(image_order[:top_k], dtype=int)
    centroid = np.nanmean(Z[idx_topk], axis=0)

    payload = {
        "roi": args.target,
        "top_k": top_k,
        "selectivity": selectivity,
        "layer_key": args.layer_key,
        "alpha": float(args.alpha),
        "bin_size_ms": int(args.bin_size_ms),
        "centroid_pc1": float(centroid[0]),
        "centroid_pc2": float(centroid[1]),
    }

    print(
        f"{args.target}: k={top_k}, layer={args.layer_key}, "
        f"centroid=({centroid[0]:.6f}, {centroid[1]:.6f})"
    )

    if args.save:
        s3_out = (
            f"{pth.SAVEDIR}/alexnet/roi_centroids/{args.layer_key.replace('.', '_')}/{args.target}.pkl"
        )
        with fsspec.open(s3_out, "wb") as f:
            pickle.dump(payload, f)
        vprint(f"Saved centroid payload to {s3_out}")


if __name__ == "__main__":
    main()
