from __future__ import annotations

import argparse
import pickle

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import manifold_dynamics.neural_utils as nu
import manifold_dynamics.paths as pth
import manifold_dynamics.tuning_utils as tut
import visionlab_utils.storage as vst


def main() -> None:
    """
    Compute eigenspectra over time for top-k, all-image, and random-k sets.

    Target format options:
      - ROI UID (4 parts): ``SesIdx.RoiIndex.AREALABEL.Categoty``
      - ROI key (3 parts): ``RoiIndex.AREALABEL.Categoty``
    """
    parser = argparse.ArgumentParser(
        description="Time-resolved eigenspectra analysis for one ROI target."
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help=(
            "ROI UID (4-part: SesIdx.RoiIndex.AREALABEL.Categoty) "
            "or ROI key (3-part: RoiIndex.AREALABEL.Categoty)."
        ),
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bin-size-ms", type=int, default=20)
    parser.add_argument("--n-pcs", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--log-scale", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    def vprint(msg: str) -> None:
        if args.verbose:
            print(msg)

    target_parts = args.target.split(".")
    if len(target_parts) not in (3, 4):
        raise ValueError(
            "Invalid --target format. Use 4-part UID (SesIdx.RoiIndex.AREALABEL.Categoty) "
            "or 3-part ROI key (RoiIndex.AREALABEL.Categoty)."
        )
    if len(target_parts) == 4:
        roi_label = f"{int(target_parts[1]):02d}.{target_parts[2]}.{target_parts[3]}"
    else:
        roi_label = args.target

    topk_local = vst.fetch(f"{pth.OTHERS}/topk_vals.pkl")
    with open(topk_local, "rb") as f:
        topk_vals = pickle.load(f)

    top_k = args.top_k
    if top_k is None:
        if roi_label not in topk_vals:
            raise ValueError(f"No top-k entry found for ROI: {roi_label}")
        top_k = int(topk_vals[roi_label]["k"])

    vprint(f"ROI target: {args.target}")
    vprint(f"Using top-k = {top_k}")

    raster_4d = nu.significant_trial_raster(
        roi_uid=args.target,
        alpha=args.alpha,
        bin_size_ms=args.bin_size_ms,
    )
    trial_avg = np.nanmean(raster_4d, axis=3)
    vprint(f"Responsive trial raster shape: {raster_4d.shape}")
    vprint(f"Trial-averaged PSTH shape: {trial_avg.shape}")

    image_order = tut.rank_images_by_response(trial_avg)
    rng = np.random.default_rng(args.random_state)

    idx_top = np.asarray(image_order[:top_k], dtype=int)
    idx_all = np.arange(trial_avg.shape[2], dtype=int)
    idx_rand = rng.choice(trial_avg.shape[2], size=top_k, replace=False)

    conditions = {
        f"Top-{top_k}": idx_top,
        "All images": idx_all,
        f"Random-{top_k}": idx_rand,
    }

    eigenspectra = {}
    cumvar = {}
    for label, idx in conditions.items():
        L = np.full((args.n_pcs, trial_avg.shape[1]), np.nan, dtype=np.float64)
        C = np.full((args.n_pcs, trial_avg.shape[1]), np.nan, dtype=np.float64)
        for t in range(trial_avg.shape[1]):
            M = trial_avg[:, t, idx]
            M = M - np.nanmean(M, axis=1, keepdims=True)
            s = np.linalg.svd(np.nan_to_num(M, nan=0.0), full_matrices=False, compute_uv=False)
            lam = (s ** 2) / max(M.shape[1] - 1, 1)
            L[:, t] = lam[: args.n_pcs]

            total = float(np.sum(lam))
            if total > 0:
                C[:, t] = np.cumsum(lam[: args.n_pcs]) / total
        eigenspectra[label] = L
        cumvar[label] = C

    transformed = {
        label: np.log1p(L) if args.log_scale else L
        for label, L in eigenspectra.items()
    }
    shared_vmax = float(max(np.nanmax(Z) for Z in transformed.values()))

    fig_heat, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True, constrained_layout=True)
    for ax, label in zip(axes, transformed):
        im = ax.imshow(
            transformed[label],
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            vmin=0.0,
            vmax=shared_vmax,
        )
        ax.set_title(label)
        ax.set_xlabel("time bin")
    axes[0].set_ylabel("PC index")
    axes[0].set_yticks(range(args.n_pcs))
    axes[0].set_yticklabels([f"PC{i + 1}" for i in range(args.n_pcs)])
    fig_heat.colorbar(
        im,
        ax=axes,
        fraction=0.025,
        pad=0.02,
        label="log(lambda)" if args.log_scale else "lambda",
    )

    fig_cum, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True, constrained_layout=True)
    for ax, label in zip(axes, cumvar):
        im = ax.imshow(
            cumvar[label],
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(label)
        ax.set_xlabel("time bin")
    axes[0].set_ylabel("PC index")
    axes[0].set_yticks(range(args.n_pcs))
    axes[0].set_yticklabels([f"PC{i + 1}" for i in range(args.n_pcs)])
    fig_cum.colorbar(
        im,
        ax=axes,
        fraction=0.025,
        pad=0.02,
        label="cumulative variance explained",
    )

    payload = {
        "target": args.target,
        "roi_label": roi_label,
        "top_k": int(top_k),
        "alpha": float(args.alpha),
        "bin_size_ms": int(args.bin_size_ms),
        "n_pcs": int(args.n_pcs),
        "random_state": int(args.random_state),
        "indices": {label: idx.tolist() for label, idx in conditions.items()},
        "eigenspectra": eigenspectra,
        "cumvar": cumvar,
    }

    summary = []
    for label in eigenspectra:
        summary.append(
            {
                "condition": label,
                "pc1_mean": float(np.nanmean(eigenspectra[label][0])),
                "pc1_peak": float(np.nanmax(eigenspectra[label][0])),
            }
        )
    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))

    if args.save:
        s3_base = f"{pth.SAVEDIR}/eigenspectra/{args.target}"
        with fsspec.open(f"{s3_base}.pkl", "wb") as f:
            pickle.dump(payload, f)
        with fsspec.open(f"{s3_base}_heatmap.png", "wb") as f:
            fig_heat.savefig(f, format="png", dpi=300, bbox_inches="tight")
        with fsspec.open(f"{s3_base}_cumvar.png", "wb") as f:
            fig_cum.savefig(f, format="png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
