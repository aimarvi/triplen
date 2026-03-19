from __future__ import annotations

import argparse
import pickle

import fsspec
import numpy as np
import pandas as pd

import manifold_dynamics.neural_utils as nu
import manifold_dynamics.paths as pth
import manifold_dynamics.tuning_utils as tut
import visionlab_utils.storage as vst


def main() -> None:
    """
    Compute local, global, and bootstrapped-random ED for one ROI target.

    This version uses all trials for the trial-averaged response tensor and does
    not perform cross-validation.
    """
    parser = argparse.ArgumentParser(
        description="Time-time ED analysis for one ROI target using all trials."
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
    parser.add_argument("--tstart", type=int, default=100)
    parser.add_argument("--tend", type=int, default=350)
    parser.add_argument("--n-random", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=0)
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

    raster_4d = nu.significant_trial_raster(
        roi_uid=args.target,
        alpha=args.alpha,
        bin_size_ms=args.bin_size_ms,
    )
    X = np.nanmean(raster_4d, axis=3)
    order = tut.rank_images_by_response(X)
    idx_local = np.asarray(order[:top_k], dtype=int)
    idx_global = np.arange(X.shape[2], dtype=int)

    vprint(f"Responsive raster shape: {raster_4d.shape}")
    vprint(f"Trial-averaged PSTH shape: {X.shape}")
    vprint(f"Using top-k = {top_k}")

    R_local, _ = tut.tuning_rdm(
        X=X,
        indices=idx_local,
        tstart=args.tstart,
        tend=args.tend,
        metric="correlation",
    )
    R_global, _ = tut.tuning_rdm(
        X=X,
        indices=idx_global,
        tstart=args.tstart,
        tend=args.tend,
        metric="correlation",
    )
    ed_local = float(tut.ED2(R_local))
    ed_global = float(tut.ED2(R_global))

    rng = np.random.default_rng(args.random_state)
    random_rows = []
    for i in range(args.n_random):
        idx_random = rng.choice(X.shape[2], size=top_k, replace=False)
        R_random, _ = tut.tuning_rdm(
            X=X,
            indices=idx_random,
            tstart=args.tstart,
            tend=args.tend,
            metric="correlation",
        )
        random_rows.append(
            {
                "roi": roi_label,
                "target": args.target,
                "condition": "random",
                "bootstrap": i,
                "top_k": int(top_k),
                "ED": float(tut.ED2(R_random)),
            }
        )

    rows = [
        {
            "roi": roi_label,
            "target": args.target,
            "condition": "local",
            "bootstrap": np.nan,
            "top_k": int(top_k),
            "ED": ed_local,
        },
        {
            "roi": roi_label,
            "target": args.target,
            "condition": "global",
            "bootstrap": np.nan,
            "top_k": int(top_k),
            "ED": ed_global,
        },
        *random_rows,
    ]
    df_out = pd.DataFrame(rows)

    print(df_out[df_out["condition"] != "random"].to_string(index=False))
    print(f"mean ED_random: {df_out.loc[df_out['condition'] == 'random', 'ED'].mean():.6f}")
    print(f"std ED_random:  {df_out.loc[df_out['condition'] == 'random', 'ED'].std():.6f}")

    if args.save:
        s3_out = f"{pth.SAVEDIR}/timextime/ed_main/{args.target}.pkl"
        with fsspec.open(s3_out, "wb") as f:
            df_out.to_pickle(f)


if __name__ == "__main__":
    main()
