import argparse
import os

import numpy as np
import pandas as pd

import manifold_dynamics.cv_timextime as cvt
import manifold_dynamics.paths as pth


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for CV timextime replication."""
    parser = argparse.ArgumentParser(
        description=(
            "Cross-validated timextime replication from raw S3 rasters: "
            "tuning RDMs + ED compression change for top-k/localizer sets."
        )
    )
    parser.add_argument(
        "--roi-uid",
        type=str,
        default=None,
        help="Single ROI uid (format: SesIdx.RoiIndex.AREALABEL.Categoty).",
    )
    parser.add_argument(
        "--roi-label",
        type=str,
        default=None,
        help=(
            "Combined ROI label (format: AREALABEL_SesIdx_Categoty in this codebase, "
            "e.g., Unknown_19_F). If set, all matching UIDs from --roi-csv are combined."
        ),
    )
    parser.add_argument(
        "--roi-csv",
        type=str,
        default=os.path.join(pth.OTHERS, "roi-uid.csv"),
        help="CSV containing ROI UIDs. Must include a column set by --uid-col.",
    )
    parser.add_argument(
        "--uid-col",
        type=str,
        default="uid",
        help="Column name in --roi-csv containing ROI uid strings.",
    )
    parser.add_argument(
        "--max-rois",
        type=int,
        default=None,
        help="Optional cap for number of ROIs loaded from --roi-csv.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=(
            "Optional fixed top-k scale. If omitted, top-k is estimated per fold "
            "from train data."
        ),
    )
    parser.add_argument(
        "--k-step",
        type=int,
        default=5,
        help="Scale step used when estimating top-k.",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=200,
        help="Maximum scale considered when estimating top-k.",
    )
    parser.add_argument(
        "--tstart",
        type=int,
        default=100,
        help="Time-window start index for time-by-time RDM (inclusive).",
    )
    parser.add_argument(
        "--tend",
        type=int,
        default=400,
        help="Time-window end index for time-by-time RDM (exclusive).",
    )
    parser.add_argument(
        "--alpha-responsive",
        type=float,
        default=0.05,
        help="P-value threshold used to keep responsive units.",
    )
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="If provided, save per-ROI CV time-by-time RDM artifacts (.npz).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(pth.SAVEDIR, "timextime-cv"),
        help="Directory where summary tables and artifacts are written.",
    )
    return parser.parse_args()


def _resolve_roi_uids(args: argparse.Namespace) -> list[str]:
    """
    Resolve ROI UID list from either a direct argument or CSV.

    Args:
        args (argparse.Namespace):
            Parsed CLI arguments.

    Returns:
        (list[str]):
            roi_uids (list[str]):
                List of ROI UID strings.
    """
    if args.roi_uid is not None:
        return [args.roi_uid]

    df = pd.read_csv(args.roi_csv)
    if args.uid_col not in df.columns:
        raise ValueError(
            f"Column '{args.uid_col}' not found in {args.roi_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    roi_uids = df[args.uid_col].astype(str).tolist()
    if args.roi_label is not None:
        labels = []
        for uid in roi_uids:
            parts = uid.split(".")
            if len(parts) != 4:
                raise ValueError(f"Unexpected uid format in CSV: {uid}")
            labels.append(f"{parts[2]}_{int(parts[1])}_{parts[3]}")
        df_labels = pd.DataFrame({"uid": roi_uids, "roi": labels})
        roi_uids = df_labels.loc[df_labels["roi"] == args.roi_label, "uid"].tolist()
        if len(roi_uids) == 0:
            raise ValueError(f"No UIDs matched roi-label '{args.roi_label}'.")

    if args.max_rois is not None:
        roi_uids = roi_uids[: args.max_rois]
    return roi_uids


def main() -> None:
    """Run the cross-validated timextime replication pipeline."""
    args = parse_args()
    roi_uids = _resolve_roi_uids(args)
    print(f"[CV] Loaded {len(roi_uids)} ROI UIDs")

    config = cvt.CvConfig(
        alpha_responsive=args.alpha_responsive,
        tstart=args.tstart,
        tend=args.tend,
        k_step=args.k_step,
        k_max=args.k_max,
    )

    if args.roi_label is not None:
        df_summary, artifacts = cvt.run_combined_roi_cv(
            roi_uids=roi_uids,
            roi_label=args.roi_label,
            top_k=args.top_k,
            config=config,
        )
        os.makedirs(args.output_dir, exist_ok=True)
        filepath_parquet = os.path.join(args.output_dir, f"{args.roi_label}_cv_tuning_ed_summary.parquet")
        filepath_csv = os.path.join(args.output_dir, f"{args.roi_label}_cv_tuning_ed_summary.csv")
        df_summary.to_parquet(filepath_parquet, index=False)
        df_summary.to_csv(filepath_csv, index=False)
        if args.save_artifacts:
            filepath_npz = os.path.join(args.output_dir, f"{args.roi_label}_cv_rdms.npz")
            npz_payload = {k: v for k, v in artifacts.items()}
            np.savez_compressed(filepath_npz, **npz_payload)
    else:
        df_summary = cvt.run_many_rois_cv(
            roi_uids=roi_uids,
            dir_output=args.output_dir,
            top_k=args.top_k,
            config=config,
            save_artifacts=args.save_artifacts,
        )
    print("[CV] Completed")
    print(df_summary.head())
    print(f"[CV] Summary written to: {args.output_dir}")


if __name__ == "__main__":
    main()
