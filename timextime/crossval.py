from __future__ import annotations

import argparse
import pickle

import fsspec

import numpy as np
import pandas as pd

import manifold_dynamics.neural_utils as nu
import manifold_dynamics.paths as pth
import manifold_dynamics.spike_response_stats as srs
import manifold_dynamics.tuning_utils as tut
import visionlab_utils.storage as vst


def main() -> None:
    """
    Run cross-validated time-time analysis for one ROI target.

    Target format options:
      - ROI UID (4 parts): ``SesIdx.RoiIndex.AREALABEL.Categoty``
        Example: ``11.04.MO1s1.O``
      - ROI key (3 parts): ``RoiIndex.AREALABEL.Categoty``
        Example: ``04.MO1s1.O``
        In this mode, all matching sessions are combined before CV.
    """
    parser = argparse.ArgumentParser(
        description="Cross-validated time-time ED analysis for one ROI target."
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

    uid_csv_uri = f"{pth.OTHERS}/roi-uid.csv"
    uid_csv_local = vst.fetch(uid_csv_uri)
    df_uid = pd.read_csv(uid_csv_local)

    roi_uids: list[str] = []
    if len(target_parts) == 4:
        roi_uids = [args.target]
        roi_label = f"{int(target_parts[1]):02d}.{target_parts[2]}.{target_parts[3]}"
    else:
        roi_index = int(target_parts[0])
        area_label = target_parts[1]
        category = target_parts[2]
        for uid in df_uid["uid"].astype(str):
            parts = uid.split(".")
            if len(parts) != 4:
                continue
            if int(parts[1]) == roi_index and parts[2] == area_label and parts[3] == category:
                roi_uids.append(uid)
        roi_uids = sorted(roi_uids, key=lambda x: int(x.split(".")[0]))
        roi_label = f"{roi_index:02d}.{area_label}.{category}"

    if len(roi_uids) == 0:
        raise ValueError(f"No matching ROI UIDs found for target: {args.target}")

    topk_local = vst.fetch(f"{pth.OTHERS}/topk_vals.pkl")
    with open(topk_local, "rb") as f:
        topk_vals = pickle.load(f)

    top_k = args.top_k
    if top_k is None:
        if roi_label not in topk_vals:
            raise ValueError(f"No top-k entry found for ROI: {roi_label}")
        top_k = int(topk_vals[roi_label]["k"])

    vprint(f"Resolved ROI target {args.target} to UIDs: {roi_uids}")
    vprint(f"Using top-k = {top_k}")

    rasters_by_uid: dict[str, np.ndarray] = {}
    for uid in roi_uids:
        raster_4d = nu.load_cached_session_raster(uid)
        raster_4d = nu.bin_to_psth(raster_4d, bin_size_ms=args.bin_size_ms)
        rasters_by_uid[uid] = raster_4d
        vprint(f"{uid}: raster shape after binning {raster_4d.shape}")

    rows: list[dict[str, object]] = []
    for fold_name, fold_index in [("A_to_B", 0), ("B_to_A", 1)]:
        train_3d_parts: list[np.ndarray] = []
        test_3d_parts: list[np.ndarray] = []
        n_units_responsive_total = 0

        for uid, raster_4d in rasters_by_uid.items():
            split_a = raster_4d[:, :, :, 0::2]
            split_b = raster_4d[:, :, :, 1::2]
            if fold_index == 0:
                train_4d, test_4d = split_a, split_b
            else:
                train_4d, test_4d = split_b, split_a

            pvals = srs.is_responsive(X=train_4d, roi_uid=uid, test_type="paired").squeeze()
            responsive_mask = np.isfinite(pvals) & (pvals < args.alpha)
            n_responsive = int(np.sum(responsive_mask))
            if n_responsive == 0:
                continue

            train_3d_uid = np.nanmean(train_4d, axis=3)[responsive_mask]
            test_3d_uid = np.nanmean(test_4d, axis=3)[responsive_mask]
            train_3d_parts.append(train_3d_uid)
            test_3d_parts.append(test_3d_uid)
            n_units_responsive_total += n_responsive

            vprint(
                f"{fold_name} {uid}: responsive={n_responsive}, "
                f"train={train_3d_uid.shape}, test={test_3d_uid.shape}"
            )

        if n_units_responsive_total < 2:
            raise ValueError(f"{fold_name}: fewer than 2 responsive units across sessions.")

        train_3d = np.concatenate(train_3d_parts, axis=0)
        test_3d = np.concatenate(test_3d_parts, axis=0)

        base = slice(50 - 50, 50 + 0)
        resp = slice(50 + 50, 50 + 220)
        scores = np.nanmean(train_3d[:, resp, :], axis=(0, 1)) - np.nanmean(
            train_3d[:, base, :], axis=(0, 1)
        )
        order = np.argsort(scores)[::-1]
        idx_topk = order[: top_k]
        idx_all = np.arange(test_3d.shape[2])

        R_topk, _ = tut.tuning_rdm(
            X=test_3d,
            indices=idx_topk,
            tstart=args.tstart,
            tend=args.tend,
            metric="correlation",
        )
        R_all, _ = tut.tuning_rdm(
            X=test_3d,
            indices=idx_all,
            tstart=args.tstart,
            tend=args.tend,
            metric="correlation",
        )

        ed_topk = float(tut.ED2(R_topk))
        ed_all = float(tut.ED2(R_all))

        rows.append(
            {
                "roi": roi_label,
                "roi_uid": "|".join(roi_uids),
                "fold": fold_name,
                "n_units_responsive_train": n_units_responsive_total,
                "top_k": int(top_k),
                "ED_topk": ed_topk,
                "ED_all": ed_all,
                "compression_topk_vs_all": 1.0 - (ed_topk / ed_all),
            }
        )

    df_out = pd.DataFrame(rows)

    if args.save:
        s3_out = f"{pth.SAVEDIR}/crossval/{args.target}.pkl"
        with fsspec.open(s3_out, "wb") as f:
            df_out.to_pickle(f)

    print(df_out.to_string(index=False))
    print(f"mean ED_topk: {df_out['ED_topk'].mean():.6f}")
    print(f"mean ED_all:  {df_out['ED_all'].mean():.6f}")



if __name__ == "__main__":
    main()
