from __future__ import annotations

import pickle
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import manifold_dynamics.neural_utils as nu
import manifold_dynamics.paths as pth
import manifold_dynamics.tuning_utils as tut
import visionlab_utils.storage as vst


# Ad-hoc configuration
ALPHA = 0.05
BIN_SIZE_MS = 20
SAVE = True
VERBOSE = True


def vprint(msg: str) -> None:
    if VERBOSE:
        print(msg)


vprint("Loading top-k values from S3...")
topk_local = vst.fetch(f"{pth.OTHERS}/topk_vals.pkl")
with open(topk_local, "rb") as f:
    topk_vals = pickle.load(f)

rows = []
missing_ed = []
failed_raster = []

for roi_key in sorted(topk_vals):
    vprint(f"Processing {roi_key}")

    try:
        ed_local = vst.fetch(f"{pth.SAVEDIR}/timextime/ed_main/{roi_key}.pkl")
        df_ed = pd.read_pickle(ed_local)
    except Exception:
        missing_ed.append(roi_key)
        continue

    local_rows = df_ed[df_ed["condition"] == "local"]
    global_rows = df_ed[df_ed["condition"] == "global"]
    if len(local_rows) != 1 or len(global_rows) != 1:
        missing_ed.append(roi_key)
        continue

    try:
        raster_4d = nu.significant_trial_raster(
            roi_uid=roi_key,
            alpha=ALPHA,
            bin_size_ms=BIN_SIZE_MS,
        )
    except Exception:
        failed_raster.append(roi_key)
        continue

    X = np.nanmean(raster_4d, axis=3)
    top_k = int(topk_vals[roi_key]["k"])
    order = tut.rank_images_by_response(X)
    resp = np.nanmean(X[:, tut.RESP, :], axis=(0, 1))
    base_mean = np.nanmean(X[:, tut.BASE, :], axis=(0, 1))
    base_std = np.nanstd(X[:, tut.BASE, :], axis=(0, 1))
    base_std = np.where(base_std == 0, np.nan, base_std)
    zscores = (resp - base_mean) / base_std

    rows.append(
        {
            "roi_key": roi_key,
            "major_selectivity": roi_key.split(".")[-1],
            "top_k": top_k,
            "local_ed": float(local_rows["ED"].iloc[0]),
            "global_ed": float(global_rows["ED"].iloc[0]),
            "avg_topk_zscore": float(np.nanmean(zscores[order][:top_k])),
            "n_units": int(raster_4d.shape[0]),
        }
    )

df_out = pd.DataFrame(rows).sort_values(
    ["major_selectivity", "roi_key"]
).reset_index(drop=True)

print(df_out.to_string(index=False))

if len(df_out) == 0:
    raise ValueError("No ROI rows were collected.")

df_model = df_out.dropna(
    subset=["local_ed", "global_ed", "top_k", "avg_topk_zscore", "n_units"]
).copy()
df_model["major_selectivity"] = pd.Categorical(df_model["major_selectivity"])

formula = "local_ed ~ global_ed + top_k + n_units + avg_topk_zscore + C(major_selectivity)"
ols = smf.ols(formula, data=df_model).fit(cov_type="HC3")

print("\nROI-level model coefficients:\n")
print(ols.summary())

if missing_ed:
    print("\nMissing ed_main:")
    print(", ".join(missing_ed))

if failed_raster:
    print("\nFailed raster:")
    print(", ".join(failed_raster))

if SAVE:
    csv_path = Path("aux_controls/sampling_strength_summary.csv")
    df_out.to_csv(csv_path, index=False)

    s3_csv = f"{pth.SAVEDIR}/aux_controls/sampling_strength_summary.csv"
    with fsspec.open(s3_csv, "w") as f:
        df_out.to_csv(f, index=False)

# save to local
#     download_csv = Path.home() / "Downloads" / "sampling_strength_summary.csv"
#     df_out.to_csv(download_csv, index=False)
