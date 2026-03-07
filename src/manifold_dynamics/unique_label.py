from __future__ import annotations

from pathlib import Path

import pandas as pd

import manifold_dynamics.paths as pth


INPUT_FILENAME = "exclude_area.xls"
OUTPUT_FILENAME = "roi-uid.csv"
ID_COLS = ["SesIdx", "RoiIndex", "AREALABEL", "Categoty"]
OUTPUT_COLS = ["uid", "y1", "y2"]


def _normalize_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Format ID columns with zero-padding for stable UID construction."""
    out = df.copy()
    out["SesIdx"] = out["SesIdx"].map(lambda x: f"{int(x):02d}")
    out["RoiIndex"] = out["RoiIndex"].map(lambda x: f"{int(x):02d}")
    return out


def build_roi_uid_table(df_source: pd.DataFrame) -> pd.DataFrame:
    """
    Build the ROI UID table from source metadata.

    The UID format is:
      ``SesIdx.RoiIndex.AREALABEL.Categoty``

    Returns:
        pd.DataFrame with columns ``uid``, ``y1``, ``y2``.
    """
    df = _normalize_id_columns(df_source)
    df["uid"] = df[ID_COLS].astype(str).agg(".".join, axis=1)
    return df[OUTPUT_COLS].copy()


def main() -> None:
    """Read ROI metadata, build UID table, and save the output CSV."""
    others_dir = Path(pth.OTHERS)
    source_path = others_dir / INPUT_FILENAME
    output_path = others_dir / OUTPUT_FILENAME

    df_source = pd.read_excel(source_path)
    df_uid = build_roi_uid_table(df_source)
    df_uid.to_csv(output_path, index=False)
    print("saved to:", output_path)


if __name__ == "__main__":
    main()
