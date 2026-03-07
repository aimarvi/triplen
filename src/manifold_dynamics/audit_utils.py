from __future__ import annotations

import os

import pandas as pd

import manifold_dynamics.paths as pth
import visionlab_utils.storage as vst


def roi_uids_for_label(roi_label: str) -> list[str]:
    """
    Resolve all session UIDs for an ROI label.

    UID format in roi-uid.csv: SesIdx.RoiIndex.AREALABEL.Categoty
    ROI label format used here: AREALABEL_RoiIndex_Categoty
    """
    uid_csv = os.path.join(pth.OTHERS, "roi-uid.csv")
    f = vst.fetch(uid_csv)
    df_uid = pd.read_csv(f)

    uids = []
    for uid in df_uid["uid"].astype(str):
        parts = uid.split(".")
        if len(parts) != 4:
            continue
        label = f"{parts[2]}_{int(parts[1])}_{parts[3]}"
        if label == roi_label:
            uids.append(uid)
    return uids
