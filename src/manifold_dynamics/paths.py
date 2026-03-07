from __future__ import annotations

from os import getenv
from pathlib import Path, PurePosixPath


def _join_path(base: str, *parts: str) -> str:
    """Join local paths and cloud URIs with stable POSIX separators."""
    if "://" in base:
        protocol, path = base.split("://", 1)
        joined = PurePosixPath(path.strip("/"))
        for part in parts:
            joined = joined / str(part).strip("/")
        return f"{protocol}://{joined}"
    return str(Path(base).joinpath(*parts))


VISLAB_USERNAME = getenv("VISLAB_USERNAME")
if not VISLAB_USERNAME:
    raise EnvironmentError("VISLAB_USERNAME is not set.")

# Root directory (bucket namespace for this user)
ROOT = _join_path("s3://visionlab-members", VISLAB_USERNAME)

# Top level data directory
DATADIR = _join_path(ROOT, "datasets", "triple-n", "V1")

# Standard subdirectories
PROCESSED = _join_path(DATADIR, "Processed")
RAW = _join_path(DATADIR, "Raw", "GoodUnitV2")
OTHERS = _join_path(DATADIR, "others")
IMAGEDIR = _join_path(DATADIR, "NSD1000_LOC")

# Figure and template directories
SAVEDIR = _join_path(ROOT, "manifold-dynamics")
TEMPLATEDIR = _join_path(ROOT, "datasets", "NMT_v2.0_sym")
