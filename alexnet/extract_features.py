from __future__ import annotations

import pickle
from pathlib import Path

import fsspec
import torch
import torchvision.models as models

import manifold_dynamics.model_utils as mut
import manifold_dynamics.paths as pth
import visionlab_utils.storage as vst


# Ad-hoc configuration
IMAGE_DIR_URI = pth.IMAGEDIR
OUTPUT_URI = f"{pth.SAVEDIR}/alexnet/alexnet_acts.pkl"
LAYER_NAMES =   [     # Example: ["features.12", "classifier.1", "classifier.2"]
      "features.2",   # pool1
      "features.5",   # pool2
      "features.7",   # relu3
      "features.9",   # relu4
      "features.11",  # relu5
      "features.12",  # pool5
      "classifier.2", # relu6 (fc6)
      "classifier.5", # relu7 (fc7)
  ]
VERBOSE = True


def vprint(msg: str) -> None:
    if VERBOSE:
        print(msg)


def _protocol_to_str(protocol) -> str:
    if isinstance(protocol, (tuple, list)):
        return str(protocol[0])
    return str(protocol)


def list_image_uris(image_dir_uri: str) -> list[str]:
    """
    List image objects under an S3 directory URI.

    Returns:
        Sorted list of S3 URIs.
    """
    fs, path = fsspec.core.url_to_fs(image_dir_uri)
    protocol = _protocol_to_str(fs.protocol)
    entries = fs.ls(path, detail=True)

    image_uris: list[str] = []
    for entry in entries:
        name = entry["name"]
        file_type = entry.get("type", "")
        suffix = Path(name).suffix.lower()
        if file_type != "file":
            continue
        if suffix != ".bmp":
            continue
        image_uris.append(f"{protocol}://{name.lstrip('/')}")

    image_uris = sorted(image_uris, key=lambda x: Path(x).name)
    return image_uris


# 1) List all NSD1000/LOC images from S3
image_uris = list_image_uris(IMAGE_DIR_URI)
if len(image_uris) == 0:
    raise ValueError(f"No image files found under {IMAGE_DIR_URI}")
vprint(f"Found {len(image_uris)} images under {IMAGE_DIR_URI}")

# 2) Fetch each image to local cache path via visionlab_utils
local_image_paths = [Path(vst.fetch(uri)) for uri in image_uris]
vprint(f"Fetched {len(local_image_paths)} images into local cache")

# 3) Build input tensor
# local_image_paths are already in stable order, so keep sort_paths=False
x = mut.load_image_tensor(local_image_paths, sort_paths=False)
vprint(f"Input tensor shape: {tuple(x.shape)}")

# 4) Load pretrained AlexNet and extract features
model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
model.eval()
acts = mut.extract_model_activations(model=model, x=x, layer_names=LAYER_NAMES)
vprint(f"Captured activations for {len(acts)} layers")

# 5) Save activations pickle to S3
with fsspec.open(OUTPUT_URI, "wb") as f:
    pickle.dump(acts, f)

print(f"Saved activations to: {OUTPUT_URI}")
