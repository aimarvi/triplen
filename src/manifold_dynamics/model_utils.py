from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image

import torch
import torchvision.transforms as T

IMAGENET_PREPROCESS = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_image_tensor(
    image_paths: Sequence[str | Path],
    transform: T.Compose | None = None,
    sort_paths: bool = True,
) -> torch.Tensor:
    """
    Load images and return a batched tensor for model input.

    Args:
        image_paths: Image file paths.
        transform: Torchvision transform pipeline. Defaults to ImageNet preprocessing.
        sort_paths: Sort paths for stable row ordering across runs.

    Returns:
        Tensor of shape (n_images, 3, 224, 224) when default transform is used.
    """
    if transform is None:
        transform = IMAGENET_PREPROCESS

    paths = [Path(p) for p in image_paths]
    if sort_paths:
        paths = sorted(paths)
    if not paths:
        raise ValueError("image_paths is empty.")

    images = [transform(Image.open(path).convert("RGB")) for path in paths]
    return torch.stack(images, dim=0)


def extract_model_activations(
    model: torch.nn.Module,
    x: torch.Tensor,
    layer_names: Iterable[str] | None = None,
) -> OrderedDict[str, torch.Tensor]:
    """
    Run a forward pass and collect layer activations.

    Args:
        model: PyTorch model in eval mode.
        x: Batched model input tensor.
        layer_names: Optional list of module names to capture.
            If None, captures all named submodules (legacy behavior).

    Returns:
        OrderedDict mapping layer name -> detached output tensor.
    """
    named_modules = dict(model.named_modules())
    if layer_names is None:
        requested_names = [name for name in named_modules if name]
    else:
        requested_names = list(layer_names)
        missing = [name for name in requested_names if name not in named_modules]
        if missing:
            raise ValueError(f"Unknown layer names: {missing}")

    activations: OrderedDict[str, torch.Tensor] = OrderedDict()
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def build_hook(name: str):
        def _hook(_module, _inp, out):
            # Detach and move to CPU for lightweight downstream serialization.
            activations[name] = out.detach().cpu()

        return _hook

    try:
        for name in requested_names:
            hooks.append(named_modules[name].register_forward_hook(build_hook(name)))

        # Standard inference context: no gradient tracking.
        with torch.no_grad():
            _ = model(x)
    finally:
        for hook in hooks:
            hook.remove()

    return activations
