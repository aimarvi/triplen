from PIL import Image
from collections import OrderedDict

import torch
import torchvision.models as models
import torchvision.transforms as T

# standard imagenet preprocessing
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def load_images(paths):
    # returns tensor [n_images, 3, 224, 224]
    imgs = [transform(Image.open(p).convert('RGB')) for p in paths]
    return torch.stack(imgs)

def get_activations(model, x):
    activations = OrderedDict()
    hooks = []

    def hook_fn(name):
        def hook(module, inp, out):
            # detach so autograd doesn't eat your ram
            activations[name] = out.detach()
        return hook

    # register hooks on all submodules
    for name, module in model.named_modules():
        if name:  # skip the top-level container
            hooks.append(module.register_forward_hook(hook_fn(name)))

    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    return activations
