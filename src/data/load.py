from pathlib import Path

import kornia as K
import torch


def load_torch_image(file_name: Path | str, device=None):
    """Loads an image and adds batch dimension"""
    if device is None:
        device = torch.device("cpu")

    img = K.io.load_image(file_name, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    return img
