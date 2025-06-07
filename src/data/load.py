from pathlib import Path

import kornia as K
import torch


def arr_to_str(a):
    """Returns ;-separated string representing the input"""
    return ";".join([str(x) for x in a.reshape(-1)])


def load_torch_image(file_name: Path | str, device=None):
    """Loads an image and adds batch dimension"""
    if device is None:
        device = torch.device("cpu")

    img = K.io.load_image(file_name, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    return img
