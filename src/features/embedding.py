from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor as T
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from src.data.load import load_torch_image


def embed_images(
    paths: list[Path],
    model_name: str,
    device: torch.device = None,
) -> T:
    """Computes image embeddings.

    Returns a tensor of shape [len(filenames), output_dim]
    """
    if device is None:
        device = torch.device("cpu")

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)

    embeddings = []

    for _, path in tqdm(enumerate(paths), desc="Global descriptors"):
        image = load_torch_image(path)

        with torch.inference_mode():
            inputs = processor(images=image, return_tensors="pt", do_rescale=False).to(
                device
            )
            outputs = model(**inputs)  # last_hidden_state and pooled

            # Max pooling over all the hidden states but the first (starting token)
            # To obtain a tensor of shape [1, output_dim]
            # We normalize so that distances are computed in a better fashion later
            embedding = F.normalize(
                outputs.last_hidden_state[:, 1:].max(dim=1)[0], dim=-1, p=2
            )

        embeddings.append(embedding.detach().cpu())
    return torch.cat(embeddings, dim=0)
