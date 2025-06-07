import itertools
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.features.embedding import embed_images


def get_pairs_exhaustive(lst: list[Any]) -> list[tuple[int, int]]:
    """Obtains all possible index pairs of a list"""
    return list(itertools.combinations(range(len(lst)), 2))


def get_image_pairs(
    paths: list[Path],
    model_name: str,
    similarity_threshold: float = 2.0,
    tolerance: int = 1000,
    min_matches: int = 20,
    exhaustive_if_less: int = 20,
    p: float = 2.0,
    device: torch.device = None,
) -> list[tuple[int, int]]:
    """Obtains pairs of similar images"""
    if device is None:
        device = torch.device("cpu")

    if len(paths) <= exhaustive_if_less:
        return get_pairs_exhaustive(paths)

    matches = []

    # Embed images and compute distances for filtering
    embeddings = embed_images(paths, model_name)
    distances = torch.cdist(embeddings, embeddings, p=p)

    # Remove pairs above similarity threshold (if enough)
    mask = distances <= similarity_threshold
    image_indices = np.arange(len(paths))

    for current_image_index in range(len(paths)):
        mask_row = mask[current_image_index]
        indices_to_match = image_indices[mask_row]

        # We don't have enough matches below the threshold, we pick most similar ones
        if len(indices_to_match) < min_matches:
            indices_to_match = np.argsort(distances[current_image_index])[:min_matches]

        for other_image_index in indices_to_match:
            # Skip an image matching itself
            if other_image_index == current_image_index:
                continue

            # We need to check if we are below a certain distance tolerance
            # since for images that don't have enough matches, we picked
            # the most similar ones (which could all still be very different
            # to the image we are analyzing)
            if distances[current_image_index, other_image_index] < tolerance:
                # Add the pair in a sorted manner to avoid redundancy
                matches.append(
                    tuple(sorted((current_image_index, other_image_index.item())))
                )

    return sorted(list(set(matches)))
