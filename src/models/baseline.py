from pathlib import Path
import pycolmap
import numpy as np
import gc
from copy import deepcopy

from src.utils.general import get_filenames
from src.features.matching import get_image_pairs
from src.features.keypoints import detect_keypoints, keypoint_distances
from src.reconstruction.colmap_utils import import_into_colmap

from omegaconf import DictConfig


def baseline(cfg: DictConfig) -> None:
    """
    Returns a list of filenames in the specified directory.

    Args:
        image_dir (Union[str, Path]): Path to the directory with images of scene.
        output_dir (Union[str, Path]): Path to the directory, where results will be saved.

    Returns:
        List[str]: A list containing filenames.
    """

    device = cfg["model"]["device"]

    image_paths = get_filenames(cfg["data"]["input_data_path"])
    results = {}
    print(f"Got {len(image_paths)} images")

    output_dir = Path(cfg["data"]["output_data_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = Path(cfg["data"]["database_path"])
    if database_path.exists():
        database_path.unlink()

    # 1. Get the pairs of images that are somewhat similar
    index_pairs = get_image_pairs(
        image_paths,
        **cfg["model"]["pair_matching_args"],
        device=device,
    )
    gc.collect()

    # 2. Detect keypoints of all images
    detect_keypoints(
        image_paths,
        output_dir,
        **cfg["model"]["keypoint_detection_args"],
        device=device,
    )
    gc.collect()

    # 3. Match  keypoints of pairs of similar images
    keypoint_distances(
        image_paths,
        index_pairs,
        output_dir,
        **cfg["model"]["keypoint_distances_args"],
        device=device,
    )
    gc.collect()

    # sleep(1)

    # 4.1. Import keypoint distances of matches into colmap for RANSAC
    import_into_colmap(
        cfg["data"]["input_data_path"],
        cfg["data"]["output_data_path"],
        database_path,
    )

    reconstruction_path = Path(cfg["data"]["reconstruction_path"])
    reconstruction_path.mkdir(parents=True, exist_ok=True)

    # 4.2. Compute RANSAC (detect match outliers)
    # By doing it exhaustively we guarantee we will find the best possible configuration
    pycolmap.match_exhaustive(database_path)

    mapper_options = pycolmap.IncrementalPipelineOptions(
        cfg["model"]["colmap_mapper_options"]
    )

    # 5.1 Incrementally start reconstructing the scene (sparse reconstruction)
    # The process starts from a random pair of images and is incrementally extended by
    # registering new images and triangulating new points.
    maps = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=cfg["data"]["input_data_path"],
        output_path=reconstruction_path,
        options=mapper_options,
    )

    print(maps)
    # clear_output(wait=False)

    # 5.2. Look for the best reconstruction: The incremental mapping offered by
    # pycolmap attempts to reconstruct multiple models, we must pick the best one
    images_registered = 0
    best_idx = None

    print("Looking for the best reconstruction")

    if isinstance(maps, dict):
        for idx1, rec in maps.items():
            print(idx1, rec.summary())
            try:
                if len(rec.images) > images_registered:
                    images_registered = len(rec.images)
                    best_idx = idx1
            except Exception:
                continue

    # Parse the reconstruction object to get the rotation matrix and translation vector
    # obtained for each image in the reconstruction
    if best_idx is not None:
        for _, im in maps[best_idx].images.items():
            key = cfg["data"]["input_data_path"] / im.name
            results[key] = {}
            results[key]["R"] = deepcopy(im.cam_from_world.rotation.matrix())
            results[key]["t"] = deepcopy(np.array(im.cam_from_world.translation))

    print(f"Registered: {len(results)} images")
    # print(f"Total: {len(data_dict[dataset][scene])} images")
    # create_submission(results, data_dict, config.base_path)
    gc.collect()
