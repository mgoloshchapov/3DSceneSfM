from pathlib import Path

import fire
import pycolmap

from src.features.keypoints import detect_keypoints
from src.features.matching import get_image_pairs
from src.reconstruction.colmap_utils import import_into_colmap

import numpy as np
import open3d as o3d


def main():
    images_list = list(Path("data/test/church/images/").glob("*.png"))[:10]
    index_pairs = get_image_pairs(images_list, "facebook/dinov2-base")

    feature_dir = Path("./sample_test_features")
    detect_keypoints(images_list, feature_dir)
    print(index_pairs)

    database_path = "colmap.db"
    images_dir = images_list[0].parent

    import_into_colmap(
        images_dir,
        feature_dir,
        database_path,
    )

    # This does RANSAC
    pycolmap.match_exhaustive(database_path)

    mapper_options = pycolmap.IncrementalPipelineOptions()
    mapper_options.min_model_size = 3
    mapper_options.max_num_models = 2

    maps = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=images_dir,
        output_path=Path.cwd() / "incremental_pipeline_outputs",
        options=mapper_options,
    )
    print(len(maps))

    print(maps[0].summary())
    for _, im in maps[0].images.items():
        print(
            "Rotation",
            im.cam_from_world.rotation.matrix(),
            "Translation:",
            im.cam_from_world.translation,
            sep="\n",
        )
        print()

    reconstruction = maps[0]

    # Extract 3D points
    points = np.array([p.xyz for p in reconstruction.points3D.values()])

    # Extract camera centers
    camera_centers = np.array(
        [img.projection_center() for img in reconstruction.images.values()]
    )

    # Create point cloud for reconstructed points
    pcd_points = o3d.geometry.PointCloud()
    pcd_points.points = o3d.utility.Vector3dVector(points)
    pcd_points.paint_uniform_color([0.0, 0.8, 0.0])  # green points clearly

    # Create point cloud for camera positions
    pcd_cameras = o3d.geometry.PointCloud()
    pcd_cameras.points = o3d.utility.Vector3dVector(camera_centers)
    pcd_cameras.paint_uniform_color([1.0, 0.0, 0.0])  # red camera centers clearly

    # Coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

    # Visualize clearly
    o3d.visualization.draw_geometries([pcd_points, pcd_cameras, coord_frame])


def test_db():
    import sqlite3

    # creating file path
    dbfile = "colmap.db"
    # Create a SQL connection to our SQLite database
    con = sqlite3.connect(dbfile)

    # creating cursor
    cur = con.cursor()

    # reading all table names
    table_list = [
        a for a in cur.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    ]
    # here is you table list
    print(table_list)

    # Be sure to close the connection
    con.close()


if __name__ == "__main__":
    # fire.Fire(main)
    fire.Fire(test_db)
