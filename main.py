# import numpy as np
# import open3d as o3d
# import fire


# def main():
#     points = np.load("misc/points.npy")
#     # camera_centers = np.load('misc/camera_centers.npy')

#     pcd_points = o3d.geometry.PointCloud()
#     pcd_points.points = o3d.utility.Vector3dVector(points)
#     pcd_points.paint_uniform_color([0.0, 0.8, 0.0])  # green points clearly

#     # # Create point cloud for camera positions
#     # pcd_cameras = o3d.geometry.PointCloud()
#     # pcd_cameras.points = o3d.utility.Vector3dVector(camera_centers)
#     # pcd_cameras.paint_uniform_color([1.0, 0.0, 0.0])  # red camera centers clearly

#     # Coordinate frame for reference
#     # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

#     # Visualize clearly
#     o3d.visualization.draw_geometries([pcd_points])


# if __name__ == "__main__":
#     fire.Fire(main())

from src.models.baseline import baseline
import fire
import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig):
    baseline(cfg)


if __name__ == "__main__":
    fire.Fire(main())
