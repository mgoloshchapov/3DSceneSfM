from pathlib import Path

from src.reconstruction.database import COLMAPDatabase
from src.reconstruction.h5_to_db import add_keypoints, add_matches


def import_into_colmap(
    path: Path,
    feature_dir: Path,
    database_path: str = "colmap.db",
) -> None:
    """Adds keypoints into colmap"""
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    single_camera = False
    fname_to_id = add_keypoints(
        db, feature_dir, path, "", "simple-pinhole", single_camera
    )
    add_matches(
        db,
        feature_dir,
        fname_to_id,
    )
    db.commit()
