from pathlib import Path
import requests
import hydra
import tarfile
import shutil


@hydra.main(config_path="../configs", config_name="data", version_base=None)
def donwload_landmarks(cfg):
    dataset_dir = Path(cfg.data.landmarks.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Output file path
    for data_name in cfg.data.landmarks.data_name:
        url = f"{cfg.data.landmarks.data_dir_url}{data_name}"
        tar_path = dataset_dir / data_name
        print(tar_path)

        # Download the tar file
        if not tar_path.exists():
            print("Downloading image archive...")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(tar_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("Download complete:", tar_path)
        else:
            print("Archive already exists:", tar_path)

        # Unpack images from directories
        with tarfile.open(tar_path, "r") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    file = tar.extractfile(member)
                    if file:
                        out_path = dataset_dir / Path(member.name).name
                        with open(out_path, "wb") as f:
                            shutil.copyfileobj(file, f)

        # Remove tar file
        tar_path.unlink()
        print("Removed archive:", tar_path)


if __name__ == "__main__":
    donwload_landmarks()
