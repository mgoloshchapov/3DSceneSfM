from src.preprocessing.rotations import train_rotation_classifier

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    train_rotation_classifier(cfg)


if __name__ == "__main__":
    main()
