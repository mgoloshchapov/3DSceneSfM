from src.models.baseline import baseline
import fire
import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig):
    baseline(cfg)


if __name__ == "__main__":
    fire.Fire(main())
