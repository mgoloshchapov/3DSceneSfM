from src.preprocessing.rotations import RotationClassifier, RotationDataModule

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor


@hydra.main(config_path="../config", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    model = RotationClassifier(
        cfg.backbone, lr=cfg.model.lr, pretrained=cfg.model.pretrained
    )

    datamodule = RotationDataModule(
        image_dir=cfg.data.landmarks.dataset_dir,
        batch_size=cfg.datamodule.batch_size,
        img_size=cfg.datamodule.img_size,
        num_workers=cfg.datamodule.num_workers,
    )

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[
            ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1),
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
