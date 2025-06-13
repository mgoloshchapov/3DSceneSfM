import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from PIL import Image
from pathlib import Path
import random

import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import fire


# # Disable proxy
# os.environ["NO_PROXY"] = "127.0.0.1,localhost"
# os.environ.pop("HTTP_PROXY", None)


class RotationDataset(Dataset):
    def __init__(self, image_dir: Path, transform=None):
        self.image_paths = list(Path(image_dir).glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        label = random.randint(0, 3)
        angle = label * 90
        img = img.rotate(angle)
        if self.transform:
            img = self.transform(img)
        return img, label


class RotationDataModule(L.LightningDataModule):
    def __init__(self, image_dir: str, batch_size: int = 64, img_size: int = 128):
        super().__init__()
        self.image_dir = Path(image_dir)
        self.batch_size = batch_size
        self.img_size = img_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )

    def setup(self, stage=None):
        full_dataset = RotationDataset(self.image_dir, self.transform)
        n = len(full_dataset)
        split = int(n * 0.8)
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [split, n - split]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )


class RotationClassifier(L.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def train_model():
    mlf_logger = MLFlowLogger(
        experiment_name="rotation-classifier",
        tracking_uri="http://127.0.0.1:8080",
        run_name="rotation-cnn-run",
    )

    datamodule = RotationDataModule(image_dir="landmark_images", batch_size=64)
    model = RotationClassifier(lr=1e-3)

    checkpoint_cb = ModelCheckpoint(
        monitor="val_acc", mode="max", save_top_k=1, filename="best-rotation-model"
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = L.Trainer(
        max_epochs=10,
        logger=mlf_logger,
        callbacks=[checkpoint_cb, lr_monitor],
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule)
    return model


if __name__ == "__main__":
    model = fire.Fire(train_model())
