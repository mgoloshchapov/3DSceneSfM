import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import lightning as L

import importlib

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import random


class RotationDataset(Dataset):
    def __init__(self, image_dir: Path, transform=None):
        self.image_paths = list(image_dir.glob("*.jpg"))
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
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
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


def load_backbone_from_config(cfg):
    module = importlib.import_module(cfg.module)
    backbone_class = getattr(module, cfg.class_name)
    model = backbone_class(pretrained=cfg._parent_.model.pretrained)

    if hasattr(model, "fc"):
        model.fc = nn.Identity()
    elif hasattr(model, "classifier"):
        model.classifier = nn.Identity()

    return model, cfg.output_features


class RotationClassifier(L.LightningModule):
    def __init__(self, backbone_cfg, lr=1e-3, pretrained=True):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone_cfg"])

        self.backbone, features = load_backbone_from_config(backbone_cfg)
        self.classifier = nn.Linear(features, 4)

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
