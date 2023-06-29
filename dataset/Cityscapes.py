"""
To load the Cityscapes Dataset for panoptic segmentation
source: https://www.cityscapes-dataset.com/
"""

from pathlib import Path
import torch
import os
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import time
from PIL import Image


class CityscapesDataset(data.Dataset):
    def __init__(self, root: str | Path, split: str = "train", transform=None):
        super().__init__()

        if type(root) == str:
            self.root_dir = Path(root)
        else:
            self.root_dir = root

        assert self.root_dir.is_dir()

        if split in {"train", "val", "test"}:
            self.label_dir = self.root_dir / "panoptic" / f"cityscapes_panoptic_{split}"
            self.image_dir = self.root_dir / "leftImg8bit" / split
        else:
            raise ValueError(
                f"Expect split to be 'train', 'val' or 'test', got {split}"
            )

        self.label_files = sorted(self.label_dir.rglob("*.png"), key=lambda p: p.name)
        self.image_files = sorted(self.image_dir.rglob("*.png"), key=lambda p: p.name)
        for label_file, image_file in zip(self.label_files, self.image_files):
            label_city, label_seq, label_frame, *_ = str(label_file.name).split("_")
            image_city, image_seq, image_frame, *_ = str(image_file.name).split("_")
            assert label_city == image_city
            assert label_seq == image_seq
            assert label_frame == image_frame
        self.label_image_file_pairs = list(zip(self.label_files, self.image_files))

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True
                    ),
                ]
            )
        else:
            self.transform = transform

    def __getitem__(self, index):
        to_tensor = transforms.ToTensor()
        label_file, image_file = self.label_image_file_pairs[index]
        label = Image.open(label_file).convert("RGB")
        label = to_tensor(label)
        image = Image.open(image_file).convert("RGB")
        image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.label_image_file_pairs)
