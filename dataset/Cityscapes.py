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

        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"Expect split to be 'train', 'val' or 'test', got {split}"
            )
        self.image_dir = self.root_dir / "leftImg8bit" / split
        self.ground_truth_dir = self.root_dir / "gtFine" / split
        self.panoptic_dir = self.root_dir / "panoptic" / f"cityscapes_panoptic_{split}"

        # get paths to all image files needed
        self.image_files = list(self.image_dir.rglob("*.png"))
        self.image_files.sort(key=lambda p: p.name)
        self.label_files = list(self.ground_truth_dir.rglob("*labelTrainIds.png"))
        self.label_files.sort(key=lambda p: p.name)
        self.instance_files = list(self.ground_truth_dir.rglob("*instanceTrainIds.png"))
        self.instance_files.sort(key=lambda p: p.name)
        self.panoptic_files = list(self.panoptic_dir.rglob("*.png"))
        self.panoptic_files.sort(key=lambda p: p.name)

        self.training_files = list(
            zip(
                self.image_files,
                self.label_files,
                self.instance_files,
                self.panoptic_files,
            )
        )

        # check they are in the same order
        for image_file, label_file, instance_file, panoptic_file in self.training_files:
            image_city, image_seq, image_frame, *_ = str(image_file.name).split("_")
            label_city, label_seq, label_frame, *_ = str(label_file.name).split("_")
            instance_city, instance_seq, instance_frame, *_ = str(
                instance_file.name
            ).split("_")
            panoptic_city, panoptic_seq, panoptic_frame, *_ = str(
                panoptic_file.name
            ).split("_")
            assert image_city == label_city == instance_city == panoptic_city
            assert image_seq == label_seq == instance_seq == panoptic_seq
            assert image_frame == label_frame == instance_frame == panoptic_frame

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
        to_tensor = transforms.PILToTensor()
        image_file, label_file, instance_file, panoptic_file = self.training_files[
            index
        ]
        image = Image.open(image_file).convert("RGB")
        image = self.transform(image)
        label = Image.open(label_file).convert("RGB")
        label = to_tensor(label)
        instance = Image.open(instance_file).convert("RGB")
        instance = to_tensor(instance)
        panoptic = Image.open(panoptic_file).convert("RGB")
        panoptic = to_tensor(panoptic)

        return image, label, instance, panoptic

    def __len__(self):
        return len(self.training_files)


def _test():
    root = r"/home/cyrus/_Project/segment/training_data"
    dataset_train = CityscapesDataset(root)
    dataset_val = CityscapesDataset(root, split="val")
    dataset_test = CityscapesDataset(root, split="test")
    print(f"{len(dataset_train)=}| {len(dataset_val)=}| {len(dataset_test)=}")
    image, label, instance, panoptic = dataset_train[1674]
    print(image[0, :3, :3])
    print(instance[0, :9, :9])


if __name__ == "__main__":
    _test()
