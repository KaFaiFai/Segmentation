"""
To load the Cityscapes Dataset for panoptic segmentation
source: https://www.cityscapes-dataset.com/
"""

from pathlib import Path
import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
from PIL import Image


class CityscapesDataset(data.Dataset):
    HEIGHT, WIDTH = 1024, 2048
    IMAGENET_MEAN = np.array((0.485, 0.456, 0.406))
    IMAGENET_STD = np.array((0.229, 0.224, 0.225))
    INPUT_CHANNELS, OUTPUT_CHANNELS = 3, 19

    def __init__(self, root: str | Path, split: str = "train", scale=1):
        super().__init__()

        if type(root) == str:
            self.root_dir = Path(root)
        else:
            self.root_dir = root
        assert self.root_dir.is_dir()
        assert scale > 0, "scale of images must be > 0"

        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"Expect split to be 'train', 'val' or 'test', got {split}"
            )
        self.image_dir = self.root_dir / "leftImg8bit" / split
        self.ground_truth_dir = self.root_dir / "gtFine" / split
        self.panoptic_dir = self.root_dir / "gtFine" / f"cityscapes_panoptic_{split}"

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

        # transform for train images and labels/instance
        # the size is scaled accordingly
        self.scale = scale
        self.size = (int(self.HEIGHT * scale), int(self.WIDTH * scale))
        self.transform_image = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.size, antialias=None),
                transforms.Normalize(
                    mean=self.IMAGENET_MEAN,
                    std=self.IMAGENET_STD,
                    inplace=True,
                ),
            ]
        )

        self.transform_mask = transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.Resize(
                    self.size,
                    interpolation=transforms.InterpolationMode.NEAREST,
                    antialias=None,
                ),
                _multiclass_mask_to_multiple_binary_masks,
            ]
        )

    def __getitem__(self, index):
        to_tensor = transforms.PILToTensor()

        image_file, label_file, instance_file, panoptic_file = self.training_files[
            index
        ]
        image = Image.open(image_file).convert("RGB")
        image = self.transform_image(image)
        label = Image.open(label_file).convert("RGB")
        label = self.transform_mask(label)
        instance = Image.open(instance_file).convert("RGB")
        instance = self.transform_mask(instance)
        panoptic = Image.open(panoptic_file).convert("RGB")
        panoptic = to_tensor(panoptic)

        return image, label, instance, panoptic

    def __len__(self):
        return len(self.training_files)

    @classmethod
    def plot_image(cls, image: torch.Tensor, save_to="image.png"):
        # reverse the normalization used in training
        inverse_mean = -cls.IMAGENET_MEAN / cls.IMAGENET_STD
        inverse_std = 1 / cls.IMAGENET_STD
        transform = transforms.Compose(
            [
                transforms.Normalize(mean=inverse_mean, std=inverse_std),
                transforms.ToPILImage(),
            ]
        )
        image = transform(image)
        image.save(save_to)

    @classmethod
    def plot_mask(cls, masks: torch.Tensor, save_to="mask.png"):
        from .CityscapesLabels import trainId_to_color

        board = torch.zeros(3, *masks.shape[1:])
        for i, mask in enumerate(masks):
            cur_color = trainId_to_color(i)
            # repeat color tensor for braodcasting to the whole board
            repeated_color = (
                torch.tensor(cur_color).repeat(*mask.shape, 1).permute((2, 0, 1))
            )
            board += mask * repeated_color

        # from int32 tensor to pillow image
        board = 255 - board.to(torch.float32)
        board = transforms.ToPILImage()(board)
        board.save(save_to)

    @classmethod
    def plot_output(cls, output: torch.Tensor, save_to="prediction.png"):
        # convert output to multiple binary masks
        output_max = torch.max(output, 0, keepdim=True).values
        prediction = (output == output_max).to(int)
        cls.plot_mask(prediction, save_to=save_to)


def _multiclass_mask_to_multiple_binary_masks(mask: torch.Tensor):
    # check if different channels contain the same masks
    if len(mask.shape) == 3:
        num_channels = mask.shape[0]
        for c in range(num_channels - 1):
            assert torch.all(mask[c] == mask[c + 1])
        mask = mask[0, :, :]

    masks = []
    num_classes = 19

    for class_id in range(num_classes):
        masks.append((mask == class_id).to(int))

    binary_masks = torch.stack(masks)
    return binary_masks


def _test():
    import os
    from dotenv import load_dotenv
    load_dotenv()
    root = os.environ['CITYSCAPES_DATASET']
    print(root)

    dataset_train = CityscapesDataset(root, scale=0.4)
    dataset_val = CityscapesDataset(root, split="val")
    dataset_test = CityscapesDataset(root, split="test")
    print(f"{len(dataset_train)=}| {len(dataset_val)=}| {len(dataset_test)=}")

    image, label, instance, panoptic = dataset_train[0]
    print(image[0, :3, :3])
    print(instance[0, :3, :3])
    print(image.shape, label.shape, instance.shape, panoptic.shape)

    CityscapesDataset.plot_image(image)
    CityscapesDataset.plot_mask(label)
    output = torch.rand(label.shape)
    CityscapesDataset.plot_output(output)


if __name__ == "__main__":
    _test()
