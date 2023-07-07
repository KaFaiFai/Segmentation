import torch
import torch.nn as nn
import torch.nn.functional as F

from math import floor


class LeNet(nn.Module):
    def __init__(self, num_class, input_resolution=224, **kwargs):
        super().__init__()
        self.num_class = num_class

        final_resolution = floor(floor(input_resolution / 2 - 2) / 2 - 2)
        final_num = 16 * final_resolution**2

        self.resize = nn.AdaptiveAvgPool2d(input_resolution)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(final_num, final_num // 10)
        self.fc2 = nn.Linear(final_num // 10, num_class)

    def forward(self, x):
        x = self.resize(x)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def __repr__(self):
        return f"LeNet({self.num_class})"


def test():
    from torchinfo import summary

    batch_size = 5
    model = LeNet(num_class=3)
    summary(model, input_size=(batch_size, 3, 128, 128))


if __name__ == "__main__":
    test()
