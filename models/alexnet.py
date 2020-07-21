import torch
import torch.nn as nn
from .modules import WCRConv2d as Conv2d, FeatsNorm as NonLinear
# from torch.nn import Conv2d, ReLU as NonLinear


__all__ = ['alexnet']


class AlexNet(nn.Module):

    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            # nn.ReLU(inplace=True),
            NonLinear(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2d(64, 192, kernel_size=5, padding=2),
            # nn.ReLU(inplace=True),
            NonLinear(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2d(192, 384, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            NonLinear(),
            Conv2d(384, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            NonLinear(),
            Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            NonLinear(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def alexnet(num_classes=10):
    return AlexNet(num_classes=num_classes)