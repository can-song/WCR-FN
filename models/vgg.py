'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init
# from .modules import WCRConv2d as Conv2d, FeatsNorm as NonLinear
# from .modules import WCRConv2d as Conv2d, GroupSaparse as NonLinear
# from torch.nn import Conv2d, ReLU as NonLinear
# from torch.nn import Conv2d
from .modules import WCRConv2d as Conv2d
# from .modules import GroupSaparse as NonLinear
from .modules import FeatsNorm as NonLinear

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]
tch = 128


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(tch*8, tch*8),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(tch*8, tch*8),
            nn.ReLU(True),
            nn.Linear(tch*8, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), NonLinear()]
            else:
                layers += [conv2d, NonLinear()]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'A': [tch, 'M', tch*2, 'M', tch*4, tch*4, 'M', tch*8, tch*8, 'M', tch*8, tch*8, 'M'],
    'B': [tch, tch, 'M', tch*2, tch*2, 'M', tch*4, tch*4, 'M', tch*8, tch*8, 'M', tch*8, tch*8, 'M'],
    'D': [tch, tch, 'M', tch*2, tch*2, 'M', tch*4, tch*4, tch*4, 'M', tch*8, tch*8, tch*8, 'M', tch*8, tch*8, tch*8, 'M'],
    'E': [tch, tch, 'M', tch*2, tch*2, 'M', tch*4, tch*4, tch*4, tch*4, 'M', tch*8, tch*8, tch*8, tch*8, 'M',
          tch*8, tch*8, tch*8, tch*8, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))
