# -*- coding: utf-8 -*-

from collections import OrderedDict
import torch.nn as nn
import torchvision.models as tv_models

from model_summary import model_summary


class CCustomNet(nn.Module):
    def __init__(self):
        super(CCustomNet, self).__init__()

        a = list()
        a.append(('conv_0', nn.Conv2d(3, 16, 3, 2, 1)))
        a.append(('bn_0', nn.BatchNorm2d(16)))
        a.append(('relu_0', nn.ReLU(inplace=True)))
        b = list()
        b.append(('conv_1', nn.Conv2d(16, 32, 3, 1, 1)))
        b.append(('bn_1', nn.BatchNorm2d(32)))
        b.append(('relu_1', nn.ReLU(inplace=True)))

        a = nn.Sequential(OrderedDict(a))
        b = nn.Sequential(OrderedDict(b))
        self.c = nn.Sequential(OrderedDict(a=a, b=b))

    def forward(self, x):
        x = self.c(x)
        return x


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.a = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1),
                               nn.BatchNorm2d(16),
                               nn.ReLU(inplace=True))
        self.b = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1),
                               nn.BatchNorm2d(32),
                               nn.ReLU(inplace=True))
        self.c = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        x = self.a(x)
        x = self.b(x)
        x = self.c(x)
        return x


def main():
    model = tv_models.resnet50()
    # model = tv_models.vgg16()
    # model = CustomModel()
    input_size = (3, 224, 224)

    model_summary(model, input_size, query_granularity=1)


if __name__ == "__main__":
    main()
