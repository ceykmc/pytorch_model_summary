# -*- coding: utf-8 -*-

import torch.nn as nn
import torchvision.models as tv_models
from pytorch_model_summary import PyTorchModelSummary


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.a = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1),
                               nn.BatchNorm2d(16),
                               nn.ReLU(inplace=True))
        self.b = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1),
                               nn.BatchNorm2d(32),
                               nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.a(x)
        x = self.b(x)
        return x


def reset_net_test():
    model = tv_models.resnet50()
    input_size = (3, 224, 224)
    model_summary = PyTorchModelSummary(model, input_size)
    model_summary.summary()


def vgg16_test():
    model = tv_models.resnet50()
    input_size = (3, 224, 224)
    model_summary = PyTorchModelSummary(model, input_size)
    model_summary.summary()


def custom_model_test():
    model = CustomModel()
    input_size = (3, 144, 160)
    model_summary = PyTorchModelSummary(model, input_size)
    model_summary.summary()


def main():
    reset_net_test()
    # vgg16_test()
    # custom_model_test()


if __name__ == "__main__":
    main()
