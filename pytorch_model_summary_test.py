# -*- coding: utf-8 -*-

import torch.nn as nn
import torchvision.models as tv_models
from pytorch_model_summary import PyTorchModelSummary
import argparse


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


def reset_net_test(max_depth):
    model = tv_models.resnet50()
    input_size = (3, 224, 224)
    model_summary = PyTorchModelSummary(model, input_size)
    model_summary.summary(max_depth)


def vgg16_test(max_depth):
    model = tv_models.vgg16()
    input_size = (3, 224, 224)
    model_summary = PyTorchModelSummary(model, input_size)
    model_summary.summary(max_depth)


def custom_model_test(max_depth):
    model = CustomModel()
    input_size = (3, 144, 160)
    model_summary = PyTorchModelSummary(model, input_size)
    model_summary.summary(max_depth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_depth', default=-1, type=int)
    args = parser.parse_args()
    reset_net_test(args.max_depth)
    # vgg16_test(args.max_depth)
    # custom_model_test(args.max_depth)


if __name__ == "__main__":
    main()
