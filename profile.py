# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


def count_conv2d(m, x, y):
    cin = m.in_channels // m.groups
    kh, kw = m.kernel_size

    # ops per output element
    kernel_mul = kh * kw * cin
    # 加法
    kernel_add = kh * kw * cin - 1
    bias_ops = 1 if m.bias is not None else 0
    mul_ops = kernel_mul
    add_ops = kernel_add + bias_ops
    # total ops
    num_out_elements = y.numel()
    total_mul_ops = num_out_elements * mul_ops * m.groups
    total_add_ops = num_out_elements * add_ops * m.groups
    total_ops = total_mul_ops + total_add_ops
    # incase same conv is used multiple times
    m.total_ops = torch.Tensor([total_ops])


def count_bn2d(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_sub = nelements
    total_div = nelements
    total_ops = total_sub + total_div

    m.total_ops = torch.Tensor([total_ops])


def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_ops = nelements

    m.total_ops = torch.Tensor([total_ops])


def count_softmax(m, x, y):
    x = x[0]

    batch_size, nfeatures = x.size()

    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    m.total_ops = torch.Tensor([total_ops])


def count_maxpool(m, x, y):
    kernel_ops = torch.prod(torch.Tensor([m.kernel_size])) - 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops = torch.Tensor([total_ops])


def count_avgpool(m, x, y):
    total_add = torch.prod(torch.Tensor([m.kernel_size])) - 1
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops = torch.Tensor([total_ops])


def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    # total_add = m.in_features - 1
    num_elements = y.numel()
    # total_ops = (total_mul + total_add) * num_elements
    total_ops = total_mul * num_elements

    m.total_ops = torch.Tensor([total_ops])


def hook_count_function(model):
    def add_hooks(m):
        if len(list(m.children())) > 0:
            return

        # for p in m.parameters():
        #     m.total_params += torch.Tensor([p.numel()])

        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(count_conv2d)
        elif isinstance(m, nn.BatchNorm2d):
            m.register_forward_hook(count_bn2d)
        elif isinstance(m, nn.ReLU):
            m.register_forward_hook(count_relu)
        elif isinstance(m, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
            m.register_forward_hook(count_maxpool)
        elif isinstance(m, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
            m.register_forward_hook(count_avgpool)
        elif isinstance(m, nn.Linear):
            m.register_forward_hook(count_linear)
        elif isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            pass
        else:
            print("Not implemented for ", m)

    model.apply(add_hooks)
