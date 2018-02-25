import argparse

import torch
import torch.nn as nn
import pandas as pd

from torch.autograd import Variable as V
# from mobilenet import CMobileNet
# from torchvision import models


def count_conv2d(m, x, y):
    x = x[0]

    cin = m.in_channels // m.groups
    cout = m.out_channels // m.groups
    kh, kw = m.kernel_size
    batch_size = x.size()[0]

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
    m.total_ops += torch.Tensor([int(total_ops)])


def count_bn2d(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_sub = nelements
    total_div = nelements
    total_ops = total_sub + total_div

    m.total_ops += torch.Tensor([int(total_ops)])


def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_ops = nelements

    m.total_ops += torch.Tensor([int(total_ops)])


def count_softmax(m, x, y):
    x = x[0]

    batch_size, nfeatures = x.size()

    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    m.total_ops += torch.Tensor([int(total_ops)])


def count_maxpool(m, x, y):
    kernel_ops = torch.prod(torch.Tensor([m.kernel_size])) - 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])


def count_avgpool(m, x, y):
    total_add = torch.prod(torch.Tensor([m.kernel_size])) - 1
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])


def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    # total_add = m.in_features - 1
    num_elements = y.numel()
    # total_ops = (total_mul + total_add) * num_elements
    total_ops = total_mul * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])


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


def profile(model, batch_size, input_size, custom_ops = {}):

    model.eval()

    def add_hooks(m):
        if len(list(m.children())) > 0:
            return
        m.register_buffer('total_ops', 0)
        m.register_buffer('total_params', 0)

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])

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

    x = V(torch.zeros((batch_size, *input_size)))
    model(x)

    def _torch_compute_summarize(model,
                                 parent_name='',
                                 show_weights=True,
                                 show_parameters=True,
                                 level=0):
        data = []
        for key, m in model._modules.items():
            # if it contains layers let call it recursively to get params and weights
            layer_type = type(m)
            is_container = layer_type in [
                torch.nn.modules.container.Container,
                torch.nn.modules.container.Sequential, torch.nn.Module,
                torch.nn.modules.ModuleList
                ]
            weights = list([tuple(p.size()) for p in m.parameters()])
            if is_container:
                data += _torch_compute_summarize(
                    m,
                    parent_name=parent_name + '=>' + key if parent_name else key,
                    show_weights=show_weights,
                    show_parameters=show_parameters,
                    level=level + 1
                )
            else:
                data.append(
                        dict(
                            key=parent_name + '#' + key,
                            type=type(m).__name__,
                            layer_name=m.__repr__(),
                            parameters=float(m.total_params),
                            weights=weights,
                            operations=float(m.total_ops)
                        )
                )
        return data
    data = _torch_compute_summarize(
        model,
        parent_name=type(model).__name__,
        show_weights=True,
        show_parameters=True,
        level=0)
    df = pd.DataFrame(data)
    total_parameters = df['parameters'].sum()
    total_operations = df['operations'].sum()
    df.loc[df.shape[0]] = dict(
                    key="Total",
                    type=None,
                    layer_name=None,
                    parameters="{:>2f} M".format(total_parameters/1e6),
                    operations="{:>2f} G".format(total_operations/1e9),
                    weights=None
                )
    df = df[['key', 'type', 'parameters', 'operations', 'weights', 'layer_name']]
    # df = df[['key', 'type', 'parameters', 'layer_name']]
    df.index.name = 'layer'
    return df


# def main():
#     # model = CNetwork()
#     # model = models.vgg16()
#     model = CMobileNet()
#     df = profile(model, 1, (3, 144, 160))
#     print(df)
#     df.to_json('./summary.json')
#     df.to_csv('summary.csv', sep=',', header=True, index=True)
#     df.to_html('summary.html')
#
#
# if __name__ == "__main__":
#     main()
