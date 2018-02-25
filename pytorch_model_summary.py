# -*- coding: utf-8 -*-

import time
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
from profile import hook_count_function
import json

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 10000)


class PyTorchModelSummary(object):
    def __init__(self, model, input_size):
        assert isinstance(model, nn.Module)
        assert isinstance(input_size, (list, tuple))

        self.summary_tree = OrderedDict(fullname='root', children=OrderedDict())  # 统计树

        self._model = model
        self._input_size = input_size
        self._origin_call = dict()  # sub module call hook

        self._hook_model()
        x = Variable(torch.rand(1, *self._input_size))
        self._model(x)

        self._collect_summary()

    @staticmethod
    def _register_buffer(module):
        assert isinstance(module, nn.Module)

        module.register_buffer('module_name', torch.IntTensor())
        module.register_buffer('input_shape', torch.IntTensor())
        module.register_buffer('output_shape', torch.IntTensor())
        module.register_buffer('nb_params', torch.IntTensor())
        module.register_buffer('total_ops', 0)
        module.register_buffer('duration', torch.FloatTensor())

    def _sub_module_call_hook(self):
        def wrap_call(module, *input, **kwargs):
            assert module.__class__ in self._origin_call

            start = time.time()
            result = self._origin_call[module.__class__](module, *input, **kwargs)
            end = time.time()
            module.duration = torch.FloatTensor([end - start])

            name = str(module.__class__).split('.')[-1].split("'")[0]
            module.module_name = torch.IntTensor([ord(c) for c in name])
            module.input_shape = torch.IntTensor(list(input[0].size())[1:])
            module.output_shape = torch.IntTensor(list(result.size())[1:])
            params = 0
            # iterate through parameters and count num params
            for name, p in module._parameters.items():
                if p is None:
                    continue
                params += torch.numel(p.data)
            module.nb_params = torch.IntTensor([params])

            return result

        for module in self._model.modules():
            if len(list(module.children())) == 0 and module.__class__ not in self._origin_call:
                self._origin_call[module.__class__] = module.__class__.__call__
                module.__class__.__call__ = wrap_call

    def _hook_model(self):
        self._model.apply(self._register_buffer)
        hook_count_function(self._model)
        self._sub_module_call_hook()

    @staticmethod
    def _test_module(name, module, prefix):
        module_name = prefix + ('.' if prefix else '') + name + '.' + \
                      str(module.__class__).split('.')[-1].split("'")[0]
        input_shape_np = module.input_shape.numpy()
        input_shape = ' '.join(['{:>3d}'] * len(input_shape_np)).format(*[e for e in input_shape_np])
        output_shape_np = module.output_shape.numpy()
        output_shape = ' '.join(['{:>3d}'] * len(output_shape_np)).format(*[e for e in output_shape_np])
        parameters_number = '{:>5d}'.format(module.nb_params.numpy()[0])
        total_ops = '{:>5d}'.format(int(module.total_ops.numpy()[0]))
        duration = '{:.5f}'.format(module.duration.numpy()[0])
        return OrderedDict(module_name=module_name,
                           input_shape=input_shape,
                           output_shape=output_shape,
                           parameters_number=parameters_number,
                           total_ops=total_ops,
                           duration=duration)

    @staticmethod
    def _get_module_summary(name, module):
        return OrderedDict(module_name=name + str(module.__class__).split('.')[-1].split("'")[0],
                           input_shape=module.input_shape.numpy().tolist(),
                           output_shape=module.output_shape.numpy().tolist(),
                           parameters_number=int(module.nb_params.numpy()[0]),
                           total_ops=int(module.total_ops.numpy()[0]),
                           duration=float(module.duration.numpy()[0]))

    @staticmethod
    def _node_to_print_format(node):
        return OrderedDict(module_name=node['module_name'],
                           input_shape=' '.join(['{:>3d}'] * len(node['input_shape'])).format(*[e for e in node['input_shape']]),
                           output_shape=' '.join(['{:>3d}'] * len(node['output_shape'])).format(*[e for e in node['output_shape']]),
                           parameters_number='{:>5d}'.format(node['parameters_number']),
                           total_ops='{:>5d}'.format(node['total_ops']),
                           duration='{:.5f}'.format(node['duration']))

    def _retrive_modules(self, model, prefix=''):
        modules = []
        for name, module in model._modules.items():
            if module is None:
                continue
            if len(list(module.children())) > 0:
                modules += self._retrive_modules(module, prefix + ('' if prefix == '' else '.') + name)
            else:
                modules.append((prefix + ('' if prefix == '' else '.') + name, module))
        return modules

    def _collect_summary(self):
        modules = self._retrive_modules(self._model)
        for name, module in modules:
            name_parts = name.split('.')
            node = self.summary_tree["children"]
            fullname = ''
            for part in name_parts:
                fullname += part
                if part not in node:
                    node[part] = {
                        "fullname": fullname,
                        "children": {}
                    }
                node = node[part]["children"]
                fullname += '.'
            node['info'] = self._get_module_summary(fullname, module)

    @staticmethod
    def _is_leaf(node):
        return 'info' in node["children"]

    def _aggregate_leaf(self, node):
        if self._is_leaf(node):
            return OrderedDict(parameters_number=node["children"]['info']['parameters_number'],
                               total_ops=node["children"]['info']['total_ops'],
                               duration=node["children"]['info']['duration'])
        else:
            parameters_number = 0
            total_ops = 0
            duration = 0
            for key in node["children"]:
                son = self._aggregate_leaf(node["children"][key])
                parameters_number += son['parameters_number']
                total_ops += son['total_ops']
                duration += son['duration']
            return OrderedDict(module_name=node["fullname"],
                               parameters_number=parameters_number,
                               total_ops=total_ops,
                               duration=duration)

    def _summary_leaf(self):
        queue = [self.summary_tree]
        result = []
        while len(queue) > 0:
            node = queue[0]
            del queue[0]
            if self._is_leaf(node):
                result.append(self._node_to_print_format(node["children"]['info']))
            else:
                for key in node["children"]:
                    queue.append(node["children"][key])
        df = pd.DataFrame(result)
        return df

    def _summary_depth(self, nodes, max_depth, depth=0):
        result = []
        for node in nodes:
            if self._is_leaf(node):
                result.append(node["children"]["info"])
            elif depth >= max_depth >= 0:
                    result.append(self._aggregate_leaf(node))
            else:
                result += self._summary_depth(list(node["children"].values()), max_depth, depth+1)
        return result

    def summary(self, max_depth):
        result = self._summary_depth([self.summary_tree], max_depth)
        df = pd.DataFrame(result)
        df['duration_percent'] = df['duration'] / df['duration'].sum()
        df['duration_percent'] = df['duration_percent'].apply(lambda x: '{:.2%}'.format(x))
        print(df)
