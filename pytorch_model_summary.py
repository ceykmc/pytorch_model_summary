# -*- coding: utf-8 -*-

import time
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
from profile import hook_count_function

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 10000)


class PyTorchModelSummary(object):
    def __init__(self, model, input_size):
        assert isinstance(model, nn.Module)
        assert isinstance(input_size, (list, tuple))

        self.summary_tree = OrderedDict(fullname='root', children=OrderedDict())

        self._model = model
        self._input_size = input_size
        self._origin_call = dict()  # sub module call hook

        self._hook_model()
        x = Variable(torch.rand(16, *self._input_size))
        self._model(x)

        self._collect_summary()

    @staticmethod
    def _register_buffer(module):
        assert isinstance(module, nn.Module)

        module.register_buffer('input_shape', torch.IntTensor())
        module.register_buffer('output_shape', torch.IntTensor())
        module.register_buffer('nb_params', torch.IntTensor([0]))
        module.register_buffer('memory', torch.FloatTensor([0.0]))
        module.register_buffer('total_ops', torch.IntTensor([0]))
        module.register_buffer('duration', torch.FloatTensor([0.0]))

    @staticmethod
    def _get_module_summary(name, module):
        input_shape_np = module.input_shape.numpy().tolist()
        input_shape = ' '.join(['{:>3d}'] * len(input_shape_np)).format(*[e for e in input_shape_np])
        output_shape_np = module.output_shape.numpy().tolist()
        output_shape = ' '.join(['{:>3d}'] * len(input_shape_np)).format(*[e for e in output_shape_np])
        return OrderedDict(module_name=name + str(module.__class__).split('.')[-1].split("'")[0],
                           input_shape=input_shape,
                           output_shape=output_shape,
                           parameters_number=int(module.nb_params.numpy()),
                           memory=float(module.memory.numpy()),
                           total_ops=int(module.total_ops.numpy()),
                           duration=float(module.duration.numpy()))

    @staticmethod
    def _pretty_format(df):
        df = df.fillna(' ')
        df['memory'] = df['memory'].apply(lambda x: '{:.2f}MB'.format(x))
        df['duration'] = df['duration'].apply(lambda x: '{:.2f}ms'.format(x * 1000))
        df['duration_percent'] = df['duration_percent'].apply(lambda x: '{:.2%}'.format(x))
        if len(df.columns) == 8:
            df.columns = ['module name', 'input shape', 'output shape', 'parameters quantity', 'memory', 'opertaion quantity', 'run time', 'run time percent']
        elif len(df.columns) == 6:
            df.columns = ['module name', 'parameters quantity', 'memory', 'opertaion quantity', 'run time', 'run time percent']
        return df

    @staticmethod
    def _is_leaf(node):
        return 'info' in node["children"]

    def _sub_module_call_hook(self):
        def wrap_call(module, *input, **kwargs):
            assert module.__class__ in self._origin_call

            start = time.time()
            result = self._origin_call[module.__class__](module, *input, **kwargs)
            end = time.time()
            module.duration = torch.FloatTensor([end - start])

            module.input_shape = torch.IntTensor(list(input[0].size())[1:])
            module.output_shape = torch.IntTensor(list(result.size())[1:])
            memory = 1
            for p in result.size():
                memory *= p
            params = 0
            # iterate through parameters and count num params
            for name, p in module._parameters.items():
                if p is None:
                    continue
                params += torch.numel(p.data)
            module.nb_params = torch.IntTensor([params])
            memory += params
            memory = memory * 4 / (1024 ** 2)
            module.memory = torch.FloatTensor([memory])
            return result

        for module in self._model.modules():
            if len(list(module.children())) == 0 and module.__class__ not in self._origin_call:
                self._origin_call[module.__class__] = module.__class__.__call__
                module.__class__.__call__ = wrap_call

    def _hook_model(self):
        self._model.apply(self._register_buffer)
        hook_count_function(self._model)
        self._sub_module_call_hook()

    def _retrieve_leaf_modules(self, model, prefix=''):
        modules = []
        for name, module in model._modules.items():
            if module is None:
                continue
            if len(list(module.children())) > 0:
                modules += self._retrieve_leaf_modules(module, prefix + ('' if prefix == '' else '.') + name)
            else:
                modules.append((prefix + ('' if prefix == '' else '.') + name, module))
        return modules

    def _collect_summary(self):
        leaf_modules = self._retrieve_leaf_modules(self._model)
        for name, leaf_module in leaf_modules:
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
            node['info'] = self._get_module_summary(fullname, leaf_module)

    def _aggregate_leaf(self, node):
        if self._is_leaf(node):
            return OrderedDict(parameters_number=node["children"]["info"]["parameters_number"],
                               memory=node["children"]["info"]["memory"],
                               total_ops=node["children"]["info"]["total_ops"],
                               duration=node["children"]["info"]["duration"])
        else:
            parameters_number = 0
            memory = 0
            total_ops = 0
            duration = 0
            for key in node["children"]:
                son = self._aggregate_leaf(node["children"][key])
                parameters_number += son["parameters_number"]
                memory += son["memory"]
                total_ops += son["total_ops"]
                duration += son["duration"]
            return OrderedDict(module_name=node["fullname"],
                               parameters_number=parameters_number,
                               memory=memory,
                               total_ops=total_ops,
                               duration=duration)

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
        total_parameters_quantity = df['parameters_number'].sum()
        total_memory = '{:.2f}MB'.format(df['memory'].sum())
        total_operation_quantity = df['total_ops'].sum()
        total_run_time = '{:.2f}ms'.format(df['duration'].sum() * 1000)
        df['duration_percent'] = df['duration'] / df['duration'].sum()
        df = self._pretty_format(df)
        df.to_excel('summary.xlsx', 'summary')
        print(df)
        print("=" * len(str(df).split('\n')[0]))
        print("total parameters quantity: {}".format(total_parameters_quantity))
        print("total memory: {}".format(total_memory))
        print("total operation quantity: {}".format(total_operation_quantity))
        print("total run time: {}".format(total_run_time))
