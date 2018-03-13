# -*- coding: utf-8 -*-

import time
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
from module_madd import compute_module_madd

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
        x = Variable(torch.rand(16, *self._input_size))  # add module duration time
        self._model.eval()
        self._model(x)

        self._collect_summary()

    @staticmethod
    def _register_buffer(module):
        assert isinstance(module, nn.Module)

        if len(list(module.children())) > 0:
            return

        module.register_buffer('input_shape', torch.zeros(3).int())
        module.register_buffer('output_shape', torch.zeros(3).int())
        module.register_buffer('parameters_quantity', torch.zeros(1).int())
        module.register_buffer('inference_memory', torch.zeros(1).long())
        module.register_buffer('MAdd', torch.zeros(1).long())
        module.register_buffer('duration', torch.zeros(1).float())

    def _sub_module_call_hook(self):
        def wrap_call(module, *input, **kwargs):
            assert module.__class__ in self._origin_call

            start = time.time()
            output = self._origin_call[module.__class__](module, *input, **kwargs)
            end = time.time()
            module.duration = torch.from_numpy(
                np.array([end - start], dtype=np.float32))

            module.input_shape = torch.from_numpy(
                np.array(input[0].size()[1:], dtype=np.int32))
            module.output_shape = torch.from_numpy(
                np.array(output.size()[1:], dtype=np.int32))

            parameters_quantity = 0
            # iterate through parameters and count num params
            for name, p in module._parameters.items():
                parameters_quantity += (0 if p is None else torch.numel(p.data))
            module.parameters_quantity = torch.from_numpy(
                np.array([parameters_quantity], dtype=np.long))

            inference_memory = 1
            for i in range(1, len(output.size()[1:])):
                inference_memory *= output.size()[i]
            # memory += parameters_number  # exclude parameter memory
            inference_memory = inference_memory * 4 / (1024 ** 2)  # shown as MB unit
            module.inference_memory = torch.from_numpy(
                np.array([inference_memory], dtype=np.float32))

            madd = compute_module_madd(module, input[0], output)
            module.MAdd = torch.from_numpy(
                np.array([madd], dtype=np.int64))
            return output

        for module in self._model.modules():
            if len(list(module.children())) == 0 and module.__class__ not in self._origin_call:
                self._origin_call[module.__class__] = module.__class__.__call__
                module.__class__.__call__ = wrap_call

    def _hook_model(self):
        self._model.apply(self._register_buffer)
        self._sub_module_call_hook()

    @staticmethod
    def _get_module_summary(name, module):
        input_shape_np = module.input_shape.numpy().tolist()
        input_shape = ' '.join(['{:>3d}'] * len(input_shape_np)).format(*[e for e in input_shape_np])
        output_shape_np = module.output_shape.numpy().tolist()
        output_shape = ' '.join(['{:>3d}'] * len(input_shape_np)).format(*[e for e in output_shape_np])
        return OrderedDict(module_name=name + str(module.__class__).split('.')[-1].split("'")[0],
                           input_shape=input_shape,
                           output_shape=output_shape,
                           parameters_quantity=module.parameters_quantity.numpy()[0],
                           memory=module.inference_memory.numpy()[0],
                           madd=module.MAdd.numpy()[0],
                           duration=module.duration.numpy()[0])

    def _retrieve_leaf_modules(self, model, prefix=''):
        modules = []
        for name, module in model._modules.items():
            if module is None:
                continue
            if len(list(module.children())) > 0:
                modules += self._retrieve_leaf_modules(
                    module, prefix + ('' if prefix == '' else '.') + name)
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
            return OrderedDict(parameters_number=node["children"]["info"]["parameters_quantity"],
                               memory=node["children"]["info"]["memory"],
                               total_madd=node["children"]["info"]["madd"],
                               duration=node["children"]["info"]["duration"])
        else:
            parameters_quantity = 0
            memory = 0
            total_madd = 0
            duration = 0
            for key in node["children"]:
                son = self._aggregate_leaf(node["children"][key])
                parameters_quantity += son["parameters_quantity"]
                memory += son["memory"]
                total_madd += son["total_madd"]
                duration += son["duration"]
            return OrderedDict(module_name=node["fullname"],
                               parameters_quantity=parameters_quantity,
                               memory=memory,
                               madd=total_madd,
                               duration=duration)

    @staticmethod
    def _is_leaf(node):
        return 'info' in node["children"]

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

    @staticmethod
    def _pretty_format(df):
        df = df.fillna(' ')
        df['memory'] = df['memory'].apply(lambda x: '{:.2f}MB'.format(x))
        df['duration'] = df['duration'].apply(lambda x: '{:.2f}ms'.format(x * 1000))
        df['duration_percent'] = df['duration_percent'].apply(lambda x: '{:.2%}'.format(x))
        del df['duration']
        df.columns = ['module name', 'input shape', 'output shape', 'parameters quantity', 'memory',
                      'MAdd', 'run time percent']
        return df

    def summary(self, max_depth):
        result = self._summary_depth([self.summary_tree], max_depth)
        df = pd.DataFrame(result)
        df['duration_percent'] = df['duration'] / df['duration'].sum()
        total_parameters_quantity = df['parameters_quantity'].sum()
        total_memory = df['memory'].sum()
        total_operation_quantity = df['madd'].sum()

        df = self._pretty_format(df)
        model_summary = str(df) + '\n'
        model_summary += "=" * len(str(df).split('\n')[0])
        model_summary += '\n'
        model_summary += "total parameters quantity: {:,}\n".format(total_parameters_quantity)
        model_summary += "total memory: {:.2f}MB\n".format(total_memory)
        model_summary += "total MAdd: {:,}\n".format(total_operation_quantity)
        print(model_summary)
        with open('model_summary.txt', 'w') as model_summary_file:
            model_summary_file.write(model_summary)
