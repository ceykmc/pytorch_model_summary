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

        self._model = model
        self._input_size = input_size
        self._origin_call = dict()  # sub module call hook

        self._hook_model()
        x = Variable(torch.rand(1, *self._input_size))
        self._model(x)

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

    def _get_model_summary(self, model, prefix=''):
        model_summary = list()
        for name, module in model._modules.items():
            if module is None:
                continue
            if len(list(module.children())) == 0:
                info = self._test_module(name, module, prefix)
                model_summary.append(info)
            else:
                model_summary += self._get_model_summary(module, prefix=prefix + ('.' if prefix else '') + name)
        return model_summary

    def summary(self):
        with open('summary.txt', 'w') as f:
            model_summary = self._get_model_summary(self._model)
            df = pd.DataFrame(model_summary)

            # add duration percent
            duration = pd.to_numeric(df['duration'])
            duration_percent = duration / duration.sum()
            df['duration_percent'] = pd.Series(['{:.3f}%'.format(v * 100) for v in duration_percent])
            # df.to_pickle('summary.pkl')
            f.write(str(df))
            f.write('\n\n')

            module_time = []
            for name, module in self._model._modules.items():
                if len(list(module.children())) == 0:
                    continue
                f.write('module {}\n'.format(name))
                model_summary = self._get_model_summary(module, name)
                df = pd.DataFrame(model_summary)

                # add duration percent
                duration = pd.to_numeric(df['duration'])
                duration_percent = duration / duration.sum()
                df['duration_percent'] = pd.Series(['{:.3f}%'.format(v * 100) for v in duration_percent])
                f.write(str(df))
                f.write('\n\n')

                module_time.append(OrderedDict(module_name=name,
                                               parameters_number=pd.to_numeric(df['parameters_number']).sum(),
                                               total_ops=pd.to_numeric(df['total_ops']).sum(),
                                               duration=duration.sum()))

            df = pd.DataFrame(module_time)

            # add duration percent
            duration = pd.to_numeric(df['duration'])
            duration_percent = duration / duration.sum()
            df['duration_percent'] = pd.Series(['{:.3f}%'.format(v * 100) for v in duration_percent])
            # df.to_pickle('summary.pkl')
            f.write(str(df))
            f.write('\n\n')


def main():
    pass


if __name__ == "__main__":
    main()
