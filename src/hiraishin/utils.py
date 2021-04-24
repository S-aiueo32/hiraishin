import importlib
import inspect

import torch.optim as optim


def get_class_name_with_shortest_module(cls):
    _target_ = cls.__module__ + '.' + cls.__name__
    for i in range(len(cls.__module__.split('.'))):
        try:
            module_name = '.'.join(cls.__module__.split('.')[:-(i + 1)])
            _ = getattr(importlib.import_module(module_name), cls.__name__)
            _target_ = module_name + '.' + cls.__name__
        except (AttributeError, ValueError):
            pass
    return _target_


def get_arguments(cls, with_kwargs: bool = False):
    params = inspect.signature(cls).parameters
    if with_kwargs:
        params = {k: '???' if (d := v.default) is inspect._empty else d for k, v in params.items()}
    else:
        params = {k: '???' for k, v in params.items() if (d := v.default) is inspect._empty}

    if issubclass(cls, optim.Optimizer):
        _ = params.pop('params')
    if issubclass(cls, optim.lr_scheduler._LRScheduler):
        _ = params.pop('optimizer')

    return params
