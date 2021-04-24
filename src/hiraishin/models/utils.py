import importlib
import inspect
from collections import OrderedDict
from logging import getLogger
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

logger = getLogger(__name__)

try:
    import hydra
    CWD = Path(hydra.utils.get_original_cwd()).resolve()
except (ImportError, ValueError):
    CWD = Path.cwd()


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


def init_weights(net: nn.Module, init_type: str = 'xavier_uniform', init_gain: float = 1.) -> None:

    def init_func(m: nn.Module):
        name = m.__class__.__name__
        if hasattr(m, 'weight') and ('Conv' in name or 'Linear' in name):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'initialization method [{init_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in name:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    logger.info(f'Weights have been initialized with (type={init_type}, gain={init_gain}).')


def load_weights(net: nn.Module, weight_path: str, net_name: str = None):
    if (weight_path := CWD.joinpath(weight_path)).exists():
        if weight_path.suffix == '.pth':
            state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
            net.load_state_dict(torch.load(weight_path), strict=False)
        if weight_path.suffix == '.ckpt':
            assert net_name is not None, 'net_name is required to load weights from checkpoints.'
            state_dict = torch.load(weight_path, map_location=torch.device('cpu'))['state_dict']
            state_dict = OrderedDict((k, v) for k, v in state_dict.items() if k.startswith(net_name))
            state_dict = OrderedDict((k.replace(net_name + '.', ''), v) for k, v in state_dict.items())
            net.load_state_dict(state_dict)
        logger.info(f'Weights have been loaded from {str(weight_path)}.')
    else:
        logger.warn(f'{str(weight_path)} does not exists.')
