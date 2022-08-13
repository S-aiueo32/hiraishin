import importlib
import inspect
from collections import OrderedDict
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Optional, Type

import hydra
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

from hiraishin.schema import Module

logger = getLogger(__name__)

try:
    CWD = Path(hydra.utils.get_original_cwd()).resolve()
except ValueError:
    CWD = Path.cwd()


def get_class_name_with_shortest_module(cls: Type[Any]) -> str:
    """Get the shortest class name."""
    _target_ = cls.__module__ + "." + cls.__name__
    for i in range(len(cls.__module__.split("."))):
        try:
            module_name = ".".join(cls.__module__.split(".")[: -(i + 1)])
            _ = getattr(importlib.import_module(module_name), cls.__name__)
            _target_ = module_name + "." + cls.__name__
        except (AttributeError, ValueError):
            pass
    return _target_


def get_arguments(cls: Type[Any], with_kwargs: bool = False) -> Dict[str, Any]:
    """Get the `__init__` parameters of the given class. The positional arguments will be filled with \"???\".
    If `with_kwargs == True`, the keyword arguments will be output with the default values."""
    params = inspect.signature(cls).parameters
    if with_kwargs:
        params = {k: "???" if v.default is inspect._empty else v.default for k, v in params.items()}
    else:
        params = {k: "???" for k, v in params.items() if v.default is inspect._empty}

    if issubclass(cls, optim.Optimizer):
        _ = params.pop("params")
    if issubclass(cls, optim.lr_scheduler._LRScheduler):
        _ = params.pop("optimizer")

    return params


def load_weights(net: Module, path: Path, net_name: Optional[str] = None) -> None:
    """Loads weights from the given path whose extension is `.ckpt` or `.pth`. For `.ckpt`, net_name is required."""
    if path.suffix == ".pth":
        net.load_state_dict(torch.load(CWD.joinpath(path)), strict=False)
    elif path.suffix == ".ckpt":
        if net_name is None:
            raise ValueError("net_name must be specified for .ckpt loading.")
        state_dict = OrderedDict(
            (k.replace(f"{net_name}.", ""), v)
            for k, v in torch.load(CWD.joinpath(path))["state_dict"].items()
            if k.startswith(net_name)
        )
        net.load_state_dict(state_dict)
    else:
        raise ValueError(".ckpt or .pth are allowed.")
    logger.info(f"[{net.__class__.__name__}] have been loaded from {str(path)}.")


class BasicWeightInitializer:
    """Initializes the weights whose class names contain \"Conv\" or \"Linear\" with built-in initializers of PyTorch."""

    def __init__(self, init_type: str = "xavier_uniform", **kwargs) -> None:
        self.init_type = init_type
        self.kwargs: Dict[str, Any] = kwargs

    def init_fn(self, m: nn.Module) -> None:
        name = m.__class__.__name__
        if hasattr(m, "weight") and ("Conv" in name or "Linear" in name):
            fn = getattr(init, f"{self.init_type}_")
            fn(m.weight.data, **self.kwargs)
            if hasattr(m, "bias"):
                assert isinstance(m.bias, nn.parameter.Parameter)
                init.constant_(m.bias.data, 0.0)
        elif "BatchNorm2d" in name:
            assert isinstance(m.weight, nn.parameter.Parameter)
            assert isinstance(m.bias, nn.parameter.Parameter)
            init.normal_(m.weight.data, 0.0, 1.0)
            init.constant_(m.bias.data, 0.0)

    def __call__(self, net: Module) -> None:
        net.apply(self.init_fn)
        logger.info(f"{net.__class__.__name__} was initialized with {self.__class__.__name__}.")
