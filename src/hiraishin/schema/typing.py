from typing import NewType

from torch import nn, optim

Module = NewType("Module", nn.Module)
Optimizer = NewType("Optimizer", optim.Optimizer)
LRScheduler = NewType("LRScheduler", optim.lr_scheduler._LRScheduler)
