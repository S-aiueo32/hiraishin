from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.init as init

from .discriminators import ImageDiscriminator, PatchDiscriminator, PixelDiscriminator
from .generators import SRCNN, SRResNet, RRDBNet, ZSSR

__all__ = [
    SRCNN.__name__,
    SRResNet.__name__,
    RRDBNet.__name__,
    ZSSR.__name__,
    ImageDiscriminator.__name__,
    PatchDiscriminator.__name__,
    PixelDiscriminator.__name__
]


def init_weights(net: nn.Module, weights: str = None, init_type: str = 'normal', init_gain: float = 0.02) -> None:

    def init_func(m: nn.Module):
        name = m.__class__.__name__
        if hasattr(m, 'weight') and ('Conv' in name or 'Linear' in name):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    f'initialization method [{init_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in name:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    if weights is not None:
        if Path(weights).exists():
            net.load_state_dict(torch.load(weights))
        else:
            assert hasattr(net, 'load')
            net.load(mode=weights)
    else:
        net.apply(init_func)
