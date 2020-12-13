import logging
import traceback
from collections import OrderedDict
from itertools import chain
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.hub as hub
from torchvision import models

logger = logging.getLogger(__name__)


class LPIPS(nn.Module):
    """
    """

    URL: str = 'https://raw.githubusercontent.com/S-aiueo32/PerceptualSimilarity/master/models/weights/v{}/{}.pth'
    KEY_PATTERNS: Dict[str, str] = {
        'lin': '',
        'model.': ''
    }
    MAX_RETRY: int = 3

    def __init__(self, net_type: str = 'alex', version: str = '0.1') -> None:

        assert version in ['0.1'], 'v0.1 is only supported now'

        super(LPIPS, self).__init__()

        self.net_type = net_type
        self.version = version

        self.net = get_network(self.net_type)
        self.lin = LinLayers(self.net.n_channels_list)

        self.load()

    def load(self):
        filename = Path(__file__).resolve().parent.joinpath(f'weights/{self.net_type}_v{self.version}.pth')

        if not filename.exists():
            for _ in range(self.MAX_RETRY):
                try:
                    url = self.URL.format(self.version, self.net_type)
                    hub.download_url_to_file(url, str(filename), progress=False)
                    logger.info(f'Weights file is downloaded from {url}')
                except Exception as e:
                    logger.warn(traceback.format_exc(e))
                raise Exception('download failed')

        state_dict = torch.load(str(filename))
        state_dict = self.convert_state_dict(state_dict)

        self.lin.load_state_dict(state_dict)

    def convert_state_dict(self, state_dict: OrderedDict):
        """renames keys of downloaded state_dict to match to this module.
        """
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            for old_pattern, new_pattern in self.KEY_PATTERNS.items():
                if old_pattern in key:
                    key = key.replace(old_pattern, new_pattern)

            new_state_dict[key.replace('weigh', 'weight')] = val

        return new_state_dict

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        feat_x, feat_y = self.net(x), self.net(y)

        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

        return torch.sum(torch.cat(res, 0), 0, True)


class LinLayers(nn.ModuleList):
    def __init__(self, n_channels_list: Sequence[int]):
        super(LinLayers, self).__init__([
            nn.Sequential(
                nn.Identity(),
                nn.Conv2d(nc, 1, 1, 1, 0, bias=False)
            ) for nc in n_channels_list
        ])

        for param in self.parameters():
            param.requires_grad = False


def get_network(net_type: str):
    if net_type == 'alex':
        return AlexNet()
    elif net_type == 'squeeze':
        return SqueezeNet()
    elif net_type == 'vgg':
        return VGG16()
    else:
        raise ValueError('choose net_type from [alex, squeeze, vgg].')


class BaseNet(nn.Module):

    layers: nn.Sequential
    target_layers: List[int]
    n_channels_list: List[int]

    def __init__(self):
        super(BaseNet, self).__init__()

        # register buffer
        self.register_buffer(
            'mean', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer(
            'std', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def set_requires_grad(self, state: bool):
        for param in chain(self.parameters(), self.buffers()):
            param.requires_grad = state

    def z_score(self, x: torch.Tensor):
        return (x - self.mean) / self.std

    @staticmethod
    def normalize_activation(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
        return x / (norm_factor + eps)

    def forward(self, x: torch.Tensor):
        x = self.z_score(x)

        output = []
        for i, (_, layer) in enumerate(self.layers._modules.items(), 1):
            x = layer(x)
            if i in self.target_layers:
                output.append(self.normalize_activation(x))
            if len(output) == len(self.target_layers):
                break
        return output


class SqueezeNet(BaseNet):
    def __init__(self):
        super(SqueezeNet, self).__init__()

        self.layers = models.squeezenet1_1(True).features
        self.target_layers = [2, 5, 8, 10, 11, 12, 13]
        self.n_channels_list = [64, 128, 256, 384, 384, 512, 512]

        self.set_requires_grad(False)


class AlexNet(BaseNet):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.layers = models.alexnet(True).features
        self.target_layers = [2, 5, 8, 10, 12]
        self.n_channels_list = [64, 192, 384, 256, 256]

        self.set_requires_grad(False)


class VGG16(BaseNet):
    def __init__(self):
        super(VGG16, self).__init__()

        self.layers = models.vgg16(True).features
        self.target_layers = [4, 9, 16, 23, 30]
        self.n_channels_list = [64, 128, 256, 512, 512]

        self.set_requires_grad(False)
