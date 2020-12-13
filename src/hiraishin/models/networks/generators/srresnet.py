import logging
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub as hub

from .blocks import ResidualBlock, UpscaleBlock

logger = logging.getLogger(__name__)


class SRResNet(nn.Module):
    """PyTorch module for MSRResNet inspired by mmsr.

    Args:
        ngf (int): The number of base filters.
        n_blocks (int): The number of residual blocks.
        scale_factor (int): scaling factor of the generator model.
    """

    URLS: Dict[str, str] = {
        'mmedit_SRResNet': 'https://download.openmmlab.com/mmediting/restorers/srresnet_srgan/msrresnet_x4c64b16_1x16_300k_div2k_20200521-61556be5.pth',  # noqa
        'mmedit_SRGAN': 'https://download.openmmlab.com/mmediting/restorers/srresnet_srgan/srgan_x4c64b16_1x16_1000k_div2k_20200606-a1f0810e.pth'  # noqa
    }
    KEY_PATTERNS: Dict[str, str] = {
        'conv_first': 'head.0',
        'unk_net': 'body',
        '.conv1.weigh': '.body.0.weigh',
        '.conv1.bias': '.body.0.bias',
        '.conv2.weigh': '.body.2.weigh',
        '.conv2.bias': '.body.2.bias',
        'upsample1.upsample_conv': 'tail.0.0',
        'upsample2.upsample_conv': 'tail.0.3',
        'conv_hr': 'tail.1',
        'conv_last': 'tail.3',
    }
    MAX_RETRY: int = 3

    def __init__(self, ngf: int = 64, n_blocks: int = 16, scale_factor: int = 4) -> None:

        super(SRResNet, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(3, ngf, 3, 1, 1),
            nn.LeakyReLU(0.1, True)
        )
        self.body = nn.Sequential(
            *[ResidualBlock(ngf) for i in range(n_blocks)]
        )
        self.tail = nn.Sequential(
            UpscaleBlock(scale_factor, ngf),
            nn.Conv2d(ngf, ngf, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ngf, 3, 3, 1, 1)
        )

    def load(self, mode: str):

        assert mode in self.URLS, f'Provided mode {mode} is not matched to mmediting weights.'

        filename = Path(__file__).resolve().parent.joinpath(f'weights/{mode}.pth')

        if not filename.exists():
            for _ in range(self.MAX_RETRY):
                try:
                    hub.download_url_to_file(self.URLS[mode], str(filename), progress=False)
                    logger.info(f'Weights file is downloaded from {self.URLS[mode]}')
                    break
                except Exception as e:
                    logger.warn(traceback.format_exc(e))
                raise Exception('download failed')

        state_dict = torch.load(str(filename))['state_dict']
        state_dict = self.extract_generator_state_dict(state_dict)
        state_dict = self.convert_state_dict(state_dict)

        self.load_state_dict(state_dict)

    @staticmethod
    def extract_generator_state_dict(state_dict: OrderedDict):
        """extracts items having `generator.` key
        """
        state_dict = tuple((k.strip(pat), v) for k, v in state_dict.items() if (pat := 'generator.') in k)
        return OrderedDict(state_dict)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.head(x)
        h = self.body(h) + h
        h = self.tail(h)
        return h + F.interpolate(x, h.size()[2:], None, 'bilinear', False)
