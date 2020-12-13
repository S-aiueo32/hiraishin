import logging
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.hub as hub

from .blocks import RRDB

logger = logging.getLogger(__name__)


class RRDBNet(nn.Module):
    """PyTorch module for RRDB inspired by mmsr.

    Args:
        ngf (int): the number of base filters.
        n_rrdbs (int): the number of residual in residual dense blocks (RRDB).
        n_rdbs (int): the number of residual dense blocks (RDB) in each RRDB.
        growth (int): the number of filters to grow for each dense convolution.
        n_convs (int): the number of dense convlution in each RDB.
    """

    URLS: Dict[str, str] = {
        'mmedit_RRDB': 'https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_psnr_x4c64b23g32_1x16_1000k_div2k_20200420-bf5c993c.pth',  # noqa
        'mmedit_ESRGAN': 'https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_x4c64b23g32_1x16_400k_div2k_20200508-f8ccaf3b.pth'  # noqa
    }
    KEY_PATTERNS: Dict[str, str] = {
        'conv_first': 'head',
        'conv_body': 'body.23',
        'conv_up1': 'tail.1',
        'conv_up2': 'tail.4',
        'conv_hr': 'tail.6',
        'conv_last': 'tail.8'
    }
    MAX_RETRY: int = 3

    def __init__(self, scale_factor: int = 4, ngf: int = 64, n_rrdbs: int = 23,
                 n_rdbs: int = 3, growth: int = 32, n_convs: int = 5) -> None:

        super(RRDBNet, self).__init__()

        self.head = nn.Conv2d(3, ngf, 3, 1, 1)
        self.body = nn.Sequential(
            *[RRDB(n_rdbs, ngf, growth, n_convs) for i in range(n_rrdbs)],
            nn.Conv2d(ngf, ngf, 3, 1, 1)
        )
        if scale_factor in [2, 3]:
            self.tail = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                nn.Conv2d(ngf, ngf, 3, 1, 1,),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ngf, ngf, 3, 1, 1,),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ngf, 3, 3, 1, 1,),
            )
        else:
            self.tail = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ngf, ngf, 3, 1, 1,),
                nn.LeakyReLU(0.2, True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ngf, ngf, 3, 1, 1,),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ngf, ngf, 3, 1, 1,),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ngf, 3, 3, 1, 1,),
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
            if key.startswith('body'):
                # 'model.keys.0' -> ['model', 'key', '0']
                key_list = key.split('.')

                # for RDB
                rdb_idx = int(key_list[2].replace('rdb', '')) - 1
                key_list[2] = f'body.{rdb_idx}'

                # for Convs
                conv_idx = int(key_list[3].replace('conv', '')) - 1
                key_list[3] = f'body.{conv_idx}.body.0' if conv_idx < 4 else f'body.{conv_idx}'

                key = '.'.join(key_list)

            for old_pattern, new_pattern in self.KEY_PATTERNS.items():
                if old_pattern in key:
                    key = key.replace(old_pattern, new_pattern)

            new_state_dict[key.replace('weigh', 'weight')] = val

        return new_state_dict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.head(x)
        h = self.body(h) + h
        h = self.tail(h)
        return h
