import logging
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.hub as hub

logger = logging.getLogger(__name__)


class SRCNN(nn.Sequential):
    """PyTorch module for SRCNN.
    """

    URLS: Dict[str, str] = {
        'mmedit': 'https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth'  # noqa
    }
    KEY_PATTERNS: Dict[str, str] = {
        'conv1': '0',
        'conv2': '2',
        'conv3': '4',
    }
    MAX_RETRY: int = 3

    def __init__(self, n_filters: Sequence[int] = (64, 32), kernel_sizes: Sequence[int] = (9, 5)) -> None:

        f1, f2 = n_filters
        k1, k3 = kernel_sizes

        super(SRCNN, self).__init__(
            nn.Conv2d(3, f1, k1, padding=4),
            nn.ReLU(True),
            nn.Conv2d(f1, f2, 1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(f2, 3, k3, padding=2)
        )

    def load(self):

        filename = Path(__file__).resolve().parent.joinpath('weights/mmedit_SRCNN.pth')

        if not filename.exists():
            for _ in range(self.MAX_RETRY):
                try:
                    hub.download_url_to_file(self.URLS['mmedit'], str(filename), progress=False)
                    logger.info(f'Weights file is downloaded from {self.URLS["mmedit"]}')
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
