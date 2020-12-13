import logging
from pathlib import Path
from typing import Dict, Union, Optional

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from .utils import get_all_images
from .transforms import (
    AlignedCompose,
    AlignedCenterCrop,
    AlignedRandomCrop,
    AlignedRandomRot90,
    AlignedRandomHorizontalFlip,
    AlignedLambda,
    SplittedProcess
)

logger = logging.getLogger(__name__)

try:
    import hydra
    CWD = Path(hydra.utils.get_original_cwd()).resolve()
    logger.info('Hydra is used.')
except (ImportError, ValueError):
    logger.info('Hydra is not initialized.')
    CWD = Path.cwd()


class DIV2KDatasetTrain(Dataset):
    """defines training dataset for DIV2K dataset.

    Args:
        dataroot (str): relative path to a root directory of DIV2K.
        mode (str): degradation mode for LR, select from ('difficult', 'mild', 'wild', 'x8').
        scale_factor (int): a scale factor, DIV2K supports (4, 8).
        patch_size (int): a patch size for random cropping.
        preupsample (bool):　wheather LR will be upscaled to HR.
    """

    def __init__(self, dataroot: str, mode: str, scale_factor: int,
                 patch_size: int, preupsample: bool = False) -> None:

        super(DIV2KDatasetTrain, self).__init__()

        assert mode in ('difficult', 'mild', 'wild', 'x8'), 'specified mode is invalid.'
        assert patch_size % scale_factor == 0, 'you should define patch_size to be able to devided by scale_factor.'

        self.dir_hr = CWD / dataroot / 'HR'
        self.dir_lr = CWD / dataroot / f'LR_{mode}'
        self.filenames_lr = [f.stem for f in get_all_images(self.dir_lr)]

        self.transforms = AlignedCompose([
            AlignedRandomCrop(patch_size, scale_factor),
            AlignedRandomRot90(),
            AlignedRandomHorizontalFlip(),
            SplittedProcess(
                for_hr=lambda x: x,
                for_lr=lambda x: TF.resize(x, [l * scale_factor for l in x.size], Image.BICUBIC) if preupsample else x
            ),
            AlignedLambda([
                ToTensor()
            ]),
        ])

    def __getitem__(self, index: str) -> Dict[str, Union[torch.Tensor, str]]:
        filename_lr = self.dir_lr / f'{self.filenames_lr[index]}.png'
        filename_hr = self.dir_hr / f'{filename_lr.stem[:4]}.png'

        img_hr = Image.open(filename_hr).convert('RGB')
        img_lr = Image.open(filename_lr).convert('RGB')

        img_hr, img_lr = self.transforms(img_hr, img_lr)

        return {'hr': img_hr, 'lr': img_lr, 'filename': filename_hr.stem}

    def __len__(self):
        return len(self.filenames_lr)


class DIV2KDatasetEval(Dataset):
    def __init__(self, dataroot: str, mode: str, scale_factor: int,
                 patch_size: Optional[int] = None, preupsample: bool = False) -> None:

        assert (patch_size is None) or (patch_size % scale_factor == 0)

        super(DIV2KDatasetEval, self).__init__()

        self.dir_hr = CWD / dataroot / 'HR'
        self.dir_lr = CWD / dataroot / f'LR_{mode}'
        self.filenames_lr = [f.stem for f in get_all_images(self.dir_lr)]

        self.transforms = AlignedCompose([
            AlignedCenterCrop(patch_size, scale_factor) if patch_size else lambda x, y: (x, y),
            SplittedProcess(
                for_hr=lambda x: x,
                for_lr=lambda x: TF.resize(x, [l * scale_factor for l in x.size[::-1]], Image.BICUBIC) if preupsample else x  # noqa
            ),
            AlignedLambda([
                ToTensor()
            ]),
        ])

    def __getitem__(self, index: str) -> Dict[str, Union[torch.Tensor, str]]:
        filename_lr = self.dir_lr / f'{self.filenames_lr[index]}.png'
        filename_hr = self.dir_hr / f'{filename_lr.stem[:4]}.png'

        img_hr = Image.open(filename_hr).convert('RGB')
        img_lr = Image.open(filename_lr).convert('RGB')

        img_hr, img_lr = self.transforms(img_hr, img_lr)

        return {'hr': img_hr, 'lr': img_lr, 'filename': filename_hr.stem}

    def __len__(self):
        return len(self.filenames_lr)
