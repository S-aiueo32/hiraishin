import logging
from pathlib import Path
from typing import Optional, Union, Dict

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose

from .transforms import TimofteAugmentation, BicubicDegradation, AdjustSize
from .utils import get_all_images

logger = logging.getLogger(__name__)

try:
    import hydra
    CWD = Path(hydra.utils.get_original_cwd()).resolve()
    logger.info('Hydra is used.')
except (ImportError, ValueError):
    logger.info('Hydra is not initialized.')
    CWD = Path.cwd()


class BicubicDatasetTrain(Dataset):
    """BicubicDatasetTrain
    """

    def __init__(self,
                 data_dir: str,
                 scale_factor: int,
                 patch_size: int,
                 preupsample: bool = False) -> None:

        super(BicubicDatasetTrain, self).__init__()

        self.filenames = get_all_images(CWD / data_dir)

        self.transform_hr = TimofteAugmentation(patch_size)
        self.transform_lr = BicubicDegradation(scale_factor, preupsample)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str]]:
        filename = self.filenames[index]
        img_hr = Image.open(filename).convert('RGB')

        img_hr = self.transform_hr(img_hr)
        img_lr = self.transform_lr(img_hr)

        img_hr = TF.to_tensor(img_hr)
        img_lr = TF.to_tensor(img_lr)

        return {'hr': img_hr, 'lr': img_lr, 'filename': filename.stem}

    def __len__(self) -> int:
        return len(self.filenames)


class BicubicDatasetEval(Dataset):
    """
    """

    def __init__(self,
                 data_dir: str,
                 scale_factor: int,
                 patch_size: Optional[int] = None,
                 preupsample: bool = False) -> None:

        assert (patch_size is None) or (patch_size % scale_factor == 0)

        super(BicubicDatasetEval, self).__init__()

        self.data_dir = Path(data_dir)
        self.filenames = sorted(get_all_images(self.data_dir))

        self.transform_hr = Compose([
            CenterCrop(patch_size) if patch_size else AdjustSize(scale_factor),
        ])
        self.transform_lr = BicubicDegradation(scale_factor, preupsample)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str]]:
        filename = self.filenames[index]
        img_hr = Image.open(filename).convert('RGB')

        img_hr = self.transform_hr(img_hr)
        img_lr = self.transform_lr(img_hr)

        img_hr = TF.to_tensor(img_hr)
        img_lr = TF.to_tensor(img_lr)

        return {'hr': img_hr, 'lr': img_lr, 'filename': filename.stem}

    def __len__(self) -> int:
        return len(self.filenames)
