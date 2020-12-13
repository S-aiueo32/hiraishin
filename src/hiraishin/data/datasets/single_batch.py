import logging
from pathlib import Path
from typing import Dict, Union

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from .transforms import (
    TimofteAugmentation,
    BicubicDegradation,
)

logger = logging.getLogger(__name__)

try:
    import hydra
    CWD = Path(hydra.utils.get_original_cwd()).resolve()
    logger.info('Hydra is used.')
except (ImportError, ValueError):
    logger.info('Hydra is not initialized.')
    CWD = Path.cwd()


class SingleBatchDatasetTrain(Dataset):
    """
    """

    def __init__(self,
                 filename: str,
                 batch_size: int,
                 scale_factor: int,
                 patch_size: int,
                 preupsample: bool = False) -> None:

        super(SingleBatchDatasetTrain, self).__init__()

        self.filename = CWD / filename
        self.batch_size = batch_size

        self.img = Image.open(self.filename).convert('RGB')

        self.transform_hr = TimofteAugmentation(patch_size)
        self.transform_lr = BicubicDegradation(scale_factor, preupsample)

    def __getitem__(self, index: str) -> Dict[str, Union[torch.Tensor]]:
        img_parent = self.transform_hr(self.img)
        img_child = self.transform_lr(img_parent)
        return {'parent': TF.to_tensor(img_parent), 'child': TF.to_tensor(img_child)}

    def __len__(self) -> int:
        return self.batch_size


class SingleBatchDatasetEval(Dataset):
    """
    """

    def __init__(self, filename: str) -> None:

        super(SingleBatchDatasetEval, self).__init__()

        self.filename = CWD / filename
        self.img = Image.open(self.filename).convert('RGB')

    def __getitem__(self, index: str) -> Dict[str, Union[torch.Tensor, str]]:
        return {'parent': TF.to_tensor(self.img)}

    def __len__(self) -> int:
        return 1
