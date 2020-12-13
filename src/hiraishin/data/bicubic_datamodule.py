from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .datasets import BicubicDatasetTrain, BicubicDatasetEval


class BicubicDataModule(pl.LightningDataModule):
    """hold-out dataset
    """

    def __init__(self, dataroot: str = 'data/DIV2K', scale_factor: int = 4, patch_size: int = 96,
                 preupsample: bool = False, batch_size: int = 1, num_workers: int = 4) -> None:

        super().__init__()

        # dataset config
        self.dataroot = Path(dataroot)
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.preupsample = preupsample

        # dataloader config
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ['fit', None]:
            self.train_dataset = BicubicDatasetTrain(
                data_dir=self.dataroot / 'train/HR',
                scale_factor=self.scale_factor,
                patch_size=self.patch_size,
                preupsample=self.preupsample,
            )
            self.val_dataset = BicubicDatasetEval(
                data_dir=self.dataroot / 'valid/HR',
                scale_factor=self.scale_factor,
                patch_size=None,
                preupsample=self.preupsample,
            )
            return self
        else:
            raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
        )
