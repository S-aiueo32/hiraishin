from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .datasets import SingleBatchDatasetTrain, SingleBatchDatasetEval


class ZeroShotDataModule(pl.LightningDataModule):

    def __init__(self, filename: str, scale_factor: int = 4, patch_size: int = 96,
                 preupsample: bool = False, batch_size: int = 1, num_workers: int = 4) -> None:

        super().__init__()

        # dataset config
        self.filename = filename
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.preupsample = preupsample

        # dataloader config
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        if stage in ['fit', None]:
            self.train_dataset = SingleBatchDatasetTrain(
                filename=self.filename,
                batch_size=self.batch_size,
                scale_factor=self.scale_factor,
                patch_size=self.patch_size,
                preupsample=self.preupsample
            )
            return self
        else:
            self.test_dataset = SingleBatchDatasetEval(self.filename)
            return self

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset)
