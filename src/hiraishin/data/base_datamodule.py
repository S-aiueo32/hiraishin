from abc import ABCMeta

from pytorch_lightning import LightningDataModule


class BaseDataModule(LightningDataModule, metaclass=ABCMeta):
    pass
