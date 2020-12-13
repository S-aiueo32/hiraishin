from .bicubic_datamodule import BicubicDataModule
from .div2k_datamodule import DIV2KDataModule
from .zero_shot_datamodule import ZeroShotDataModule


__all__ = [
    BicubicDataModule.__name__,
    DIV2KDataModule.__name__,
    ZeroShotDataModule.__name__,
]
