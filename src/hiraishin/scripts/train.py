import warnings

from hydra.utils import instantiate
from omegaconf import DictConfig

from pytorch_lightning import LightningModule, LightningDataModule, Trainer, seed_everything

from .utils import resolve

warnings.simplefilter('ignore', UserWarning)


def app(config: DictConfig) -> None:

    seed_everything(338)

    config = resolve(config)

    dm: LightningDataModule = instantiate(config.data)
    model: LightningModule = instantiate(config.model)
    trainer: Trainer = instantiate(config.trainer)

    dm.setup('fit')
    trainer.fit(model, datamodule=dm)
