import inspect
import logging
from abc import ABCMeta, abstractmethod
from itertools import chain
from typing import List
from pathlib import Path

import hydra
from omegaconf import OmegaConf, DictConfig

import torch
import pytorch_lightning as pl

from .networks import init_weights
from .losses import PSNR, SSIM, LPIPS

logger = logging.getLogger(__name__)


class BaseModel(pl.LightningModule, metaclass=ABCMeta):

    def initialize(self, config: DictConfig):
        """calls __init__() of BaseModel with sub-module information.
        """
        super(self.__class__, self).__init__(config, inspect.getmodule(self).__name__ + '.' + self.__class__.__name__)

    def __init__(self, config: DictConfig, _target_: str):

        super().__init__()

        self.config = config
        self.save_hyperparameters()

        self.define_networks()
        self.define_losses()
        self.define_optimizers()
        self.define_schedulers()
        self.define_metrics()

        self.validation()

    def define_networks(self) -> None:
        """
        """
        for key, value in self.config.networks.items():
            setattr(self, key, hydra.utils.instantiate(value.args))
            init_weights(getattr(self, key), **value.init)

    def define_losses(self) -> None:
        """
        """
        for key, value in self.config.losses.items():
            setattr(self, key, hydra.utils.instantiate(value.args))
            setattr(self, key.replace('criterion', 'weight'), value.weight)

    def define_optimizers(self):
        """
        """
        for key, value in self.config.optimizers.items():
            params = chain(*[getattr(self, net).parameters() for net in value.params])
            setattr(self, key, hydra.utils.instantiate(value.args, params))

    def define_schedulers(self):
        """
        """
        for key, value in self.config.schedulers.items():
            optimizer = getattr(self, value.optimizer)
            setattr(self, key, hydra.utils.instantiate(value.args, optimizer))

    def define_metrics(self):
        """defines metric calculators. currently PSNR, SSIM and LPIPS are defined here.
        """
        self.psnr = PSNR()
        self.ssim = SSIM()
        self.lpips = LPIPS()

    def validation(self):
        """validates provided configs.
        """
        is_valid: List[bool] = []
        for k, v in self.__class__.__dict__.get('__annotations__').items():
            if not hasattr(self, k):
                logger.warning(f'class variable {k} is not initialized.')
                is_valid.append(False)
                continue
            if not isinstance(getattr(self, k), v):
                logger.warning(f'type of class variable {k} is not matched with {v}')
                is_valid.append(False)
                continue
            is_valid.append(True)
        if all(is_valid):
            logger.info('model initialization successed!')
        else:
            raise ValueError('model initialization failed because of above errors.')

    @abstractmethod
    def configure_optimizers(self, *args, **kwargs):
        """write a method returns optimizers and schedulers as you want．
        """
        pass

    @staticmethod
    @abstractmethod
    def get_inputs(*args, **kwargs):
        """
        """
        pass

    @staticmethod
    @abstractmethod
    def get_gt(*args, **kwargs):
        """
        """
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        """write a prediction routine which can be used in validation/test loop
        """
        pass

    @abstractmethod
    def training_step(self, *args, **kwargs):
        """write a training routine incluing data unpackings, predictions, loss computations etc.
        """
        pass

    @torch.no_grad()
    def validation_step(self, batch, batch_idx: int):
        outputs = self.forward(self.get_inputs(batch))
        if (gt := self.get_gt(batch)) is not None:
            if min(self.config.val_range) == -1.:
                outputs, gt = (outputs + 1) / 2, (gt + 1) / 2
            psnr = self.psnr(outputs, gt)
            ssim = self.ssim(outputs, gt)
            lpips = self.lpips(outputs, gt, retPerLayer=False, normalize=True)
            return {'psnr': psnr, 'ssim': ssim, 'lpips': lpips, 'output': outputs}
        else:
            raise NotImplementedError

    @torch.no_grad()
    def validation_epoch_end(self, outputs):
        psnr = torch.stack([x['psnr'] for x in outputs]).mean()
        ssim = torch.stack([x['ssim'] for x in outputs]).mean()
        lpips = torch.stack([x['lpips'] for x in outputs]).mean()

        self.log_dict(
            dictionary={
                'psnr/val': psnr.item(),
                'ssim/val': ssim.item(),
                'lpips/val': lpips.item()
            }
        )

    @torch.no_grad()
    def test_step(self, batch, batch_idx: int):
        outputs = self.forward(*self.get_inputs(batch))
        if (gt := self.get_gt(batch)) is not None:
            if min(self.config.val_range) == -1.:
                outputs, gt = (outputs + 1) / 2, (gt + 1) / 2
            psnr = self.psnr(outputs, gt)
            ssim = self.ssim(outputs, gt)
            lpips = self.lpips(outputs, gt, True)
            return {'psnr': psnr, 'ssim': ssim, 'lpips': lpips, 'output': outputs}
        else:
            raise NotImplementedError

    @torch.no_grad()
    def test_epoch_end(self, outputs):
        psnr = torch.stack([x['psnr'] for x in outputs]).mean()
        ssim = torch.stack([x['ssim'] for x in outputs]).mean()
        lpips = torch.stack([x['lpips'] for x in outputs]).mean()

        self.log_dict(
            dictionary={
                'psnr/test': psnr.item(),
                'ssim/test': ssim.item(),
                'lpips/test': lpips.item()
            }
        )

    def log_image(self, tag: str, imgs: torch.Tensor) -> None:
        img, *_ = self.normalize_tensor(imgs)[:1, ...]
        self.logger.experiment.add_image(tag, img, self.global_step)

    def normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        _min, _max = min(self.config.val_range), max(self.config.val_range)
        return (tensor - _min) / (_max - _min)

    @classmethod
    def generate_config(cls, output_dir: str):

        config_dict = {
            '_taregt_': cls.__module__ + '.' + cls.__name__,
            'config': {
                'scale_factor': 4,
                'preupsample': False,
                'val_range': [0., 1.],
                'networks': {},
                'losses': {},
                'optimizers': {},
                'schedulers': {},
            }
        }

        for k, v in cls.__dict__.get('__annotations__').items():
            if 'net' in k:
                config_dict['config']['networks'][k] = {
                    'args': {'_target_': '???', },
                    'weights': {'weights': None, 'init_type': 'normal', 'init_gain': 0.02}
                }
            if 'criterion' in k:
                config_dict['config']['losses'][k] = {
                    'args': {'_target_': '???', },
                    'weight': 1.
                }
            if 'optimizer' in k:
                config_dict['config']['optimizers'][k] = {
                    'args': {'_target_': '???', 'lr': 1e-3},
                    'params': []
                }
            if 'scheduler' in k:
                config_dict['config']['schedulers'][k] = {
                    'args': {'_target_': '???'},
                    'optimizer': '???'
                }

        filename = Path(output_dir).joinpath(f'model/{cls.__name__.rstrip("Model").lower()}.yaml')
        filename.parent.mkdir(parents=True, exist_ok=True)

        OmegaConf.save(OmegaConf.create(config_dict), filename)
