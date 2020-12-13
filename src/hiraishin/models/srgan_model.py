from typing import Optional

from omegaconf import DictConfig
from overrides import overrides

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .base_model import BaseModel


class SRGANModel(BaseModel):

    # networks
    net_G: Module
    net_D: Module
    # losses
    criterion_rec: Module
    criterion_per: Module
    criterion_gan: Module
    # optimizers
    optimizer_G: Optimizer
    optimizer_D: Optimizer
    # schedulers
    scheduler_G: _LRScheduler
    scheduler_D: _LRScheduler

    def __init__(self, config: DictConfig) -> None:
        self.initialize(config)

    def configure_optimizers(self):
        return [self.optimizer_D, self.optimizer_G], [self.scheduler_D, self.scheduler_G]

    @staticmethod
    @overrides
    def get_inputs(batch):
        return batch['lr']

    @staticmethod
    @overrides
    def get_gt(batch):
        return batch['hr']

    @overrides
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net_G(x)

    @overrides
    def training_step(self, batch, batch_idx: int, optimizer_idx: Optional[int] = None) -> torch.Tensor:
        img_hr: torch.Tensor = batch['hr']
        img_lr: torch.Tensor = batch['lr']
        img_sr: torch.Tensor = self.net_G(img_lr)

        if optimizer_idx == 0:
            d_loss_real = self.criterion_gan(self.net_D(img_hr), True)
            d_loss_fake = self.criterion_gan(self.net_D(img_sr), False)

            d_loss = 1 + d_loss_real + d_loss_fake

            self.log_dict(
                dictionary={
                    'd_loss/train': d_loss,
                    'd_loss_real/train': d_loss_real,
                    'd_loss_fake/train': d_loss_fake,
                },
                on_step=True
            )

            return d_loss

        if optimizer_idx == 1:
            g_loss_rec = self.criterion_rec(img_sr, img_hr) * self.weight_rec
            g_loss_per = self.criterion_per(img_sr, img_hr) * self.weight_per
            g_loss_gan = self.criterion_gan(self.net_D(img_sr), True) * self.weight_gan

            g_loss = g_loss_rec + g_loss_per + g_loss_gan

            self.log_dict(
                dictionary={
                    'g_loss/train': g_loss,
                    'g_loss_rec/train': g_loss_rec,
                    'g_loss_per/train': g_loss_per,
                    'g_loss_gan/train': g_loss_gan,
                },
                on_step=True
            )

            self.log_image('img_hr/train', img_hr)
            self.log_image('img_sr/train', img_sr)

            return g_loss
