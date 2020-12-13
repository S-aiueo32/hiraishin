import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from omegaconf import DictConfig

from .base_model import BaseModel


class SRCNNModel(BaseModel):

    # networks
    net_G: Module
    # losses
    criterion_rec: Module
    # optimizers
    optimizer_G: Optimizer
    # schedulers
    scheduler_G: _LRScheduler

    def __init__(self, config: DictConfig) -> None:
        self.initialize(config)

    def configure_optimizers(self):
        return [self.optimizer_G], [self.scheduler_G]

    @staticmethod
    def get_inputs(batch):
        return (batch['lr'], )

    @staticmethod
    def get_gt(batch):
        return batch['hr']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net_G(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        img_hr: torch.Tensor = batch['hr']
        img_lr: torch.Tensor = batch['lr']
        img_sr: torch.Tensor = self.net_G(img_lr)

        loss = self.criterion_rec(img_sr, img_hr)

        self.log('loss/train', loss, on_step=True)

        self.log_image('img_hr/train', img_hr)
        self.log_image('img_sr/train', img_sr)

        return loss
