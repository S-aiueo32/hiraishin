import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from omegaconf import DictConfig

from .base_model import BaseModel


class ZSSRModel(BaseModel):

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
        return batch['child']

    @staticmethod
    def get_gt(batch):
        return batch['parent']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net_G(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        inputs = self.get_inputs(batch)
        output = self.net_G(inputs)
        gt = self.get_gt(batch)

        loss = self.criterion_rec(output, gt)

        self.log('loss/train', loss, on_step=True)

        self.logger.experiment.add_image('img_parent/train', output[0, ...], self.global_step)
        self.logger.experiment.add_image('img_sr/train', output[0, ...], self.global_step)

        return loss
