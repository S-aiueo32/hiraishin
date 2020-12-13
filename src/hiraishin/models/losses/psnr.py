import warnings

import kornia
import torch
import torch.nn as nn


class PSNR(nn.Module):
    """
    """

    def __init__(self, max_val: float = 1.0) -> None:

        super(PSNR, self).__init__()

        self.max_val = max_val
        self.criterion = kornia.losses.PSNRLoss(max_val)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        assert x.shape[1] == 3
        if not (torch.all(0. <= x) and torch.all(x <= self.max_val)):
            warnings.warn(f'input tensor is not normalize to [0., {self.max_val}].')

        x = kornia.color.rgb_to_ycbcr(x)[:, :1, :, :]
        y = kornia.color.rgb_to_ycbcr(y)[:, :1, :, :]

        return self.criterion(x, y)
