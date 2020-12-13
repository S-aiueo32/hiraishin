import torch
from kornia.losses import TotalVariation


class TVLoss(TotalVariation):
    """
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).mean()
