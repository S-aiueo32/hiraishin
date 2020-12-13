import torch
import torch.nn as nn


class ZSSR(nn.Module):
    def __init__(self, ngf: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, ngf, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(ngf, ngf, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(ngf, ngf, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(ngf, ngf, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(ngf, ngf, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(ngf, ngf, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(ngf, ngf, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(ngf, 3, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)
