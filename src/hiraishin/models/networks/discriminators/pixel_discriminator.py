import torch.nn as nn


class PixelDiscriminator(nn.Sequential):
    """
    """

    def __init__(self,
                 in_channels: int,
                 ndf: int = 32) -> None:

        super().__init__(
            nn.Conv2d(in_channels, ndf, kernel_size=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
