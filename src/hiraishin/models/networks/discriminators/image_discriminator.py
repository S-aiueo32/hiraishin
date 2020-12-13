import torch.nn as nn


class ImageDiscriminator(nn.Sequential):
    """
    """

    def __init__(self,
                 in_channels: int,
                 ndf: int = 32) -> None:

        def ConvBlock(in_channels, out_channels, stride):
            out = [
                nn.Conv2d(in_channels, out_channels, 3, stride, 1),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(out_channels),
            ]
            return out

        super().__init__(
            nn.Conv2d(in_channels, ndf, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),

            *ConvBlock(ndf, ndf, 2),

            *ConvBlock(ndf, ndf * 2, 1),
            *ConvBlock(ndf * 2, ndf * 2, 2),

            *ConvBlock(ndf * 2, ndf * 4, 1),
            *ConvBlock(ndf * 4, ndf * 4, 2),

            *ConvBlock(ndf * 4, ndf * 8, 1),
            *ConvBlock(ndf * 8, ndf * 8, 2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ndf * 8, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Sigmoid()
        )
