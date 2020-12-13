import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Basic residual block.

    Args:
        nf (int): the number of filters.
    """

    def __init__(self, nf: int = 64) -> None:
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block.

    Args:
        nf (int): the number of base filters.
        n_rdbs (int): the number of residual dense blocks (RDB) in each RRDB.
        growth (int): the number of filters to grow for each dense convolution.
        n_convs (int): the number of dense convlution in each RDB.
    """

    def __init__(self, n_rdbs: int = 3, nf: int = 64, growth: int = 32, n_convs: int = 5) -> None:
        super(RRDB, self).__init__()
        self.body = nn.Sequential(
            *[RDB(nf, growth, n_convs) for i in range(n_rdbs)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.2 * self.body(x)


class RDB(nn.Module):
    """Residual Dense Block.

    Args:
        nf (int): the number of base filters.
        growth (int): the number of filters to grow for each dense convolution.
        n_convs (int): the number of dense convlution in each RDB.
    """

    def __init__(self, nf: int = 64, growth: int = 32, n_convs: int = 5) -> None:

        super(RDB, self).__init__()
        _n_convs = n_convs - 1
        self.body = nn.Sequential(
            *[DenseConv2d(nf + i * growth, growth) for i in range(_n_convs)],
            nn.Conv2d(nf + (_n_convs) * growth, nf, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.2 * self.body(x)


class DenseConv2d(nn.Module):
    """Dense convolutional layer.

    Args:
        nf (int): the number of filters.
        growth (int): the number of filters to grow for each dense convolution.
    """

    def __init__(self, nf: int = 64, growth: int = 32) -> None:
        super(DenseConv2d, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, growth, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat((x, self.body(x)), 1)
        return out


class UpscaleBlock(nn.Sequential):
    """Upscale block with sub-pixel convolution.

    Args:
        scale_factor (int): the upscale factor.
        nf (int): the number of filters.
    """

    def __init__(self, scale_factor: int, nf: int = 64):
        if scale_factor in [2, 3]:
            super(UpscaleBlock, self).__init__(
                nn.Conv2d(nf, nf * (scale_factor ** 2), 3, 1, 1),
                nn.PixelShuffle(scale_factor)
            )
        elif scale_factor in [4]:
            super(UpscaleBlock, self).__init__(
                nn.Conv2d(nf, nf * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1, True),
                nn.Conv2d(nf, nf * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1, True)
            )
        else:
            raise ValueError('scale factor x2, x3 and x4 is supported now.')
