from .gan_loss import GANLoss
from .ty_loss import TVLoss
from .vgg_loss import VGGLoss
from .psnr import PSNR
from .ssim import SSIM
from .lpips import LPIPS

__all__ = [
    GANLoss.__name__,
    TVLoss.__name__,
    VGGLoss.__name__,
    PSNR.__name__,
    SSIM.__name__,
    LPIPS.__name__
]
