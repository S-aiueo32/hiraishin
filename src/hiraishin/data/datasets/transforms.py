import functools
import random
from typing import Callable, Sequence, Union

import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms

from torchsnooper import snoop


class TimofteAugmentation(transforms.Compose):
    """A composed transform of the augmentation proposed by Timofte et al.
    See the paper: https://arxiv.org/abs/1511.02228.
    Args:
        patch_size (int): the patch size used in random crop
    """

    def __init__(self, patch_size: int) -> None:
        super(TimofteAugmentation, self).__init__([
            transforms.RandomCrop(patch_size),
            transforms.RandomChoice([
                functools.partial(TF.rotate, angle=0),
                functools.partial(TF.rotate, angle=90),
                functools.partial(TF.rotate, angle=180),
                functools.partial(TF.rotate, angle=270),
            ]),
            transforms.RandomHorizontalFlip(),
        ])


class AdjustSize:
    """A transformation to adjust the image size to the scale factor.
    Args:
        scale_factor (int): the upscale factor
        mode (str): the size adjustment method, select from ["crop", "resize", "pad"]
    """

    def __init__(self, scale_factor: int, mode: str = 'crop') -> None:
        assert mode in ['crop', 'resize', 'pad']
        self.scale_factor = scale_factor
        self.mode = mode

    def __call__(self, x: Image.Image) -> Image.Image:
        if self.mode == 'crop':
            w, h = (l - l % self.scale_factor for l in x.size)
            return x.crop((0, 0, w, h))
        if self.mode == 'resize':
            w, h = (l - l % self.scale_factor for l in x.size)
            return x.resize((w, h), resample=Image.BICUBIC)
        if self.mode == 'pad':
            w, h = x.size
            pad_w = (self.scale_factor - w % self.scale_factor) % self.scale_factor
            pad_h = (self.scale_factor - h % self.scale_factor) % self.scale_factor
            padding = (0, 0, pad_h, pad_w)
            return TF.pad(x, padding, padding_mode='reflect')


class BicubicDegradation:
    """A simple bicubic degradation for the LR image generation.
    Args:
        scale_factor (int): the downscale factor
        preupscaple (bool): whether the input size of the model is adjusted to the output size
    """

    def __init__(self, scale_factor: int, preupsample: bool = False) -> None:
        self.scale_factor = scale_factor
        self.preupsample = preupsample

    def __call__(self, x: Image.Image) -> Image.Image:
        size = x.size[::-1]
        new_size = [l // self.scale_factor for l in size]
        x = TF.resize(x, new_size, Image.BICUBIC)
        if self.preupsample:
            return TF.resize(x, size, Image.BICUBIC)
        return x


class AlignedCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img_hr: Image.Image, img_lr: Image.Image) -> Sequence[Image.Image]:
        for t in self.transforms:
            img_hr, img_lr = t(img_hr, img_lr)
        return img_hr, img_lr


class AlignedRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_hr: Image.Image, img_lr: Image.Image) -> Sequence[Image.Image]:
        if random.random() < self.p:
            return TF.hflip(img_hr), TF.hflip(img_lr)
        return img_hr, img_lr

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class AlignedRandomRot90:
    def __init__(self) -> None:
        pass

    def __call__(self, img_hr: Image.Image, img_lr: Image.Image) -> Sequence[Image.Image]:
        k = random.randint(0, 3)
        return TF.rotate(img_hr, 90 * k), TF.rotate(img_lr, 90 * k)


class AlignedRandomCrop:
    def __init__(self, size: Union[int, Sequence[int]], scale_factor: int) -> None:
        assert size % scale_factor == 0
        self.size = (size, size) if isinstance(size, int) else size
        self.scale_factor = scale_factor

    @staticmethod
    def get_params(img: Image.Image, output_size: Sequence[int]) -> Sequence[int]:
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    @snoop()
    def __call__(self, img_hr: Image.Image, img_lr: Image.Image) -> Sequence[Image.Image]:
        params_lr = self.get_params(img_lr, [s // self.scale_factor for s in self.size])
        params_hr = [p * self.scale_factor for p in params_lr]
        return TF.crop(img_hr, *params_hr), TF.crop(img_lr, *params_lr)


class AlignedCenterCrop:
    def __init__(self, size: Union[int, Sequence[int]], scale_factor: int) -> None:
        assert size % scale_factor == 0
        self.size = (size, size) if isinstance(size, int) else size
        self.scale_factor = scale_factor

    @staticmethod
    def get_params(img: Image.Image, output_size: Sequence[int]) -> Sequence[int]:
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img_hr: Image.Image, img_lr: Image.Image) -> Sequence[Image.Image]:
        if self.size:
            params_lr = self.get_params(img_lr, [s // self.scale_factor for s in self.size])
            params_hr = [p * self.scale_factor for p in params_lr]
            return TF.crop(img_hr, *params_hr), TF.crop(img_lr, *params_lr)
        return img_hr, img_lr


class AlignedLambda:
    def __init__(self, transforms: Sequence[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, img_hr: Image.Image, img_lr: Image.Image) -> Sequence[Image.Image]:
        for t in self.transforms:
            img_hr = t(img_hr)
            img_lr = t(img_lr)
        return img_hr, img_lr


class ConditionedProcess:
    def __init__(self, transform: Callable, exec: bool) -> None:
        self.exec = exec
        if self.exec:
            self.transform = transform

    def __call__(self, *args, **kwargs):
        if self.exec:
            return self.transform(*args, **kwargs)
        else:
            return args


class SplittedProcess:
    def __init__(self, for_hr: Callable, for_lr: Callable) -> None:
        self.for_hr = for_hr
        self.for_lr = for_lr

    def __call__(self, img_hr: Image.Image, img_lr: Image.Image) -> Sequence[Image.Image]:
        return self.for_hr(img_hr), self.for_lr(img_lr)
