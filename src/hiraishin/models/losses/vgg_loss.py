import logging
from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.vgg as vgg

logger = logging.getLogger(__name__)


class VGGLoss(nn.Module):
    """computes MSE between internal features of VGG feature extractor.
    """

    def __init__(self, net_type: str = 'vgg19', target_layer: str = 'relu5_4') -> None:
        super(VGGLoss, self).__init__()
        self.model = VGG(net_type=net_type)
        self.target_layer = target_layer

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_feat, *_ = self.model(x, [self.target_layer]).values()
        y_feat, *_ = self.model(y, [self.target_layer]).values()
        return F.mse_loss(x_feat, y_feat)


class VGG(nn.Module):
    """defines VGGs provided by torchvision. The forward() method picks the features of specified layer.
    """

    NAMES: Dict[str, List[str]] = {
        'vgg11': [
            'conv1_1', 'relu1_1', 'pool1',
            'conv2_1', 'relu2_1', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'pool5',
        ],
        'vgg13': [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'pool5',
        ],
        'vgg16': [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1',
            'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3',
            'conv4_1', 'relu4_1',
            'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
            'conv5_1', 'relu5_1',
            'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5',
        ],
        'vgg19': [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
            'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2',
            'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
            'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
        ]
    }

    def __init__(self, net_type: str, requires_grad: bool = False) -> None:
        super(VGG, self).__init__()

        features = getattr(vgg, net_type)(True).features
        self.names = self.NAMES[net_type.rstrip('_bn')]
        if 'bn' in net_type:
            self.names = self.insert_bn(self.names)

        self.net = nn.Sequential(OrderedDict([(k, v) for k, v in zip(self.names, features)]))

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.register_buffer('vgg_mean', torch.tensor([[[0.485]], [[0.456]], [[0.406]]], requires_grad=False))
        self.register_buffer('vgg_std', torch.tensor([[[0.229]], [[0.224]], [[0.225]]], requires_grad=False))

    @staticmethod
    def insert_bn(names: List[str]) -> List[str]:
        """inserts 'bn' after conv key.
        """

        names_bn = []
        for name in names:
            names_bn.append(name)
            if 'conv' in name:
                pos = name.replace('conv', '')
                names_bn.append('bn' + pos)
        return names_bn

    def z_score(self, x: torch.Tensor) -> torch.Tensor:
        x = x.sub(self.vgg_mean.detach())
        x = x.div(self.vgg_std.detach())
        return x

    def forward(self, x: torch.Tensor, targets: List[str]) -> Dict[str, torch.Tensor]:

        assert all([t in self.names for t in targets]), 'Specified name does not exist.'

        if torch.all(x < 0.) and torch.all(x > 1.):
            logger.warn('Input tensor is not normalize to [0, 1].')

        x = self.z_score(x)

        out_dict = OrderedDict()
        for key, layer in self.net._modules.items():
            x = layer(x)
            if key in targets:
                out_dict.update({key: x})
            if len(out_dict) == len(targets):  # to reduce wasting computation
                break

        return out_dict
