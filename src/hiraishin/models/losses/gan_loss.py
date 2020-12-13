import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """
    """

    def __init__(self, gan_mode: str, target_real_label: float = 1.0, target_fake_label: float = 0.0) -> None:

        super(GANLoss, self).__init__()

        assert gan_mode in ['lsgan', 'vannila', 'wgangp'], \
            'gan_mode must be selected from ["lsgan",  "vannila", "wgangp"].'

        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgangp':
            self.loss = None

        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: torch.Tensor) -> torch.Tensor:
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction).detach()

    def forward(self, prediction: torch.Tensor, target_is_real: torch.Tensor) -> torch.Tensor:
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            return self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                return -prediction.mean()
            else:
                return prediction.mean()
