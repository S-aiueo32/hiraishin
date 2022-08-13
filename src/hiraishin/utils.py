from pathlib import Path
from typing import NewType, Union

import hydra
import torch
from omegaconf import DictConfig

from hiraishin.models import BaseModel

M = NewType("M", BaseModel)


def load_from_checkpoint(path: Union[str, Path]) -> M:
    ckpt = torch.load(str(path))

    hparams: DictConfig = ckpt["hyper_parameters"]
    if "_target_" not in hparams:
        raise KeyError("_target_ key is required.")

    model: M = hydra.utils.instantiate(hparams, _recursive_=False)
    model.load_state_dict(ckpt["state_dict"])

    return model
