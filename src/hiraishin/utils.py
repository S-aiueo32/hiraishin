from pathlib import Path
from typing import TypeVar, Union

import hydra
import torch

from hiraishin.models import BaseModel

Model = TypeVar("Model", bound=BaseModel)


def load_from_checkpoint(path: Union[str, Path]) -> Model:
    ckpt = torch.load(str(path))

    hparams = ckpt["hyper_parameters"]
    if "_target_" not in hparams:
        raise KeyError("_target_ key does not exist in the checkpoint.")

    model: Model = hydra.utils.instantiate(ckpt["hyper_parameters"], _recursive_=False)
    model.load_state_dict(ckpt["state_dict"])

    return model
