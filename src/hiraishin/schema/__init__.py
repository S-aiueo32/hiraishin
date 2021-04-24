from typing import Union

from omegaconf import OmegaConf, DictConfig

from .model import ModelConfig


def validate(config: DictConfig, schema: Union[ModelConfig]):
    config = OmegaConf.to_container(config)
    _ = schema(**config)


__all__ = (
    # modules
    ModelConfig.__name__,
    # methods
    validate.__name__,
)
