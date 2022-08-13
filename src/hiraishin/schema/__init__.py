from typing import Type

from omegaconf import DictConfig, OmegaConf

from .pydantic import (
    Instantiable,
    LossConfig,
    ModelConfig,
    ModelConfigBody,
    ModuleConfig,
    NetworkConfig,
    OptimizerConfig,
    SchedulerConfig,
    WeightsConfig,
)
from .typing import LRScheduler, Module, Optimizer


def validate(config: DictConfig, schema: Type[ModelConfig]):
    _ = schema.parse_obj(OmegaConf.to_container(config))


__all__ = (
    # pydantic
    "Instantiable",
    "LossConfig",
    "ModelConfig",
    "ModelConfigBody",
    "ModuleConfig",
    "NetworkConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "WeightsConfig",
    # typing
    "LRScheduler",
    "Module",
    "Optimizer",
    # methods
    "validate",
)
