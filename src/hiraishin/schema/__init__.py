from omegaconf import (
    OmegaConf,
    DictConfig,
)

from .common import (
    Instantiable,
    ModuleConfig,
)
from .model import (
    ModelConfig,
    ModelConfigBody,
    NetworkConfig,
    WeightsConfig,
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
)


def validate(config: DictConfig, schema: ModelConfig):
    config = OmegaConf.to_container(config)
    _ = schema(**config)


__all__ = (
    # modules
    Instantiable.__name__,
    ModuleConfig.__name__,
    ModelConfig.__name__,
    ModelConfigBody.__name__,
    NetworkConfig.__name__,
    WeightsConfig.__name__,
    LossConfig.__name__,
    OptimizerConfig.__name__,
    SchedulerConfig.__name__,
    # methods
    validate.__name__,
)
