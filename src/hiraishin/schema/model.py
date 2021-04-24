from typing import Dict, List, Optional, Union

from pydantic import BaseModel, FilePath

from .common import Instantiable, ModuleConfig


class NetworkConfig(ModuleConfig):

    class InitConfig(BaseModel):
        weight_path: Optional[Union[FilePath, Dict[str, FilePath]]] = None
        init_type: Optional[str] = None
        init_gain: Optional[float] = None

    init: InitConfig


class LossConfig(ModuleConfig):
    weight: Optional[float] = 1.


class OptimizerConfig(ModuleConfig):

    class SchedulerConfig(BaseModel):
        args: Instantiable
        interval: str = 'epoch'
        frequency: int = 1
        strict: bool = True
        monitor: Optional[str] = None

    params: List[str] = []
    scheduler: Optional[SchedulerConfig] = None


class ModelConfig(Instantiable):

    class ConfigBody(BaseModel):
        networks: List[NetworkConfig]
        losses: List[LossConfig]
        optimizers: List[OptimizerConfig]
        modules: Optional[List[ModuleConfig]] = None

    config: ConfigBody
