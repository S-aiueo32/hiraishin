from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Extra, FilePath, Field, validator

from hiraishin.schema.common import Instantiable, ModuleConfig


class WeightsConfig(BaseModel):
    initializer: Optional[Instantiable] = None
    path: Optional[Union[FilePath, Dict[str, FilePath]]] = None

    @validator("path")
    def check_extensions(cls, v: Union[Path, Dict[str, Path]]):
        if isinstance(v, Path):
            if v.stem not in [".ckpt", ".pth"]:
                raise ValueError(".ckpt or .pth are allowed.")
        if isinstance(v, Path):
            if any(path.suffix != ".pth" for path in v.values()):
                raise ValueError("Only .pth is allowed for partial weights.")
        return v


class NetworkConfig(ModuleConfig):
    weights: WeightsConfig


class LossConfig(ModuleConfig):
    weight: float = 1.0


class SchedulerConfig(BaseModel):
    args: Instantiable
    interval: str = "epoch"
    frequency: int = 1
    strict: bool = True
    monitor: Optional[str] = None


class OptimizerConfig(ModuleConfig):
    params: List[str] = []
    scheduler: Optional[SchedulerConfig] = None


class ModelConfigBody(BaseModel, extra=Extra.allow):
    networks: Dict[str, NetworkConfig]
    losses: Dict[str, LossConfig]
    optimizers: Dict[str, OptimizerConfig]


class ModelConfig(Instantiable):
    recursive: Optional[bool] = Field(False, alias="_recursive_")
    config: ModelConfigBody
