import importlib
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Extra, Field, FilePath, validator


class Instantiable(BaseModel, extra=Extra.allow):
    target: str = Field(..., alias="_target_")

    @validator("target")
    def is_importable(cls, v: str):
        *module_name, cls_name = v.split(".")
        try:
            _ = getattr(importlib.import_module(".".join(module_name)), cls_name)
            return v
        except (AttributeError, ValueError):
            raise ValueError(f"previded _target_ ({v}) is not importable.")


class ModuleConfig(BaseModel, extra=Extra.allow):
    args: Instantiable


class WeightsConfig(BaseModel):
    initializer: Optional[Instantiable] = None
    path: Optional[Union[FilePath, Dict[str, FilePath]]] = None

    @validator("path")
    def check_extensions(cls, v: Union[Path, Dict[str, Path]]):
        if isinstance(v, Path):
            if v.stem not in [".ckpt", ".pth"]:
                raise ValueError(".ckpt or .pth are allowed.")
        if isinstance(v, dict):
            if any(path.suffix != ".pth" for path in v.values()):
                raise ValueError("Only .pth is allowed for partial weights.")
        return v


class NetworkConfig(ModuleConfig):
    weights: WeightsConfig


class LossConfig(ModuleConfig):
    weight: float = 1.0


class SchedulerConfig(BaseModel):
    args: Instantiable
    interval: Literal["epoch", "step"] = "epoch"
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
