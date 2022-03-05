import importlib

from pydantic import BaseModel, Extra, Field, validator


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
