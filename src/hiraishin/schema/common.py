from typing import Optional

from pydantic import BaseModel, Extra, Field


class Instantiable(BaseModel):
    target: str = Field(..., alias='_target_')
    recursive: Optional[bool] = Field(True, alias='_recursive_')

    class Config:
        extra = Extra.allow


class ModuleConfig(BaseModel):
    name: str = '???'
    args: Instantiable

    class Config:
        extra = Extra.allow
