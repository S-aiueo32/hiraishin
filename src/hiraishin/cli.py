import importlib
import sys
from pathlib import Path
from typing import NewType, Type

import click

from hiraishin.models import BaseModel

M = NewType("M", BaseModel)


@click.group()
def cmd() -> None:
    pass


@cmd.command()
@click.argument("model_class", type=str)
@click.option("--output_dir", type=click.Path(exists=True, file_okay=False), default="./config/model")
@click.option("--with_kwargs", is_flag=True, default=False)
def generate(model_class: str, output_dir: str, with_kwargs: bool):

    sys.path.append(str(Path.cwd()))

    *modules, cls = model_class.split(".")
    Model: Type[M] = getattr(importlib.import_module(".".join(modules)), cls)
    Model.generate(output_dir, with_kwargs)


if __name__ == "__main__":
    cmd()
