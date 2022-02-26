import importlib
from typing import Type

import click

from hiraishin.models import BaseModel


@click.group()
def cmd() -> None:
    pass


@cmd.command()
@click.argument(
    "model_class",
    type=str,
)
@click.option(
    "--output_dir",
    type=click.Path(exists=True, file_okay=False),
    default="./config",
)
@click.option(
    "--with_kwargs",
    is_flag=True,
    default=False,
)
def generate(model_class: str, output_dir: str, with_kwargs: bool):
    *modules, cls = model_class.split(".")
    Model: Type[BaseModel] = getattr(importlib.import_module(".".join(modules)), cls)
    Model.generate(output_dir, with_kwargs)


if __name__ == "__main__":
    cmd()
