import importlib
import sys
from os.path import relpath, dirname
from pathlib import Path

import click
from omegaconf import OmegaConf
from hydra.experimental import compose, initialize

from hiraishin.models import BaseModel
from hiraishin.scripts.train import app


@click.group()
def cmd() -> None:
    pass


@cmd.command()
@click.option('--config_path', type=str, default='config')
@click.option('--config_name', type=str, default='train')
@click.option('--overrides', type=str, default='')
def train(config_path: str, config_name: str, overrides: str) -> None:

    initialize(config_path=relpath(config_path, dirname(__file__)))

    config = compose(config_name, overrides=overrides.split(), return_hydra_config=True)

    OmegaConf.resolve(config)
    OmegaConf.set_struct(config, False)
    config_hydra = config.pop('hydra')

    run_dir = Path(config_hydra.run.dir)
    run_dir.mkdir(exist_ok=True, parents=True)

    app(config)


@cmd.command()
@click.argument('model_class', type=str)
@click.option('--output_dir', type=str, default='./config')
@click.option('--with_kwargs', is_flag=True, default=False)
def configen(model_class: str, output_dir: str, with_kwargs: bool):

    sys.path.append(str(Path.cwd()))

    *modules, cls = model_class.split('.')
    Model: BaseModel = getattr(importlib.import_module('.'.join(modules)), cls)
    Model.configen(output_dir, with_kwargs)


if __name__ == '__main__':
    cmd()
