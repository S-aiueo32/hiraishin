import re
from io import StringIO
from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue

from hiraishin.models import BaseModel
from hiraishin.schema import ModelConfig, validate


class ToyModel(BaseModel):
    """Simple toy model class"""

    net: torch.nn.Linear
    criterion: torch.nn.CrossEntropyLoss
    optimizer: torch.optim.Adam
    scheduler: torch.optim.lr_scheduler.ExponentialLR

    def __init__(self, config: ModelConfig):
        self.initialize(config)


class MultipleOptimizersModel(BaseModel):
    """Model class with multiple optimizers"""

    net: torch.nn.Linear
    criterion: torch.nn.CrossEntropyLoss
    optimizer_1: torch.optim.Adam
    optimizer_2: torch.optim.Adam

    def __init__(self, config: ModelConfig):
        self.initialize(config)


class AloneShcdulerModel(BaseModel):
    """Invalid model class"""

    net: torch.nn.Linear
    criterion: torch.nn.CrossEntropyLoss
    optimizer: torch.optim.Adam
    scheduler_: torch.optim.lr_scheduler.ExponentialLR

    def __init__(self, config: ModelConfig):
        self.initialize(config)


@pytest.mark.usefixtures('tmpdir')
class TestModel:

    @pytest.mark.usefixtures('shared_datadir')
    def test_instantiate(self, tmpdir, shared_datadir):
        """Check model instantiation."""
        config = OmegaConf.load(shared_datadir.joinpath('toy.yaml'))
        model = instantiate(config)
        assert isinstance(model, ToyModel)

    def test_instantiate_with_missing_values(self, tmpdir):
        """Check if raising error with missing values."""
        ToyModel.configen(tmpdir)
        config = OmegaConf.load(Path(tmpdir).joinpath('model/toy.yaml'))
        validate(config, ModelConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = instantiate(config)

    def test_configen(self, tmpdir):
        """Check if generating valid configs for pydantic."""
        ToyModel.configen(tmpdir)
        config = OmegaConf.load(Path(tmpdir).joinpath('model/toy.yaml'))
        validate(config, ModelConfig)

    @pytest.mark.usefixtures('monkeypatch')
    def test_configen_with_multiple_optimizers(self, tmpdir, monkeypatch):
        """Check if generating configs with multiple optimizers."""

        camel2snake = r'(?<!^)(?=[A-Z])'
        cls_name_snake = re.sub(camel2snake, '_', MultipleOptimizersModel.__name__.strip("Model")).lower()

        # The order of optimizer configuration must match the order of optimization.
        # Current order:
        #         0: optimizer_1
        #         1: optimizer_2

        # Proceed? [Y/n]: Y
        monkeypatch.setattr('sys.stdin', StringIO('Y'))
        MultipleOptimizersModel.configen(tmpdir)

        model_config: ModelConfig = OmegaConf.load(Path(tmpdir).joinpath(f'model/{cls_name_snake}.yaml'))
        assert model_config.config.optimizers[0].name == 'optimizer_1'
        assert model_config.config.optimizers[1].name == 'optimizer_2'

        # Proceed? [Y/n]: n
        # Input order (example: 0,1): 1,0
        monkeypatch.setattr('sys.stdin', StringIO('\n'.join(['n', '1,0'])))
        MultipleOptimizersModel.configen(tmpdir)

        model_config: ModelConfig = OmegaConf.load(Path(tmpdir).joinpath(f'model/{cls_name_snake}.yaml'))
        assert model_config.config.optimizers[0].name == 'optimizer_2'
        assert model_config.config.optimizers[1].name == 'optimizer_1'

        # Proceed? [Y/n]: n
        # Input order (example: 0,1): 1,1
        # Invalid input!
        # Input order (example: 0,1): 1,0
        monkeypatch.setattr('sys.stdin', StringIO('\n'.join(['x', 'n', '1,1', '1,0'])))
        MultipleOptimizersModel.configen(tmpdir)

    def test_configen_with_alone_scheduler(self, tmpdir):
        """Check error handling for alone scheduler."""
        with pytest.raises(ValueError):
            AloneShcdulerModel.configen(tmpdir)
