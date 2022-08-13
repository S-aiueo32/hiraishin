from io import StringIO
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue
from torch.utils.data import DataLoader, Dataset

from hiraishin.models import BaseModel
from hiraishin.schema import ModelConfig, validate
from hiraishin.utils import load_from_checkpoint


class ToyDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(0.0)

    def __len__(self) -> int:
        return 1


class ToyDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(ToyDataset())
        return dataloader


class ToyModel(BaseModel):
    """Simple toy model class"""

    net: torch.nn.Linear
    criterion: torch.nn.CrossEntropyLoss
    optimizer: torch.optim.Adam
    scheduler: torch.optim.lr_scheduler.ExponentialLR

    def __init__(self, config: DictConfig):
        super().__init__(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, *args, **kwargs) -> torch.Tensor:
        return self.forward(batch)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return torch.tensor(0.0)

    def validation_step_end(self, *args, **kwargs) -> torch.Tensor:
        return torch.tensor(0.0)


class DummyClass:
    def __init__(self, one, two=2) -> None:
        pass


class NotReservedParamsModel(BaseModel):
    """Model class with multiple optimizers"""

    foo: str
    foo_: str = "foo"

    bar: int
    bar_: int = 0

    buzz: float
    buzz_: float = 0.0

    fizz: dict
    fizz_: dict = {
        "one": 1,
        "two": 2,
    }

    dummy: DummyClass

    def __init__(self, config: DictConfig):
        super().__init__(config)


class MultipleOptimizersModel(BaseModel):
    """Model class with multiple optimizers"""

    net: torch.nn.Linear
    criterion: torch.nn.CrossEntropyLoss
    optimizer_1: torch.optim.Adam
    optimizer_2: torch.optim.Adam

    def __init__(self, config: DictConfig):
        super().__init__(config)


class AloneShcdulerModel(BaseModel):
    """Invalid model class"""

    net: torch.nn.Linear
    criterion: torch.nn.CrossEntropyLoss
    optimizer: torch.optim.Adam
    scheduler_: torch.optim.lr_scheduler.ExponentialLR

    def __init__(self, config: DictConfig):
        super().__init__(config)


@pytest.mark.usefixtures("tmpdir")
class TestModel:
    @pytest.mark.usefixtures("shared_datadir")
    def test_instantiate(self, tmpdir, shared_datadir):
        """Check model instantiation."""
        config = OmegaConf.load(shared_datadir.joinpath("ToyModel.yaml"))
        model = instantiate(config)
        assert isinstance(model, ToyModel)

    def test_instantiate_with_missing_values(self, tmpdir):
        """Check if raising error with missing values."""
        ToyModel.generate(tmpdir)
        config = OmegaConf.load(Path(tmpdir).joinpath("ToyModel.yaml"))
        assert isinstance(config, DictConfig)
        validate(config, ModelConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = instantiate(config)

    @pytest.mark.usefixtures("shared_datadir")
    def test_load(self, tmpdir, shared_datadir):
        """Check model instantiation."""
        config = OmegaConf.load(shared_datadir.joinpath("ToyModel.yaml"))
        model = instantiate(config)

        trainer = pl.Trainer(
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(model, datamodule=ToyDataModule())

        # save checkpoint
        ckpt_path = Path(tmpdir).joinpath("checkpoint.ckpt")
        trainer.save_checkpoint(ckpt_path)

        # load models
        load_from_checkpoint(ckpt_path)

    def test_generate(self, tmpdir):
        """Check if generating valid configs for pydantic."""
        ToyModel.generate(tmpdir)
        config = OmegaConf.load(Path(tmpdir).joinpath("ToyModel.yaml"))
        assert isinstance(config, DictConfig)
        validate(config, ModelConfig)

    def test_generate_with_not_reserved_params(self, tmpdir):
        """Check if generating valid configs with not-reserved parameters."""
        NotReservedParamsModel.generate(tmpdir, with_kwargs=True)
        config = OmegaConf.load(Path(tmpdir).joinpath("NotReservedParamsModel.yaml"))
        assert isinstance(config, DictConfig)
        validate(config, ModelConfig)

        with pytest.raises(MissingMandatoryValue):
            _ = config.config.foo
        assert config.config.foo_ == "foo"

        with pytest.raises(MissingMandatoryValue):
            _ = config.config.bar
        assert config.config.bar_ == 0

        with pytest.raises(MissingMandatoryValue):
            _ = config.config.buzz
        assert config.config.buzz_ == 0.0

        with pytest.raises(MissingMandatoryValue):
            _ = config.config.fizz.items()
        assert config.config.fizz_ == {"one": 1, "two": 2}

        assert config.config.dummy._target_ == "tests.test_models.DummyClass"
        with pytest.raises(MissingMandatoryValue):
            _ = config.config.dummy.one
        assert config.config.dummy.two == 2

    @pytest.mark.usefixtures("monkeypatch")
    def test_generate_with_multiple_optimizers(self, tmpdir, monkeypatch):
        """Check if generating configs with multiple optimizers."""

        # The order of optimizer configuration must match the order of optimization.
        # Current order:
        #         0: optimizer_1
        #         1: optimizer_2

        # Proceed? [Y/n]: Y
        monkeypatch.setattr("sys.stdin", StringIO("Y"))
        MultipleOptimizersModel.generate(tmpdir)

        model_config = OmegaConf.load(Path(tmpdir).joinpath(f"{MultipleOptimizersModel.__name__}.yaml"))
        assert list(model_config.config.optimizers.keys()) == [
            "optimizer_1",
            "optimizer_2",
        ]

        # Proceed? [Y/n]: n
        # Input order (example: 0,1): 1,0
        monkeypatch.setattr("sys.stdin", StringIO("\n".join(["n", "1,0"])))
        MultipleOptimizersModel.generate(tmpdir)

        model_config = OmegaConf.load(Path(tmpdir).joinpath(f"{MultipleOptimizersModel.__name__}.yaml"))
        assert list(model_config.config.optimizers.keys()) == [
            "optimizer_2",
            "optimizer_1",
        ]

        # Proceed? [Y/n]: n
        # Input order (example: 0,1): 1,1
        # Invalid input!
        # Input order (example: 0,1): 1,0
        monkeypatch.setattr("sys.stdin", StringIO("\n".join(["x", "n", "1,1", "1,0"])))
        MultipleOptimizersModel.generate(tmpdir)

    def test_generate_with_alone_scheduler(self, tmpdir):
        """Check error handling for alone scheduler."""
        with pytest.raises(ValueError):
            AloneShcdulerModel.generate(tmpdir)
