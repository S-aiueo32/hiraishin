# Hiraishin
A thin PyTorch-Lightning wrapper for building configuration-based DL pipelines with Hydra.

# Dependencies
- PyTorch Lightning
- Hydra
- Pydantic
- etc.

# Installation

```shell
$ pip install -U hiraishin
```

# Basic workflow
## 1. Model initialization with type annotations
Define a model class that has training components of PyTorch as instance variables.

```python
import torch.nn as nn
import torch.optim as optim

from hiraishin.models import BaseModel


class ToyModel(BaseModel):

    net: nn.Linear
    criterion: nn.CrossEntropyLoss
    optimizer: optim.Adam
    scheduler: optim.lr_schedulers.ExponentialLR

    def __init__(self, config: DictConfig) -> None:
        self.initialize(config)  # call `initialize()` instead of `super()__init__()`
```

We recommend that they have the following prefix to indicate their role.

- `net` for networks. It must be a subclass of `nn.Module` to initialize and load weights.
- `criterion` for loss functions. 
- `optimizer` for optimizers. It must be subclass of `Optimizer`.
- `scheduler` for schedulers. It must be subclass of `_LRScheduler` and the suffix must match to the corresponding optimizer.

If you need to define modules besides the above components (e.g. tokenizers), feel free to define them. The modules will be defined with the names you specify.

## 2. Generating configuration file
Hiraishin has the functionality to generate configuration files on the command line.
If the above class was written in `model.py` at the same level as the current directory, you can generate it with the following command.

```shell
$ hiraishin configen model.ToyModel
The config has been generated! --> config/model/toy.yaml
```

Let's take a look at the generated file.
The positional arguments are filled with `???` that indicates mandatory parameters in Hydra.
We recommend overwriting them with the default value, otherwise, you must give them through command-line arguments for every run.

```yaml
_target_: model.ToyModel
_recursive_: false
config:
  networks:
  - name: net
    args:
      _target_: torch.nn.Linear
      _recursive_: true
      in_features: ???  # -> 1
      out_features: ???  # -> 1
    init:
      weight_path: null
      init_type: null
      init_gain: null
  losses:
  - name: criterion
    args:
      _target_: torch.nn.CrossEntropyLoss
      _recursive_: true
    weight: 1.0
  optimizers:
  - name: optimizer
    args:
      _target_: torch.optim.Adam
      _recursive_: true
    params:
    - ???  # -> net
    scheduler:
      args:
        _target_: torch.optim.lr_scheduler.ExponentialLR
        _recursive_: true
        gamma: ???  # -> 1
      interval: epoch
      frequency: 1
      strict: true
      monitor: null
  modules: null
```

## 3. Training routines definition
The rest of model definition is only defining your training routine along with the style of PyTorch Lightning.
```python
class ToyModel(BaseModel):
    
    ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, *args, **kwargs) -> torch.Tensor:
        x, target = batch
        pred = self.forward(x)
        loss = self.criterion(pred, target)
        self.log('loss/train', loss)
        return loss
```

## 4. Model Instantiation
The defined model can be instantiated from configuration file. Try to train and test models!
```python
from hydra.utils import inatantiate
from omegeconf import OmegaConf


def app():
    ...

    config = OmegaConf.load('config/model/toy.yaml')
    model = inatantiate(config)

    print(model)
    # ToyModel(
    #     (net): Linear(in_features=1, out_features=1, bias=True)
    #     (criterion): CrossEntropyLoss()
    # )

    trainer.fit(model, ...)
```

# License
Hiraishin is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text.
