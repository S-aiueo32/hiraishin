import itertools
from abc import ABCMeta
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Callable, TypeVar

import torch.nn as nn
import torch.optim as optim
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from overrides import final
from pytorch_lightning import LightningModule

from hiraishin import schema
from hiraishin.models.utils import (
    BasicWeightInitializer,
    load_weights,
    get_arguments,
    get_class_name_with_shortest_module,
)

logger = getLogger(__name__)


Module = TypeVar(
    "Module",
    bound=nn.Module,
)
Optimizer = TypeVar(
    "Optimizer",
    bound=optim.Optimizer,
)
LRScheduler = TypeVar(
    "LRScheduler",
    bound=optim.lr_scheduler._LRScheduler,
)


class BaseModel(LightningModule, metaclass=ABCMeta):
    @final
    @classmethod
    def _target_(cls) -> str:
        """Get the shortest module name that is importable from the current directory."""
        return get_class_name_with_shortest_module(cls)

    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        self.config = schema.ModelConfigBody(**OmegaConf.to_container(config))

        self.save_hyperparameters()
        self.hparams["_target_"] = self._target_()
        self.hparams["_recursive_"] = False

        self.register_components()
        self.validate()

    @final
    def register_components(self) -> None:
        # Firstly, set non-reserved parameteres
        for name, config in self.config.dict(by_alias=True).items():
            if name in schema.ModelConfigBody.__fields__.keys():
                # skips networks, losses and optimizers
                continue
            if isinstance(config, dict) and "_target_" in config:
                # set with instantiation
                setattr(self, name, instantiate(config))
            else:
                # set other parameters
                setattr(self, name, config)

        # define training components
        self.define_networks()
        self.define_losses()
        self.define_optimizers()

    @final
    def define_networks(self) -> None:
        for name, config in self.config.networks.items():
            if not name.startswith("net"):
                raise NameError(
                    'Configurations for networks must have the prefix "net".'
                )

            # network definition
            net: Module = instantiate(config.args.dict(by_alias=True))

            # initialize weights
            if config.weights.initializer is not None:
                initializer: Callable[[Module], None] = instantiate(
                    config.weights.initializer.dict(by_alias=True)
                )
            else:
                # initialize with xavier_uniform_(gain=1.0) by default
                initializer = BasicWeightInitializer(
                    init_type="xavier_uniform",
                    gain=1.0,
                )
            initializer(net)

            # loading pretrained models
            if isinstance(config.weights.path, Path):  # for whole network
                load_weights(net, config.weights.path, name)
            if isinstance(config.weights.path, dict):
                for (mod_name, path) in config.weights.path.items():  # for modules
                    load_weights(getattr(net, mod_name), path)

            setattr(self, name, net)

    @final
    def define_losses(self) -> None:
        for name, config in self.config.losses.items():
            if not name.startswith("criterion"):
                raise NameError(
                    'Configurations for loss functions must have the prefix "criterion".'
                )
            criterion: Module = instantiate(config.args.dict(by_alias=True))
            setattr(self, name, criterion)
            setattr(self, name.replace("criterion", "weight"), config.weight)

    @final
    def define_optimizers(self) -> None:
        for name, config in self.config.optimizers.items():
            if not name.startswith("optimizer"):
                raise NameError(
                    'Configurations for optimizers must have the prefix "optimizer".'
                )
            targets: List[Module] = [getattr(self, net) for net in config.params]
            optimizer: Optimizer = instantiate(
                config.args.dict(by_alias=True),
                itertools.chain(*[net.parameters() for net in targets]),
            )
            setattr(self, name, optimizer)

            if config.scheduler is not None:
                scheduler: LRScheduler = instantiate(
                    config.scheduler.args.dict(by_alias=True),
                    optimizer=optimizer,
                )
            else:
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
            setattr(self, name.replace("optimizer", "scheduler"), scheduler)

    @final
    def configure_optimizers(
        self,
    ) -> Tuple[List[Optimizer], List[Dict[str, Any]]]:
        optim_list, sched_list = [], []
        for name, config in self.config.optimizers.items():
            optimizer: Optimizer = getattr(self, name)
            optim_list.append(optimizer)

            scheduler: LRScheduler = getattr(
                self, name.replace("optimizer", "scheduler")
            )
            sched_dict = {
                "scheduler": scheduler,
                "interval": config.scheduler.interval,
                "frequency": config.scheduler.frequency,
                "name": f"lr/{name}",
            }
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                sched_dict.update(
                    {
                        "reduce_on_plateau": True,
                        "monitor": config.scheduler.monitor or "loss/val",
                        "strict": config.scheduler.strict,
                    }
                )
            sched_list.append(sched_dict)
        return optim_list, sched_list

    @final
    def validate(self) -> None:
        """Checks wheather annotated variables is defined correctly. Additional variables will be allowed."""
        is_valid: List[bool] = []
        for k, v in self.__class__.__dict__.get("__annotations__").items():
            if not hasattr(self, k):
                logger.warning(f"class variable {k} is not initialized.")
                is_valid.append(False)
                continue
            if not isinstance(getattr(self, k), v):
                logger.warning(f"type of class variable {k} is not matched with {v}")
                is_valid.append(False)
                continue
            is_valid.append(True)
        if all(is_valid):
            logger.info("model initialization successed!")
        else:
            raise ValueError("model initialization failed because of above errors.")

    @final
    @classmethod
    def generate(cls, output_dir: Union[str, Path], with_kwargs: bool = False) -> None:
        """Gererates a configuration file from type annotations."""

        annotations: Dict[str, Any] = cls.__dict__.get("__annotations__")

        networks: Dict[str, schema.NetworkConfig] = {}
        for name, _cls in annotations.items():
            if not name.startswith("net"):
                continue
            networks.update(
                {
                    name: schema.NetworkConfig(
                        args=schema.Instantiable(
                            _target_=get_class_name_with_shortest_module(_cls),
                            **get_arguments(_cls, with_kwargs),
                        ),
                        weights=schema.WeightsConfig(),
                    )
                }
            )

        losses: Dict[str, schema.LossConfig] = {}
        for name, _cls in annotations.items():
            if not name.startswith("criterion"):
                continue
            losses.update(
                {
                    name: schema.LossConfig(
                        args=schema.Instantiable(
                            _target_=get_class_name_with_shortest_module(_cls),
                            **get_arguments(_cls, with_kwargs),
                        ),
                        weight=1.0,
                    )
                }
            )

        optimizers: List[schema.OptimizerConfig] = {}
        for name in annotations.keys():
            if not name.startswith("scheduler"):
                continue
            # check wheather the scheduler has the corresponding optimizer
            if name.replace("scheduler", "optimizer") not in annotations:
                raise ValueError(
                    "The scheduler can be defined with the correcponding optimizer."
                )
        for name, _cls in annotations.items():
            if not name.startswith("optimizer"):
                continue

            # scheduler
            if name.replace("optimizer", "scheduler") in annotations:
                sched_cls = annotations.get(name.replace("optimizer", "scheduler"))
                scheduler = schema.SchedulerConfig(
                    args=schema.Instantiable(
                        _target_=get_class_name_with_shortest_module(sched_cls),
                        **get_arguments(sched_cls, with_kwargs),
                    )
                )

            else:
                scheduler = None

            # optimizer
            optimizers.update(
                {
                    name: schema.OptimizerConfig(
                        args=schema.Instantiable(
                            _target_=get_class_name_with_shortest_module(_cls),
                            **get_arguments(_cls, with_kwargs),
                        ),
                        params=["???"],
                        scheduler=scheduler,
                    )
                }
            )

        if len(optimizers) > 1:
            print(
                "The order of optimizer configurations must match the order of optimization. (e.g., in GANs training.)"
            )
            print("Current order:")
            for i, optimizer_name in enumerate(optimizers.keys()):
                print(f"\t{i}: {optimizer_name}")

            while True:
                print("Proceed? [Y/n]: ", end="")
                proceed = input()
                if proceed in ["", "Y", "n"]:
                    break
                print("Invalid input!")

            if proceed == "n":
                indices = list(range(len(optimizers)))
                example_str = str(indices).lstrip("[").rstrip("]").replace(" ", "")
                while True:
                    try:
                        print(f"Input order (example: {example_str}): ", end="")
                        order_str = f"[{input()}]"
                        order = eval(order_str)
                        if set(order) != set(indices):
                            raise ValueError
                        optimizers = {
                            list(optimizers.keys())[o]: list(optimizers.values())[o]
                            for o in order
                        }
                        break
                    except Exception:
                        print("Invalid input!")
                        pass

        others: Dict[str, Union[schema.Instantiable, int, str, float, dict]] = {}
        for name, _cls in annotations.items():
            if any(
                name.startswith(prefix)
                for prefix in ["net", "criterion", "optimizer", "scheduler"]
            ):
                continue
            elif _cls.__module__ == "builtins":
                if _cls in (int, str, float, bool):
                    if hasattr(cls, name):
                        others.update({name: getattr(cls, name)})
                    else:
                        others.update({name: "???"})
                elif _cls == list:
                    if hasattr(cls, name):
                        others.update({name: getattr(cls, name)})
                    else:
                        others.update({name: ["???"]})
                elif _cls == dict:
                    if hasattr(cls, name):
                        others.update({name: getattr(cls, name)})
                    else:
                        others.update({name: {"???": "???"}})
                else:
                    raise TypeError(
                        "Supported built-in type is int, str, float or dict."
                    )
            else:
                others.update(
                    {
                        name: schema.Instantiable(
                            _target_=get_class_name_with_shortest_module(_cls),
                            **get_arguments(_cls, with_kwargs),
                        ),
                    }
                )

        m = schema.ModelConfig(
            _target_=cls._target_(),
            _recursive_=False,
            config=schema.ModelConfigBody(
                networks=networks,
                losses=losses,
                optimizers=optimizers,
                **others,
            ),
        ).dict(by_alias=True)

        filename = Path(output_dir).joinpath(f"{cls.__name__}.yaml")
        filename.parent.mkdir(parents=True, exist_ok=True)

        OmegaConf.save(OmegaConf.create(m), filename)
        print(f"The config has been generated! --> {filename}")
