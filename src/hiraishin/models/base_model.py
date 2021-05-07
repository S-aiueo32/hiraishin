import inspect
import itertools
import re
from collections import OrderedDict
from abc import ABCMeta
from logging import getLogger
from typing import Any, Dict, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from overrides import final
from pytorch_lightning import LightningModule

from hiraishin.schema import ModelConfig
from hiraishin.utils import get_arguments, get_class_name_with_shortest_module

logger = getLogger(__name__)

try:
    import hydra
    CWD = Path(hydra.utils.get_original_cwd()).resolve()
except (ImportError, ValueError):
    CWD = Path.cwd()


def init_weights(net: nn.Module, init_type: str = 'xavier_uniform', init_gain: float = 1.) -> None:

    def init_func(m: nn.Module):
        name = m.__class__.__name__
        if hasattr(m, 'weight') and ('Conv' in name or 'Linear' in name):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'initialization method [{init_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in name:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    logger.info(f'Weights have been initialized with (type={init_type}, gain={init_gain}).')


def load_weights(net: nn.Module, weight_path: str, net_name: str = None):
    if (weight_path := CWD.joinpath(weight_path)).exists():
        if weight_path.suffix == '.pth':
            state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
            net.load_state_dict(torch.load(weight_path), strict=False)
        if weight_path.suffix == '.ckpt':
            assert net_name is not None, 'net_name is required to load weights from checkpoints.'
            state_dict = torch.load(weight_path, map_location=torch.device('cpu'))['state_dict']
            state_dict = OrderedDict((k, v) for k, v in state_dict.items() if k.startswith(net_name))
            state_dict = OrderedDict((k.replace(net_name + '.', ''), v) for k, v in state_dict.items())
            net.load_state_dict(state_dict)
        logger.info(f'Weights have been loaded from {str(weight_path)}.')
    else:
        logger.warn(f'{str(weight_path)} does not exists.')


class BaseModel(LightningModule, metaclass=ABCMeta):

    @final
    def initialize(self, config: ModelConfig.ConfigBody):
        """calls __init__() of BaseModel with sub-module information.
        """
        super(self.__class__, self).__init__(self.__class__._target_(), config)

    @final
    @classmethod
    def _target_(cls) -> str:
        """Get the shortest module name that is importable from the current directory.
        """
        return get_class_name_with_shortest_module(cls)

    def __init__(self, _target_: str, config: ModelConfig.ConfigBody):

        assert (caller := inspect.stack()[1]).function == 'initialize' or caller.filename == __file__,\
            f'The constructor of BaseModel can only be called through initialize() from the subclass {_target_}.'

        super().__init__()

        self.config = config
        self.save_hyperparameters()

        self.define_networks()
        self.define_losses()
        self.define_optimizers()

        self.validation()

    @final
    def define_networks(self):
        for net_config in self.config.networks:
            # network definition
            net = instantiate(net_config.args)

            # network initialization with random weights
            kwargs = {}
            if 'init_type' in net_config.init and net_config.init.init_type is not None:
                kwargs.update({'init_type': net_config.init.init_type})
            if 'init_gain' in net_config.init and net_config.init.init_gain is not None:
                kwargs.update({'init_gain': net_config.init.init_gain})
            init_weights(net, **kwargs)

            # loading pretrained models
            if 'weight_path' in net_config.init:
                if isinstance(net_config.init.weight_path, str):  # for whole network
                    load_weights(net, net_config.init.weight_path, net_config.name)
                if isinstance(net_config.init.weight_path, DictConfig):  # for partial modules
                    for module_name, path in net_config.init.weight_path.items():
                        if Path(path).suffix != '.pth':
                            raise ValueError('The weights for partial modules must be pure .pth')
                        load_weights(getattr(net, module_name), path)

            # set as an attribute
            setattr(self, net_config.name, net)

    @final
    def define_losses(self):
        for loss_config in self.config.losses:
            setattr(self, loss_config.name, instantiate(loss_config.args))
            if 'weight' in loss_config:
                setattr(self, loss_config.name.replace('criterion', 'weight'), loss_config.weight)

    @final
    def define_optimizers(self):
        for config in self.config.optimizers:
            targets: List[nn.Module] = [getattr(self, net) for net in config.params]
            optimizer = instantiate(config.args, itertools.chain(*[net.parameters() for net in targets]))
            setattr(self, config.name, optimizer)

            if 'scheduler' in config and config.scheduler is not None:
                scheduler = instantiate(config.scheduler.args, optimizer)
            else:
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.)
            setattr(self, config.name.replace('optimizer', 'scheduler'), scheduler)

    @final
    def define_modules(self):
        for module_config in self.config.modules:
            setattr(self, module_config.name, instantiate(module_config.args))

    @final
    def configure_optimizers(self):
        optim_list, sched_list = [], []
        for config in self.config.optimizers:
            optimizer = getattr(self, config.name)
            optim_list.append(optimizer)

            sched_dict = {
                'scheduler': instantiate(config.scheduler.args, optimizer),
                'interval': config.scheduler.interval if 'interval' in config.scheduler else 'epoch',
                'name': f'lr/{config.name}'
            }
            if isinstance(sched_dict['scheduler'], optim.lr_scheduler.ReduceLROnPlateau):
                sched_dict.update({
                    'reduce_on_plateau': True,
                    'monitor': config.scheduler.monitor if 'monitor' in config.scheduler else 'loss/val',
                    'strict': config.scheduler.strict if 'strict' in config.scheduler else True
                })
            sched_list.append(sched_dict)
        return optim_list, sched_list

    @final
    def validation(self):
        is_valid: List[bool] = []
        for k, v in self.__class__.__dict__.get('__annotations__').items():
            if not hasattr(self, k):
                logger.warning(f'class variable {k} is not initialized.')
                is_valid.append(False)
                continue
            if not isinstance(getattr(self, k), v):
                logger.warning(f'type of class variable {k} is not matched with {v}')
                is_valid.append(False)
                continue
            is_valid.append(True)
        if all(is_valid):
            logger.info('model initialization successed!')
        else:
            raise ValueError('model initialization failed because of above errors.')

    @final
    @classmethod
    def configen(cls, output_dir: str, with_kwargs: bool = False):

        from hiraishin.schema import common, model

        annotations: Dict[str, Any] = cls.__dict__.get('__annotations__')

        networks: List[model.NetworkConfig] = []
        for name, _cls in annotations.items():
            if 'net' not in name:
                continue
            networks.append(
                model.NetworkConfig(
                    name=name,
                    args=common.Instantiable(
                        _target_=get_class_name_with_shortest_module(_cls),
                        **get_arguments(_cls, with_kwargs)
                    ),
                    init=model.NetworkConfig.InitConfig()
                )
            )

        losses: List[model.LossConfig] = []
        for name, _cls in annotations.items():
            if 'criterion' not in name:
                continue
            losses.append(
                model.LossConfig(
                    name=name,
                    args=common.Instantiable(
                        _target_=get_class_name_with_shortest_module(_cls),
                        **get_arguments(_cls, with_kwargs)
                    ),
                    weight=1.
                )
            )

        optimizers: List[model.OptimizerConfig] = []
        for name in annotations.keys():
            if 'scheduler' not in name:
                continue
            # check wheather the scheduler has the corresponding optimizer
            if name.replace('scheduler', 'optimizer') not in annotations:
                raise ValueError('Scheduler must not be defined alone.')
        for name, _cls in annotations.items():
            if 'optimizer' not in name:
                continue
            # scheduler
            if name.replace('optimizer', 'scheduler') in annotations:
                sched_cls = annotations.get(name.replace('optimizer', 'scheduler'))
                scheduler = model.OptimizerConfig.SchedulerConfig(
                    args=common.Instantiable(
                        _target_=get_class_name_with_shortest_module(sched_cls),
                        **get_arguments(sched_cls, with_kwargs)
                    )
                )

            else:
                scheduler = None
            # optimizer
            optimizers.append(
                model.OptimizerConfig(
                    name=name,
                    args=common.Instantiable(
                        _target_=get_class_name_with_shortest_module(_cls),
                        **get_arguments(_cls, with_kwargs)
                    ),
                    params=['???'],
                    scheduler=scheduler
                )
            )

        if len(optimizers) > 1:
            print('The order of optimizer configuration must match the order of optimization.')
            print('Current order:')
            for i, optimizer in enumerate(optimizers):
                print(f'\t{i}: {optimizer.name}')

            print('Proceed? [Y/n]: ', end='')
            while True:
                if (proceed := input()) in ['', 'Y', 'n']:
                    break
                print('Invalid input!')
                print('Proceed? [Y/n]: ', end='')

            if proceed == 'n':
                indices = list(range(len(optimizers)))
                example_str = str(indices).lstrip("[").rstrip("]").replace(" ", "")
                print(f'Input order (example: {example_str}): ', end='')
                while True:
                    try:
                        order_str = f'[{input()}]'
                        order = eval(order_str)
                        if sorted(order) != indices:
                            raise ValueError
                        optimizers = [optimizers[o] for o in order]
                        break
                    except Exception:
                        print('Invalid input!')
                        print(f'Input order (example: {example_str}): ', end='')
                        pass

        modules: List[model.ModuleConfig] = []
        for name, _cls in annotations.items():
            if any(prefix in name for prefix in ['net', 'criterion', 'optimizer', 'scheduler']):
                continue
            modules.append(
                model.ModuleConfig(
                    name=name,
                    args=common.Instantiable(
                        _target_=get_class_name_with_shortest_module(_cls),
                        **get_arguments(_cls)
                    ),
                )
            )

        m = ModelConfig(
            _target_=cls._target_(),
            _recursive_=False,
            config=ModelConfig.ConfigBody(
                networks=networks,
                losses=losses,
                optimizers=optimizers,
                modules=modules if len(modules) > 0 else None,
            )
        ).dict(by_alias=True)

        camel2snake = r'(?<!^)(?=[A-Z])'
        cls_name_snake = re.sub(camel2snake, '_', cls.__name__.strip("Model")).lower()

        filename = Path(output_dir).joinpath(f'model/{cls_name_snake}.yaml')
        filename.parent.mkdir(parents=True, exist_ok=True)

        OmegaConf.save(OmegaConf.create(m), filename)
        print(f'The config has been generated! --> {filename}')
