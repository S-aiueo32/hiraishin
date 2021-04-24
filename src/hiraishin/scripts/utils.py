from omegaconf import DictConfig, OmegaConf


def resolve(config: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(config, resolve=True))
