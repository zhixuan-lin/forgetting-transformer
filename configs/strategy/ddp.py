from . import StrategyConfig
from omegaconf import MISSING
from dataclasses import dataclass


@dataclass
class DDPConfig(StrategyConfig):
    _target_: str = "lightning.fabric.strategies.DDPStrategy"
