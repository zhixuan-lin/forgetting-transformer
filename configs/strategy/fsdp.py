from . import StrategyConfig
from omegaconf import MISSING
from dataclasses import dataclass


@dataclass
class FSDPConfig(StrategyConfig):
    _target_: str = "lightning.fabric.strategies.FSDPStrategy"
    state_dict_type: str = "full"  # We don't want any trouble later
    sharding_strategy: str = "FULL_SHARD"  # We don't want any trouble later
    cpu_offload: bool = False
