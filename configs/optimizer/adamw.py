from dataclasses import dataclass, field
from omegaconf import OmegaConf, MISSING
from typing import List, Any

from . import OptimizerConfig


@dataclass
class AdamWConfig(OptimizerConfig):
    _target_: str = "torch.optim.AdamW"
    lr: float = MISSING
    betas: List[float] = MISSING
    weight_decay: float = MISSING
