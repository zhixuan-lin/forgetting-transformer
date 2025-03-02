from . import ScheduleConfig
from omegaconf import MISSING
from dataclasses import dataclass


@dataclass
class WarmupLinearScheduleConfig(ScheduleConfig):
    _target_: str = 'forgetting_transformer.schedule.warmup_linear_decay_schedule'
    init_value: float = MISSING
    peak_value: float = MISSING
    warmup_steps: int = MISSING
    decay_steps: int = MISSING
    end_value: float = MISSING
