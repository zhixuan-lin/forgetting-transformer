from . import ScheduleConfig
from omegaconf import MISSING
from dataclasses import dataclass


@dataclass
class WarmupOneMinusSqrtScheduleConfig(ScheduleConfig):
    _target_: str = 'forgetting_transformer.schedule.warmup_one_minus_sqrt_schedule'
    init_value: float = MISSING
    peak_value: float = MISSING
    warmup_steps: int = MISSING
    total_steps: int = MISSING
    anneal_steps: int = MISSING
    end_value: float = MISSING
