from . import ScheduleConfig
from omegaconf import MISSING
from dataclasses import dataclass


@dataclass
class ConstantScheduleConfig(ScheduleConfig):
    _target_: str = 'forgetting_transformer.schedule.constant_schedule'
    value: float = MISSING
