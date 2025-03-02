from dataclasses import dataclass, field
from omegaconf import OmegaConf, MISSING
from typing import List, Any, Literal, Optional, Union
from pathlib import Path
from hydra.core.config_store import ConfigStore
from configs.optimizer import OptimizerConfig
from configs.schedule import ScheduleConfig
from configs.model import ModelConfig
from configs.datamodule import DataModuleConfig
from configs.utils import auto_register
from configs.strategy import StrategyConfig

@dataclass
class WandbConfig:
    project: str = "forgetting-transformer"
    mode: str = "offline"
    log_dir: str = MISSING

@dataclass
class FabricConfig:
    devices: Union[int, str] = "auto"
    precision: str = 'bf16-mixed'


@dataclass
class TrainConfig:
    max_tokens: int = MISSING
    grad_acc_tokens: int = MISSING
    max_grad_norm: float = MISSING
    gradient_checkpointing: bool = False

    bias_weight_decay: bool = False
    normalization_weight_decay: bool = False
    conv_weight_decay: bool = True

@dataclass
class EvalConfig:
    min_val_length: int = 512


@dataclass
class Config:
    defaults: List[Any] = field(
        default_factory=lambda: [
            {"model": "???"},
            {"optimizer": "???"},
            {"schedule": "???"},
            {"datamodule": "???"},
            {"strategy": "???"},
            # If we don't do these hydra will mess up python logging
            # Also must none. `disabled` mess up other libraries.
            {"override hydra/job_logging": "none"},
            {"override hydra/hydra_logging": "none"},
            "_self_",
        ]
    )

    # https://github.com/facebookresearch/hydra/issues/2049
    # If we don't do this hydra will create an annoying directory
    hydra: Any = field(default_factory=lambda: {"run": {"dir": "${output_dir}"}})

    exp: str = "debug"
    tag: str = "debug"
    seed: int = 0

    # Only used for saving HF model
    hf_load_dir: Optional[str] = None
    hf_save_dir: Optional[str] = None
    hf_load_step: Optional[int] = None

    # Everything (config, metrics, checkpoints etc) except for wandb log will be saved here
    output_dir: str = MISSING
    # Any dataset should reside here
    data_dir: str = MISSING
    # Don't forget to set wandb.log_dir as well

    # When resuming, we first try to load the latest checkpoint from output_dir / 'checkpoints'. If nothing
    # found, we try to start from fork_step from fork_dir if it is not None.
    resume: bool = MISSING
    fork_dir: Optional[str] = None
    fork_step: Optional[int] = None

    log_interval: int = MISSING
    eval_interval: int = MISSING
    final_eval: bool = True
    skip_eval: bool = True
    # Save checkpoints every this steps. We only keep the latest checkpoint
    checkpoint_interval: int = MISSING  
    # Eval results with training loss
    train_eval_interval: int = MISSING
    # Besides the latest checkpoint, also keeps permanent checkpoints at these
    # interval
    checkpoint_keep_interval: int = MISSING

    # Regular hierarhical config
    fabric: FabricConfig = FabricConfig()
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()
    wandb: WandbConfig = WandbConfig()

    # Meant to decided by default list
    strategy: StrategyConfig = MISSING
    model: ModelConfig = MISSING
    schedule: ScheduleConfig = MISSING
    datamodule: DataModuleConfig = MISSING
    optimizer: OptimizerConfig = MISSING

cs = ConfigStore.instance()
cs.store(name='config', node=Config)
config_root = Path(__file__).parent
for base_class in [
    OptimizerConfig,
    ModelConfig,
    DataModuleConfig,
    ScheduleConfig,
    StrategyConfig,
]:
    auto_register(base_class, config_root)
