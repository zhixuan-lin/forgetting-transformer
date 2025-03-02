from dataclasses import dataclass, field
from omegaconf import OmegaConf, MISSING
from typing import List, Any, Optional

from . import ModelConfig


@dataclass
class Mamba2ArgConfig:
    _target_: str = "forgetting_transformer.model.mamba2.configuration_mamba2.Mamba2Config"
    num_heads: int = MISSING
    head_dim: int = MISSING
    vocab_size: int = MISSING
    hidden_size: int = MISSING
    state_size: int = MISSING
    num_hidden_layers: int = MISSING
    layer_norm_epsilon: float = 1e-5
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    expand: int = 2
    conv_kernel: int = 4
    n_groups: int = 1
    use_bias: bool = False
    use_conv_bias: bool = True
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    residual_in_fp32: bool = True
    time_step_rank: str = "auto"
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 1e-4
    time_step_limit=(0.0, float("inf"))
    rescale_prenorm_residual: bool = True
    use_cache: bool = True
    rms_norm: bool = True
    chunk_size: int = 256
    fuse_cross_entropy: bool = True
    tie_word_embeddings: bool = False




@dataclass
class Mamba2Config(ModelConfig):
    _target_: str = "forgetting_transformer.model.mamba2.modeling_mamba2.Mamba2ForCausalLM"
    config: Mamba2ArgConfig = Mamba2ArgConfig()
