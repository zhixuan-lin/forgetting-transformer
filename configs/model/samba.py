from dataclasses import dataclass, field
from omegaconf import OmegaConf, MISSING
from typing import List, Any, Optional, Dict

from . import ModelConfig

@dataclass
class SambaAttnConfig:
    num_kv_heads: Optional[int] = None
    num_heads: int = MISSING
    window_size: Optional[int] = MISSING
    layers: Optional[List[int]] = MISSING

@dataclass
class SambaArgConfig:
    _target_: str = "forgetting_transformer.model.samba.configuration_samba.SambaConfig"
    vocab_size: int = MISSING
    hidden_size: int = MISSING
    state_size: int = 16
    num_hidden_layers: int = MISSING
    norm_eps=1e-5
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    expand: int = 2
    conv_kernel: int = 4
    use_bias: bool = False
    use_conv_bias: bool = True
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    residual_in_fp32: bool = False
    time_step_rank: str = "auto"
    time_step_scale: float = 1.0
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_init_scheme: str = "random"
    time_step_floor: float = 1e-4
    max_position_embeddings: Optional[int] = None
    attn: SambaAttnConfig = SambaAttnConfig()
    attn_hidden_ratio: Optional[float] = 4
    mamba_hidden_ratio: Optional[float] = 3
    use_cache: bool = True
    fuse_norm: bool = True
    fuse_cross_entropy: bool = True
    tie_word_embeddings: bool = False
    rope_base: float = MISSING
    rescale_prenorm_residual: bool = True  # To be consistent with other impl




@dataclass
class SambaConfig(ModelConfig):
    _target_: str = "forgetting_transformer.model.samba.modeling_samba.SambaForCausalLM"
    config: SambaArgConfig = SambaArgConfig()
