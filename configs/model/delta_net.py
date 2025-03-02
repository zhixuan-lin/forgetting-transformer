from dataclasses import dataclass, field
from omegaconf import OmegaConf, MISSING
from typing import List, Any, Optional

from . import ModelConfig


@dataclass
class DeltaNetArgConfig:
    _target_: str = "forgetting_transformer.model.delta_net.configuration_delta_net.DeltaNetConfig"
    vocab_size: int = MISSING
    hidden_size: int = MISSING
    expand_k: int = 1
    expand_v: int = 1
    use_gate: bool = False
    use_short_conv: bool = True
    conv_size: int = 4
    use_beta: bool = True
    use_output_norm: bool = True
    hidden_ratio: Optional[int] = 4
    intermediate_size: Optional[int] = None
    num_hidden_layers: int = MISSING
    num_heads: int = MISSING
    attn_mode: str = "chunk"
    qk_norm: str = 'l2'
    qk_activation: str = 'silu'
    hidden_act: str = "swish"
    max_position_embeddings: Optional[int] = None
    norm_first: bool = False
    norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02
    fuse_cross_entropy: bool = True




@dataclass
class DeltaNetConfig(ModelConfig):
    _target_: str = "forgetting_transformer.model.delta_net.modeling_delta_net.DeltaNetForCausalLM"
    config: DeltaNetArgConfig = DeltaNetArgConfig()
