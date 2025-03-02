from dataclasses import dataclass, field
from omegaconf import OmegaConf, MISSING
from typing import List, Any, Optional

from . import ModelConfig


@dataclass
class ForgettingTransformerArgConfig:
    _target_: str = "forgetting_transformer.model.forgetting_transformer.configuration_forgetting_transformer.ForgettingTransformerConfig"
    vocab_size: int = MISSING  # Should be provided programmatically
    hidden_size: int = MISSING
    hidden_ratio: float = 4
    intermediate_size: Optional[int] = None
    num_hidden_layers: int = MISSING
    num_heads: int = MISSING
    num_kv_heads: Optional[int] = None
    hidden_act: str = "swish"
    window_size: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    initializer_range: float = 0.02
    elementwise_affine: Optional[bool] = True
    norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    fuse_norm: bool = True
    fuse_cross_entropy: bool = True
    rope_base: float = 500000
    use_rope: bool = False
    use_output_gate: bool = False
    ogate_act: str = "sigmoid"
    fgate_type: str = "full"
    fgate_bias_init: bool = False
    decay_time_min: Optional[float] = None
    decay_time_max: Optional[float] = None
    use_output_norm: bool = False
    qk_norm: bool = False

    use_k_shift: bool = False
    use_v_shift: bool = False




@dataclass
class ForgettingTransformerConfig(ModelConfig):
    _target_: str = "forgetting_transformer.model.forgetting_transformer.modeling_forgetting_transformer.ForgettingTransformerForCausalLM"
    config: ForgettingTransformerArgConfig = ForgettingTransformerArgConfig()
