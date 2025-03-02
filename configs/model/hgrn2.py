from dataclasses import dataclass, field
from omegaconf import OmegaConf, MISSING
from typing import List, Any, Optional

from . import ModelConfig


@dataclass
class HGRN2ArgConfig:
    _target_: str = "forgetting_transformer.model.hgrn2.configuration_hgrn2.HGRN2Config"
    vocab_size: int = MISSING
    hidden_size: int = MISSING
    num_hidden_layers: int = MISSING
    attn_mode: str = "chunk"
    num_heads: Optional[int] = None
    expand_ratio: Optional[int] = MISSING
    use_short_conv: bool = False
    conv_size: int = 4
    use_lower_bound: bool = True
    hidden_ratio: Optional[int] = 4
    intermediate_size: Optional[int] = None
    hidden_act: str = "swish"
    max_position_embeddings: Optional[int] = None
    elementwise_affine: Optional[bool] = True
    norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02
    fuse_cross_entropy: bool = True




@dataclass
class HGRN2Config(ModelConfig):
    _target_: str = "forgetting_transformer.model.hgrn2.modeling_hgrn2.HGRN2ForCausalLM"
    config: HGRN2ArgConfig = HGRN2ArgConfig()
