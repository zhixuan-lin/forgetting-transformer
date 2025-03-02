# -*- coding: utf-8 -*-

from typing import Optional

from transformers.configuration_utils import PretrainedConfig


class ForgettingTransformerConfig(PretrainedConfig):

    model_type = 'forgetting_transformer'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        hidden_ratio: Optional[float] = 4,
        intermediate_size: Optional[int] = None,
        num_hidden_layers: int = 24,
        num_heads: int = 32,
        num_kv_heads: int = None,
        hidden_act: str = "swish",
        window_size: Optional[int] = None,
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        attention_bias: bool = False,
        fuse_norm: bool = True,
        fuse_cross_entropy: bool = True,
        rope_base: float = 500000.0,
        use_rope: bool = False,
        use_output_gate: bool = False,
        ogate_act: str = "sigmoid",
        fgate_type: str = "full",
        fgate_bias_init: bool = False,
        decay_time_min: Optional[float] = None,
        decay_time_max: Optional[float] = None,
        use_output_norm: bool = False,
        qk_norm: bool = False,
        use_k_shift: bool = False,
        use_v_shift: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.window_size = window_size
        self.max_position_embeddings = max_position_embeddings

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_norm = fuse_norm
        self.rope_base = rope_base
        self.use_rope = use_rope
        self.use_output_gate = use_output_gate
        self.ogate_act = ogate_act
        self.fgate_type = fgate_type
        self.fgate_bias_init = fgate_bias_init
        self.decay_time_min = decay_time_min
        self.decay_time_max = decay_time_max
        self.use_output_norm = use_output_norm
        self.qk_norm = qk_norm
        self.use_k_shift = use_k_shift
        self.use_v_shift = use_v_shift

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
