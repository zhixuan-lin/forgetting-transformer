# -*- coding: utf-8 -*-

import math
from typing import Dict, Optional

from transformers.configuration_utils import PretrainedConfig
try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    DictConfig, OmegaConf = None, None


class SambaConfig(PretrainedConfig):

    model_type = "samba-project_fox"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 2304,
        state_size: int = 16,
        num_hidden_layers: int = 18,
        norm_eps=1e-5,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        expand: int = 2,
        conv_kernel: int = 4,
        use_bias: bool = False,
        use_conv_bias: bool = True,
        hidden_act: str = "silu",
        initializer_range: str = 0.02,
        residual_in_fp32: bool = False,
        time_step_rank: str = "auto",
        time_step_scale: float = 1.0,
        time_step_min: float = 0.001,
        time_step_max: float = 0.1,
        time_step_init_scheme: str = "random",
        time_step_floor: float = 1e-4,
        max_position_embeddings: int = 2048,
        attn: Optional[Dict] = None,
        attn_hidden_ratio: Optional[float] = 4,
        mamba_hidden_ratio: Optional[float] = 3,
        rescale_prenorm_residual: bool = False,
        use_cache: bool = True,
        fuse_norm: bool = True,
        fuse_cross_entropy: bool = True,
        tie_word_embeddings: bool = False,
        rope_base: float = 500000.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.num_hidden_layers = num_hidden_layers
        self.norm_eps = norm_eps
        self.conv_kernel = conv_kernel
        self.expand = expand
        self.intermediate_size = int(expand * self.hidden_size)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.use_bias = use_bias
        self.use_conv_bias = use_conv_bias
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.time_step_rank = math.ceil(self.hidden_size / 16) if time_step_rank == "auto" else time_step_rank
        self.time_step_scale = time_step_scale
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_init_scheme = time_step_init_scheme
        self.time_step_floor = time_step_floor
        self.max_position_embeddings = max_position_embeddings
        self.attn_hidden_ratio = attn_hidden_ratio
        self.mamba_hidden_ratio = mamba_hidden_ratio
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.residual_in_fp32 = residual_in_fp32
        self.use_cache = use_cache
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_norm = fuse_norm
        self.rope_base = rope_base

        if attn is not None:
            if isinstance(attn, (DictConfig)):
                attn = OmegaConf.to_container(attn)
            if not isinstance(attn, dict):
                raise ValueError("attn must be a dictionary")
            if 'layers' not in attn:
                raise ValueError("Layer indices must be provided to initialize hybrid attention layers")
            if 'num_heads' not in attn:
                raise ValueError("Number of heads must be provided to initialize hybrid attention layers")
            # attn['num_heads'] = attn.get('num_kv_heads', 18)
            # attn['num_kv_heads'] = attn.get('num_kv_heads', attn['num_heads'])
            # attn['window_size'] = attn.get('window_size', 2048)
        # else:
            # raise ValueError("attn must not be None")
        self.attn = attn

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
