# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

# from fla.layers.attn import Attention
from fla.modules import FusedCrossEntropyLoss, RMSNorm
from fla.modules.layernorm import group_norm_fn
from fla.modules.activations import swiglu_linear

from fla.modules import RotaryEmbedding
from einops import rearrange

from .configuration_forgetting_transformer import ForgettingTransformerConfig
from forgetting_transformer.ops.forgetting_attention import forgetting_attention
from .fgate_cache import FgateDynamicCache
from .glu_linear import glu_linear
from .token_shift import token_shift

from functools import partial

logger = logging.get_logger(__name__)


class ShiftLinear(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        bias: bool,
        shift_bias: bool = False
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        assert self.output_dim % self.num_heads == 0

        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.shift_proj = nn.Linear(input_dim, num_heads, bias=shift_bias)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.input_dim}, {self.output_dim})"
        return s

    def forward(self, x: torch.Tensor, shift_state: Optional[torch.Tensor]) -> torch.Tensor:
        assert x.ndim == 3, "Input must be (B, T, D)"
        B, T, D = x.size()
        out = self.linear(x)
        # (B, T, H, 1)
        alpha = torch.sigmoid(self.shift_proj(x).float()).float()
        # left, right, top, bottom (B, T=H, D=W)
        # out_prev = nn.functional.pad(out, (0, 0, 1, -1))
        # out_prev = torch.roll(out, shifts=1, dims=1)
        
        out_per_head = rearrange(out, 'b t (h d) -> b t h d', h=self.num_heads)
        if T > 1:
            # TODO: note in this case cache is not used
            result_per_head = token_shift(out_per_head, alpha, 1.0 - alpha)
        else:
            shift_state_per_head = rearrange(shift_state, 'b (h d) -> b 1 h d', h=self.num_heads)
            result_per_head = (alpha[..., None] * shift_state_per_head + (1 - alpha[..., None]) * out_per_head)

        result_per_head = result_per_head.to(out.dtype)

        if shift_state is not None:
            shift_state.copy_(out[:, -1, :])

        result = rearrange(result_per_head, 'b t h d -> b t (h d)', h=self.num_heads)
        return result

class GroupRMSNorm(nn.Module):
    def __init__(
        self,
        num_groups: int,
        hidden_size: int,
        elementwise_affine: bool = True,
        bias: bool = False,
        eps: float = 1e-5
    ) -> GroupRMSNorm:
        super().__init__()

        if hidden_size % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')

        self.num_groups = num_groups
        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        self.register_parameter("weight", None)
        self.register_parameter("bias", None)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
            if bias:
                self.bias = nn.Parameter(torch.zeros(hidden_size))

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.num_groups}, {self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        return group_norm_fn(
            x,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=True,
            num_groups=self.num_groups
        )

class ForgettingAttentionLayer(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        window_size: Optional[int] = None,
        max_position_embeddings: Optional[int] = None,
        use_rope: bool = False,
        rope_base: float = 500000.0,
        use_output_gate: bool = False,
        ogate_act: str = "sigmoid",
        fgate_type: str = "full",
        fgate_bias_init: bool = False,
        decay_time_min: Optional[float] = None,
        decay_time_max: Optional[float] = None,
        use_output_norm: bool = False,
        norm_eps: float = 1e-6,
        qk_norm: bool = False,
        qk_norm_share_param_across_head: bool = False,
        use_k_shift: bool = False,
        use_v_shift: bool = False,
        log_pruning_tolerance: Optional[float] = None,
        initializer_range: float = 0.02,
        layer_idx: int = None
    ):
        """
        Forgetting Attention layer.

        Arguments:
            - hidden_size: Input dimension and qkv dimension
            - num_heads: Number of heads
            - num_kv_heads: Not used. Should be None 
            - window_size: Not used. Should be None
            - max_position_embeddings: Not used. Should be None
            - use_rope: Whether to use RoPE. Default is False
            - rope_base: the theta hyperparameter in RoPE. This has no effect if
                  use_rope=False
            - use_output_gate: Whether to use output gates. Note that using output gates
                  introduces extra parameters and you may want to reduce parameters from
                  other components (e.g., MLPs)
            - ogate_act: Activation for the output gate. Either "sigmoid" or "silu"
            - fgate_type: Forget gate type. The following are supported:
                - "full": The default data-dependent forget gate
                - "bias_only": The data-independent forget gate
                - "fixed": Forget gates with fixed values
                - "none": Not using forget gates. Equivalent to forget gates with all
                  ones.
            - fgate_bias_init: Whether to use special initalization for the bias terms in 
                  the forget gate. This should only be used with fgate types in 
                  ["bias_only", "fixed"].
            - decay_time_min: T_min for the forget gate bias initialization. See paper
                  for details.
            - decay_time_max: T_max for the forget gate bias initalization. See paper
                  for details.
            - use_output_norm: Whether to use output normalization.
            - norm_eps: Epsilon for the RMSNorms
            - qk_norm: Whether to use qk_norm
            - qk_norm_share_param_across_head: In QK-norm, whether to share the RMSNorm
                scaling parameters across heads. This is just for backward compatibility.
            - use_k_shift: Whether to use data-dependent key shift
            - use_v_shift: Whether to use data-dependent value shift
            - log_pruning_tolerance: The natural logarithm of the pruning tolerance
                  hyperparameter epsilon. We recommend setting it to -10.0
            - initializer_range: standard deviation for initialization
            - layer_idx: The block index of this layer. Needed for KV-cache
        """
        super().__init__()

        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            raise NotImplementedError("GQA has not been tested.")
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.hidden_size = hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.window_size = window_size
        self.max_position_embeddings = max_position_embeddings
        self.log_pruning_tolerance = log_pruning_tolerance
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        if use_k_shift:
            self.k_proj = ShiftLinear(self.hidden_size, self.kv_dim, self.num_heads, bias=False)
        else:
            self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)

        if use_v_shift:
            self.v_proj = ShiftLinear(self.hidden_size, self.kv_dim, self.num_heads, bias=False)
        else:
            self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)

        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.use_k_shift = use_k_shift
        self.use_v_shift = use_v_shift


        device = next(self.parameters()).device
        # Forget gate
        assert fgate_type in ["full", "bias_only", "fixed", "none"]
        self.fgate_type = fgate_type
        self.fgate_bias_init = fgate_bias_init
        if fgate_type == "full":
            assert not fgate_bias_init
            self.fgate_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)
        elif fgate_type == "bias_only":
            self.fgate_bias = nn.Parameter(torch.zeros(size=(self.num_heads,), device=device))
            self.fgate_bias._no_weight_decay = True
        elif fgate_type == "fixed":
            assert fgate_bias_init, "You must set fgate_bias_init = True with fixed fgate"
            fgate_bias = torch.zeros(size=(self.num_heads,), device=device)
            self.register_buffer("fgate_bias", fgate_bias)
        elif fgate_type == "none":
            pass
        else:
            raise ValueError(f"Unknown fgate type {fgate_type}")

                

        # Forget gate intialization for data-independent and fixed forget gates
        if fgate_bias_init:
            assert decay_time_min is not None and decay_time_max is not None
            assert decay_time_min > 0 and decay_time_max > 0
            with torch.no_grad():
                log_decay_time = torch.linspace(math.log(decay_time_min), math.log(decay_time_max), steps=self.num_heads)
                decay_time = torch.exp(log_decay_time)
                # Such that t = -1 / log(sigmoid(b)) 
                bias_init = -torch.log(torch.expm1(1 / decay_time))
                self.fgate_bias.copy_(bias_init)
        else:
            assert decay_time_min is None and decay_time_max is None

        if use_output_gate:
            self.ogate_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.ogate_act = ogate_act
            assert ogate_act in ["silu", "sigmoid"]
        else:
            self.ogate_proj = None

        if use_output_norm:
            self.output_norm = GroupRMSNorm(num_groups=self.num_heads, hidden_size=self.hidden_size, eps=norm_eps)
        else:
            self.output_norm = None


        if use_rope:
            self.rotary = RotaryEmbedding(self.head_dim, base=rope_base)
        else:
            self.rotary = None


        self.qk_norm = qk_norm
        self.qk_norm_share_param_across_head = qk_norm_share_param_across_head
        if qk_norm:
            if self.qk_norm_share_param_across_head:
                # This is an incorrect implemention kept just for backward compatibility
                self.q_norm = RMSNorm(self.head_dim)
                self.k_norm = RMSNorm(self.head_dim)
            else:
                self.q_norm = GroupRMSNorm(num_groups=self.num_heads, hidden_size=self.hidden_size)
                self.k_norm = GroupRMSNorm(num_groups=self.num_heads, hidden_size=self.hidden_size)

        self.initializer_range = initializer_range
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        # This will actually be overwritten by outer init.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        We assume that during decoding attention mask is always 1. Otherwise it won't work.
        """
        batch_size, q_len, _ = hidden_states.size()
        if use_cache:
            key_shift_state = past_key_values.key_shift_cache[self.layer_idx]
            value_shift_state = past_key_values.value_shift_cache[self.layer_idx]
        else:
            key_shift_state = value_shift_state = None

        # Shift states are updated in place
        q = self.q_proj(hidden_states)
        if self.use_k_shift:
            k = self.k_proj(hidden_states, key_shift_state)
        else:
            k = self.k_proj(hidden_states)
        if self.use_v_shift:
            v = self.v_proj(hidden_states, value_shift_state)
        else:
            v = self.v_proj(hidden_states)

        if self.qk_norm and (not self.qk_norm_share_param_across_head):
            q = self.q_norm(q).to(q.dtype)
            k = self.k_norm(k).to(k.dtype)

        q = rearrange(q, '... (h d) -> ... h d', h=self.num_heads)
        k = rearrange(k, '... (h d) -> ... h d', h=self.num_kv_heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_kv_heads)


        if self.qk_norm and (self.qk_norm_share_param_across_head):
            q = self.q_norm(q).to(q.dtype)
            k = self.k_norm(k).to(k.dtype)


        seqlen_offset, max_seqlen = 0, q.shape[1]
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                # to deliminate the offsets of padding tokens
                seqlen_offset = (seqlen_offset + attention_mask.sum(-1) - attention_mask.shape[-1])
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        if self.rotary is not None:
            q, k = self.rotary(q, k, seqlen_offset, max_seqlen)

        if self.fgate_type == "full":
            fgate_logit = self.fgate_proj(hidden_states)
            fgate_logit = rearrange(fgate_logit, "b t h -> b h t")
            log_fgate = torch.nn.functional.logsigmoid(fgate_logit.float())
        elif self.fgate_type == "none":
            log_fgate = torch.zeros((batch_size, self.num_heads, q_len), dtype=torch.float32, device=hidden_states.device)
        else:
            assert self.fgate_type in ["fixed", "bias_only"]
            fgate_logit = torch.broadcast_to(self.fgate_bias, (batch_size, q_len, self.num_heads))
            fgate_logit = rearrange(fgate_logit, "b t h -> b h t")
            log_fgate = torch.nn.functional.logsigmoid(fgate_logit.float())

        k = rearrange(k, 'b t h d -> b h t d')
        if past_key_values is not None:
            k, v, log_fgate = past_key_values.update(k, v, log_fgate, self.layer_idx)
        # k, v = rearrange(k, 'b h t d -> b t h d'), rearrange(v, 'b h t d -> b t h d')
        q = rearrange(q, 'b t h d -> b h t d')

        if self.num_kv_groups > 1:
            assert False
            k = rearrange(k.unsqueeze(-2).repeat(1, 1, 1, self.num_kv_groups, 1), 'b t h g d -> b t (h g) d')
            v = rearrange(v.unsqueeze(-2).repeat(1, 1, 1, self.num_kv_groups, 1), 'b t h g d -> b t (h g) d')

        if self.log_pruning_tolerance is not None:
            # TODO: normally this should be per-head. Unfortunately we made a mistake
            # and share the RMSNorm parameters across different heads
            with torch.no_grad():
                B, _, T = log_fgate.size()
                if self.qk_norm:
                    if self.qk_norm_share_param_across_head:
                        assert self.q_norm.weight.size() == self.k_norm.weight.size() == (self.head_dim,)
                        logit_upper_bound = self.q_norm.weight.abs().max()  * self.k_norm.weight.abs().max() * math.sqrt(self.head_dim)
                    else:
                        assert self.q_norm.weight.size() == self.k_norm.weight.size() == (self.hidden_size,)
                        logit_upper_bound = self.q_norm.weight.view(self.num_heads, self.head_dim).abs().max(dim=-1).values  * self.k_norm.weight.view(self.num_heads, self.head_dim).abs().max(dim=-1).values * math.sqrt(self.head_dim)
                        assert logit_upper_bound.size() == (self.num_heads,)

                    adaptive_threshold = -(2 * logit_upper_bound + math.log(T)) + self.log_pruning_tolerance
                else:
                    max_q_norm = torch.linalg.vector_norm(q, dim=-1).max(dim=-1).values
                    max_k_norm = torch.linalg.vector_norm(k, dim=-1).max(dim=-1).values
                    assert max_q_norm.size() == max_k_norm.size() == (B, self.num_heads)
                    logit_upper_bound = max_q_norm * max_k_norm / math.sqrt(self.head_dim)
                    adaptive_threshold = -(2 * logit_upper_bound + math.log(T)) + self.log_pruning_tolerance
        else:
            adaptive_threshold = None
            

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            B, _, T = log_fgate.size()
            assert attention_mask.size() == (B, T), ((B, T), attention_mask.size())
            seq_start = T - attention_mask.sum(dim=-1)
            o = forgetting_attention(
                q, k, v,
                log_fgate,
                head_first=True,
                seq_start=seq_start,
                sm_scale=1 / math.sqrt(self.head_dim),
                adaptive_threshold=adaptive_threshold
            )
            o = rearrange(o, "b h t d -> b t h d")
        else:
            o = forgetting_attention(
                q, k, v,
                log_fgate,
                head_first=True,
                sm_scale=1 / math.sqrt(self.head_dim),
                adaptive_threshold=adaptive_threshold
            )
            o = rearrange(o, "b h t d -> b t h d")

        o = o.reshape(batch_size, q_len, self.hidden_size)

        if self.output_norm is not None:
            o = self.output_norm(o)

        if self.ogate_proj is not None:
            # ogate = self.ogate act(self.ogate_proj(hidden_states))
            # o = o * ogate
            # ogate = act_gate(self.ogate_proj(hidden_states), o)
            ogate_logit = self.ogate_proj(hidden_states)
            dtype = ogate_logit.dtype
            if self.ogate_act == "silu":
                o = swiglu_linear(ogate_logit, o, self.o_proj.weight.to(dtype), self.o_proj.bias.to(dtype) if self.o_proj.bias is not None else self.o_proj.bias)
            elif self.ogate_act == "sigmoid":
                o = glu_linear(ogate_logit, o, self.o_proj.weight.to(dtype), self.o_proj.bias.to(dtype) if self.o_proj.bias is not None else self.o_proj.bias)
            else:
                raise ValueError(f"Unknown ogate act {self.ogate_act}")
        else:
            o = self.o_proj(o)

        if not output_attentions:
            attentions = None
        else:
            SAVE_HEADS = [0, 1, 2, 3]
            # (B, H, T, T)
            score = q[:, SAVE_HEADS] @ k[:, SAVE_HEADS].mT
            log_lambda = torch.cumsum(log_fgate, dim=-1)
            decay_bias = (log_lambda[:, SAVE_HEADS, :, None] - log_lambda[:, SAVE_HEADS, None, :]).to(torch.bfloat16)
            # normalized_score = torch.softmax(score, dim=-1)
            attentions = (score, decay_bias)

        return o, attentions, past_key_values

    def init_shift_state(self, batch_size: int):
        param = next(self.parameters())
        state = dict()
        try:
            dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else torch.float32
        except TypeError:
            # Support legacy torch version
            dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else torch.float32
        if self.use_k_shift:
            state['key_shift'] = param.new_zeros(batch_size, self.kv_dim, dtype=dtype)
        else:
            state['key_shift'] = None
        if self.use_v_shift:
            state['value_shift'] = param.new_zeros(batch_size, self.kv_dim, dtype=dtype)
        else:
            state['value_shift'] = None
        return state


class ForgettingTransformerMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[float] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'swish'
    ) -> ForgettingTransformerMLP:
        super().__init__()

        self.hidden_size = hidden_size
        # the final number of params is `hidden_ratio * hidden_size^2`
        # `intermediate_size` is chosen to be a multiple of 256 closest to `2/3 * hidden_size * hidden_ratio`
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]
        self.hidden_act = hidden_act
        assert hidden_act in ["swish", "sigmoid"]

    def forward(self, x):
        y = self.gate_proj(x)
        gate, y = y.chunk(2, -1)
        # TODO: maybe wrap swiglu_linear in custom_fwd/custom_bwd
        if self.hidden_act == "swish":
            return swiglu_linear(
                gate, y,
                self.down_proj.weight.to(y.dtype),
                self.down_proj.bias.to(y.dtype) if self.down_proj.bias is not None else self.down_proj.bias
            )
        elif self.hidden_act == "sigmoid":
            return glu_linear(
                gate, y,
                self.down_proj.weight.to(y.dtype),
                self.down_proj.bias.to(y.dtype) if self.down_proj.bias is not None else self.down_proj.bias
            )
        else:
            raise ValueError()


class ForgettingTransformerBlock(nn.Module):
    def __init__(self, config: ForgettingTransformerConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.attn_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
        self.attn = ForgettingAttentionLayer(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            window_size=config.window_size,
            max_position_embeddings=config.max_position_embeddings,
            rope_base=config.rope_base,
            use_rope=config.use_rope,
            use_output_gate=config.use_output_gate,
            ogate_act=config.ogate_act,
            fgate_type=config.fgate_type,
            fgate_bias_init=config.fgate_bias_init,
            decay_time_min=config.decay_time_min,
            decay_time_max=config.decay_time_max,
            use_output_norm = config.use_output_norm,
            norm_eps=config.norm_eps,
            qk_norm=config.qk_norm,
            qk_norm_share_param_across_head=config.qk_norm_share_param_across_head,
            use_k_shift=config.use_k_shift,
            use_v_shift=config.use_v_shift,
            log_pruning_tolerance=config.log_pruning_tolerance,
            initializer_range=config.initializer_range,
            layer_idx=layer_idx
        )
        self.mlp_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
        self.mlp = ForgettingTransformerMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act
        )

    def forward_attn(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        # residual handled outside of this
        # residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions
        )
        return hidden_states, attentions, past_key_values

    def forward_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ):
        hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        gradient_checkpointing: bool = False
        # **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states


        if gradient_checkpointing:
            forward_attn = partial(torch.utils.checkpoint.checkpoint, self.forward_attn, use_reentrant=False)
            forward_mlp = partial(torch.utils.checkpoint.checkpoint, self.forward_mlp, use_reentrant=False)
        else:
            forward_attn = self.forward_attn
            forward_mlp = self.forward_mlp

        hidden_states, attentions, past_key_values = forward_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions
        )

        hidden_states = forward_mlp(
            hidden_states,
            residual,
        )

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attentions,)

        if use_cache:
            outputs += (past_key_values,)

        return outputs



class ForgettingTransformerPreTrainedModel(PreTrainedModel):

    config_class = ForgettingTransformerConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ['ForgettingTransformerBlock']

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
    ):
        # if isinstance(module, (nn.Linear, nn.Conv1d)):
        if isinstance(module, (nn.Linear)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class ForgettingTransformerModel(ForgettingTransformerPreTrainedModel):

    def __init__(self, config: ForgettingTransformerConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([ForgettingTransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # if output_attentions:
            # warnings.warn(
                # "`ForgettingTransformerModel` does not support output attention weights now, so `output_attentions` is set to `False`."
            # )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if use_cache:
            # use_legacy_cache = not isinstance(past_key_values, Cache)
            # if use_legacy_cache:
                # past_key_values = FgateDynamicCache.from_legacy_cache(past_key_values)
            if past_key_values is None:
                past_key_values = FgateDynamicCache()
                for layer_idx, layer in enumerate(self.layers):
                    shift_state = layer.attn.init_shift_state(
                        batch_size=input_ids.size(0),
                    )
                    past_key_values.update_shift_cache(
                        key_shift_state=shift_state["key_shift"],
                        value_shift_state=shift_state["value_shift"],
                        layer_idx=layer_idx
                    )
            else:
                assert isinstance(past_key_values, FgateDynamicCache)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_attns = {} if output_attentions else None
        next_decoder_cache = None

        for layer_id, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                gradient_checkpointing=self.gradient_checkpointing and self.training
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                OUTPUT_ATTN_LAYERS = [0, 7, 15, 23]
                if layer_id in OUTPUT_ATTN_LAYERS:
                    # all_attns += (layer_outputs[1],)
                    all_attns[layer_id] = layer_outputs[1]

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            # next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
            next_cache = next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attns
        )


class ForgettingTransformerForCausalLM(ForgettingTransformerPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = ForgettingTransformerModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # only last token for `inputs_ids` if the `past_key_values` is passed along.
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard.
            # Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {'input_ids': input_ids.contiguous()}

        model_inputs.update({
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
            'attention_mask': attention_mask,
        })
        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]

        loss = None
        if labels is not None:
            if self.config.fuse_cross_entropy:
                loss_fct = FusedCrossEntropyLoss(inplace_backward=True, reduction='none')
            else:
                loss_fct = nn.CrossEntropyLoss(reduction='none')
            logits = self.lm_head(hidden_states)
            # Enable model parallelism
            labels = labels.to(logits.device)
            # labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], loss_fct.ignore_index)), 1)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
            loss = loss.view(*labels.size())
            del logits
            logits = None
        else:
            logits = self.lm_head(hidden_states)

        if not return_dict:
            raise NotImplementedError
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
