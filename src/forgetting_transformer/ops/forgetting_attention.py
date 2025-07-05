"""
Implementation of Forgetting Attention.

Our code is adapted from https://github.com/FlagOpen/FlagAttention/blob/ee91638dec6da8c00c4113d179f469e0ffcd5852/src/flag_attn/flash.py. The code is modified to implement Forgetting Attention.

The original license info from FlagAttention:

Copyright 2023 BAAI

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import pytest
import math
import torch
import triton
import triton.language as tl
from einops import rearrange
from typing import Optional


__all__ = ["forgetting_attention"]


# File flash.py
def maybe_contiguous(x):
    # only when the inner most dimension is contiguous can LDGSTS be used
    # so inner-dimension contiguity is enforced.
    return x.contiguous() if x.stride(-1) != 1 else x

def rounded_multiple(a, b):
    return (a + b - 1) // b * b

# --------------------------- public API ---------------------------
class ForgettingAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, log_fgate, seq_start, causal, sm_scale, return_log_normalizer):
        assert causal, "Only causal attention is supported"
        Dq, Dk, Dv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Dq == Dk == Dv, "feature size of q, k, v should be equal"
        assert Dk in {16, 32, 64, 128}, "We only support head dims in {16, 32, 64, 128}"

        B, H, M, D = q.shape
        if seq_start is not None:
            has_seq_start = True
            assert seq_start.shape == (B,)
        else:
            has_seq_start = False
            seq_start = torch.zeros((B,), device=q.device, dtype=torch.long)
        N = k.shape[2]
        assert log_fgate.shape == (B, H, N)
        log_fgate = log_fgate.float()
        if has_seq_start:
            log_fgate = log_fgate.clone()
            # We absolutely don't want masked value to affect result. If we
            # don't do this then it could via affecting numerical precision of
            # cumsum
            mask_index = (torch.arange(N, device=q.device)[None, None, :] < seq_start[:, None, None])
            mask_index = torch.broadcast_to(mask_index, log_fgate.size())
            log_fgate[mask_index] = 0.0

        log_lambda = torch.cumsum(log_fgate, dim=-1, dtype=log_fgate.dtype).float()

        Hk, Hv = k.shape[1], v.shape[1]
        assert Hk == Hv, "num of heads in k and v should be equal"
        assert H == Hk, "groupped query attention has not been tested. You can uncomment this if you know what you are doing."
        assert H % Hk == 0, "number of heads in q must be a multiple of that in k & v"
        num_groups = H // Hk

        P_SEQ = N - M
        larger_m = M > N
        assert (not larger_m), "The key/value tensors must be longer than the query tensor"

        if sm_scale is None:
            sm_scale = 1. / math.sqrt(D)

        # contiguity
        q, k, v = maybe_contiguous(q), maybe_contiguous(k), maybe_contiguous(v)

        # to work around https://github.com/openai/triton/issues/2441
        device = torch.cuda.device_of(q)

        with torch.cuda.device(device):

            config = get_fwd_config(B, H, M, N, D, causal)
            BLOCK_M, BLOCK_N, num_stages, num_warps = config

            divisible_m = M % BLOCK_M == 0
            divisible_n = N % BLOCK_N == 0
            # consider using 3d grid to avoid div & rem
            grid = (triton.cdiv(M, BLOCK_M), H, B)
            o = torch.empty_like(q)
            L = torch.empty((B, H, M), device=q.device, dtype=torch.float32)
            _fwd_kernel[grid](
                q, k, v, log_lambda, seq_start, sm_scale,
                L, o,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                log_lambda.stride(0), log_lambda.stride(1), log_lambda.stride(2),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                B, H, M, N, P_SEQ, num_groups,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D,
                IS_CAUSAL=causal, LARGER_M=larger_m, HAS_SEQ_START=has_seq_start,
                DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n,
                num_warps=num_warps, num_stages=num_stages,
            )

        # autograd context maintenance
        ctx.save_for_backward(q, k, v, o, L, log_lambda, seq_start)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.has_seq_start = has_seq_start

        has_extra_return = return_log_normalizer
        if has_extra_return:
            outs = (
                o,
                L if return_log_normalizer else None,
            )
            return outs
        return o

    @staticmethod
    def backward(ctx, do, *ignored):
        q, k, v, o, L, log_lambda, seq_start = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        causal = ctx.causal
        has_seq_start = ctx.has_seq_start

        B, H, M, D = q.shape
        N = k.shape[2]
        Hk = k.shape[1]
        num_groups = H // Hk
        P_SEQ = N - M
        larger_m = M > N

        if sm_scale is None:
            sm_scale = 1. / math.sqrt(D)

        # to work around https://github.com/openai/triton/issues/2441
        device = torch.cuda.device_of(q)
        with torch.cuda.device(device):
            config = get_bwd_config(B, H, M, N, D, causal)
            BLOCK_M, BLOCK_N, num_stages, num_warps = config

            divisible_m = M % BLOCK_M == 0
            divisible_n = N % BLOCK_N == 0

            delta = torch.empty_like(L)
            grid = (triton.cdiv(M, BLOCK_M), H, B)
            _bwd_preprocess[grid](
                o, do,
                delta,
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                delta.stride(0), delta.stride(1), delta.stride(2),
                M,
                BLOCK_M=BLOCK_M, D_HEAD=D,
                DIVISIBLE_M=divisible_m,
            )

            # NOTE that dk & dv always have the same number of heads as q, instead of q.
            BLOCK_M, BLOCK_N, num_stages, num_warps = get_bwd_kv_config(B, H, M, N, D, causal)
            divisible_m = M % BLOCK_M == 0
            divisible_n = N % BLOCK_N == 0

            dk = torch.empty((B, H, N, D), dtype=k.dtype, device=q.device)
            dv = torch.empty((B, H, N, D), dtype=v.dtype, device=q.device)
            dlog_lambda = torch.empty((B, H, N), dtype=log_lambda.dtype, device=q.device)
            grid = (triton.cdiv(N, BLOCK_N), H, B)
            _bwd_kv_kernel[grid](
                q, k, v, log_lambda, seq_start, sm_scale, do,
                dk, dv, dlog_lambda,
                L, delta,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                log_lambda.stride(0), log_lambda.stride(1), log_lambda.stride(2),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                dlog_lambda.stride(0), dlog_lambda.stride(1), dlog_lambda.stride(2),
                B, H, M, N, P_SEQ,
                num_groups,
                BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N=BLOCK_N, CAUSAL=causal,
                DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n, HAS_SEQ_START=has_seq_start,
                num_stages=num_stages, num_warps=num_warps,
            )

            BLOCK_M, BLOCK_N, num_stages, num_warps = get_bwd_q_config(B, H, M, N, D, causal)
            divisible_m = M % BLOCK_M == 0
            divisible_n = N % BLOCK_N == 0
            dq = torch.zeros_like(q)
            grid = (triton.cdiv(M, BLOCK_M), H, B)
            _bwd_q_kernel[grid](
                q, k, v, log_lambda, seq_start, sm_scale, do,
                dq, dlog_lambda,
                L, delta,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                log_lambda.stride(0), log_lambda.stride(1), log_lambda.stride(2),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                dlog_lambda.stride(0), dlog_lambda.stride(1), dlog_lambda.stride(2),
                B, H, M, N, P_SEQ,
                num_groups,
                BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N=BLOCK_N,
                CAUSAL=causal, LARGER_M=larger_m, HAS_SEQ_START=has_seq_start,
                DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n,
                num_stages=num_stages, num_warps = num_warps,
            )
            dk = dk.reshape((B, Hk, num_groups, N, D)).sum(2)
            dv = dv.reshape((B, Hk, num_groups, N, D)).sum(2)
        dcumsum = torch.cumsum(dlog_lambda, dim=-1, dtype=log_lambda.dtype)
        dlog_fgate = dlog_lambda + dcumsum[..., -1:] - dcumsum
        dlog_fgate = dlog_fgate.float()
        return dq, dk, dv, dlog_fgate, None, None, None, None, None, None, None


def forgetting_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_fgate: torch.Tensor,
    *,
    head_first: bool = False,
    seq_start: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
):
    """
    A FlashAttention-based implementation of Forgetting Attention. 

    Note:
    - We recommand bfloat16/float16 for q, k, v and float32 for log_fgate. float32 for 
      q, k, v is also supported, but the kernel will not use tensor cores if q, k, v are
      in float32 (which would be slow).
    - We only support seqlen_q <= seqlen_k
    - We only support causal attention
    - Head dimension must be in one of {16, 32, 64, 128}

    Arguments:
        - q: (batch_size, seqlen_q, num_heads, head_dim) unless head_first=True.
        - k: (batch_size, seqlen_k, num_heads, head_dim) unless head_first=True.
        - v: (batch_size, seqlen_k, num_heads, head_dim) unless head_first=True.
        - log_fgate: (batch_size, seqlen_k, num_heads) unless head_first=True. 
              This should be the **log** of the forget gates. This is typically the 
              output of torch.nn.functional.logsigmoid.
        - head_first: if True, the order the num_heads and seqlen_* axis of the all 
              FloatTensor inputs and outputs should be (num_heads, seq_len_*) instead of
              (seq_len_*, num_heads)
        - seq_start: If not None, should be LongTensor with shape (batch_size,) 
              and range in [0, seq_len_k). For each batch index batch_id, no attention 
              will be allocated to tokens before the token index seq_start[batch_id]. 
              This is useful for left-padded inputs.
        - sm_scale: The scaling of attention scores before applying softmax. If
              None, it defaults to (1.0 / math.sqrt(head_dim))

    Returns:
        out (torch.Tensor): (batch_size, seqlen_q, num_heads, head_dim) unless head_first=True.
    """
    if not head_first:
        q, k, v = [rearrange(item, "b t h d -> b h t d") for item in (q, k, v)]
        log_fgate = rearrange(log_fgate, "b t h -> b h t")
    out = ForgettingAttention.apply(q, k, v, log_fgate, seq_start, True, sm_scale, False)
    if not head_first:
        out = rearrange(out, "b h t d -> b t h d")
    return out


# --------------------------- Forward ---------------------------
# NOTE: this function can be overwritten at runtime to use your custom config
def get_fwd_config(B, H, M, N, D, causal):
    assert causal
    if torch.cuda.get_device_capability() == (8, 0):
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 32, 3, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 4, 4
    elif torch.cuda.get_device_capability() == (9, 0):
        # H100
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 8
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 2, 8
    elif torch.cuda.get_device_capability() == (8, 6):
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
        else: # causal
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
    elif torch.cuda.get_device_capability() == (8, 9):
        # L40S
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 2, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
    else:
        BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
    return (BLOCK_M, BLOCK_N, num_stages, num_warps)


@triton.jit
def _fwd_kernel(
    Q, K, V, LOG_LAMBDA, SEQ_START, sm_scale,
    L, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_log_lambda_z, stride_log_lambda_h, stride_log_lambda_n,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, M, N, P_SEQ,
    num_groups,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr, LARGER_M: tl.constexpr, HAS_SEQ_START: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    # -- grid id --
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    log2e: tl.constexpr = 1.4426950408889634
    loge2: tl.constexpr = 0.6931471805599453
    qk_scale = sm_scale * log2e

    # offset pointers for (batch, head)
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    LOG_LAMBDA += off_z * stride_log_lambda_z + off_h * stride_log_lambda_h
    O += off_z * stride_oz + off_h * stride_oh
    L += (off_z * H + off_h) * M # l's shape is (B, H, M)

    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + offs_m_base
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)


    # initialize pointers to value-like data
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk) # (BLOCK_M, BLOCK_DMODEL)
    log_lambda_out_ptrs = LOG_LAMBDA + (P_SEQ + offs_m) * stride_log_lambda_n
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok) # (BLOCK_M, BLOCK_DMODEL)
    l_ptrs = L + offs_m

    # initialize pointer to m and l, fp32 for accumulators
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # load q
    if DIVISIBLE_M:
        q = tl.load(q_ptrs, cache_modifier=".cg")
        log_lambda_out = tl.load(log_lambda_out_ptrs, cache_modifier=".cg")
    else:
        mask_m = offs_m < M
        q = tl.load(q_ptrs, mask=mask_m[:, None], cache_modifier=".cg")
        log_lambda_out = tl.load(log_lambda_out_ptrs, mask=mask_m, cache_modifier=".cg")

    #Dot I trick: to place q in registers, it saves shared memory
    # if BLOCK_DMODEL < 128:
    #     I = tl.where(offs_k[:, None] == offs_k,
    #                  tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 1.0, dtype=input_dtype),
    #                  tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 0.0, dtype=input_dtype))
    #     q = tl.dot(q, I, input_precision="ieee").to(input_dtype)
    # else:
    #     I = tl.where(offs_m_base[:, None] == offs_m_base,
    #                  tl.full((BLOCK_M, BLOCK_M), 1.0, dtype=input_dtype),
    #                  tl.full((BLOCK_M, BLOCK_M), 0.0, dtype=input_dtype))
    #     q = tl.dot(I, q, input_precision="ieee").to(input_dtype)

    # NOTE: Loop-Bound-For-N
    # The indices in m-dimension that this block may access is in `[start_m * BLOCK_M, (start_m + 1) * BLOCK_M)`.
    # According to the rule of causal masking, then max index in n-dimension that this block may access
    # is `P_SEQ + (start_m + 1) * BLOCK_M`.
    # However, the upper bound of index in n-dimension should never exceed the sequence length of k/v(`P_SEQ + N_CTX`).
    # `P_SEQ + (start_m + 1) * BLOCK_M` may be larger than `N`.
    # At this case, there would be illegal memory access when loading k & v tiles
    # if mask_n is not applied for loading(only when `DIVISIBLE_N`` is true).
    # See also https://github.com/FlagOpen/FlagAttention/pull/8
    if IS_CAUSAL:
        hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
        if LARGER_M:
            hi = tl.maximum(0, hi)
    else:
        hi = N

    offs_n_init = offs_n_base
    if HAS_SEQ_START:
        SEQ_START += off_z
        seq_start = tl.load(SEQ_START)
        lo = tl.minimum(seq_start, hi)
        lo = (lo // BLOCK_N) * BLOCK_N
        offs_n_init += lo
    else:
        lo = 0
        seq_start = 0

    # loop over k, v and update accumulators
    k_ptrs = K + (offs_k[:, None] * stride_kk + offs_n_init[None, :] * stride_kn) # (BLOCK_DMODEL, BLOCK_N)
    v_ptrs = V + (offs_n_init[:, None] * stride_vn + offs_k[None, :] * stride_vk) # (BLOCK_N, BLOCK_DMODEL)
    log_lambda_in_ptrs = LOG_LAMBDA + (offs_n_init * stride_log_lambda_n) # (BLOCK_N, BLOCK_DMODEL)
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base

        # -- load k, v --
        if DIVISIBLE_N:
            k = tl.load(k_ptrs, cache_modifier=".cg")
            v = tl.load(v_ptrs, cache_modifier=".cg")
            log_lambda_in = tl.load(log_lambda_in_ptrs, cache_modifier=".cg")
        else:
            mask_n = offs_n < N
            k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")
            v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier=".cg")
            log_lambda_in = tl.load(log_lambda_in_ptrs, mask=mask_n, cache_modifier=".cg")

        # -- compute qk ---
        # s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s = tl.dot(q, k, input_precision="ieee") * qk_scale
        decay_bias = log_lambda_out[:, None] - log_lambda_in[None, :]
        s += decay_bias * log2e

        if not DIVISIBLE_N:
            s = tl.where(mask_n[None, :], s, float("-inf"))
        if IS_CAUSAL:
            causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))
        if HAS_SEQ_START:
            s = tl.where(offs_n[None, :] >= seq_start, s, float("-inf"))


        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2((m_i - m_i_new))
        p = tl.math.exp2(s - m_i_new[:, None])

        # -- compute partial sumexpn before applying dropout
        p_sum = tl.sum(p, 1)


        # -- scale and update acc: acc *= alpha[:, None]--
        acc *= alpha[:, None]
        acc += tl.dot(p.to(input_dtype), v, input_precision="ieee")

        # -- update m_i and l_i --
        l_i = l_i * alpha + p_sum
        m_i = m_i_new
        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        log_lambda_in_ptrs += BLOCK_N * stride_log_lambda_n

    # write back l & o
    if IS_CAUSAL and (LARGER_M or HAS_SEQ_START):
        is_empty_line = (offs_m + P_SEQ) < seq_start
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
        l = tl.where(is_empty_line, float("-inf"), m_i * loge2 + tl.log(l_i))
    else:
        acc = acc * (1.0 / l_i[:, None])
        l = m_i * loge2 + tl.log(l_i) # log(normalizer)


    if DIVISIBLE_M:
        tl.store(l_ptrs, l, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(input_dtype), cache_modifier=".cg")
    else:
        tl.store(l_ptrs, l, mask=mask_m, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(input_dtype), mask=mask_m[:, None], cache_modifier=".cg")


# --------------------------- Backward ---------------------------
# NOTE: this function can be overwritten at runtime to use your custom config
def get_bwd_config(B, H, M, N, D, causal):
    if torch.cuda.get_device_capability() == (9, 0):
        if not causal:
            BLOCK_M = 128 if D <= 64 else 64
            BLOCK_N = 64
            num_stages = 2
            num_warps = 4
        else:
            BLOCK_M = 64
            BLOCK_N = 64
            num_stages = 3 if D <= 64 else 2
            num_warps = 4
    elif torch.cuda.get_device_capability() == (8, 0):
        if not causal:
            BLOCK_M = 128 if D <= 64 else 64
            BLOCK_N = 64
            num_stages = 2
            num_warps = 4
        else:
            BLOCK_M = 64
            BLOCK_N = 64
            num_stages = 3 if D <= 64 else 2
            num_warps = 4
    elif torch.cuda.get_device_capability() == (8, 6): # tune for RTX-3090, device_capability(8, 6)
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 8
        else:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 2, 4
    else:
        BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 1, 4
    return (BLOCK_M, BLOCK_N, num_stages, num_warps)

def get_bwd_kv_config(B, H, M, N, D, causal):
    assert causal
    if torch.cuda.get_device_capability() == (8, 0): # A100
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 4, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 128, 4, 8
    elif torch.cuda.get_device_capability() == (8, 6): # tune for RTX-3090, device_capability(8, 6)
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 2, 4
    elif torch.cuda.get_device_capability() == (8, 9): # L40S
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 128, 4, 8
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 128, 2, 8
    elif torch.cuda.get_device_capability() == (9, 0): # H100
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
    else:
        BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
    return (BLOCK_M, BLOCK_N, num_stages, num_warps)

def get_bwd_q_config(B, H, M, N, D, causal):
    assert causal
    if torch.cuda.get_device_capability() == (8, 0): # A100
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 4, 8
    elif torch.cuda.get_device_capability() == (8, 6): # tune for RTX-3090, device_capability(8, 6)
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 2, 4
    elif torch.cuda.get_device_capability() == (8, 9): # L40S
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 4, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 3, 4
    elif torch.cuda.get_device_capability() == (9, 0): # H100
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 4, 8
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 2, 8
    else:
        BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
    return (BLOCK_M, BLOCK_N, num_stages, num_warps)


@triton.jit
def _bwd_preprocess(
    Out, DO,
    Delta,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dz, stride_dh, stride_dm,
    M,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
):
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    Out += off_z * stride_oz + off_h * stride_oh
    DO += off_z * stride_doz + off_h * stride_doh
    Delta += off_z * stride_dz + off_h * stride_dh

    # compute (Out * Dout).sum() for vector interpretation
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)

    # load
    o_ptrs = Out + off_m[:, None] * stride_om + off_n[None, :] * stride_ok
    do_ptrs = DO + off_m[:, None] * stride_dom + off_n[None, :] * stride_dok

    if DIVISIBLE_M:
        o = tl.load(o_ptrs).to(tl.float32)
        do = tl.load(do_ptrs).to(tl.float32)
    else:
        mask_m = off_m < M
        o = tl.load(o_ptrs, mask=mask_m[:, None]).to(tl.float32)
        do = tl.load(do_ptrs, mask=mask_m[:, None]).to(tl.float32)

    # compute
    delta = tl.sum(o * do, axis=1)

    # write-back
    d_ptrs = Delta + off_m * stride_dm
    if DIVISIBLE_M:
        tl.store(d_ptrs, delta)
    else:
        tl.store(d_ptrs, delta, mask=mask_m)


@triton.jit
def _bwd_kv_kernel(
    Q, K, V, LOG_LAMBDA, SEQ_START, sm_scale, DO,
    DK, DV, DLOG_LAMBDA,
    L,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_log_lambda_z, stride_log_lambda_h, stride_log_lambda_n,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    stride_dlog_lambda_z, stride_dlog_lambda_h, stride_dlog_lambda_n,
    Z, H, M, N, P_SEQ,
    num_groups,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr, HAS_SEQ_START: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    # -- grid id --
    start_n = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    # offset pointers for (batch, head)
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    LOG_LAMBDA += off_z * stride_log_lambda_z + off_h * stride_log_lambda_h
    DO += off_z * stride_doz + off_h * stride_doh

    # offset pointers for batch/head
    DK += off_z * stride_dkz + off_h * stride_dkh
    DV += off_z * stride_dvz + off_h * stride_dvh
    DLOG_LAMBDA += off_z * stride_dlog_lambda_z + off_h * stride_dlog_lambda_h

    # offset pointers for batch/head
    D += (off_z * H + off_h) * M
    L += (off_z * H + off_h) * M

    if CAUSAL:
        lo = tl.maximum(start_n * BLOCK_N - P_SEQ, 0)
        lo = (lo // BLOCK_M) * BLOCK_M
    else:
        lo = 0

    offs_m_init = lo + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    # initialize pointers to value-like data
    q_ptrs = Q + (offs_m_init[:, None] * stride_qm + offs_k[None, :] * stride_qk) # (BLOCK_M, BLOCK_DMODEL)
    log_lambda_out_ptrs = LOG_LAMBDA + (P_SEQ + offs_m_init) * stride_log_lambda_n # (BLOCK_N, BLOCK_DMODEL)
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk) # (BLOCK_N, BLOCK_DMODEL)
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk) # (BLOCK_N, BLOCK_DMODEL)
    log_lambda_in_ptrs = LOG_LAMBDA + (offs_n * stride_log_lambda_n) # (BLOCK_N, BLOCK_DMODEL)
    do_ptrs = DO + (offs_m_init[:, None] * stride_dom + offs_k[None, :] * stride_dok) # (BLOCK_M, BLOCK_DMODEL)

    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_k[None, :] * stride_dvk) # (BLOCK_N, BLOCK_DMODEL)
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk) # (BLOCK_N, BLOCK_DMODEL)
    dlog_lambda_in_ptrs = DLOG_LAMBDA + (offs_n * stride_dlog_lambda_n) # (BLOCK_N, BLOCK_DMODEL)

    # k and v stay in SRAM throughout
    if DIVISIBLE_N:
        v = tl.load(v_ptrs)
        k = tl.load(k_ptrs)
        log_lambda_in = tl.load(log_lambda_in_ptrs)
    else:
        mask_n = offs_n < N
        v = tl.load(v_ptrs, mask=mask_n[:, None])
        k = tl.load(k_ptrs, mask=mask_n[:, None])
        log_lambda_in = tl.load(log_lambda_in_ptrs, mask=mask_n)

    # If the N block doesn't contain seq_start, no need to loop
    if HAS_SEQ_START:
        SEQ_START += off_z
        seq_start = tl.load(SEQ_START)
        hi = tl.where(start_n * BLOCK_N + BLOCK_N >= seq_start - 1, M, lo)
    else:
        hi = M

    # initialize dk amd dv
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dlog_lambda_in = tl.zeros([BLOCK_N], dtype=tl.float32)

    # loop over a col
    for start_m in range(lo, hi, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + offs_m_base
        causal_mask = (P_SEQ + offs_m[None, :]) >= (offs_n[:, None]) # (BLOCK_M, BLOCK_N)

        # load q1, k1, q2, k2, v, do on-chip
        if DIVISIBLE_M:
            q = tl.load(q_ptrs)
            log_lambda_out = tl.load(log_lambda_out_ptrs)
        else:
            mask_m = offs_m < M
            valid_mask = mask_m[None, :] # & mask_n
            q = tl.load(q_ptrs, mask=mask_m[:, None])
            log_lambda_out = tl.load(log_lambda_out_ptrs, mask=mask_m)
        # recompute p = softmax(qk * sm_scale, dim=-1)
        # s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        sT = tl.dot(k, tl.trans(q), input_precision="ieee") * qk_scale
        decay_bias = log_lambda_out[None, :] - log_lambda_in[:, None]
        sT += decay_bias * log2e
        # NOTE: since softmax in backward is pointwise, the normalizer has been saved in fwd)
        # So masking on s is not needed.
        # s = tl.where(valid_mask, s , float("-inf"))
        # if CAUSAL:
        #     s = tl.where(causal_mask, s, float("-inf"))

        # -- recompute p ---
        if DIVISIBLE_M:
            l = tl.load(L + offs_m)
        else:
            l = tl.load(L + offs_m, mask=mask_m)
        pT = tl.math.exp2(sT - l[None, :] * log2e) # (BLOCK_M, BLOCK_N)

        if not DIVISIBLE_M:
            pT = tl.where(valid_mask, pT, 0.0)
        if CAUSAL:
            pT = tl.where(causal_mask, pT, 0.0)

        # compute dv = dot(p, do)
        if DIVISIBLE_M:
            do = tl.load(do_ptrs)
        else:
            do = tl.load(do_ptrs, mask=mask_m[:, None]) # (BLOCK_M, BLOCK_DMODEL)


        dv += tl.dot(pT.to(input_dtype), do, input_precision="ieee") # (BLOCK_N, BLOCK_DMODEL)  # still correct

        # compute dp = dot(v, do)
        if DIVISIBLE_M:
            delta = tl.load(D + offs_m)
        else:
            delta = tl.load(D + offs_m, mask=mask_m)
        # dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dpT = tl.dot(v, tl.trans(do), input_precision="ieee")


        # compute ds = p * (dp - delta[:, None])
        dsT = pT * (dpT - delta[None, :]) # (BLOCK_M, BLOCK_N)

        if not DIVISIBLE_M:
            dsT = tl.where(valid_mask, dsT, 0.0)
        if CAUSAL:
            dsT = tl.where(causal_mask, dsT, 0.0)

        # compute dk = dot(ds.T, q) masking
        dk += tl.dot(dsT.to(input_dtype), q, input_precision="ieee")
        dlog_lambda_in += -tl.sum(dsT, axis=1)

        # increment pointers
        q_ptrs += BLOCK_M * stride_qm
        log_lambda_out_ptrs += BLOCK_M * stride_log_lambda_n
        do_ptrs += BLOCK_M * stride_dom

    dk *= sm_scale
    if HAS_SEQ_START:
        # Mask out 
        seq_mask = (offs_n >= seq_start)
        dk = tl.where(seq_mask[:, None], dk, 0.0)
        dv = tl.where(seq_mask[:, None], dv, 0.0)
        dlog_lambda_in = tl.where(seq_mask, dlog_lambda_in, 0.0)
    if DIVISIBLE_N:
        tl.store(dk_ptrs, dk.to(input_dtype)) # (BLOCK_N, BLOCK_DMODEL)
        tl.store(dv_ptrs, dv.to(input_dtype)) # (BLOCK_N, BLOCK_DMODEL,)
        tl.store(dlog_lambda_in_ptrs, dlog_lambda_in.to(tl.float32)) # (BLOCK_N, BLOCK_DMODEL,)
    else:
        tl.store(dk_ptrs, dk.to(input_dtype), mask=mask_n[:, None]) # (BLOCK_N, BLOCK_DMODEL)
        tl.store(dv_ptrs, dv.to(input_dtype), mask=mask_n[:, None]) # (BLOCK_N, BLOCK_DMODEL)
        tl.store(dlog_lambda_in_ptrs, dlog_lambda_in.to(tl.float32), mask=mask_n) # (BLOCK_N, BLOCK_DMODEL,)


@triton.jit
def _bwd_q_kernel(
    Q, K, V, LOG_LAMBDA, SEQ_START, sm_scale, DO,
    DQ, DLOG_LAMBDA,
    L,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_log_lambda_z, stride_log_lambda_h, stride_log_lambda_n,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    stride_dlog_lambda_z, stride_dlog_lambda_h, stride_dlog_lambda_n,
    Z, H, M, N, P_SEQ,
    num_groups,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, LARGER_M: tl.constexpr, HAS_SEQ_START: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    # -- grid id --
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    # offset pointers for (batch, head)
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    LOG_LAMBDA += off_z * stride_log_lambda_z + off_h * stride_log_lambda_h
    DO += off_z * stride_doz + off_h * stride_doh
    D += (off_z * H + off_h) * M
    L += (off_z * H + off_h) * M

    # offset pointers for batch/head
    DQ += off_z * stride_dqz + off_h * stride_dqh
    DLOG_LAMBDA += off_z * stride_dlog_lambda_z + off_h * stride_dlog_lambda_h

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    # initialize pointers to value-like data
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk) # (BLOCK_M, BLOCK_DMODEL)
    log_lambda_out_ptrs = LOG_LAMBDA + (P_SEQ + offs_m) * stride_log_lambda_n

    dq_ptrs = DQ + (offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqk) # (BLOCK_M, BLOCK_DMODEL)
    dlog_lambda_out_ptrs = DLOG_LAMBDA + (P_SEQ + offs_m) * stride_dlog_lambda_n
    do_ptrs = DO + (offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok) # (BLOCK_M, BLOCK_DMODEL)

    # pointer to row-wise quantities in value-like data
    d_ptrs = D + offs_m
    l_ptrs = L + offs_m

    # load q: it will stay in SRAM throughout
    if DIVISIBLE_M:
        q = tl.load(q_ptrs)
        do = tl.load(do_ptrs)
        delta = tl.load(d_ptrs)
        l = tl.load(l_ptrs)
        log_lambda_out = tl.load(log_lambda_out_ptrs)
    else:
        mask_m = offs_m < M
        q = tl.load(q_ptrs, mask=mask_m[:, None])
        do = tl.load(do_ptrs, mask=mask_m[:, None])
        delta = tl.load(d_ptrs, mask=mask_m)
        l = tl.load(l_ptrs, mask=mask_m)
        log_lambda_out = tl.load(log_lambda_out_ptrs, mask=mask_m)

    # initialize dq
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    dlog_lambda_out = tl.zeros([BLOCK_M], dtype=tl.float32)

    # loop over k, v and update accumulator
    # see note "Loop-Bound-For-N"
    if CAUSAL:
        hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
        if LARGER_M:
            hi = tl.maximum(0, hi)
    else:
        hi = N

    offs_n_base = tl.arange(0, BLOCK_N)
    offs_n_init = offs_n_base
    if HAS_SEQ_START:
        SEQ_START += off_z
        seq_start = tl.load(SEQ_START)
        lo = tl.minimum(seq_start, hi)
        lo = (lo // BLOCK_N) * BLOCK_N
        offs_n_init += lo
    else:
        lo = 0
    k_ptrs = K + (offs_n_init[:, None] * stride_kn + offs_k[None, :] * stride_kk) # (BLOCK_N, BLOCK_DMODEL)
    v_ptrs = V + (offs_n_init[:, None] * stride_vn + offs_k[None, :] * stride_vk) # (BLOCK_N, BLOCK_DMODEL)
    log_lambda_in_ptrs = LOG_LAMBDA + (offs_n_init * stride_log_lambda_n)

    # loop over a row
    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + offs_n_base

        # load k1, k2, v on chip
        if DIVISIBLE_N:
            v = tl.load(v_ptrs)
            k = tl.load(k_ptrs)
            log_lambda_in = tl.load(log_lambda_in_ptrs)
        else:
            mask_n = offs_n < N
            v = tl.load(v_ptrs, mask=mask_n[:, None])
            k = tl.load(k_ptrs, mask=mask_n[:, None])
            log_lambda_in = tl.load(log_lambda_in_ptrs, mask=mask_n)


        # recompute p = softmax(qk * sm_scale, dim=-1)
        if not DIVISIBLE_N:
            valid_mask = mask_n[None, :] # & mask_m[:, None]
        if CAUSAL:
            causal_mask = (P_SEQ + offs_m[:, None]) >= (offs_n[None, :]) # (BLOCK_M, BLOCK_N)
        # s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s = tl.dot(q, tl.trans(k), input_precision="ieee") * qk_scale
        decay_bias = log_lambda_out[:, None] - log_lambda_in[None, :]
        s += decay_bias * log2e

        # NOTE: since softmax in backward is pointwise, the normalizer has been saved in fwd)
        # So masking on s is not needed.
        # if CAUSAL:
        #     s = tl.where(causal_mask & valid_mask, s, float("-inf"))
        # else:
        #     s = tl.where(valid_mask, s, float("-inf"))
        p = tl.math.exp2(s - l[:, None] * log2e) # (BLOCK_M, BLOCK_N)

        # compute dp = dot(v, do)
        # dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp = tl.dot(do.to(input_dtype), tl.trans(v), input_precision="ieee")


        # no need to mask dp
        # if CAUSAL:
        #     dp = tl.where(causal_mask & valid_mask, dp, 0.0)
        # else:
        #     dp = tl.where(valid_mask, dp, 0.0)

        # compute ds = p * (dp - delta[:, None])
        # move scale out to dq at last
        ds = p * (dp - delta[:, None]) # (BLOCK_M, BLOCK_N)

        # mask ds to ensure no small values
        if not DIVISIBLE_N:
            ds = tl.where(valid_mask, ds, 0.0)
        if CAUSAL:
            ds = tl.where(causal_mask, ds, 0.0)
        if HAS_SEQ_START:
            ds = tl.where(offs_n[None, :] >= seq_start, ds, 0.0)

        dq += tl.dot(ds.to(input_dtype), k, input_precision="ieee")
        dlog_lambda_out += tl.sum(ds, axis=1)

        # increment pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        log_lambda_in_ptrs += BLOCK_N * stride_log_lambda_n

    dq *= sm_scale
    if DIVISIBLE_M:
        tmp = tl.load(dlog_lambda_out_ptrs)
    else:
        tmp = tl.load(dlog_lambda_out_ptrs, mask=mask_m)
    dlog_lambda_out += tmp
    if DIVISIBLE_M:
        tl.store(dq_ptrs, dq.to(input_dtype))
        tl.store(dlog_lambda_out_ptrs, dlog_lambda_out)
    else:
        tl.store(dq_ptrs, dq.to(input_dtype), mask=mask_m[:, None])
        tl.store(dlog_lambda_out_ptrs, dlog_lambda_out, mask=mask_m)



@pytest.mark.parametrize("Z, H, M, N, HEAD_DIM", [(4, 2, 1020, 2098, 64), (4, 2, 1024, 2048, 64)])
@pytest.mark.parametrize("causal", [True])
def test_op(Z, H, M, N, HEAD_DIM, causal, dtype=torch.bfloat16):
    torch.manual_seed(24)
    q = (torch.empty((Z, H, M, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    fgate_logit = torch.empty((Z, H, N), dtype=torch.float32, device="cuda").uniform_(5, 10)
    log_fgate = torch.nn.functional.logsigmoid(fgate_logit).requires_grad_()
    seq_start = torch.randint(low=0, high=N, size=(Z,), dtype=torch.long, device="cuda")
    # seq_start = torch.randint(low=0, high=10, size=(Z,), dtype=torch.long, device="cuda")
    # seq_start = torch.full(fill_value=0, size=(Z,), dtype=torch.long, device="cuda")
    sm_scale = 0.5
    dout = torch.randn_like(q)
    # reference implementation
    P_SEQ = N - M
    mask = torch.tril(torch.ones((M, N), device="cuda"), diagonal=P_SEQ)
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p = p.float()

    log_lambda = torch.cumsum(log_fgate, dim=-1)
    decay_bias = log_lambda[..., -M:, None] - log_lambda[..., None, :]
    p = p + decay_bias
    if causal:
        p[:, :, mask == 0] = float("-inf")

    attention_mask = torch.arange(N, device="cuda") < seq_start[:, None, None, None]
    p = torch.where(attention_mask, float("-inf"), p)
    p = torch.softmax(p.float(), dim=-1).to(dtype)
    p = p.clone()
    p[torch.isnan(p)] = 0.0
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    ref_dlog_fgate, log_fgate.grad = log_fgate.grad.clone(), None
    # triton implementation
    tri_out = forgetting_attention(q, k, v, log_fgate, head_first=True, seq_start=seq_start, sm_scale=sm_scale)
    tri_out = tri_out.to(dtype)

    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    tri_dlog_fgate, log_fgate.grad = log_fgate.grad.clone(), None
    # compare
    # assert torch.allclose(tri_log_normalizer[~torch.isnan(tri_log_normalizer)], ref_log_normalizer[~torch.isnan(ref_log_normalizer)], atol=1e-2, rtol=0)
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0), (ref_out - tri_out).abs().max()
    rtol = 0
    # Relative tolerance workaround for known hardware limitation of MI200 GPU.
    # For details see https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    # if torch.version.hip is not None and triton.runtime.driver.active.get_current_target().arch == "gfx90a":
        # rtol = 1e-2
    assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=rtol), (ref_dv - tri_dv).abs().max()
    assert torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=rtol), (ref_dk - tri_dk).abs().max()
    assert torch.allclose(ref_dq, tri_dq, atol=1e-2, rtol=rtol), (ref_dq - tri_dq).abs().max()
    assert torch.allclose(ref_dlog_fgate, tri_dlog_fgate, atol=1e-2, rtol=rtol), (ref_dlog_fgate - tri_dlog_fgate).abs().max()

try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')
BATCH, N_HEADS, HEAD_DIM = 4, 32, 128
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd", "bwd"]:
# for mode in ["bwd"]:
    # for causal in [True, False]:
    for causal in [True]:
        if mode == "bwd" and not causal:
            continue
        configs.append(
            triton.testing.Benchmark(
                x_names=["N_CTX"],
                # x_vals=[2**i for i in range(10, 15)],
                x_vals=[2**i for i in range(14, 15)],
                line_arg="provider",
                # line_vals=["triton-fp16", "flag"] + (["flash"] if HAS_FLASH else []),
                # line_names=["Triton [FP16]", "Flag"] + (["Flash-2"] if HAS_FLASH else []),
                line_vals=["flag"] + (["flash"] if HAS_FLASH else []),
                line_names=["Flag"] + (["Flash-2"] if HAS_FLASH else []),
                styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                ylabel="ms",
                plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}",
                args={
                    "H": N_HEADS,
                    "BATCH": BATCH,
                    "HEAD_DIM": HEAD_DIM,
                    "mode": mode,
                    "causal": causal,
                },
            ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, causal, mode, provider, device="cuda"):
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    dtype = torch.bfloat16
    if "flag" in provider:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fgate_logit = torch.empty((BATCH, H, N_CTX), dtype=torch.float32, device="cuda").uniform_(5, 10)
        log_fgate = torch.nn.functional.logsigmoid(fgate_logit).requires_grad_()
        # if mode == "fwd" and "fp8" in provider:
        #     q = q.to(torch.float8_e5m2)
        #     k = k.to(torch.float8_e5m2)
        #     v = v.permute(0, 1, 3, 2).contiguous()
        #     v = v.permute(0, 1, 3, 2)
        #     v = v.to(torch.float8_e5m2)
        sm_scale = 1.3
        fn = lambda: forgetting_attention(q, k, v, log_fgate, head_first=True, sm_scale=sm_scale)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops / ms * 1e-9


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    bench_flash_attention.run(save_path=".", print_data=True)
