# Forgetting Transformer

Official PyTorch implementation of 
* [Forgetting Transformer: Softmax Attention with a Forget Gate](https://arxiv.org/abs/2503.02130) (ICLR 2025)
* [Adaptive Computation Pruning for the Forgetting Transformer](https://arxiv.org/abs/2504.06949) (COLM 2025)

This repository contains the implementation of Forgetting Attention and the Forgetting Transformer (FoX), with support for Adaptive Computation Pruning (ACP). In particular, we provide **an efficient Triton kernel of Forgetting Attention that could be used as a (almost) drop-in replacement for the regular FlashAttention kernel**. Besides this official repository, [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) also has a Forgetting Attention kernel implementation that supports variable-length inputs (see [here](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/forgetting_attn)). We also provide training code, evaluation code, and model checkpoints to reproduce the results in the FoX paper, including all the baselines.

This README describes how you can use this repository as a library. This allows you to import and use the Forgetting Attention kernel and the FoX layer/model. For instructions on training, evaluation, and HuggingFace checkpoints, see [REPRODUCE.md](REPRODUCE.md). Note different usage of this repository requires different dependencies.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## Installation

Python 3.10 or above is recommended.

First, install this repository as a regular Python package:

```bash
# We recommend you keep track of the commit hash you used. We may introduce breaking changes in the future.
# First uninstall to prevent potential issues
pip uninstall forgetting_transformer && pip install -U git+https://github.com/zhixuan-lin/forgetting-transformer
```

Then, install the rest of the dependencies. If you only want to use the Forgetting Attention kernel (e.g., as a replacement for the FlashAttention kernel), you need the following (we pin the `torch` version to ensure that this works; you don't have to):

```bash
pip install pytest einops numpy
pip install torch==2.4.0
```

If you want to use the FoX time-mixing layer or the complete FoX model, you additionally need the `transformers` package (again the version is pinned just in case):

```bash
pip install transformers==4.44.0
```


## The Forgetting Attention Kernel

The documentation for `forgetting_attention` is as follows:

```python
from forgetting_transformer import forgetting_attention
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
```


Here is a simple example demonstrating the usage:

```python
import torch
from forgetting_transformer import forgetting_attention

batch_size = 4
num_heads = 12
seq_len = 512
head_dim = 64
dtype = torch.bfloat16
device = "cuda"

q = torch.randn((batch_size, seq_len, num_heads, head_dim), dtype=dtype, device=device, requires_grad=True)
k = torch.randn((batch_size, seq_len, num_heads, head_dim), dtype=dtype, device=device, requires_grad=True)
v = torch.randn((batch_size, seq_len, num_heads, head_dim), dtype=dtype, device=device, requires_grad=True)
# You can use a tiny linear layer to get `fgate_logit`.
# For example, let `x` be the attention input with shape (batch_size, seq_len, hidden_size) 
# which is also used to compute `q`, `k` and `v`. You can get `fgate_logit` as follows
#     In your model's `__init__`: `self.fgate_proj = nn.Linear(hidden_size, num_heads, bias=True)`
#     In your model's `forward`:  `fgate_logit = self.fgate_proj(x)`
fgate_logit = torch.randn((batch_size, seq_len, num_heads), dtype=dtype, device=device, requires_grad=True)
log_fgate = torch.nn.functional.logsigmoid(fgate_logit.float())

out = forgetting_attention(q, k, v, log_fgate)
assert out.size() == (batch_size, seq_len, num_heads, head_dim)
out.sum().backward()
```

Note that our kernel only supports input batches that contain sequences of the same length. You can find another implementation that supports variable-length inputs in the [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) repository (see [here](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/forgetting_attn)).

### Adaptive Computation Pruning

It is highly recommended you use Forgetting Attention with Adaptive Computation Pruning, especially for long-context pretraining. The core API change compared to the original Forgetting Attention kernel is the `adaptive_threshold` argument, which corresponds to the $\delta$ threshold in the paper. Here is an example demonstrating how you should set the threshold, depending on whether QK-norm is used:

```python
import torch
from forgetting_transformer import forgetting_attention
import math
from torch import nn
from einops import rearrange

batch_size = 4
num_heads = 12
seq_len = 512
head_dim = 64
dtype = torch.bfloat16
device = "cuda"

q = torch.randn((batch_size, seq_len, num_heads, head_dim), dtype=dtype, device=device, requires_grad=True)
k = torch.randn((batch_size, seq_len, num_heads, head_dim), dtype=dtype, device=device, requires_grad=True)
v = torch.randn((batch_size, seq_len, num_heads, head_dim), dtype=dtype, device=device, requires_grad=True)
# You can use a tiny linear layer to get `fgate_logit`.
# For example, let `x` be the attention input with shape (batch_size, seq_len, hidden_size) 
# which is also used to compute `q`, `k` and `v`. You can get `fgate_logit` as follows
#     In your model's `__init__`: `self.fgate_proj = nn.Linear(hidden_size, num_heads, bias=True)`
#     In your model's `forward`:  `fgate_logit = self.fgate_proj(x)`
fgate_logit = torch.randn((batch_size, seq_len, num_heads), dtype=dtype, device=device, requires_grad=True)
log_fgate = torch.nn.functional.logsigmoid(fgate_logit.float())


USE_QK_NORM = False
if USE_QK_NORM:
    class GroupRMSNorm(nn.Module):
        """Naive implementation of grouped RMSNorm"""
        def __init__(self, hidden_size: int, num_groups: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.hidden_size = hidden_size
            self.num_groups = num_groups
            self.eps = eps

        def forward(self, x):
            assert x.size(-1) == self.hidden_size, x.size(-1)
            x = rearrange(x, '... (g d) -> ... g d', g=self.num_groups)
            weight = rearrange(self.weight, '(g d) -> g d', g=self.num_groups)
            rstd = x.float().square().mean(dim=-1, keepdim=True).sqrt()
            out = x / rstd * weight
            out = rearrange(out, '... g d -> ... (g d)')
            out = out.to(x.dtype)
            return out

    # Apply QK-norm
    q_norm = GroupRMSNorm(hidden_size=num_heads * head_dim, num_groups=num_heads).to(device)
    k_norm = GroupRMSNorm(hidden_size=num_heads * head_dim, num_groups=num_heads).to(device)
    q, k = [rearrange(entry, '... h d -> ... (h d)') for entry in (q, k)]
    q = q_norm(q)
    k = k_norm(k)
    q, k = [rearrange(entry, '... (h d) -> ... h d', h=num_heads) for entry in (q, k)]

# exp(log_pruning_tolerance) bounds the maximum total attention weights that could be pruned
log_pruning_tolerance = -10.0

with torch.no_grad():
    # Calculate an upper bound of attention logits
    if USE_QK_NORM:
        # If we use QK-norm, it is easily to get an upper bound of q/k L2-norm
        max_q_norm = q_norm.weight.view(num_heads, head_dim).abs().max(dim=-1).values * math.sqrt(head_dim)
        max_k_norm = k_norm.weight.view(num_heads, head_dim).abs().max(dim=-1).values * math.sqrt(head_dim)
    else:
        # Otherwise we could calculate the max L2 norms manually 
        max_q_norm = torch.linalg.vector_norm(q, dim=-1).max(dim=-2).values
        max_k_norm = torch.linalg.vector_norm(k, dim=-1).max(dim=-2).values
        assert max_q_norm.size() == max_k_norm.size() == (batch_size, num_heads)

    logit_upper_bound = max_q_norm * max_k_norm / math.sqrt(head_dim)
    adaptive_threshold = -(2 * logit_upper_bound + math.log(seq_len)) + log_pruning_tolerance


out = forgetting_attention(q, k, v, log_fgate, adaptive_threshold=adaptive_threshold)
assert out.size() == (batch_size, seq_len, num_heads, head_dim)
```

If you want to directly use the FoX time-mixing layer and model classes instead of the Triton kernel, enabling ACP is as easy as providing a `log_pruning_tolerance=-10.0` argument. See the following for examples.


## FoX Time-Mixing Layer and Model


**WARNINGS**: 
1. We only support `attention_mask` that implements left padding. Passing `attention_mask` that implements right padding to the model would lead to incorrect results.
2. Decoding with `attention_mask` is supported but has not been thoroughly tested. If it is necessary to use `attention_mask` during decoding, we recommend you test this functionality and make sure it works for your use case.

Usage example for the FoX time-mixing layer `ForgettingAttentionLayer`:

```python
import torch
from forgetting_transformer.model import ForgettingAttentionLayer

batch_size = 4
seq_len = 512
hidden_size = 1536
dtype = torch.float32
device = "cuda"
x = torch.randn((batch_size, seq_len, hidden_size), dtype=dtype, device=device, requires_grad=True)

# Configuration for the 760M FoX (Pro) model, with ACP enabled
layer = ForgettingAttentionLayer(
    hidden_size=hidden_size,
    num_heads=24,
    use_output_gate=True,
    use_output_norm=True,
    qk_norm=True,
    use_k_shift=True,
    use_v_shift=True,
    log_pruning_tolerance=-10.0  # This sets the epsilon parameter in ACP to exp(-10)
).to(device)

out, *rest = layer(x)
assert out.size() == (batch_size, seq_len, hidden_size)
print(layer)
# ForgettingAttentionLayer(
#   (q_proj): Linear(in_features=1536, out_features=1536, bias=False)
#   (k_proj): ShiftLinear(1536, 1536)
#   (v_proj): ShiftLinear(1536, 1536)
#   (o_proj): Linear(in_features=1536, out_features=1536, bias=False)
#   (fgate_proj): Linear(in_features=1536, out_features=24, bias=True)
#   (ogate_proj): Linear(in_features=1536, out_features=1536, bias=False)
#   (output_norm): FusedGroupNormGated(24, 1536, is_rms_norm=True, eps=1e-06, activation=sigmoid)
#   (q_norm): GroupNorm(24, 1536, is_rms_norm=True, eps=1e-05)
#   (k_norm): GroupNorm(24, 1536, is_rms_norm=True, eps=1e-05)
# )

```

Usage example for the complete FoX model `ForgettingTransformerForCausalLM`:

```python
import torch
from forgetting_transformer.model import ForgettingTransformerConfig, ForgettingTransformerForCausalLM


batch_size = 4
seq_len = 512
hidden_size = 1536
vocab_size = 32768
bos_token_id = 0
device = "cuda"

# Configuration for the 760M FoX (Pro) model
config = ForgettingTransformerConfig(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    num_hidden_layers=24,
    num_heads=24,
    use_output_gate=True,
    use_output_norm=True,
    qk_norm=True,
    use_k_shift=True,
    use_v_shift=True,
    hidden_ratio=3.5,   # output gates introduce extra params so we reduce MLP hidden size
    log_pruning_tolerance=-10.0  # This sets the epsilon parameter in ACP to exp(-10)
)
model = ForgettingTransformerForCausalLM(config).to(device)

labels = torch.randint(0, vocab_size, size=(batch_size, seq_len), device=device)
input_ids = torch.roll(labels, shifts=1, dims=-1)
input_ids[:, 0] = bos_token_id
out = model(input_ids=input_ids, labels=labels)
assert out.loss.size() == (batch_size, seq_len)
# Logits are not returned (to save memory) if labels are given
assert out.logits is None
# To get logits don't provide labels
out = model(input_ids=input_ids)
assert out.logits.size() == (batch_size, seq_len, vocab_size)


print(model)

# ForgettingTransformerForCausalLM(
#   (model): ForgettingTransformerModel(
#     (embeddings): Embedding(32768, 1536)
#     (layers): ModuleList(
#       (0-23): 24 x ForgettingTransformerBlock(
#         (attn_norm): RMSNorm(1536, eps=1e-06)
#         (attn): ForgettingAttentionLayer(
#           (q_proj): Linear(in_features=1536, out_features=1536, bias=False)
#           (k_proj): ShiftLinear(1536, 1536)
#           (v_proj): ShiftLinear(1536, 1536)
#           (o_proj): Linear(in_features=1536, out_features=1536, bias=False)
#           (fgate_proj): Linear(in_features=1536, out_features=24, bias=True)
#           (ogate_proj): Linear(in_features=1536, out_features=1536, bias=False)
#           (output_norm): FusedGroupNormGated(24, 1536, is_rms_norm=True, eps=1e-06, activation=sigmoid)
#           (q_norm): GroupNorm(24, 1536, is_rms_norm=True, eps=1e-05)
#           (k_norm): GroupNorm(24, 1536, is_rms_norm=True, eps=1e-05)
#         )
#         (mlp_norm): RMSNorm(1536, eps=1e-06)
#         (mlp): ForgettingTransformerMLP(
#           (gate_proj): Linear(in_features=1536, out_features=7168, bias=False)
#           (down_proj): Linear(in_features=3584, out_features=1536, bias=False)
#           (act_fn): SiLU()
#         )
#       )
#     )
#     (norm): RMSNorm(1536, eps=1e-06)
#   )
#   (lm_head): Linear(in_features=1536, out_features=32768, bias=False)
# )

```


## Acknowledgements

All the model implementations are based on [flash-linear-attention](https://github.com/fla-org/flash-linear-attention). The Forgetting Attention kernel is based on the Triton FlashAttention kernel implemented in [FlagAttention](https://github.com/FlagOpen/FlagAttention). The repository structure is inspired by [litgpt](https://github.com/Lightning-AI/litgpt) and the [FlashAttention](https://github.com/Dao-AILab/flash-attention) training code. Some components are also taken from these these repositories.

## Citation

If you use this code, please consider citing the following:

```
@inproceedings{
lin2025forgetting,
title={Forgetting Transformer: Softmax Attention with a Forget Gate},
author={Zhixuan Lin and Evgenii Nikishin and Xu He and Aaron Courville},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=q2Lnyegkr8}
}

@inproceedings{
lin2025adaptive,
title={Adaptive Computation Pruning for the Forgetting Transformer},
author={Zhixuan Lin and Johan Obando-Ceron and Xu He and Aaron Courville},
booktitle={Second Conference on Language Modeling},
year={2025},
url={https://openreview.net/forum?id=xNj14CY5S1}
}
```



