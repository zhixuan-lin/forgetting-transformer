# Forgetting Transformer

Official PyTorch implementation of [Forgetting Transformer: Softmax Attention with a Forget Gate](https://openreview.net/forum?id=q2Lnyegkr8) (ICLR 2025).

This repository contains the implementation of the Forgetting Attention and the Forgetting Transformer (FoX). In particular, we provide an efficient Triton implementation of the Forgetting Attention that could be used as a (almost) drop-in replacement for the regular FlashAttention kernel. 

(WIP) We will also provide training and evaluation code to reproduce the results in the paper, including all the baselines.

## Installation and Quickstart

Python 3.10 or above is recommended.

If you just want to use the Forgetting Attention kernel and the FoX model, you can install this repository as a regular Python package:

```bash
# First uninstall to prevent potential issues
pip uninstall forgetting_transformer && pip install -U git+https://github.com/zhixuan-lin/forgetting-transformer
```

If you want to run the training code or modify the code, it is best to clone this repository and do an [editable install](https://setuptools.pypa.io/en/latest/userguide/development_mode.html):

```bash
git clone git@github.com:zhixuan-lin/forgetting-transformer.git
cd forgetting-transformer
pip install --editable .
```

For the first installation method we recommend you keep track of the commit hash you used. We may introduce breaking changes in the future.

Note that both installation methods DO NOT install any dependencies by default. The needed dependencies depend on what you want to use and will be explained below.

### The Forgetting Attention kernel

If you only want to use the Forgetting Attention kernel (e.g., as a replacement for the FlashAttention kernel), you need to install the following (we pin the versions to ensure that this works; you don't have to):

```bash
pip install pytest einops numpy
pip install torch==2.4.0
```

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
    An FlashAttention-based implementation of Forgetting Attention. 

    For now:
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
        out (torch.Tensor): (batch_size, num_heads, seqlen_q, head_dim) unless head_first=True.
    """
```


Here is a simple example demonstrating the usage:

```python
import torch
from forgetting_transformer import forgetting_attention

batch_size = 4
num_heads = 12
seqlen = 512
head_dim = 64
dtype = torch.bfloat16
device = "cuda"

q = torch.randn((batch_size, seqlen, num_heads, head_dim), dtype=dtype, device=device, requires_grad=True)
k = torch.randn((batch_size, seqlen, num_heads, head_dim), dtype=dtype, device=device, requires_grad=True)
v = torch.randn((batch_size, seqlen, num_heads, head_dim), dtype=dtype, device=device, requires_grad=True)
# You can use a tiny linear layer to get fgate_logit
fgate_logit = torch.randn((batch_size, seqlen, num_heads), dtype=torch.float32, device=device)
log_fgate = torch.nn.functional.logsigmoid(fgate_logit.float())

out = forgetting_attention(q, k, v, log_fgate)
assert out.size() == (batch_size, seqlen, num_heads, head_dim)
```

### FoX Time-Mixing Layer and Model

If you want to use the FoX time-mixing layer or the whole model, you need the following dependencies (again versions are pinned just in case):

```bash
pip install pytest einops numpy
pip install torch==2.4.0
pip install transformers==4.44.0
# No guarantee other commits would work; we may fix this later
pip install --no-deps --force-reinstall git+https://github.com/sustcsonglin/flash-linear-attention.git@1c5937eeeb8b0aa17bed5ee6dae345b353196bd4
```

Usage example for the time-mixing layer `ForgettingAttentionLayer`:

```python
import torch
from forgetting_transformer.model import ForgettingAttentionLayer

batch_size = 4
seqlen = 512
hidden_size = 1536
dtype = torch.float32
device = "cuda"
x = torch.randn((batch_size, seqlen, hidden_size), dtype=dtype, device=device, requires_grad=True)

# Configuration for the 760M FoX (Pro) model
layer = ForgettingAttentionLayer(
    hidden_size=hidden_size,
    num_heads=24,
    use_output_gate=True,
    use_output_norm=True,
    qk_norm=True,
    use_k_shift=True,
    use_v_shift=True,
).to(device)

out, *rest = layer(x)
assert out.size() == (batch_size, seqlen, hidden_size)
print(layer)
# ForgettingAttentionLayer(
#   (q_proj): Linear(in_features=1536, out_features=1536, bias=False)
#   (k_proj): ShiftLinear(1536, 1536)
#   (v_proj): ShiftLinear(1536, 1536)
#   (o_proj): Linear(in_features=1536, out_features=1536, bias=False)
#   (fgate_proj): Linear(in_features=1536, out_features=24, bias=True)
#   (ogate_proj): Linear(in_features=1536, out_features=1536, bias=False)
#   (output_norm): GroupRMSNorm(24, 1536, eps=1e-06)
#   (q_norm): RMSNorm(64, eps=1e-05)
#   (k_norm): RMSNorm(64, eps=1e-05)
# )
```

Usage example for the (complete) model `ForgettingTransformerForCausalLM`:

```python
import torch
from forgetting_transformer.model import ForgettingTransformerConfig, ForgettingTransformerForCausalLM


batch_size = 4
seqlen = 512
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
    hidden_ratio=3.5   # output gates introduce extra params so we reduce MLP hidden size
)
model = ForgettingTransformerForCausalLM(config).to(device)

labels = torch.randint(0, vocab_size, size=(batch_size, seqlen), device=device)
input_ids = torch.roll(labels, shifts=1, dims=-1)
input_ids[:, 0] = bos_token_id
out = model(input_ids=input_ids, labels=labels)
assert out.loss.size() == (batch_size, seqlen)
# Logits are not returned (to save memory) if labels are given
assert out.logits is None
# To get logits don't provide labels
out = model(input_ids=input_ids)
assert out.logits.size() == (batch_size, seqlen, vocab_size)


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
#           (output_norm): GroupRMSNorm(24, 1536, eps=1e-06)
#           (q_norm): RMSNorm(64, eps=1e-05)
#           (k_norm): RMSNorm(64, eps=1e-05)
#         )
#         (mlp_norm): RMSNorm(1536, eps=1e-06)
#         (mlp): ForgettingTransformerMLP(
#           (gate_proj): Linear(in_features=1536, out_features=8192, bias=False)
#           (down_proj): Linear(in_features=4096, out_features=1536, bias=False)
#           (act_fn): SiLU()
#         )
#       )
#     )
#     (norm): RMSNorm(1536, eps=1e-06)
#   )
#   (lm_head): Linear(in_features=1536, out_features=32768, bias=False)
# )

```

## Training and Evaluation

Work in progress. This will be updated soon.

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
```

