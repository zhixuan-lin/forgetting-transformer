# Forgetting Transformer

Official PyTorch implementation of [Forgetting Transformer: Softmax Attention with a Forget Gate](https://openreview.net/forum?id=q2Lnyegkr8) (ICLR 2025).

This repository contains the implementation of Forgetting Attention and the Forgetting Transformer (FoX). In particular, we provide an efficient Triton kernel of Forgetting Attention that could be used as a (almost) drop-in replacement for the regular FlashAttention kernel. Besides this official repository, [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) also has a Forgetting Attention kernel implementation that supports variable-length inputs (see [here](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/forgetting_attn)).

We also provide training code, evaluation code, and model checkpoints to reproduce the results in the paper, including all the baselines.

## Changelog

* [2025-03-31] In our paper experiments we accidentally share the RMSNorm scaling parameters across heads in the QK-norm implementation. Normally there should be a total of `num_heads * head_dim` scaling parameters but in our experiments there were only `head_dim` scaling parameters shared across the `num_heads` heads. We have verified that this has no observable impact on performance. Nevertheless, we have added a `qk_norm_share_param_across_head` argument that controls this behavior and set the default value to `False` because it makes more sense. Note that for backward compatibility, in our provided checkpoints `qk_norm_share_param_across_head` is still set to `True` (otherwise the weights cannot be loaded).

## Installation and Quickstart

Python 3.10 or above is recommended.

If you just want to use the Forgetting Attention kernel and the FoX layer/model, you can install this repository as a regular Python package:

```bash
# We recommend you keep track of the commit hash you used. We may introduce breaking changes in the future.
# First uninstall to prevent potential issues
pip uninstall forgetting_transformer && pip install -U git+https://github.com/zhixuan-lin/forgetting-transformer
```

If you want to do training/evaluation or modify the code, you need to clone this repository and do an [editable install](https://setuptools.pypa.io/en/latest/userguide/development_mode.html):

```bash
git clone git@github.com:zhixuan-lin/forgetting-transformer.git
cd forgetting-transformer
pip install --editable .
```

Note that both installation methods DO NOT install any dependencies by default. The needed dependencies depend on what you want to use and will be explained below.

### The Forgetting Attention Kernel

If you only want to use the Forgetting Attention kernel (e.g., as a replacement for the FlashAttention kernel), you need to install the following (we pin the versions to ensure that this works; you don't have to):

```bash
pip install pytest einops numpy
pip install torch==2.4.0  # This also installs triton==3.0.0
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

### FoX Time-Mixing Layer and Model


**WARNINGS**: 
1. We only support `attention_mask` that implements left padding. Passing `attention_mask` that implements right padding to the model would lead to incorrect results.
2. Decoding with `attention_mask` is supported but has not been thoroughly tested. If it is necessary to use `attention_mask` during decoding, we recommend you test this functionality and make sure it works for your use case.



If you want to use the FoX time-mixing layer or the complete FoX model, you need the following dependencies (again versions are pinned just in case):

```bash
pip install pytest einops numpy
pip install torch==2.4.0
pip install transformers==4.44.0
# No guarantee other commits would work; we may fix this later
pip install --no-deps --force-reinstall git+https://github.com/sustcsonglin/flash-linear-attention.git@1c5937eeeb8b0aa17bed5ee6dae345b353196bd4
```

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
assert out.size() == (batch_size, seq_len, hidden_size)
print(layer)
# ForgettingAttentionLayer(
#   (q_proj): Linear(in_features=1536, out_features=1536, bias=False)
#   (k_proj): ShiftLinear(1536, 1536)
#   (v_proj): ShiftLinear(1536, 1536)
#   (o_proj): Linear(in_features=1536, out_features=1536, bias=False)
#   (fgate_proj): Linear(in_features=1536, out_features=24, bias=True)
#   (ogate_proj): Linear(in_features=1536, out_features=1536, bias=False)
#   (output_norm): GroupRMSNorm(24, 1536, eps=1e-06)
#   (q_norm): GroupRMSNorm(24, 1536, eps=1e-05)
#   (k_norm): GroupRMSNorm(24, 1536, eps=1e-05)
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
    hidden_ratio=3.5   # output gates introduce extra params so we reduce MLP hidden size
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
#           (output_norm): GroupRMSNorm(24, 1536, eps=1e-06)
#           (q_norm): GroupRMSNorm(24, 1536, eps=1e-05)
#           (k_norm): GroupRMSNorm(24, 1536, eps=1e-05)
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



## Model Checkpoints

For reproducibility and research purposes, we provide model checkpoints for our main experiments. These are 760M-parameter-scale models trained on 48B tokens from [LongCrawl64](https://manifestai.com/articles/longcrawl64/) with a training context length of 16k tokens. 

Note that these are small models trained on a small number of tokens.  Also, as a
long-context dataset for research purposes, LongCrawl64 is **not** designed for optimal
downstream task performance (it also has a strange tokenization process, see
[here](https://github.com/zhixuan-lin/forgetting-transformer/blob/main/src/forgetting_transformer/tokenizer.py)).
Therefore, these models are only suitable for research purposes (e.g., inspecting forget gate values). Also, if you want to compare FoX with other models trained in another setting with another dataset, **you should definitely train FoX on your own dataset under your own setting for the comparison**.

These checkpoints can be downloaded from [this HuggingFace collection](https://huggingface.co/collections/zhixuan-lin/forgetting-transformer-paper-checkpoints-67d0ded3caa418ff0cc16ba4). Here is a usage example:

```python
import forgetting_transformer.model  # Needed to register the model classes
import forgetting_transformer.tokenizer  # Needed to register the tokenizer class
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("zhixuan-lin/fox-pro-760m-longcrawl64-48b")
tokenizer = AutoTokenizer.from_pretrained("zhixuan-lin/fox-pro-760m-longcrawl64-48b", add_bos_token=True, clean_up_tokenization_spaces=False)

# Generation using HF api
prompt = "The best thing to do in San Francisco is"
model = model.cuda()
encoded = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    output = model.generate(
        encoded,
        max_new_tokens=30,
    )[0]
pred = tokenizer.decode(output, skip_special_tokens=True)
print(pred)

# Of course you can also compute the logits or loss given proper inputs
batch_size, seq_len = encoded.shape
labels = encoded
input_ids = torch.roll(labels, shifts=1, dims=-1)
input_ids[:, 0] = tokenizer.bos_token_id  # 50256
out = model(input_ids=input_ids, labels=labels)
assert out.loss.size() == (batch_size, seq_len)
# Logits are not returned (to save memory) if labels are given
assert out.logits is None
# To get logits don't provide labels
out = model(input_ids=input_ids)
assert out.logits.size() == (batch_size, seq_len, tokenizer.vocab_size)
```



## Training and Evaluation

### Dependencies

First, make sure you've done an editable installation of this repository if you haven't:

```bash
git clone git@github.com:zhixuan-lin/forgetting-transformer.git
cd forgetting-transformer
pip install --editable .
```

Then install the rest of the dependencies for training and evaluation:

```bash
pip install -r requirements-dev.txt
pip install --no-deps --force-reinstall git+https://github.com/sustcsonglin/flash-linear-attention.git@1c5937eeeb8b0aa17bed5ee6dae345b353196bd4
```

### Data Preparation

We provide code for training on [LongCrawl64](https://manifestai.com/articles/longcrawl64/). First, download the dataset using `gsutil` (the downloading instructions are from the [LongCrawl64](https://manifestai.com/articles/longcrawl64/) website):

```bash
DATA_DIR="./data"  # You can use any other path
mkdir -p ${DATA_DIR}/longcrawl64
# Install gsutil
curl https://sdk.cloud.google.com | bash
GSUTIL_PARALLEL_THREAD_COUNT=5 GSUTIL_PARALLEL_PROCESS_COUNT=5 gsutil -m cp -r 'gs://longcrawl64/*.zarr' ${DATA_DIR}/longcrawl64
```

The dataset is around 800GB so it will take a while. Make sure the directory structure looks like this after the download:

```
$DATA_DIR
└── longcrawl64
    ├── heldout.zarr
    └── train.zarr
```

If you just want to run evaluation with our checkpoints, you can only download the validation set (5.9GB):

```bash
DATA_DIR="./data"  # You can use any other path
mkdir -p ${DATA_DIR}/longcrawl64
# Install gsutil
curl https://sdk.cloud.google.com | bash
GSUTIL_PARALLEL_THREAD_COUNT=5 GSUTIL_PARALLEL_PROCESS_COUNT=5 gsutil -m cp -r 'gs://longcrawl64/heldout.zarr' ${DATA_DIR}/longcrawl64
```

### Training

We provide training configurations for all the baselines for the main 760M-param/48B-token setting in `configs/experiments/longcrawl64`. We also provide additional configurations for the 760M-param/16B-token, 360M-param/7.5B-token, and 125M-param/2.7B-token settings used for our analysis experiments. For example, to train a 760M-param FoX (Pro) on 48B tokens from LongCrawl64, you can run the following:

```bash
OUTPUT_DIR="./output/model/fox_pro_760m_48b"  # You can set this to any other path
WANDB_DIR="./output/wandb"  # You can set this to any other path
mkdir -p $OUTPUT_DIR
mkdir -p $WANDB_DIR
fabric run train.py \
    --devices 4 \
    --num-nodes 1 \
    --node-rank 0 \
    --main-address localhost \
    --main-port 1234 \
    +experiment/longcrawl64/forgetting_transformer=pro_760m_48b \
    seed=0 \
    exp=demo \
    tag=fox_pro_760m_48b \
    output_dir=$OUTPUT_DIR \
    data_dir=$DATA_DIR \
    wandb.log_dir=$WANDB_DIR \
    wandb.mode=online \
    resume=true
```

The above is for single-node training with 4 GPUs. For multi-node training you need to set `--num-nodes`, `--node-rank`, `--main-address`, and `--main-port` properly and launch the `fabric run` command on every node. Please refer to the [lightning fabric CLI documentation](https://lightning.ai/docs/fabric/stable/fundamentals/launch.html#launch-with-the-cli) for instructions on how to set these flags for multi-node training.

Checkpoints will be saved to `$OUTPUT_DIR` periodically. We support resuming interrupted training runs with `resume=true` (the default). Setting `resume=false` would cause existing contents in `$OUTPUT_DIR` to be removed before training starts.

### Saving the Model in Hugging Face Format

For evaluation we require models to be saved in Hugging Face format. After training is finished, you can save the trained model in HuggingFace format using `save_model.py`:

```bash
HF_LOAD_DIR=$OUTPUT_DIR
HF_SAVE_DIR="./output/hf/fox-pro-760m-48b"
mkdir -p $HF_SAVE_DIR
python save_model.py \
    --hf_load_dir=$HF_LOAD_DIR \
    --hf_save_dir=$HF_SAVE_DIR
```

This saves the model to `$HF_SAVE_DIR`. After you save the model, you can load the saved model as follows:

```python
import forgetting_transformer.model  # Needed to register the model classes
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrain("./output/hf/fox-pro-760m-48b")
```

### Evaluation

In `eval/`, we provide code for the following evaluation that we use in the paper: 
* Per-token loss
* Needle-in-a-haystack retrieval task
* Short-context downstream tasks from [Language Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
* Long-context downstream tasks from [LongBench](https://github.com/THUDM/LongBench/)

Before you run evaluation make sure you've saved the model in Hugging Face format using `save_model.py`. You could also use our provided checkpoints to run the evaluation. Please see the `README.md` file in each subdirectories of `eval/` for instructions.

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

