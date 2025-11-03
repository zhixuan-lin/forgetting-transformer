# Checkpoints, Training, and Evaluation

Please note that loading checkpoints and training/evaluation require different dependencies. See the corresponding sections for instructions.

## Model Checkpoints

For reproducibility and research purposes, we provide model checkpoints for our main experiments. These are 760M-parameter-scale models trained on 48B tokens from [LongCrawl64](https://manifestai.com/articles/longcrawl64/) with a training context length of 16k tokens. 

Note that these are small models trained on a small number of tokens.  Also, as a
long-context dataset for research purposes, LongCrawl64 is **not** designed for optimal
downstream task performance (it also has a strange tokenization process, see
[here](https://github.com/zhixuan-lin/forgetting-transformer/blob/main/src/forgetting_transformer/tokenizer.py)).
Therefore, these models are only suitable for research purposes (e.g., inspecting forget gate values). Also, if you want to compare FoX with other models trained in another setting with another dataset, **you should definitely train FoX on your own dataset under your own setting for the comparison**.

### Dependenceies

Python 3.10 or above is recommended. Install the following:

```bash
# First uninstall to prevent potential issues
pip uninstall forgetting_transformer && pip install -U git+https://github.com/zhixuan-lin/forgetting-transformer
pip install pytest einops numpy
pip install torch==2.4.0
pip install transformers==4.44.0

# No guarantee other commits would work
pip install --no-deps --force-reinstall git+https://github.com/sustcsonglin/flash-linear-attention.git@1c5937eeeb8b0aa17bed5ee6dae345b353196bd4
flash-attn==2.6.3    # Needed for transformer LLaMA
causal-conv1d==1.4.0 # For Mamba-2 and DeltaNet
mamba-ssm==2.2.2     # For Mamba-2
```

### Usage

The checkpoints can be downloaded from [this HuggingFace collection](https://huggingface.co/collections/zhixuan-lin/forgetting-transformer-paper-checkpoints-67d0ded3caa418ff0cc16ba4). Here is a usage example:

```python
import forgetting_transformer.model.register_all  # Needed to register the model classes
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

First, clone this repository and do an [editable install](https://setuptools.pypa.io/en/latest/userguide/development_mode.html):

```bash
# First uninstall to prevent potential issues
pip uninstall forgetting_transformer
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
gcloud auth login  # Login if you haven't
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
gcloud auth login  # Login if you haven't
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
import forgetting_transformer.model.register_all  # Needed to register the model classes
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

