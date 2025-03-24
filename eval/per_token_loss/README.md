# Per-token loss

This directory contains code to compute and plot per-token loss using LongCrawl64. 

Before you run the evaluation, make sure you have downloaded the heldout set of LongCrawl64. If you haven't, run the following:

```bash
DATA_DIR="./data"  # You can use any other path
mkdir -p ${DATA_DIR}/longcrawl64
# Install gsutil
curl https://sdk.cloud.google.com | bash
GSUTIL_PARALLEL_THREAD_COUNT=5 GSUTIL_PARALLEL_PROCESS_COUNT=5 gsutil -m cp -r 'gs://longcrawl64/heldout.zarr' ${DATA_DIR}/longcrawl64
```


After this you can run the evaluation as follows:

```bash
DATA_DIR="./data"  # Or whatever path that contains the LongCrawl64 dataset
SAVE_DIR="./results"  # You can use any other path
fabric run run_per_token_loss.py \
   --devices 1 \
   --model  "fox-pro-760m-longcrawl64-48b" \
   --model_path "zhixuan-lin/fox-pro-760m-longcrawl64-48b" \
   --data_path $DATA_DIR/longcrawl64 \
   --save_dir $SAVE_DIR \
   --resume \
   --save_interval 128
```
We also support multi-gpu evaluation and resuming. However, resuming requires that you use the same number of GPUs as the resumed evaluation run, **otherwise the rseults would be incorrect**.

After this, you can plot the per token loss:

```bash
RESULT_DIR=$SAVE_DIR  
FIGURE_DIR="./figures"  # You can use any other path
python plot_per_token_loss.py \
    --result_dir $RESULT_DIR \
    --figure_dir $FIGURE_DIR
```

You can change `MODELS` in `plot_per_token_loss.py` to specify the set of models for which you want to plot the per-token loss.
