# Language Model Evaluation Harness

This directory contains the code for evaluating trained models on several tasks from [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness).

## Usage

Example usage:

```bash
export SAVE_DIR="./results"  # You can use any other path
python run_lm_eval.py \
    --model "fox-pro-760m-longcrawl64-48b" \
    --model_path "zhixuan-lin/fox-pro-760m-longcrawl64-48b" \
    --device_id 0 \
    --max_len 16384 \
    --batch_size 16 \
    --save_dir $SAVE_DIR
```

After you've got the results, you can generate a latex table:

```bash
python table_lm_eval.py --result_dir $SAVE_DIR
```

You can change the `MODELS` list in `table_lm_eval.py` to specify what models to include in your table.

Note that we observe the evaluation results to be non-deterministic, likely due to GPU non-determinism. Therefore the results you obtain may not exactly match those reported in the paper. However, the difference should be small.

## Citation

If you use this code, consider citing Language Evaluation Harness:

```
@misc{eval-harness,
  author       = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and Black, Sid and DiPofi, Anthony and Foster, Charles and Golding, Laurence and Hsu, Jeffrey and Le Noac'h, Alain and Li, Haonan and McDonell, Kyle and Muennighoff, Niklas and Ociepa, Chris and Phang, Jason and Reynolds, Laria and Schoelkopf, Hailey and Skowron, Aviya and Sutawika, Lintang and Tang, Eric and Thite, Anish and Wang, Ben and Wang, Kevin and Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = 07,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v0.4.3},
  doi          = {10.5281/zenodo.12608602},
  url          = {https://zenodo.org/records/12608602}
}
```
