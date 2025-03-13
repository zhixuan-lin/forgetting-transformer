# LongBench

This directory contains the code for evaluation on LongBench. The code is adapted from the original [LongBench-v1 repository](https://github.com/THUDM/LongBench/blob/main/LongBench/README.md).

## Usage

Usage example:

```bash
python pred.py --model "fox-pro-760m-longcrawl64-48b" --model_path "zhixuan-lin/fox-pro-760m-longcrawl64-48b" --max_length 15500
python eval.py --model "fox-pro-760m-longcrawl64-48b"
```

After you run these, results will be saved to `./pred`. You can create a latex table using:


```bash
python table_longbench.py
```

You can change `MODELS` in `table_longbench.py` to specify which models you want to include in the table.


Note that we observe the evaluation results to be non-deterministic, likely due to GPU non-determinism. Therefore the results you obtain may not exactly match those reported in the paper. However, the difference should be small.

## Citation

If you use this code, consider citing LongBench:
```
@article{bai2023longbench,
  title={LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding},
  author={Bai, Yushi and Lv, Xin and Zhang, Jiajie and Lyu, Hongchang and Tang, Jiankai and Huang, Zhidian and Du, Zhengxiao and Liu, Xiao and Zeng, Aohan and Hou, Lei and Dong, Yuxiao and Tang, Jie and Li, Juanzi},
  journal={arXiv preprint arXiv:2308.14508},
  year={2023}
}
```
When citing LongBench, please kindly consider citing the original dataset papers. The relevant citation information is listed [here](refs/ref.bib).
