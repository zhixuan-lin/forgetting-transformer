# Needle-in-the-Haystack Evaluation


This directory contains the code for the needle in the haystack experiments in the paper. The code is adapted from [the needle test in LongAlign](https://github.com/THUDM/LongAlign/tree/main/Needle_test).

## Usage

First, generate the prompts for the easy and the standard mode:

```bash
python prompt.py --config config-prompt-easy.yaml --exp max_len_32k_easy
python prompt.py --config config-prompt-standard.yaml --exp max_len_32k_standard
```

Then you can run the actual retrieval task. For example, This is how you can evaluate FoX (Pro):

```bash
python pred.py --exp max_len_32k_easy --model "fox-pro-760m-longcrawl64-48b" --model_path "zhixuan-lin/fox-pro-760m-longcrawl64-48b" --device_id 0 
python pred.py --exp max_len_32k_standard --model "fox-pro-760m-longcrawl64-48b" --model_path "zhixuan-lin/fox-pro-760m-longcrawl64-48b" --device_id 0 
```

The results would be saved to `./pred`. After this we need to use `gpt-4o-2024-08-06` to score the retrieval results. This require an OpenAI API key be set in `$API_KEY`. Then you can run the following:

```bash
python eval.py --exp max_len_32k_easy --model fox-pro-760m-longcrawl64-48b --api-key $API_KEY
python eval.py --exp max_len_32k_standard --model fox-pro-760m-longcrawl64-48b --api-key $API_KEY
```

The scores would be saved to `./results`. After this you can visualize the results as follows:

```bash
FIGURE_DIR="./figures"  # You can use any other path
python plot_niah.py --figure_dir=$FIGURE_DIR
```

You can change `MODEL_LIST` in `plot_niah.py` to specify the set of models for which you want to visualize results.

## Citation

If you use this code, consider citing LongAlign:

```
@inproceedings{bai2024longalign,
    title = "{L}ong{A}lign: A Recipe for Long Context Alignment of Large Language Models",
    author = "Bai, Yushi and Lv, Xin and Zhang, Jiajie and He, Yuze and Qi, Ji and Hou, Lei and Tang, Jie and Dong, Yuxiao and Li, Juanzi",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.74",
    doi = "10.18653/v1/2024.findings-emnlp.74",
    pages = "1376--1395",
}
```
