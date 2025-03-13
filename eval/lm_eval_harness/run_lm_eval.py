from typing import Callable, Dict, Union, Optional, Tuple, NamedTuple, Any, List
import logging
from pathlib import Path
import rich
import rich.syntax

import torch
import os
import os.path as osp
from torch import nn
import colorlog
from datetime import datetime
import jsonlines
import lm_eval
from lm_eval.models.huggingface import HFLM

import json
import pprint
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizerFast, LlamaTokenizer
import forgetting_transformer.tokenizer
import forgetting_transformer.model
import pickle




def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, required=True, choices=["mamba2-760m", "fot-760m", "hgrn2-760m", "delta_net-760m", "transformer-760m", "fot-qk-norm-760m"])
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--device_id', type=int, required=True)
    parser.add_argument('--max_len', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    args = parser.parse_args()

    assert args.model == Path(args.model_path).name, f"Model name '{args.model}' is different from the last component of model path '{args.path}'. You can delete this assertion if you are sure this is correct."
    model_name = args.model
    device_id = args.device_id
    max_len = args.max_len
    batch_size = args.batch_size
    save_dir = Path(args.save_dir) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model_path

    device = torch.device(f"cuda:{device_id}")


    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, add_bos_token=True, clean_up_tokenization_spaces=False)
    assert max_len == 16384, "Just in case. You can delete this."
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)



    # TODO: note that the models are trained with BOS. Therefore, in principle in all
    # evaluation BOS should be added. However, for wikitext perplexity eval except for
    # the first rolling window, no BOS is added. This is fine in our case since our 16k
    # context length covers most wikitext docs. However, if you use a short training
    # context length with BOS, you will need to modify HFLM to implement the correct
    # behavior.
    hflm = HFLM(
        pretrained=model,
        batch_size=batch_size,
        tokenizer=tokenizer,
        max_length=max_len,
        add_bos_token=True,  # This is basically whether to use add_special_tokens
    )

    task_manager = lm_eval.tasks.TaskManager()

    # Setting `task_manager` to the one above is optional and should generally be
    # done
    # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
    # `simple_evaluate` will instantiate its own task_manager if it is set to None
    # here.

    with torch.cuda.device(device):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                results = lm_eval.simple_evaluate( # call simple_evaluate
                    model=hflm,
                    # tasks=["wikitext"], 
                    tasks=["wikitext", "lambada_openai", "piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "boolq", "sciq", "copa", "openbookqa"], 
                    # tasks=["winogrande"], 
                    # tasks=["scrolls_narrativeqa", "scrolls_qasper", "scrolls_quality"], 
                    # tasks=[

                        # "scrolls_govreport",  # 10min for mamba2
                        # "scrolls_qmsum",# 4min for mamba2
                        # "scrolls_summscreenfd", <10min

                        # "scrolls_qasper",

                        # "scrolls_quality",
                        # "scrolls_contractnli",

                        # "scrolls_narrativeqa",
                    # ], 
                    # tasks=["wikitext"], 
                    # tasks=["lambada_openai"], 
                    num_fewshot=0,
                    task_manager=task_manager,
                    device="cuda"
                )
    pprint.pprint(results["results"])
    save_path = save_dir / "results.json"
    with save_path.open("w") as f:
        json.dump(results["results"], f, indent=4)
    print(f"Results saved to {save_path}")
    # import ipdb; ipdb.set_trace()
if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

