import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import yaml
import os
import glob
import json
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
import tqdm
import argparse
# from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
# replace_llama_attn_with_flash_attn()
import forgetting_transformer.model.register_all
import forgetting_transformer.tokenizer

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, required=True, choices=["llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k", "mamba2-760m", "fot-760m", "hgrn2-760m", "delta_net-760m", "transformer-760m", "fot-qk-norm-760m"])
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--device_id', type=int, required=True)
    return parser.parse_args(args)

def pred(model_name, model, tokenizer, input_data, device, max_new_tokens=1024, temperature=0.1):
    # prompt = input_data[0]['content']+'\n'+input_data[1]['content']
    prompt = input_data[1]['content']
    history = []
    # if "internlm" in model_name or "chatglm" in model_name or "longalign-6b" in model_name:
    #     response, history = model.chat(tokenizer, prompt, history=history, max_new_tokens=max_new_tokens, temperature=temperature)
    #     return response
    # elif "longalign-7b" in model_name or "longalign-13b" in model_name:
    #     if history == []:
    #         prompt = f"[INST]{prompt}[/INST]"
    #     else:
    #         prompt = history+"\n\n"+f"[INST]{prompt}[/INST]"
    # elif "mistral" in model_name or "mixtral" in model_name:
    #     if history == []:
    #         prompt = f"<s>[INST] {prompt} [/INST]"
    #     else:
    #         prompt = history+f"</s> [INST] {prompt} [/INST]"
    # elif "longchat" in model_name or "vicuna" in model_name:
    #     from fastchat.model import get_conversation_template
    #     conv = get_conversation_template("vicuna")
    #     conv.append_message(conv.roles[0], prompt)
    #     conv.append_message(conv.roles[1], None)
    #     prompt = conv.get_prompt()
    input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
    context_length = input.input_ids.shape[-1]
    with torch.cuda.device(device):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model.generate(
                **input,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                temperature=temperature,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
    return pred.strip()

def load_model_and_tokenizer(path, device):
    valid_path = path.lower()
    if "longchat" in valid_path or "vicuna" in valid_path:
        from fastchat.model import load_model
        model, _ = load_model(path, device='cpu', num_gpus=0, load_8bit=False, cpu_offloading=False, debug=False)
        model = model.to(device)
        model = model.bfloat16()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    elif "mistral" in valid_path or "mixtral" in valid_path:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, use_flash_attention_2=True, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
        model.generation_config = GenerationConfig.from_pretrained(path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, add_bos_token=True, clean_up_tokenization_spaces=False)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).to(device)
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    args = parse_args()
    with open('config-pred.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    assert args.model == Path(args.model_path).name, f"Model name '{args.model}' is different from the last component of model path '{args.path}'. You can delete this assertion if you are sure this is correct."
    model_provider = config['model']['model_provider']
    # model_name = config['model']['model_name']
    # model2path = json.load(open("config-model2path.json", "r"))
    # model2path[args.model] = args.model_path
    print(f"Evaluating {args.model} from {args.model_path}")
    # model_name = model2path[args.model]
    model_name = args.model_path
    # try:
    # except KeyError:
    #     model_name = f"{args.prefix}/{args.model}"
    prompt_dir = str(Path(config['prompt_dir']) / args.exp)
    save_dir = str(Path(config['save_dir']) / args.exp / args.model)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(f'cuda:{args.device_id}')

    model, tokenizer = load_model_and_tokenizer(model_name, device)

    for filename in tqdm.tqdm(sorted(glob.glob(f'{prompt_dir}/{model_provider}_*_prompts.json'))):
        basename = os.path.basename(filename)
        newname = basename.replace('.json', '.txt').replace('_prompts', '')
        result_path = Path(save_dir) / newname
        sentinel_path = Path(f"{result_path}.done")
        if sentinel_path.exists():
            assert result_path.exists()
            print(f"{sentinel_path} exists. Skipping {filename}")
            continue

        with open(filename, 'r') as f:
            prompts = json.load(f)

        result = pred(model_name.lower(), model, tokenizer, prompts, device)

        with result_path.open("w") as f:
            f.write(result)
        with sentinel_path.open("w") as f:
            pass


