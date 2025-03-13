import yaml
import os
import json
import re

import time
import requests
import argparse
import tqdm
from pathlib import Path


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, required=True, choices=["llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k", "mamba2-760m", "fot-760m", "hgrn2-760m", "delta_net-760m", "transformer-760m"])
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--api-key', type=str, required=True)
    return parser.parse_args(args)

def pred_openai(model_name, msg, api_key):
    tries = 0
    while tries < 5:
        tries += 1
        try:
            headers = {
                'Authorization': f"Bearer {api_key}"
            }
            resp = requests.post("https://api.openai.com/v1/chat/completions", json = {
                "model": model_name,
                "messages": msg,
                "temperature": 0.
            }, headers=headers, timeout=120)
            if resp.status_code != 200:
                raise Exception(resp.text)
            resp = resp.json()
            break
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            if "maximum context length" in str(e):
                raise e
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            time.sleep(1)
    else:
        print("Max tries. Failed.")
        return
    
    return resp["choices"][0]["message"]["content"]


# USER_TEMPLATE = '''[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. {criteria}[Ground truth]\n{reference}\nBegin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n[Question]\n{input}\n\n[The Start of Assistant\'s Answer]\n{prediction}\n[The End of Assistant\'s Answer]'''
USER_TEMPLATE = '''[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by small language model to a question displayed below. {criteria}[Ground truth]\n{reference}\nRemember that the model is small and not instruction-tuned so it may output additional garbage content and have strange formatting. These should absolutely not affect your rating as long as the ground truth is included in the model's response. Only give your rating based on the relevant (if any) part of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]". Some bad models might give empty response in which case you should give the lowest score.\n\n[Question]\n{input}\n\n[The Start of the Model\'s Answer]\n{prediction}\n[The End of the Model\'s Answer]'''
SYSTEM_TEMPLATE = 'You are a helpful assistant.'
CRITERIA = {
    "accuracy": """
    Score 1: The answer is completely unrelated to the reference.
    Score 3: The answer has minor relevance but does not align with the reference.
    Score 5: The answer has moderate relevance but contains inaccuracies.
    Score 7: The answer aligns with the reference but has minor omissions.
    Score 10: The answer is completely accurate and aligns perfectly with the reference.
    Only respond with a numberical score
    """
}

def get_criteria():
    cri = 'For this evaluation, you should primarily consider the following criteria:\n'
    for key, value in CRITERIA.items():
        cri += f'{key}: {value}\n'

    return cri

def get_user_template(input, prediction, reference, criteria):
    return USER_TEMPLATE.format(
        input=input,
        prediction=prediction,
        reference=reference,
        criteria=criteria
    )

if __name__ == '__main__':
    args = parse_args()
    with open('config-eval.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    
    api_key = args.api_key
    pred_dir = str(Path(config['pred_dir']) / args.exp / args.model)
    save_dir = str(Path(config['save_dir']) / args.exp / args.model)
    model_name = config['model']['model_name']
    model_provider = config['model']['model_provider']
    criteria = get_criteria()
    reference = config['prompt']['needle']
    input = config['prompt']['retrieval_question']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    result_dict = {}

    for filename in tqdm.tqdm(sorted(x for x in os.listdir(pred_dir) if x.endswith('.txt'))):
        time.sleep(2)
        if not filename.endswith('.txt'):
            continue

        sentinel_path = Path(pred_dir) / f"{filename}.done"
        if not sentinel_path.exists():
            print(f"Sentinal {sentinel_path} not found. Skipping {filename}")
            continue

        with open(f'{pred_dir}/{filename}', 'r') as f:
            data = f.read().strip()

        prediction = data
        user_template = get_user_template(input, prediction, reference, criteria)

        if model_provider == 'OpenAI':
            msg = [{
                    "role": "system",
                    "content": SYSTEM_TEMPLATE
                }, {
                    "role": "user",
                    "content": user_template
                }
            ]
            result = pred_openai(model_name, msg, api_key)
            
        else:
            raise NotImplementedError(f'Not implemented model provider: {model_provider}')
        
        pattern = r"\[\[(\d+)\]\]"
        match = re.search(pattern, result)
        score = int(match.group(1)) if match else None

        result_dict[filename.replace('.txt', '')] = {
            'prediction': prediction,
            'score': score
        }

    with open(f'{save_dir}/{model_provider}_{model_name}_eval.json', 'w') as f:
        json.dump(result_dict, f, indent=4)

