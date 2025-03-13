import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import subprocess

def generate_latex_table(results_dict):
    """
    Generates LaTeX code for a table from a dictionary of results.

    Parameters:
    - results_dict: A dictionary mapping model names to either:
        - float/int values
        - dictionaries of task results

    Returns:
    - A string containing LaTeX code for the table using booktabs.
    """
    # Check if all values are dictionaries
    # Values are dictionaries
    # Collect all task names
    # task_names = set()
    # for tasks in results_dict.values():
        # task_names.update(tasks.keys())
    # task_names = sorted(task_names)
    
    # Initialize data structures for best and second best per task
    task_names = TASK_LIST
    task_best_values = {}
    task_second_best_values = {}
    # For each task, collect all results
    for task in task_names:
        task_results = [tasks.get(task, None) for tasks in results_dict.values()]
        # Remove missing values and duplicates
        valid_results = [r for r in task_results if r is not None]
        unique_results = sorted(set(valid_results), reverse=True)
        # Get best and second best values
        if unique_results:
            task_best_values[task] = unique_results[0]
            if len(unique_results) > 1:
                task_second_best_values[task] = unique_results[1]
            else:
                task_second_best_values[task] = None
        else:
            task_best_values[task] = None
            task_second_best_values[task] = None

    # Begin table
    num_columns = len(task_names) + 1  # +1 for model column
    column_format = 'l' + 'r' * len(task_names)
    table_header = (
        "\\begin{table}\n"
        r"\caption{Evalution results on LongBench. All models have roughly $760$M non-embedding parameters and are trained on roughly $48$B tokens on LongCrawl64. Bold and underlined numbers indicate the best and the second-best results, respectively.}" + "\n"
        r"\label{tab:long-bench}" + "\n"
        "\\begin{center}\n"
        r"\resizebox{\columnwidth}{!}{" + "\n"
        f"\\begin{{tabular}}{{{column_format}}}\n"
        "\\toprule\n"
        r"\multirow{5}{*}{Model} & \multicolumn{3}{c}{Single-Document QA} & \multicolumn{3}{c}{Multi-Document QA}& \multicolumn{3}{c}{Summarization}& \multicolumn{3}{c}{Few-shot Learning}& \multicolumn{2}{c}{Code} \\" + "\n"
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}\cmidrule(lr){14-15}" + "\n"
    )
    # Escape underscores in task names
    task_labels = [TASK_TO_LABEL[task] for task in task_names]
    task_labels = [f"\\rotatebox[origin=c]{{45}}{{{task}}}" for task in task_labels]
    # Header row
    header_row = " & " + " & ".join(task_labels) + " \\\\\n\\midrule\n"
    table_body = ""
    for model, tasks in results_dict.items():
        # Escape underscores in model name
        # model_label = model.replace('_', '\\_')
        model_label = MODEL_TO_LABEL.get(model, model)
        row = [model_label]
        for task in task_names:
            result = tasks.get(task, None)
            if result is None:
                result_str = ''
            else:
                # Format the result based on best and second best
                if result == task_best_values[task]:
                    result_str = f"\\textbf{{{result}}}"
                elif result == task_second_best_values[task]:
                    result_str = f"\\underline{{{result}}}"
                else:
                    result_str = f"{result}"
            row.append(result_str)
        table_body += " & ".join(row) + " \\\\\n"
    table_footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "}\n"
        "\\end{center}\n"
        "\\end{table}"
    )
    latex_table = table_header + header_row + table_body + table_footer
    return latex_table

if __name__ == "__main__":

    RESULT_DIR = Path("./pred")
    MODELS = [
        "fox-pro-760m-longcrawl64-48b",
        "transformer-pro-760m-longcrawl64-48b",
        "fox-llama-760m-longcrawl64-48b",
        "transformer-llama-760m-longcrawl64-48b",
        "mamba2-760m-longcrawl64-48b",
        "hgrn2-760m-longcrawl64-48b",
        "delta_net-760m-longcrawl64-48b",
    ]


    MODEL_TO_LABEL = {
        "fox-pro-760m-longcrawl64-48b": "FoX (Pro)",
        "fox-llama-760m-longcrawl64-48b": "FoX (LLaMA)",
        "transformer-pro-760m-longcrawl64-48b": "Transformer (Pro)",
        "transformer-llama-760m-longcrawl64-48b": "Transformer (LLaMA)",
        "mamba2-760m-longcrawl64-48b": "Mamba-2",
        "hgrn2-760m-longcrawl64-48b": "HGRN2",
        "delta_net-760m-longcrawl64-48b": "DeltaNet",
    }

    TASK_LIST = [
        "narrativeqa",
        "qasper",
        "multifieldqa_en",

        "hotpotqa",
        "2wikimqa",
        "musique",

        "gov_report",
        "qmsum",
        "multi_news",

        "trec",
        "triviaqa",
        "samsum",
        # "passage_count",
        # "passage_retrieval_en",
        "lcc",
        "repobench-p",
    ]
    TASK_TO_LABEL = {
        "narrativeqa": "NarrativeQA",
        "qasper": "Qasper",
        "multifieldqa_en": "MFQA",
        "hotpotqa": "HotpotQA",
        "2wikimqa": "2WikiMQA",
        "musique": "Musique",
        "gov_report": "GovReport",
        "qmsum": "QMSum",

        "multi_news": "MultiNews",
        "trec": "TREC",
        "triviaqa": "TriviaQA",
        # "passage_count": 0.0,
        # "passage_retrieval_en": 0.0,
        "lcc": "LCC",
        "repobench-p": "RepoBench-P",
        "samsum": "SamSum"
    }

    


    result_dict = {}
    for model_name in MODELS:
        path = RESULT_DIR / model_name / "result.json"
        with path.open("r") as f:
            data = json.load(f)
            data = {k: v for k, v in data.items() if k in TASK_TO_LABEL}
            result_dict[model_name] = data

    table = generate_latex_table(result_dict)
    print(table)
    subprocess.run("pbcopy", text=True, input=table)
    print("Copied to clipboard.")
