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
    - results_dict: A dictionary mapping algorithm names to either:
        - float/int values
        - dictionaries of metric results

    Returns:
    - A string containing LaTeX code for the table using booktabs.
    """
    # Check if all values are dictionaries
    if all(isinstance(v, dict) for v in results_dict.values()):
        # Values are dictionaries
        # Collect all metric names
        metric_names = set()
        for metrics in results_dict.values():
            metric_names.update(metrics.keys())
        metric_names = sorted(metric_names)
        
        # Initialize data structures for best and second best per metric
        metric_best_values = {}
        metric_second_best_values = {}
        # For each metric, collect all results
        for metric in metric_names:
            metric_results = [metrics.get(metric, None) for metrics in results_dict.values()]
            # Remove missing values and duplicates
            valid_results = [r for r in metric_results if r is not None]
            unique_results = sorted(set(valid_results), reverse=True)
            # Get best and second best values
            if unique_results:
                metric_best_values[metric] = unique_results[0]
                if len(unique_results) > 1:
                    metric_second_best_values[metric] = unique_results[1]
                else:
                    metric_second_best_values[metric] = None
            else:
                metric_best_values[metric] = None
                metric_second_best_values[metric] = None

        # Begin table
        num_columns = len(metric_names) + 1  # +1 for Algorithm column
        column_format = 'l' + 'r' * len(metric_names)
        table_header = (
            "\\begin{table}[ht]\n"
            "\\centering\n"
            f"\\begin{{tabular}}{{{column_format}}}\n"
            "\\toprule\n"
        )
        # Escape underscores in metric names
        escaped_metric_names = [metric.replace('_', '\\_') for metric in metric_names]
        # Header row
        header_row = "Algorithm & " + " & ".join(escaped_metric_names) + " \\\\\n\\midrule\n"
        table_body = ""
        for algorithm, metrics in results_dict.items():
            # Escape underscores in algorithm name
            escaped_algorithm = algorithm.replace('_', '\\_')
            row = [escaped_algorithm]
            for metric in metric_names:
                result = metrics.get(metric, None)
                if result is None:
                    result_str = ''
                else:
                    # Format the result based on best and second best
                    if result == metric_best_values[metric]:
                        result_str = f"\\textbf{{{result}}}"
                    elif result == metric_second_best_values[metric]:
                        result_str = f"\\underline{{{result}}}"
                    else:
                        result_str = f"{result}"
                row.append(result_str)
            table_body += " & ".join(row) + " \\\\\n"
        table_footer = (
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "\\caption{Your caption here}\n"
            "\\label{tab:your_label}\n"
            "\\end{table}"
        )
        latex_table = table_header + header_row + table_body + table_footer
        return latex_table
    elif all(isinstance(v, (int, float)) for v in results_dict.values()):
        # Values are numeric
        # Collect all results
        all_results = list(results_dict.values())
        # Remove duplicates
        unique_results = sorted(set(all_results), reverse=True)
        # Get best and second best values
        best_value = unique_results[0]
        second_best_value = unique_results[1] if len(unique_results) > 1 else None
        # Start building the table
        table_header = (
            "\\begin{table}[ht]\n"
            "\\centering\n"
            "\\begin{tabular}{lr}\n"
            "\\toprule\n"
            "Algorithm & Result \\\\\n"
            "\\midrule\n"
        )
        table_body = ""
        for algorithm, result in results_dict.items():
            # Escape underscores in algorithm name
            escaped_algorithm = algorithm.replace('_', '\\_')
            # Format the result
            if result == best_value:
                result_str = f"\\textbf{{{result}}}"
            elif result == second_best_value:
                result_str = f"\\underline{{{result}}}"
            else:
                result_str = f"{result}"
            table_body += f"{escaped_algorithm} & {result_str} \\\\\n"
        table_footer = (
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "\\caption{Your caption here}\n"
            "\\label{tab:your_label}\n"
            "\\end{table}"
        )
        latex_table = table_header + table_body + table_footer
        return latex_table
    else:
        raise ValueError("Values in the dictionary must be all numeric or all dictionaries.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    args = parser.parse_args()

    RESULT_DIR = Path(args.result_dir)

    MODELS = [
        "fox-pro-760m-longcrawl64-48b",
        # "transformer-pro-760m-longcrawl64-48b",
        # "fox-llama-760m-longcrawl64-48b",
        # "transformer-llama-760m-longcrawl64-48b",
        # "mamba2-760m-longcrawl64-48b",
        # "hgrn2-760m-longcrawl64-48b",
        # "delta_net-760m-longcrawl64-48b",
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

    TASK_TO_METRIC = {
        "lambada_openai_perp": "perplexity,none",
        "wikitext": "word_perplexity,none",

        "arc_challenge": "acc_norm,none",
        "hellaswag": "acc_norm,none",
        "openbookqa": "acc_norm,none",

        "arc_easy": "acc,none",
        "boolq": "acc,none",
        "copa": "acc,none",
        "lambada_openai": "acc,none",
        "piqa": "acc,none",
        "sciq": "acc,none",
        "winogrande":"acc,none" 
    }

    METRIC_TO_LABEL = {
        "perplexity,none": r"ppl$\downarrow$",
        "word_perplexity,none": r"ppl$\downarrow$",
        "acc,none": r"acc$\uparrow$",
        "acc_norm,none": r"acc-n$\uparrow$",
    }

    TASK_TO_LABEL = {
        "lambada_openai_perp": "LMB.",
        "wikitext": "Wiki.",

        "arc_challenge": "ARC-c",
        "hellaswag": "Hella.",
        "openbookqa": "OBQA",

        "arc_easy": "ARC-e",
        "boolq": "BoolQ",
        "copa": "COPA",
        "lambada_openai": "LMB.",
        "piqa": "PIQA",
        "sciq": "SciQA",
        "winogrande":"Wino." 
    }

    AVG_LIST = [
        "arc_challenge",
        "hellaswag",
        "openbookqa",

        "arc_easy",
        "boolq",
        "copa",
        "lambada_openai",
        "piqa",
        "sciq",
        "winogrande",
    ]

    TASK_LIST = [
        "wikitext",
        "lambada_openai_perp",

        "lambada_openai",
        "piqa",
        "hellaswag",
        "winogrande",
        "arc_easy",
        "arc_challenge",
        "copa",
        "openbookqa",

        "sciq",
        "boolq",

    ]


    result_dict = {}
    for model_name in MODELS:
        path = RESULT_DIR / model_name / "results.json"
        with path.open("r") as f:
            data = json.load(f)
            metric_dict = {}
            for task, metric in TASK_TO_METRIC.items():
                if task == "lambada_openai_perp":
                    item = data["lambada_openai"][metric]
                else:
                    item = data[task][metric]
                if "acc" in metric:
                    item *= 100
                metric_dict[task] = item
            result_dict[model_name] = metric_dict

    best_dict = {}
    second_best_dict = {}

    for task in TASK_LIST:
        value_list = set(result_dict[model][task] for model in MODELS)
        assert len(value_list) >= 1
        reverse = (task in ["wikitext", "lambada_openai_perp"])
        value_list = sorted(value_list, reverse=reverse)
        best_dict[task] = value_list[-1]
        second_best_dict[task] = value_list[-2] if len(value_list) > 1 else None


    avg_dict = {}
    for model in MODELS:
        avg_list = []
        for task in TASK_LIST:
            if task in AVG_LIST:
                avg_list.append(result_dict[model][task])
        # print(avg_list)
        avg_dict[model] = np.mean(avg_list)
    avg_list = sorted(set(avg_dict.values()))
    avg_best = avg_list[-1]
    avg_second_best = avg_list[-2] if len(avg_list) > 1 else None

    toprule = r"\toprule"
    task_row = [r"\textbf{Model}"] + [r"\textbf{{{}}}".format(TASK_TO_LABEL[task]) for task in TASK_LIST] + [r"\textbf{Avg}"]
    task_row = ' & '.join(task_row)
    task_row += " \\\\"
    metric_row = [""] + [METRIC_TO_LABEL[TASK_TO_METRIC[task]] for task in TASK_LIST] + [r"$\uparrow$"]
    metric_row = ' & '.join(metric_row)
    metric_row += " \\\\"
    midrule = r"\midrule"

    rows = [toprule, task_row, metric_row, midrule]

    for model in MODELS:
        model_row = [MODEL_TO_LABEL.get(model, model)]

        avg_metric_list = []
        for task in TASK_LIST:
            # if task == "lambada_openai_perp":
                # task = "lambada_openai"
            value = result_dict[model][task]
            metric_str = f"{value:.2f}"
            if value == best_dict[task]:
                metric_str = f"\\textbf{{{metric_str}}}"
            elif value == second_best_dict[task]:
                metric_str = f"\\underline{{{metric_str}}}"
            model_row += [metric_str]
            # if task in AVG_LIST:
                # avg_metric_list.append(result_dict[model][task])

        # model_row += [f"{np.mean(avg_metric_list):.2f}"]
        avg_value = avg_dict[model]
        avg_str = f"{avg_value:.2f}"
        if avg_value == avg_best:
            avg_str = f"\\textbf{{{avg_str}}}"
        elif avg_value == avg_second_best:
            avg_str = f"\\underline{{{avg_str}}}"
        model_row += [avg_str]

        model_row = " & ".join(model_row)
        model_row += " \\\\"
        rows += [model_row]

    rows += [r"\bottomrule"]

    column_format = 'l|' + 'cc|' + 'c' * (len(TASK_LIST) - 2) + '|c'
    tabular_content = "\n".join(rows)
    table=(
r"""
\begin{table}
\caption{Evaluation results on LM-eval-harness.}
\label{table:lm-eval}
\begin{center}
\resizebox{\columnwidth}{!}{
"""

f"""
\\begin{{tabular}}{{{column_format}}}
{tabular_content}
"""

r"""
\end{tabular}
}
\end{center}
\end{table}
"""
    )


    # table = generate_latex_table(result_dict)
    print(table)
    # subprocess.run("pbcopy", text=True, input=table)
    # print("Copied to clipboard.")
