import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
import os
import re
import glob
import argparse
from pathlib import Path
import seaborn

def setup_style():
    SMALL_SIZE = 18
    MEDIUM_SIZE = 18
    LARGE_SIZE = 18

    # General configuration: https://matplotlib.org/stable/api/matplotlib_configuration_api.html#default-values-and-styling
    # Default rcParams: https://matplotlib.org/stable/tutorials/introductory/customizing.html#the-default-matplotlibrc-file
    # plt.rc vs plt.rcParams: TLDR: plt.rc updates plt.rcParams: https://stackoverflow.com/questions/67148006/what-is-the-difference-between-matplotlib-rc-and-matplotlib-rcparams-and-wh

    # Style-sheets: https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
    # plt.style.use('bmh')
    
    seaborn.set_style("whitegrid", {'grid.linestyle': '--'})
    seaborn.set_palette('colorblind')
    # plt.style.use('tableau-colorblind10')
    # You don't need latex: https://matplotlib.org/3.5.0/tutorials/text/mathtext.html
    # plt.rc('text', usetex=True)
    # Font: https://matplotlib.org/stable/tutorials/text/text_props.html#default-font
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.sans-serif'] = ['Dejavu Sans', 'Computer Modern Sans Serif', 'Helvetica',  'sans-serif']  # first one will be used. Dejavu is just the default
    plt.rcParams['font.serif'] = ['Liberation Serif', 'Computer Modern Roman', 'Times New Roman', 'serif']
    plt.rcParams['font.weight'] = 'normal'  # or light, bold
    plt.rcParams['font.size'] = SMALL_SIZE
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['axes.titlesize'] = MEDIUM_SIZE  # fontsize of the axes title
    plt.rcParams['axes.labelsize'] = SMALL_SIZE  # fontsize of the x and y labels
    plt.rcParams["axes.grid"] = True
    plt.rcParams['axes.formatter.useoffset'] = False  # Don't offset. See https://matplotlib.org/stable/api/ticker_api.html#matplotlib.ticker.ScalarFormatter.set_useOffset
    plt.rcParams['figure.dpi'] = 300  # default is 100

    plt.rcParams['xtick.labelsize'] = SMALL_SIZE  # fontsize of the tick labels
    plt.rcParams['ytick.labelsize'] = SMALL_SIZE  # fontsize of the tick labels

    plt.rcParams['legend.fontsize'] = SMALL_SIZE  # legend fontsize
    plt.rcParams['figure.titlesize'] = MEDIUM_SIZE  # fontsize of the figure title
    plt.rcParams['lines.linewidth'] = 1.5

    # Color cycle
    # Specifying colors: https://matplotlib.org/3.1.0/tutorials/colors/colors.html
    # Specifying line styles: https://matplotlib.org/3.5.0/gallery/lines_bars_and_markers/linestyles.html
    # Cycler: https://matplotlib.org/stable/tutorials/intermediate/color_cycle.html
    from cycler import cycler
    # prop_cycle = cycler(color=['r', 'g', 'b', 'y']) + cycler(linestyle=['-', '--', ':', '-.'])
    # prop_cycle = cycler(color=['tab:red', 'tab:green', 'tab:blue']) * cycler(linestyle=['-', '--', ':', '-.'])
    # CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
    #               '#f781bf', '#a65628', '#984ea3',
    #               '#999999', '#e41a1c', '#dede00']
    prop_cycle = plt.rcParams['axes.prop_cycle']  # The default one
    plt.rcParams['axes.prop_cycle'] = prop_cycle

if __name__ == '__main__':
    setup_style()
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure_dir", type=str, required=True)
    args = parser.parse_args()
    EXP_LIST = ["max_len_32k_easy", "max_len_32k_standard"]
    # EXP_LIST = ["debug"]
    MODEL_LIST = [
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
    EXP_TO_LABEL = {
        "easy_max_len_32k": "Easy",
        "standard_max_len_32k": "Standard"
    }

    # args = parse_args()
    RESULT_ROOT = Path("./results")
    for exp in EXP_LIST:
        FIGURE_DIR = Path(args.figure_dir) / exp
        if not FIGURE_DIR.exists():
            FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        for model in MODEL_LIST:
            print(exp, model)
            # exp = exp
            # model = model
            # Using glob to find all json files in the directory
            result_dir = RESULT_ROOT / exp / model
            assert result_dir.is_dir(), result_dir
            # json_files = glob.glob(f"results/*.json")
            json_files = list(result_dir.glob("*.json"))
            # print(json_files)

            # vis_dir = Path("vis") / exp / model

            # if not vis_dir.exists():
                # vis_dir.mkdir(parents=True, exist_ok=True)

            # if not os.path.exists('vis'):
                # os.makedirs('vis')

            # Iterating through each file and extract the 3 columns we need
            for file in json_files:
                file = str(file)
                # List to hold the data
                data = []

                with open(file, 'r') as f:
                    json_data = json.load(f)
                    
                    for k in json_data:
                        pattern = r"_len_(\d+)_"
                        match = re.search(pattern, k)
                        context_length = int(match.group(1)) if match else None

                        pattern = r"depth_(\d+)"
                        match = re.search(pattern, k)
                        document_depth = eval(match.group(1))/100 if match else None

                        score = json_data[k]['score']

                        # Appending to the list
                        data.append({
                            "Document Depth": document_depth,
                            "Context Length": context_length,
                            "Score": score
                        })

                # Creating a DataFrame
                df = pd.DataFrame(data)

                pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
                pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score") # This will turn into a proper pivot
                
                # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
                cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

                # Create the heatmap with better aesthetics
                # plt.figure(figsize=(10, 6))  # Can adjust these dimensions as needed
                plt.figure(figsize=(6, 6))  # Can adjust these dimensions as needed
                sns.heatmap(
                    pivot_table,
                    # annot=True,
                    fmt="g",
                    cmap=cmap,
                    cbar=False,
                    cbar_kws={'label': 'Score'},
                    vmin=1,
                    vmax=10,
                    linewidths=0.5
                )
                if "32k" in exp:
                    # TRAIN_LEN = 16384
                    plt.axvline(x=5.5, color='black', linestyle='--', linewidth=4)
                    # plt.annotate('Training Context Length', xy=(5.5, 5.5), xytext=(6.0, 5.5),
                                 # arrowprops=dict(facecolor='black', arrowstyle='->'),
                                 # fontsize=20, color='black')
                if "64k" in exp:
                    # TRAIN_LEN = 16384
                    plt.axvline(x=3.0, color='black', linestyle='--', linewidth=4)
                    # plt.annotate('Training Context Length', xy=(5.5, 5.5), xytext=(6.0, 5.5),
                                 # arrowprops=dict(facecolor='black', arrowstyle='->'),
                                 # fontsize=20, color='black')


                # More aesthetics
                # plt.title(f'Pressure Testing\nFact Retrieval Across Context Lengths ("Needle In A HayStack")')  # Adds a title
                model_label = MODEL_TO_LABEL.get(model, model)
                exp_label = EXP_TO_LABEL.get(exp, exp)
                title = f"{model_label}, {exp_label}"
                plt.title(title, fontsize=25)  # Adds a title
                # plt.xlabel('Token Limit', fontsize=25)  # X-axis label
                plt.xlabel('Document Length', fontsize=25)  # X-axis label
                plt.ylabel('Depth Percent', fontsize=25)  # Y-axis label
                plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
                plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
                plt.tight_layout()  # Fits everything neatly into the figure area

                base_name = f"needle_{exp}_{model}"
                png_path = FIGURE_DIR / f"{base_name}.png"
                pdf_path = FIGURE_DIR / f"{base_name}.pdf"
                plt.savefig(png_path, bbox_inches="tight")
                plt.savefig(pdf_path, bbox_inches="tight")
                print(f"Saved to {png_path}")
                # plt.savefig(f"{vis_dir}/{file.split('/')[-1].replace('.json', '')}.png")

