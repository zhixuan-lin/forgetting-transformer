import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

import seaborn
import matplotlib.pyplot as plt
import numpy as np
import argparse

def smooth_array(arr, window_size):
    # Initialize an empty array to store the smoothed values
    smoothed_arr = np.zeros_like(arr)
    
    # Half of the window size to know how many elements to take left and right
    half_window = window_size // 2
    
    # Iterate through each element in the array
    for i in range(len(arr)):
        # Calculate the left and right bounds of the window, making sure not to go out of bounds
        left = max(0, i - half_window)
        right = min(len(arr), i + half_window + 1)
        
        # Compute the average of the window
        smoothed_arr[i] = np.mean(arr[left:right])
    
    return smoothed_arr

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

if __name__ == "__main__":
    setup_style()
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--figure_dir", type=str, required=True)
    args = parser.parse_args()

    MODELS = [
        "fox-pro-760m-longcrawl64-48b",
        # "transformer-pro-760m-longcrawl64-48b",
        # "fox-llama-760m-longcrawl64-48b",
        # "transformer-llama-760m-longcrawl64-48b",
        # "mamba2-760m-longcrawl64-48b",
        # "hgrn2-760m-longcrawl64-48b",
        # "delta_net-760m-longcrawl64-48b",
    ]
    RESULT_DIR = Path(args.result_dir)
    FIGURE_DIR = Path(args.figure_dir)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    MODEL_TO_LABEL = {
        "fox-pro-760m-longcrawl64-48b": "FoX (Pro)",
        "fox-llama-760m-longcrawl64-48b": "FoX (LLaMA)",
        "transformer-pro-760m-longcrawl64-48b": "Transformer (Pro)",
        "transformer-llama-760m-longcrawl64-48b": "Transformer (LLaMA)",
        "mamba2-760m-longcrawl64-48b": "Mamba-2",
        "hgrn2-760m-longcrawl64-48b": "HGRN2",
        "delta_net-760m-longcrawl64-48b": "DeltaNet",
    }
    MODEL_TO_COLOR = {
        "fox-pro-760m-longcrawl64-48b": "C0",
        "transformer-pro-760m-longcrawl64-48b": "C1",
        "fox-llama-760m-longcrawl64-48b": "C2",
        "transformer-llama-760m-longcrawl64-48b": "C3",
        "mamba2-760m-longcrawl64-48b": "C4",
        "hgrn2-760m-longcrawl64-48b": "C8",
        "delta_net-760m-longcrawl64-48b": "C9",
    }

    plot_data = {}
    for model in MODELS:
        model_dir = RESULT_DIR / model
        # path_list = sorted(model_dir.glob("*.pt"))
        # path = path_list[-1]
        # with path.open("rb") as f:
        #     state = torch.load(f, map_location="cpu")
        # loss_per_token = (state["total_loss"] / state["seq_count"]).numpy()

        path_list = sorted(model_dir.glob("*.npz"))
        assert len(path_list) == 1, model_dir
        path = path_list[-1]
        data = np.load(path)
        # if "eval_tokens" in data:
            # assert data["eval_tokens"] == 2 * 2 ** 30, "Just in case. You can delete this."
        if "eval_len" in data:
            assert data["eval_len"] == 65536, "Just in case. You can delete this"
        loss_per_token = data["val/loss_per_token"]
        plot_data[model] = loss_per_token

    # fig, ax = plt.subplots(1, 1)
    # fig, ax = plt.subplots(1, 1, figsize=(6.4, 5.2))
    fig, axes = plt.subplots(1, 2, figsize=(6.4 * 2 + 2, 4.8), gridspec_kw={"wspace": 0.2})
    ax = axes[0]
    T_min = 128
    T_max = 65536
    for model, data in plot_data.items():
        data = smooth_array(data, window_size=101)
        label = MODEL_TO_LABEL.get(model, model)
        color = MODEL_TO_COLOR.get(model, None)
        ax.plot(np.arange(T_min, T_max), data[T_min:T_max], label=label, linewidth=2.5, color=color)

    # Plot the vertical line
    TRAIN_LEN = 16384
    ax.axvline(x=TRAIN_LEN, color='black', linestyle='--', linewidth=2.5)

    # Annotate the vertical line
    # ax.set_yticks(np.arange(1.65, 2.05, 0.05))
    ax.set_ylim(1.47, 1.67)
    ax.set_xticks([8192 * (i) for i in range(9)], ["0" if n == 0 else f"{8 * (n)}k" for n in range(9)])
    ax.set_xlabel("Token index $i$")
    ax.set_ylabel("Loss $L(i)$")
    ax = axes[1]
    T_min = 128
    T_max = 65536
    for model, data in plot_data.items():
        # data = smooth_array(data, window_size=101)
        data = np.cumsum(data) / (np.arange(len(data)) + 1)
        data = np.exp(data)
        label = MODEL_TO_LABEL.get(model, model)
        color = MODEL_TO_COLOR.get(model, None)
        ax.plot(np.arange(T_min, T_max), data[T_min:T_max], label=label, linewidth=2.5, color=color)

    # Plot the vertical line
    TRAIN_LEN = 16384
    ax.axvline(x=TRAIN_LEN, color='black', linestyle='--', linewidth=2.5)

    # Annotate the vertical line
    # ax.annotate('Training Context Length', xy=(TRAIN_LEN, 6.5), xytext=(TRAIN_LEN+10000, 6.5),
                 # arrowprops=dict(facecolor='black', arrowstyle='->'),
                 # fontsize=12, color='black')
    ax.set_ylim(4.4, 5.5)
    # ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    # fig.legend(, loc='upper center')

    leg = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.00), ncols=4, fancybox=True, shadow=True)
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    ax.set_xticks([8192 * (i) for i in range(9)], ["0" if n == 0 else f"{8 * (n)}k" for n in range(9)])
    ax.set_xlabel("Validation context length $l$")
    ax.set_ylabel("Perplexity $P(l)$")
    # ax.set_xscale("log")

    # base_name = "main_perplexity_64k_line"
    base_name = "main_loss_64k_line"
    png_path = FIGURE_DIR / f"{base_name}.png"
    pdf_path = FIGURE_DIR / f"{base_name}.pdf"
    # fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved to {png_path}")

