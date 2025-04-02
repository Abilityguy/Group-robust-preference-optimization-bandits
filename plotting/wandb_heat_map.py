import wandb
import re
import ast
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import os


def extract_parameters(run_name, config):
    """Extracts relevant parameters from the run name."""
    match = re.match(r"([^,]+),([^,]+),([^,]+),([^,]+),(\[[\d.,-]+\]),(\[[\d.,-]+\]),([^,]+),IS=(True|False)_\d+", run_name)
    ipo_grad_type = config['ipo_grad_type']
    
    if match:
        algorithm = match.group(1)
        noise_ratio_str = match.group(5)
        weights_str = match.group(6)
        importance_sampling_str = match.group(8)

        noise_ratio = ast.literal_eval(noise_ratio_str)[0]
        data_size = ast.literal_eval(weights_str)[2]
        importance_sampling = importance_sampling_str == "True"

        return algorithm, noise_ratio, data_size, importance_sampling, ipo_grad_type
    else:
        return None, None, None, None, None
    

def fetch_and_process_data(project_name):
    """Fetches and processes data from wandb."""
    api = wandb.Api()
    runs = api.runs(project_name)
    data = {}
    for run in tqdm(runs):
        if run.state == "finished":
            config = run.config
            algorithm, noise_ratio, data_size, importance_sampling, ipo_grad_type = extract_parameters(run.name, config)
            if algorithm is not None:
                max_val_grp_loss = run.history(keys=["max_val_grp_loss"])["max_val_grp_loss"].dropna()
                if len(max_val_grp_loss) > 0:
                    last_worst_group_loss = max_val_grp_loss.iloc[-1]
                    key = (algorithm, ipo_grad_type, importance_sampling, noise_ratio, data_size)
                    if key not in data:
                        data[key] = []
                    data[key].append(last_worst_group_loss)

    # Average over seeds
    averaged_data = {}
    for key, values in data.items():
        assert len(values) == 10, f"len(values)={len(values)} is not equal to 10"
        averaged_data[key] = np.mean(values)

    return averaged_data


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs, aspect='auto')
    im.set_clim(vmin=0.1, vmax=0.7)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(range(data.shape[1]), labels=col_labels,
                  rotation=-30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on bottom.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        # threshold = im.norm(data.max())/2.
        thresold = im.norm(0.7)/2

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def main():
    project_name = "noisy_exp_paper_2"  # Replace with your WandB project name
    save_dir = "heatmaps" # directory to save the heatmaps.
    os.makedirs(save_dir, exist_ok=True) #create directory if it doesn't exist.

    data = fetch_and_process_data(project_name)

    noise_ratios = sorted(list(set([key[3] for key in data.keys()])), reverse=True) # To ensure noise ratios increase as we go away from origin
    data_sizes = sorted(list(set([key[4] for key in data.keys()])))

    title_map = {
        ('rdpo', 'justdpo', True): 'DPO w/ IS',
        ('rdpo', 'justdpo', False): 'Group Robust DPO',
        ('rdpo', 'noisy_dpo', False): 'Noise Robust, Group Robust DPO',
        ('dpo', 'justdpo', False): 'DPO',
        ('cgd', 'justdpo', False): 'CGD',
    }

    cmap_list = ['Wistia', 'YlGn', 'PuOr', 'magma_r']
    for cmap_val in cmap_list:
        for title_key in title_map.keys():
            filtered_data = np.zeros((len(noise_ratios), len(data_sizes)))
                    
            for i in range(len(noise_ratios)):
                for j in range(len(data_sizes)):
                    key = (*title_key, noise_ratios[i], data_sizes[j])
                    filtered_data[i, j] = data[key]

            fig, ax = plt.subplots()
            
            im, cbar = heatmap(
                filtered_data,
                noise_ratios,
                data_sizes,
                ax=ax,
                cmap=cmap_val,
                cbarlabel="Max Validation Group Loss"
            )

            texts = annotate_heatmap(im, valfmt="{x:.2f}", threshold=1.0)

            fig.tight_layout()

            ax.set_title(title_map[title_key])
            ax.set_xlabel('Data Size (Third Group)')
            ax.set_ylabel('Noise Level (First Group)')

            os.makedirs(os.path.join(save_dir, cmap_val), exist_ok=True)
            plt.savefig(os.path.join(save_dir, cmap_val, f"{'_'.join(str(x) for x in title_key)}.png"), bbox_inches='tight', dpi=1000)

if __name__ == "__main__":
    main()