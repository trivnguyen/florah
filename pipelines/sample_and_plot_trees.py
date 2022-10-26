
import os
import h5py
import json
import argparse

import numpy as np
import torch
import scipy.interpolate as interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt
import corner
import seaborn as sns
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

plt.style.use('/mnt/home/tnguyen/mplstyle/latex_plot_style.mplstyle')

import utils
from merger_tree_ml.utils import io_utils
from merger_tree_ml.models import torchutils
from merger_tree_ml.envs import DEFAULT_RUN_PATH, DEFAULT_DATASET_PATH


# Parser cmd argument
def parse_cmd():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # required args
    parser.add_argument(
        "--model-arch", required=True, type=str, help="Model architecture")
    parser.add_argument(
        "--box-name", required=True, type=str, help="Box name")
    parser.add_argument(
        "--run-name", required=True, type=str,
        help="Name of run. Output is saved at \"run_prefix/run_name\".")
    parser.add_argument(
        "--dataset-name", required=True, type=str,
        help="Name of dataset. Read at \"dataset_prefix/dataset_name\"")

    # optional args
    parser.add_argument(
        "--run-prefix", required=False, type=str, default=DEFAULT_RUN_PATH,
        help="Prefix of run. Output is saved at \"run_prefix/run_name\".")
    parser.add_argument(
        "--dataset-prefix", required=False, type=str,
        default=DEFAULT_DATASET_PATH,
        help="Prefix of dataset. Read at \"dataset_prefix/dataset_name\".")
    parser.add_argument(
        "--plot-prefix", required=False, type=str, default=utils.PLOT_PREFIX,
        help="Prefix of plot directory")
    parser.add_argument(
        "--run-version", required=False, type=str, default="best",
        help="Run version")
    parser.add_argument(
        "--multiplier", required=False, type=int, default=10,
        help="Generation size: data_size * multiplier")

    return parser.parse_args()

def plot_trees_comparison(
        trees_nn, trees_nbody, roots_nn, roots_nbody, time=None,
        plot_dim=0, sub_root=False, xlabel="", ylabel="",
        legend=True, save_path=None
    ):
    """ Plot trees. Comparing N-body and NN trees"""

    num_bins = len(trees_nn)
    fontsize = 20
    colors = sns.color_palette("rocket", num_bins)

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 8), sharex=True, sharey=True)

    # Plot DL
    ax = axes[0]
    for i, (roots, trees) in enumerate(zip(roots_nn, trees_nn)):
        if sub_root:
            x = trees[..., plot_dim] - roots[..., plot_dim, np.newaxis]
        else:
            x = trees[..., plot_dim]

        lo, med, hi = np.nanpercentile(x, [16, 50, 84], 0)
        if time is None:
            time = np.arange(len(med))
        lme = ax.plot(time, med, color=colors[i], lw=3)
        llo = ax.plot(time, lo, color=colors[i], ls='--', alpha=0.5, lw=3)
        lhi = ax.plot(time, hi, color=colors[i], ls='-.', alpha=0.5, lw=3)

    # plot N-body trees
    ax = axes[1]
    for i, (roots, trees) in enumerate(zip(roots_nbody, trees_nbody)):
        if sub_root:
            x = trees[..., plot_dim] - roots[..., plot_dim, np.newaxis]
        else:
            x = trees[..., plot_dim]

        lo, med, hi = np.percentile(x, [16, 50, 84], 0)
        if time is None:
            time = np.arange(len(med))
        lme = ax.plot(time, med, color=colors[i], lw=3)
        llo = ax.plot(time, lo, color=colors[i], ls='--', alpha=0.5, lw=3)
        lhi = ax.plot(time, hi, color=colors[i], ls='-.', alpha=0.5, lw=3)

    axes[0].set_xlabel(xlabel, fontsize=fontsize)
    axes[1].set_xlabel(xlabel, fontsize=fontsize)
    axes[0].set_ylabel(ylabel, fontsize=fontsize)
    axes[0].set_title(r'\bf DL Trees', fontsize=fontsize, y=1.01)
    axes[1].set_title(r'\bf N-body Trees', fontsize=fontsize, y=1.01)

    for ax in axes:
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which='major', ls='--', color='k', alpha=0.5)
    axes[0].invert_xaxis()

    if legend:
        axes[0].legend(
            [lme[0], llo[0], lhi[0]], ['Median', 'Low -68%', 'High-68%'],
            ncol=2, loc='best')

    fig.tight_layout()
    fig.subplots_adjust(wspace=0)

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig, axes


def plot_trees_difference(
        trees_nn, trees_nbody, roots_nn, roots_nbody, time=None,
        plot_dim=0, sub_root=False, xlabel="", ylabel="",
        legend=True, save_path=None
    ):

    num_bins = len(trees_nn)
    fontsize = 20
    colors = sns.color_palette("rocket", num_bins)
    fig, ax = plt.subplots(1, figsize=(10, 5), sharex=True, sharey=True)

    for i in range(num_bins):
        x_nn = trees_nn[i][..., plot_dim]
        x_nbody = trees_nbody[i][..., plot_dim]
        if sub_root:
            x_nn = x_nn - roots_nn[i][..., plot_dim, np.newaxis]
            x_nbody = x_nbody - roots_nbody[i][..., plot_dim, np.newaxis]

        lo_nn, med_nn, hi_nn = np.percentile(x_nn, [16, 50, 84], 0)
        lo_nbody, med_nbody, hi_nbody = np.percentile(x_nbody, [16, 50, 84], 0)
        if time is None:
            time = np.arange(len(med_nn))
        lme = ax.plot(time, med_nn - med_nbody, color=colors[i], lw=3)
        llo = ax.plot(time, lo_nn - lo_nbody, color=colors[i], ls='--', alpha=0.5, lw=3)
        lhi = ax.plot(time, hi_nn - hi_nbody, color=colors[i], ls='-.', alpha=0.5, lw=3)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(which='major', ls='--', color='k', alpha=0.5)
    ax.invert_xaxis()

    if legend:
        ax.legend(
            [lme[0], llo[0], lhi[0]], ['Median', 'Low -68%', 'High-68%'],
            ncol=2, loc='best', fontsize=20
        )

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig, ax

def main():
    FLAGS = parse_cmd()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get best checkpoint
    checkpoint_path, loss = io_utils.get_best_checkpoint(
        io_utils.get_run(
            FLAGS.run_name, prefix=FLAGS.run_prefix, version=FLAGS.run_version))

    print("read checkpoint from {}".format(checkpoint_path))

    # get model architecture and read in checkpoint
    model_arch = utils.get_model_arch(FLAGS.model_arch)
    model = model_arch.DataModule.load_from_checkpoint(checkpoint_path)
    model = model.to(DEVICE)
    model = model.eval()

    # read in trees and root data
    trees_padded, seq_len = utils.read_dataset(
        FLAGS.dataset_name, FLAGS.dataset_prefix)
    roots = trees_padded[:, 0]
    times = trees_padded[..., -1]
    max_len = np.max(seq_len)
    time_max_len = times[np.argmax(seq_len)]

    if not model.transform.use_t:
        roots = roots[:, :-1]
        time_sample = None
    else:
        time_sample = time_max_len

    # get bins
    bins = utils.DEFAULT_BINS[FLAGS.box_name]
    widths = utils.DEFAULT_WIDTHS[FLAGS.box_name]
    num_bins = len(bins)

    # get N-body trees and generate DL trees from roots
    trees_nbody = []
    roots_nbody = []
    trees_nn = []
    roots_nn = []
    print("Low   High   Selected")
    for i in range(num_bins):
        low = bins[i] - widths[i]
        high = bins[i] + widths[i]
        select = (low <= roots[..., 0]) & (roots[..., 0] < high)
        print(low, high, np.sum(select))

        # N-body trees
        trees_nbody.append(trees_padded[select])
        roots_nbody.append(roots[select])

        # DL trees
        root = np.repeat(roots[select], FLAGS.multiplier, axis=0)
        roots_nn.append(root)
        if FLAGS.model_arch == "AttentionMAF":
            trees_nn.append(
                torchutils.sample_trees_attention(
                    model, root, max_len=max_len, time=time_sample, device=DEVICE))
        elif FLAGS.model_arch == "RecurrentMAF":
            trees_nn.append(
                torchutils.sample_trees_recurrent(
                    model, root, max_len=max_len, time=time_sample, device=DEVICE))
        else:
            raise RunTimeError(
                "model arch {} has no sampling function".format(FLAGS.model_arch))

    # Start plotting
    plot_dir = os.path.join(FLAGS.plot_prefix, FLAGS.run_name)
    os.makedirs(plot_dir, exist_ok=True)

    # Plot tree comparison side-by-side
    plot_trees_comparison(
        trees_nn, trees_nbody, roots_nn, roots_nbody, time=time_max_len,
        plot_dim=0, sub_root=True,
        xlabel=r"Time Variable",
        ylabel=r"$\log_{10}(M / M_0)$",
        legend=True,
        save_path=os.path.join(plot_dir, "mass_evo_comparison.png")
    )
    plot_trees_comparison(
        trees_nn, trees_nbody, roots_nn, roots_nbody, time=time_max_len,
        plot_dim=1, sub_root=False,
        xlabel=r"Time Variable",
        ylabel=r"$C_\mathrm{vir}$",
        legend=True,
        save_path=os.path.join(plot_dir, "cvir_evo_comparison.png")
    )

    # Plot tree difference
    plot_trees_difference(
        trees_nn, trees_nbody, roots_nn, roots_nbody, time=time_max_len,
        plot_dim=0, sub_root=True,
        xlabel=r"Time Variable",
        ylabel=r"$\Delta\{\log_{10}(M / M_0)\}$",
        legend=True,
        save_path=os.path.join(plot_dir, "mass_evo_difference.png")
    )
    plot_trees_difference(
        trees_nn, trees_nbody, roots_nn, roots_nbody, time=time_max_len,
        plot_dim=1, sub_root=False,
        xlabel=r"Time Variable",
        ylabel=r"$\Delta C_\mathrm{vir}$",
        legend=True,
        save_path=os.path.join(plot_dir, "cvir_evo_difference.png")
    )


if __name__ == "__main__":
    main()
