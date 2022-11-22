
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
from scipy.stats import binned_statistic
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

plt.style.use('/mnt/home/tnguyen/mplstyle/latex_plot_style.mplstyle')

import utils
from merger_tree_ml import models
from merger_tree_ml.models import torchutils
from merger_tree_ml.utils import io_utils
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
        "--is-svar", required=False, action="store_true",
        help="Enable if tree is Svar")
    parser.add_argument(
        "--multiplier", required=False, type=int, default=10,
        help="Generation size: data_size * multiplier")

    return parser.parse_args()

def get_binned_stats(x, y, bins):

    mask = (~np.isnan(x)) & (~np.isnan(y))
    x = x[mask]
    y = y[mask]

    mean, bins, _ = binned_statistic(x, y, statistic='mean', bins=bins)
    stdv, _, _ = binned_statistic(x, y, statistic='std', bins=bins)
    bins_ce = 0.5 * (bins[1:] + bins[:-1])

    mask = (~np.isnan(mean)) & (~np.isnan(stdv))
    bins_ce = bins_ce[mask]
    mean = mean[mask]
    stdv = stdv[mask]

    return bins_ce, mean, stdv

def plot_mass_concentration(
        trees_nn, trees_nbody, time, x_dim=0, y_dim=1,
        xlabel="", ylabel="", save_path=None):

    num_bins = len(time)
    ncol = 5
    nrow = num_bins // ncol + 1
    bins = np.arange(5, 15, 0.5)

    fig, axes = plt.subplots(
        nrow, ncol, figsize=(ncol*5, nrow*5),
        sharex=True, sharey=True)

    for i in range(num_bins):
        ax = axes.ravel()[i]

        # DL trees
        x_nn = trees_nn[:, i, x_dim]   # mass
        y_nn = trees_nn[:, i, y_dim]   # concentration
        bins_ce, mean, stdv = get_binned_stats(x_nn, y_nn, bins)
        ax.errorbar(
            bins_ce, mean, yerr=stdv, fmt='o', capsize=5, lw=1, zorder=1,
            label='DL')

        # Nbody trees
        x_nbody = trees_nbody[:, i, x_dim]   # mass
        y_nbody = trees_nbody[:, i, y_dim]   # concentration
        bins_ce, mean, stdv = get_binned_stats(x_nbody, y_nbody, bins)
        ax.errorbar(
            bins_ce, mean, yerr=stdv, fmt='o', capsize=5, lw=1, zorder=1,
            label='N-body')

        # legend
        ax.text(0.1, 0.9, r'$T$ = {{{:.2f}}}'.format(time[i]), fontsize=30,
               horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)

    # format axes
    for i in range(nrow):
        axes[i, 0].set_ylabel(ylabel, fontsize=40)
    for i in range(ncol):
        axes[-1, i].set_xlabel(xlabel, fontsize=40)
    # axes[0, 0].set_ylim(0, 12)

    for ax in axes.ravel():
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
        ax.grid(which='major', ls='--', alpha=0.5)
        ax.tick_params(labelsize=30)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')

    return fig, axes


def main(
        model_arch, box_name, run_name, dataset_name,run_prefix,
        dataset_prefix, plot_prefix, run_version, is_svar, multiplier
    ):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get best checkpoint
    checkpoint_path, loss = io_utils.get_best_checkpoint(
        io_utils.get_run(
            run_name, prefix=run_prefix, version=run_version))

    print("read checkpoint from {}".format(checkpoint_path))

    # get model architecture and read in checkpoint
    model_arch = models.get_model_arch(model_arch)
    model = model_arch.DataModule.load_from_checkpoint(checkpoint_path)
    model = model.to(DEVICE)
    model = model.eval()

    # read in trees and root data
    trees_padded, seq_len = utils.read_dataset(
        dataset_name, dataset_prefix)
    times = trees_padded[..., -1].copy()
    max_len = np.max(seq_len)
    time_max_len = times[np.argmax(seq_len)]

    if model.transform.use_t:
        roots = trees_padded[:, 0].copy()
        time_sample = time_max_len
    elif model.transform.use_dt:
        roots = trees_padded[:, 0].copy()
        roots[:, -1] = trees_padded[:, 1, -1] - roots[:, -1]
        time_sample = time_max_len[1:] - time_max_len[:-1]
        time_sample = np.append(time_sample, 0)
    else:
        roots = trees_padded[:, 0, :-1].copy()
        time_sample = None

    # get N-body trees and generate DL trees from roots
    if is_svar:
        svar_to_mass, mass_to_svar = utils.create_interpolator(
            time_max_len, is_omega=True)
        roots_svar = roots.copy()

        roots[..., 0] = svar_to_mass(roots[..., 0], time_max_len[0])
        trees_padded[..., 0] = svar_to_mass(trees_padded[..., 0], time_max_len)
        trees_nbody = trees_padded
        roots_nbody = roots

        roots_nn = np.repeat(roots_nbody, multiplier, axis=0)
        trees_nn = torchutils.sample_trees(
            model, roots=np.repeat(roots_svar, multiplier, axis=0),
            max_len=max_len, time=time_sample, device=DEVICE)
        trees_nn[..., 0] = svar_to_mass(trees_nn[..., 0], time_max_len)
    else:
        trees_nbody = trees_padded
        roots_nbody = roots
        roots_nn = np.repeat(roots_nbody, multiplier, axis=0)
        trees_nn = torchutils.sample_trees(
            model, roots=roots_nn, max_len=max_len,
            time=time_sample, device=DEVICE)

    # Start plotting
    plot_dir = os.path.join(plot_prefix, run_name)
    os.makedirs(plot_dir, exist_ok=True)

    # Plot mass accretion distribution
    plot_mass_concentration(
        trees_nn, trees_nbody, time_max_len,
        x_dim=0, y_dim=1,
        xlabel=r"$\log_{10} M$",
        ylabel=r"$C_\mathrm{vir}$",
        save_path=os.path.join(plot_dir, "mass_concentration.png")
    )
    plt.close()


if __name__ == "__main__":
    FLAGS = parse_args()
    main(
        model_arch=FLAGS.model_arch,
        box_name=FLAGS.box_name,
        run_name=FLAGS.run_name,
        dataset_name=FLAGS.dataset_name,
        run_prefix=FLAGS.run_prefix,
        dataset_prefix=FLAGS.dataset_prefix,
        plot_prefix=FLAGS.plot_prefix,
        run_version=FLAGS.run_version,
        is_svar=FLAGS.is_svar,
        multiplier=FLAGS.multiplier
    )
