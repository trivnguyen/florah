
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

def calc_t_form(trees, t, x=0.5):
    """ Calculate formation time at mass fraction x"""
    dtrees = trees[..., :1] - trees
    tx= np.array([
        interpolate.interp1d(
            dtree, t, bounds_error=False, fill_value=np.nan, assume_sorted=True, kind='linear')(
                -np.log10(x)) for dtree in dtrees
    ])
    return tx.T

def plot_accretion_mass(
        trees_nn, trees_nbody, plot_dim=0, log=True,
        xlabel=r"$\log_{10} (M_i/M_{i+1})$", ylabel="frequency",
        save_path=None
    ):

    num_bins = len(trees_nn)
    nrows = num_bins // 2 + num_bins % 2
    ncols = num_bins // 2

    fig, axes = plt.subplots(
        ncols, nrows, figsize=(nrows * 4, ncols * 3.5),
        sharex=True, sharey=True)

    for i in range(num_bins):
        ax = axes.ravel()[i]

        # plot N-body trees accreted mass distribution
        x = -np.diff(trees_nbody[i][..., plot_dim]).ravel()
        _, bins, _  = ax.hist(
            x, 'auto', histtype='step', density=True,
            color='darkblue', lw=2, label='N-body')

        # plot DL trees accreted mass distribution
        x = -np.diff(trees_nn[i][..., plot_dim]).ravel()
        ax.hist(
            x, bins, histtype='step', density=True,
            color='firebrick', lw=2, label='DL')

    for i in range(ncols):
        axes[i, 0].set_ylabel(ylabel)
    for i in range(nrows):
        axes[-1, i].set_xlabel(xlabel)

    if log:
        axes[0, 0].set_yscale('log')
    axes[0, 0].set_ylim(1e-3, 1e3)

    # for i in range(n_bins):
        # ax = axes.ravel()[i]
        # ax.text(0.45, 0.9, r'$\log_{10} M = $' + ' {}'.format(x_ce[i]), fontsize=20,
               # horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
        # ax.text(0.45, 0.75, r'$N = $' + ' {}'.format(len(trees_nbody[i])), fontsize=20,
               # horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.55, 1.05))

    # ax.legend()
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')

    return fig, axes

def plot_tform(
        trees_nn, trees_nbody, time, plot_dim=0,
        ylabel="frequency", save_path=None
    ):

    num_bins = len(trees_nn)
    nrows = 3
    ncols = num_bins

    # calculate formation time
    tform_nn = []
    tform_nbody = []
    for i in range(num_bins):
        # N-body trees
        t25, t50, t75 = calc_t_form(
            trees_nbody[i][..., plot_dim], time, x=(0.25, 0.5, 0.75))
        tform_nbody.append(np.stack([t25, t50, t75]))

        # DL trees
        t25, t50, t75 = calc_t_form(
            trees_nn[i][..., plot_dim], time, x=(0.25, 0.5, 0.75))
        tform_nn.append(np.stack([t25, t50, t75]))

    # plot
    fig, axes = plt.subplots(
        ncols, nrows, figsize=(nrows * 4, ncols * 3.5),
        sharex=True, sharey="row")

    for i in range(ncols):
        _, bins, _ = axes[i, 0].hist(
            tform_nbody[i][0], 'auto', density=True, histtype='step', lw=2,
            color='darkblue', label=r'N-body')
        axes[i, 0].hist(
            tform_nn[i][0], bins, density=True, histtype='step', lw=2,
            color='firebrick', label=r'DL')

        _, bins, _ = axes[i, 1].hist(
            tform_nbody[i][1], 'auto', density=True, histtype='step', lw=2,
            color='darkblue', label=r'N-body')
        axes[i, 1].hist(
            tform_nn[i][1], bins, density=True, histtype='step', lw=2,
            color='firebrick', label=r'DL')

        _, bins, _ = axes[i, 2].hist(
            tform_nbody[i][2], 'auto', density=True, histtype='step', lw=2,
            color='darkblue', label=r'N-body')
        axes[i, 2].hist(
            tform_nn[i][2], bins, density=True, histtype='step', lw=2,
            color='firebrick', label=r'DL')

    axes[-1, 0].set_xlabel(r'$t_{25}$')
    axes[-1, 1].set_xlabel(r'$t_{50}$')
    axes[-1, 2].set_xlabel(r'$t_{75}$')

    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)

    for ax in axes.ravel():
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
        ax.grid(which='major', ls='--', alpha=0.5)

    axes[0, 1].legend(loc='upper center', fontsize=20, ncol=1)

    # for i, ax in enumerate(axes[:, 1]):
    #     ax.set_title(
    #         r'$\log_{{10}} M = {{{0}}}; N_\mathrm{{trees}} = {{{1}}}$'.format(
    #               x_ce[i], len(trees_nbody[i])),
    #         pad=10, fontsize=20)

#     axes.set_title(0.2, 0.9, r'$\log_{10} M = $' + ' {}'.format(x_ce[i]), fontsize=20,
#            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
#     ax.text(0.2, 0.7, r'$N = $' + ' {}'.format(len(trees_nbody[i])), fontsize=20,
#            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)

    fig.suptitle(r'\bf Formation Time')
    fig.tight_layout()
    fig.subplots_adjust()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')

    return fig, axes

def plot_tform_spread(
        trees_nn, trees_nbody, time, plot_dim=0,
        ylabel="frequency", save_path=None
    ):

    num_bins = len(trees_nn)
    nrows = num_bins // 2 + num_bins % 2
    ncols = num_bins // 2

    # calculate formation time
    tform_spread_nn = []
    tform_spread_nbody = []
    for i in range(num_bins):
        # N-body trees
        t25, t75 = calc_t_form(
            trees_nbody[i][..., plot_dim], time, x=(0.25, 0.75))
        tform_spread_nbody.append(t75 - t25)

        # DL trees
        t25, t75 = calc_t_form(
            trees_nn[i][..., plot_dim], time, x=(0.25, 0.75))
        tform_spread_nn.append(t75 - t25)

    # plot
    fig, axes = plt.subplots(
        ncols, nrows, figsize=(nrows * 4, ncols * 3.5),
        sharex=True, sharey="row")

    for i in range(num_bins):
        ax = axes.ravel()[i]
        _, bins, _ = ax.hist(
            tform_spread_nbody[i], 20, density=True, histtype='step',
            lw=2, color='darkblue', label=r'N-body')
        ax.hist(
            tform_spread_nn[i], bins, density=True, histtype='step',
            lw=2, color='firebrick', label=r'DL')

    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    for ax in axes[-1, :]:
        ax.set_xlabel(r'$t_{75} - t_{25}$', )

    for ax in axes.ravel():
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
        ax.grid(which='major', ls='--', alpha=0.5)

    # for i in range(n_bins):
    #     ax = axes.ravel()[i]
    #     ax.text(0.1, 0.9, r'$\log_{10} M = $' + ' {}'.format(x_ce[i]), fontsize=20,
    #            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    #     ax.text(0.1, 0.8, r'$N = $' + ' {}'.format(len(trees_nbody[i])), fontsize=20,
    #            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.55, 1.05))
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)

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

    # get bins
    bins = utils.DEFAULT_BINS[box_name]
    widths = utils.DEFAULT_WIDTHS[box_name]
    num_bins = len(bins)

    # get N-body trees and generate DL trees from roots
    trees_nbody = []
    roots_nbody = []
    trees_nn = []
    roots_nn = []
    print("Low   High   Selected")

    if is_svar:
        svar_to_mass, mass_to_svar = utils.create_interpolator(
            time_max_len, is_omega=True)

        roots_svar = roots.copy()
        roots[..., 0] = svar_to_mass(roots[..., 0], time_max_len[0])
        trees_padded[..., 0] = svar_to_mass(trees_padded[..., 0], time_max_len)

        for i in range(num_bins):
            low = bins[i] - widths[i]
            high = bins[i] + widths[i]
            select = (low <= roots[..., 0]) & (roots[..., 0] < high)
            print(low, high, np.sum(select))

            if np.sum(select) == 0:
                continue

            # N-body trees
            trees_nbody.append(trees_padded[select])
            roots_nbody.append(roots[select])

            # DL trees
            root = np.repeat(roots[select], multiplier, axis=0)
            root_svar = np.repeat(roots_svar[select], multiplier, axis=0)
            roots_nn.append(root)
            tree = torchutils.sample_trees(
                model, roots=root_svar, max_len=max_len,
                time=time_sample, device=DEVICE
            )
            tree[..., 0] = svar_to_mass(tree[..., 0], time_max_len)
            trees_nn.append(tree)
    else:
        for i in range(num_bins):
            low = bins[i] - widths[i]
            high = bins[i] + widths[i]
            select = (low <= roots[..., 0]) & (roots[..., 0] < high)
            print(low, high, np.sum(select))

            if np.sum(select) == 0:
                continue

            # N-body trees
            trees_nbody.append(trees_padded[select])
            roots_nbody.append(roots[select])

            # DL trees
            root = np.repeat(roots[select], multiplier, axis=0)
            roots_nn.append(root)
            trees_nn.append(
                torchutils.sample_trees(
                    model, roots=root, max_len=max_len,
                    time=time_sample, device=DEVICE
                    )
                )

    # Start plotting
    plot_dir = os.path.join(plot_prefix, run_name)
    os.makedirs(plot_dir, exist_ok=True)

    # Plot mass accretion distribution
    plot_accretion_mass(
        trees_nn, trees_nbody,
        plot_dim=0, log=True,
        xlabel=r"$\log_{10} (M_i/M_{i+1})$",
        ylabel="frequency",
        save_path=os.path.join(plot_dir, "mass_accreted_distribution_log.png")
    )
    plot_accretion_mass(
        trees_nn, trees_nbody,
        plot_dim=0, log=False,
        xlabel=r"$\log_{10} (M_i/M_{i+1})$",
        ylabel="frequency",
        save_path=os.path.join(plot_dir, "mass_accreted_distribution.png")
    )

    # Plot accretion time
    plot_tform(
        trees_nn, trees_nbody, time_max_len,
        plot_dim=0,
        ylabel="frequency",
        save_path=os.path.join(plot_dir, "formation_time.png")
    )
    plot_tform_spread(
        trees_nn, trees_nbody, time_max_len,
        plot_dim=0,
        ylabel="frequency",
        save_path=os.path.join(plot_dir, "formation_time_spread.png")
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
