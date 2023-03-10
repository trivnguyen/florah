#!/usr/bin/env python
# coding: utf-8

from typing import Optional, Union

import os
import sys
import h5py
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import scipy.interpolate as interpolate
from scipy.stats import binned_statistic
from torch import FloatTensor

from florah import physics, utils
from florah.models import recurrent_maf, attention_maf, torchutils
from florah.config import DEFAULT_RUN_PATH, DEFAULT_RAW_DATASET_PATH

# DEFINE GLOBAL VARIABLES
FLAGS = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))
LOGGER.setLevel(logging.DEBUG)
DEFAULT_BINS = {
    "GUREFT05": [7, 7.5, 8, 8.5],
    "GUREFT15": [8.5, 9, 9.5, 10],
    "GUREFT35": [9.5, 10, 10.5, 11],
    "GUREFT90": [10.4, 10.8, 11.2, 11.6],
    "GUREFT": [7, 8, 9, 10, 11],
    "BP": [11, 12, 13, 14],
    "VSMDPL": [11, 12, 13, 14],
    "TNG50": [8.5, 9, 9.5, 10, 10.5],
    "TNG100": [9.5, 10, 10.5, 11, 11.5],
    "TNG300": [10, 11, 12, 13, 14],
    "ALL": [0, ]
}
DEFAULT_WIDTHS = {
    "GUREFT05": [0.25, 0.25, 0.25, 0.25],
    "GUREFT15": [0.25, 0.25, 0.25, 0.25],
    "GUREFT35": [0.25, 0.25, 0.25, 0.25],
    "GUREFT90": [0.2, 0.2, 0.2, 0.2],
    "GUREFT": [0.2, 0.2, 0.2, 0.2, 0.2],
    "BP": [0.2, 0.2, 0.2, 0.2],
    "VSMDPL": [0.2, 0.2, 0.2, 0.2],
    "TNG50": [0.2, 0.2, 0.2, 0.2, 0.2],
    "TNG100": [0.2, 0.2, 0.2, 0.2, 0.2],
    "TNG300": [0.2, 0.2, 0.2, 0.2, 0.2],
    "ALL": [1000, ]
}

# PARSE CMD ARGS
def parse_cmd():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Required args
    parser.add_argument(
        '--output', required=True, type=str,
        help="Path to output"
    )
    parser.add_argument(
        "--bin-name", required=False, default=None, type=str,
        help="Name of mass bins to sample from"
    )
    # Optional args
    # input/output args
    parser.add_argument(
        "--dataset-name", required=False, type=str, default="",
        help="Name of dataset. Read at \"dataset_prefix/dataset_name\""
    )
    parser.add_argument(
        "--dataset-prefix", required=False, type=str,
        default=DEFAULT_RAW_DATASET_PATH,
        help="Prefix of dataset. Read at \"dataset_prefix/dataset_name\"."
    )
    parser.add_argument(
        "--run-prefix", required=False, type=str, default=DEFAULT_RUN_PATH,
        help="Prefix of run. Output is saved at \"run_prefix/run_name\"."
    )
    parser.add_argument(
        "--run-name", required=False, type=str, default="default",
        help="Name of run. Output is saved at \"run_prefix/run_name\"."
    )
    parser.add_argument(
        "--run-version", required=False, type=str, default='best',
        help="Run version"
    )
    # sampling args
    # parser.add_argument(
    #     "--num-min-halo", required=False, default=0, type=int,
    #     help="Number of minimum particles per halo"
    # )
    # parser.add_argument(
    #     "--num-min-root", required=False, default=500, type=int,
    #     help="Number of minimum particles for the root halo"
    # )
    parser.add_argument(
        "--num-times", required=False, default=1, type=int,
        help="Number of time steps"
    )
    parser.add_argument(
        "--max-tree-len", required=False, default=None, type=int,
        help="Maximum length of generated trees"
    )
    parser.add_argument(
        "--multiplier", required=False, default=1, type=int,
        help="Multiplier to number of trees in dataset"
    )
    parser.add_argument(
        "--min-step", required=False, default=1, type=int,
        help="Minimum time step for sampling"
    )
    parser.add_argument(
        "--max-step", required=False, default=2, type=int,
        help="Maximum time step for sampling"
    )
    parser.add_argument(
        "--ini-step", required=False, default=0, type=int,
        help="Initial step"
    )
    return parser.parse_args()


def get_root_features(node_features, in_channels=2, ini=0):
    """ Get root features of trees from node features"""
    num_data = len(node_features['mass'])  # number of trees
    roots = []  # list for all roots
    for i in range(num_data):
        # get mass and length
        m_tree = np.log10(node_features['mass'][i])
        len_tree = len(m_tree)
        if ini > len_tree:
            continue

        # get other properties
        # TODO: find a better way to include channels
        c_tree = node_features['rvir'][i] / node_features['rs'][i]
        if in_channels == 4:
            vrms_tree = node_features['vrms'][i]
            vmax_tree = node_features['vmax'][i]
            data = np.array([
                m_tree[ini], c_tree[ini], vrms_tree[ini], vmax_tree[ini]])
        else:
            data = np.array([m_tree[ini], c_tree[ini]])
        roots.append(data)
    roots = np.stack(roots)
    return roots

# MAIN
def main(FLAGS):
    """ Sample merger trees from an input dataset distribution """

    # Read in NN checkpoints
    checkpoint_path, loss = utils.io_utils.get_best_checkpoint(
        utils.io_utils.get_run(FLAGS.run_name, version=FLAGS.run_version))
    model = recurrent_maf.DataModule.load_from_checkpoint(checkpoint_path)
    model = model.to(DEVICE)
    in_channels = model.hparams.model_hparams['in_channels']

    # Read in dataset and meta data
    # node features
    path = utils.io_utils.get_dataset(
        os.path.join(FLAGS.dataset_name, "raw_data.h5"),
        prefix=FLAGS.dataset_prefix, is_dir=False
    )
    LOGGER.info(f'Read input data from {path}')
    node_features = utils.io_utils.read_dataset(path)[0]
    # root features
    root_features = get_root_features(
        node_features, ini=FLAGS.ini_step, in_channels=in_channels)

    # Generate trees by bins
    bins_ce = DEFAULT_BINS[FLAGS.bin_name]
    bins_wd = DEFAULT_WIDTHS[FLAGS.bin_name]
    num_bins = len(bins_ce)

    # sample time
    all_time_samples = []
    all_time_indices = []
    all_trees = []
    LOGGER.info("Sampling trees...")
    for _ in range(FLAGS.num_times):
        time = utils.ppr_utils.get_ztable(
            FLAGS.dataset_name, return_scale_factor=True)
        if FLAGS.max_tree_len is None:
            FLAGS.max_tree_len = len(time)
        time_indices = utils.ppr_utils.sample_cumulative(
            FLAGS.max_tree_len, step_min=FLAGS.min_step, step_max=FLAGS.max_step+1,
            ini=FLAGS.ini_step)
        time_samples = time[time_indices]

        all_time_samples.append(time_samples)
        all_time_indices.append(time_indices)

        LOGGER.info(f"Num time step: {len(time_indices)}")
        LOGGER.info(f"Time indices: {time_indices}")
        LOGGER.info(f"Time samples: {time_samples}")

        # sample trees
        trees = []  # list for all trees

        for i in range(num_bins):
            bins_lo = bins_ce[i] - bins_wd[i]
            bins_hi = bins_ce[i] + bins_wd[i]
            select = (
                (bins_lo <= root_features[..., 0]) &
                (root_features[..., 0] < bins_hi)
            )

            LOGGER.info(f"- Bins [{bins_lo} {bins_hi}]: {select.sum()}")

            # if no trees match the criteria, add empty list
            if select.sum() == 0:
                trees.append(np.array([]))
                continue

            roots_bins = np.repeat(root_features[select], FLAGS.multiplier, axis=0)
            roots_bins = roots_bins.astype(np.float32)
            times = np.repeat(time_samples.reshape(1, -1), len(roots_bins), axis=0)
            times = times[..., np.newaxis]

            # generate trees and add to list
            trees.append(torchutils.sample_trees(model, roots_bins, times))

        all_trees.append(trees)

    # Write trees to file
    os.makedirs(os.path.dirname(FLAGS.output), exist_ok=True)
    with h5py.File(FLAGS.output, 'w') as f:
        f.attrs.update({
            'num_bins': num_bins,
            'num_times': FLAGS.num_times,
            'bins_ce': bins_ce,
            'bins_wd': bins_wd,
            'node_properties': ['log_mass', 'cvir']
        })

        for i in range(FLAGS.num_times):
            gr = f.create_group(f'time_gr{i}')
            gr.create_dataset('time', data=all_time_samples[i])
            gr.create_dataset('time_indices', data=all_time_indices[i])
            trees = all_trees[i]
            for j in range(num_bins):
                ggr = gr.create_group(f'gr{j}')
                ggr.attrs.update({
                    'bins_ce': bins_ce[j],
                    'bins_wd': bins_wd[j],
                    'num_trees': len(trees[j])
                })
                if len(trees[j]) > 0:
                    ggr.create_dataset('log_mass', data=trees[j][..., 0])
                    ggr.create_dataset('cvir', data=trees[j][..., 1])
                    if in_channels == 4:
                        ggr.create_dataset('vrms', data=trees[j][..., 2])
                        ggr.create_dataset('vmax', data=trees[j][..., 2])
                else:
                    ggr.create_dataset('log_mass', data=np.array([]))
                    ggr.create_dataset('cvir', data=np.array([]))
                    ggr.create_dataset('vrms', data=np.array([]))
                    ggr.create_dataset('vmax', data=np.array([]))

if __name__ == "__main__":
    FLAGS = parse_cmd()
    main(FLAGS)
