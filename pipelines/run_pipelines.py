
import os
import h5py
import json
import argparse

import utils
import sample_and_plot_trees
import compute_statistics
from merger_tree_ml.envs import DEFAULT_RUN_PATH, DEFAULT_DATASET_PATH

ALL_PIPELINES = {
    "sample_and_plot_trees": sample_and_plot_trees,
    "compute_statistics": compute_statistics,
}

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


if __name__ == "__main__":
    """ Run all pipelines """
    FLAGS = parse_cmd()

    print("Running all pipelines")
    print("---------------------")
    for pipeline in ALL_PIPELINES:
        print("Running: {}".format(pipeline))
        print("----------------------------------")
        ALL_PIPELINES[pipeline].main()
