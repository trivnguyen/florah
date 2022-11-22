
import os
import h5py
import json
import argparse

import utils
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
        "--pipeline", required=False, type=str,
        help="Pipeline to run. If not given, run all pipelines")
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


if __name__ == "__main__":
    """ Run all pipelines """
    FLAGS = parse_cmd()


    if FLAGS.pipeline is not None:
        print("Running pipeline: {}".format(FLAGS.pipeline))
        if FLAGS.pipeline not in utils.ALL_PIPELINES:
            raise KeyError("Pipeline {} does not exist".format(FLAGS.pipeline))
        utils.ALL_PIPELINES[FLAGS.pipeline].main(
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

    else:
        print("Running all pipelines")
        print("---------------------")
        for pipeline in utils.ALL_PIPELINES:
            print("Running: {}".format(pipeline))
            print("----------------------------------")
            utils.ALL_PIPELINES[pipeline].main(
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
