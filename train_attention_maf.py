#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import shutil
import argparse
import logging
import h5py

import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.loggers import CSVLogger

from merger_tree_ml import utils
from merger_tree_ml.models import attention_maf
from merger_tree_ml.envs import DEFAULT_RUN_PATH, DEFAULT_DATASET_PATH


# Set logger
def set_logger():
    """ Set up logger to STDOUT """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

# Parser cmd argument
def parse_cmd():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # input/output args
    parser.add_argument(
        "--dataset-prefix", required=False, type=str,
        default=DEFAULT_DATASET_PATH,
        help="Prefix of dataset. Read at \"dataset_prefix/dataset_name\".")
    parser.add_argument(
        "--dataset-name", required=False, type=str, default="",
        help="Name of dataset. Read at \"dataset_prefix/dataset_name\"")
    parser.add_argument(
        "--run-prefix", required=False, type=str, default=DEFAULT_RUN_PATH,
        help="Prefix of run. Output is saved at \"run_prefix/run_name\".")
    parser.add_argument(
        "--run-name", required=False, type=str, default="default",
        help="Name of run. Output is saved at \"run_prefix/run_name\".")
    parser.add_argument(
        "--run-version", required=False, type=str,
        help="Run version")
    parser.add_argument(
        "--resume", required=False, action="store_true",
        help="Enable to resume previous run. Version number is required.")
    # model args
    parser.add_argument(
        "--in-channels", required=True, type=int,
        help="Number of input channels")
    parser.add_argument(
        "--out-channels", required=True, type=int,
        help="Number of output channels")
    parser.add_argument(
        "--hidden-features", required=False, type=int, default=64,
        help="Number of hidden features for self-attention blocks")
    parser.add_argument(
        "--num-blocks", required=False, type=int, default=1,
        help="Number of self-attention blocks")
    parser.add_argument(
        "--num-heads", required=False, type=int, default=1,
        help="Number of attention heads in self-attention blocks")
    parser.add_argument(
        "--hidden-features-flows", required=False, type=int, default=128,
        help="Number of hidden features of MAF transformations")
    parser.add_argument(
        "--num-layers-flows", required=False, type=int, default=4,
        help="Number of MAF transformations")
    parser.add_argument(
        "--num-blocks-flows", required=False, type=int, default=2,
        help="Number of blocks per MAF transformation")
    # transfrom args
    parser.add_argument(
        "--subtract-dim", required=False, type=int, nargs="+", default=[0, ],
        help="Dimension to subtract Y from X")
    parser.add_argument(
        "--use-time", required=False, action="store_true",
        help="If True, add time parameters to X")
    parser.add_argument(
        "--use-dtime", required=False, action="store_true",
        help="If True, add time difference parameters to X")
    # training args
    parser.add_argument(
        "--batch-size", required=False, type=int, default=1024,
        help="Batch size")
    parser.add_argument(
        "--max-epochs", required=False, type=int, default=10000,
        help="Maximum number of epochs. Stop training automatically if exceeds")
    parser.add_argument(
        "--es-patience", required=False, type=int, default=100,
        help="Patience for early stopping")
    parser.add_argument(
        "--num-workers", required=False, type=int, default=1,
        help="Number of workers")
    # optimizer args
    parser.add_argument(
        "--optimizer", required=False, type=str, default="Adam",
        help="Type of optimizer to use")
    parser.add_argument(
        "--learning-rate", required=False, type=float, default=5e-4,
        help="Learning rate of optimizer")
    parser.add_argument(
        "--adam-betas", required=False, type=float, nargs="+",
        default=(0.9, 0.98), help="Coefficients used for computing running "\
                                  "averages of gradient and its square")
    # scheduler args
    parser.add_argument(
        "--scheduler", required=False, type=str, default="AttentionScheduler",
        help="Type of LR scheduler to use")
    parser.add_argument(
        "--warmup-steps", required=False, type=int, default=5000,
        help="Warm up step for Attention LR scheduler")
    return parser.parse_args()


def main():
    """ Train Attention-MAF model for merger tree generation """
     # Parse cmd args
    FLAGS = parse_cmd()
    LOGGER = set_logger()

    # Create data module
    in_channels = FLAGS.in_channels + int(FLAGS.use_time) + int(FLAGS.use_dtime)
    model_hparams = {
        "in_channels": in_channels,
        "out_channels": FLAGS.out_channels,
        "hidden_features": FLAGS.hidden_features,
        "num_blocks": FLAGS.num_blocks,
        "num_heads": FLAGS.num_heads,
        "num_layers_flows": FLAGS.num_layers_flows,
        "hidden_features_flows": FLAGS.hidden_features_flows,
        "num_blocks_flows": FLAGS.num_blocks_flows
    }
    transform_hparams = {
        "nx": FLAGS.in_channels,
        "ny": FLAGS.out_channels,
        "sub_dim": FLAGS.subtract_dim,
        "use_t": FLAGS.use_time,
        "use_dt": FLAGS.use_dtime
    }
    optimizer_hparams = {
        "optimizer": {
            "optimizer": FLAGS.optimizer,
            "lr": FLAGS.learning_rate,
            "betas": FLAGS.adam_betas
        },
        "scheduler": {
            "scheduler": FLAGS.scheduler,
            "dim_embed": FLAGS.hidden_features,
            "warmup_steps": FLAGS.warmup_steps,
        }
    }
    model = attention_maf.DataModule(
        model_hparams, transform_hparams, optimizer_hparams)

    LOGGER.info(f"Dataset path: {FLAGS.dataset_prefix}/{FLAGS.dataset_name}")
    LOGGER.info(f"Run path: {FLAGS.run_prefix}/{FLAGS.run_name}")
    LOGGER.info(f"Version: {FLAGS.run_version}")

    # Read in features and labels and preprocess
    train_path = utils.io_utils.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_prefix, train=True)
    val_path = utils.io_utils.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_prefix, train=False)
    raw_train_features  = utils.io_utils.read_dataset(train_path)[0]
    raw_val_features = utils.io_utils.read_dataset(val_path)[0]

    # Preprocess
    train_features = model.transform(raw_train_features, fit=True)
    val_features = model.transform(raw_val_features, fit=False)
    train_ds = TensorDataset(*train_features)
    val_ds = TensorDataset(*val_features)

    # Create DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=FLAGS.batch_size, shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_ds, batch_size=FLAGS.batch_size, shuffle=False,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Create pytorch_lightning trainer
    if FLAGS.resume:
        if FLAGS.run_version is None:
            raise ValueError(f"run version is required to resume training")
        run_path = utils.get_run(
            FLAGS.run_name, prefix=FLAGS.run_prefix, version=FLAGS.run_version)
        ckpt_path, _ = utils.get_best_checkpoint(run_path)

    trainer = pl.Trainer(
        default_root_dir=os.path.join(FLAGS.run_prefix, FLAGS.run_name),
        accelerator="auto",
        devices=1,
        max_epochs=FLAGS.max_epochs,
        logger=CSVLogger(
            FLAGS.run_prefix, name=FLAGS.run_name, version=FLAGS.run_version),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                filename="{epoch}-{val_loss:.4f}", save_weights_only=True,
                mode="min", monitor="val_loss"),
            pl.callbacks.LearningRateMonitor("epoch"),
            pl.callbacks.early_stopping.EarlyStopping(
                monitor="val_loss", min_delta=0.00, patience=FLAGS.es_patience,
                mode="min", verbose=True),
        ],
    )

    # Start training
    trainer.fit(
        model=model, train_dataloaders=train_loader,
        val_dataloaders=val_loader)

    LOGGER.info("Done!")


if __name__ == "__main__":
   main()
