#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import shutil
import argparse
import logging
import h5py

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.loggers import CSVLogger

from merger_tree_ml import utils
from merger_tree_ml.models import recurrent_maf
from merger_tree_ml.envs import DEFAULT_RUN_PATH, DEFAULT_DATASET_PATH


# Set logger
def set_logger():
    ''' Set up stdv out logger and file handler '''
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
        "--dataset-names", required=False, type=str, nargs='+', default=["", ],
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
        help="Enable to resume previous run. Version number is required")
    # model and graph args
    parser.add_argument(
        "--in-channels", required=True, type=int,
        help="Number of input channels")
    parser.add_argument(
        "--out-channels", required=False, type=int,
        help="Number of output channels")
    parser.add_argument(
        "--time-embed-channels", required=False, type=int,
        help="Number of time embedding channels")
    parser.add_argument(
        "--rnn-name", required=False, type=str.upper, default="GRU",
        help="Type of RNN layers")
    parser.add_argument(
        "--hidden-features", required=False, type=int, default=64,
        help="Number of hidden RNN features")
    parser.add_argument(
        "--num-layers", required=False, type=int, default=2,
        help="Number of RNN transformations")
    parser.add_argument(
        "--hidden-features-flows", required=False, type=int, default=128,
        help="Number of hidden features of MAF transformations")
    parser.add_argument(
        "--num-layers-flows", required=False, type=int, default=4,
        help="Number of MAF transformations")
    parser.add_argument(
        "--num-blocks", required=False, type=int, default=2,
        help="Number of blocks per MAF transformation")
    # transfrom args
    parser.add_argument(
        "--subtract-dim", required=False, type=int, nargs="+", default=[0, ],
        help="Dimension to subtract Y from X")
    # training args
    parser.add_argument(
        "--batch-size", required=False, type=int, default=1024,
        help="Batch size")
    parser.add_argument(
        "--max-epochs", required=False, type=int, default=1000,
        help="Maximum number of epochs. Stop training automatically if exceeds")
    parser.add_argument(
        "--es-stopping", required=False, type=int, default=40,
        help="Patience for early stopping")
    parser.add_argument(
        "--num-workers", required=False, type=int, default=1,
        help="Number of workers")
    # optimizer args
    parser.add_argument(
        "--optimizer", required=False, type=str, default="AdamW",
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
        "--scheduler", required=False, type=str, default="ReduceLROnPlateau",
        help="Type of LR scheduler to use")
    parser.add_argument(
        "--scheduler-patience", required=False, type=int, default=20,
        help="Patience for LR scheduler")

    return parser.parse_args()

def main():
    # Parse cmd args
    FLAGS = parse_cmd()
    LOGGER = set_logger()

    # Create data module
    if FLAGS.out_channels is None:
        FLAGS.out_channels = FLAGS.in_channels
    model_hparams = {
        "in_channels": FLAGS.in_channels,
        "out_channels": FLAGS.out_channels,
        "time_embed_channels": FLAGS.time_embed_channels,
        "num_layers": FLAGS.num_layers,
        "hidden_features": FLAGS.hidden_features,
        "num_layers_flows": FLAGS.num_layers_flows,
        "hidden_features_flows": FLAGS.hidden_features_flows,
        "num_blocks": FLAGS.num_blocks,
        "rnn_name": FLAGS.rnn_name,
        "rnn_hparams": {},
    }
    transform_hparams = {
        "nx": FLAGS.in_channels,
        "ny": FLAGS.out_channels,
        "sub_dim": FLAGS.subtract_dim,
    }
    optimizer_hparams = {
        "optimizer": {
            "optimizer": FLAGS.optimizer,
            "lr": FLAGS.learning_rate,
            "betas": FLAGS.adam_betas
        },
        "scheduler": {
            "scheduler": FLAGS.scheduler,
            "patience": FLAGS.scheduler_patience
        }
    }
    model = recurrent_maf.DataModule(
        model_hparams, transform_hparams, optimizer_hparams)

    LOGGER.info(f"Run path: {FLAGS.run_prefix}/{FLAGS.run_name}")
    LOGGER.info(f"Version: {FLAGS.run_version}")

    # Read in features and labels and preprocess
    raw_train_features = {}
    raw_val_features = {}
    for dset_name in FLAGS.dataset_names:
        LOGGER.info(f"Dataset path: {FLAGS.dataset_prefix}/{dset_name}")
        train_path = utils.io_utils.get_dataset(
            dset_name, FLAGS.dataset_prefix, train=True)
        val_path = utils.io_utils.get_dataset(
            dset_name, FLAGS.dataset_prefix, train=False)

        dset_train_features  = utils.io_utils.read_dataset(train_path)[0]
        dset_val_features = utils.io_utils.read_dataset(val_path)[0]

        for key in dset_train_features:
            if raw_train_features.get(key) is None:
                raw_train_features[key] = []
            if raw_val_features.get(key) is None:
                raw_val_features[key] = []
            raw_train_features[key].append(dset_train_features[key])
            raw_val_features[key].append(dset_val_features[key])

    for key in raw_train_features:
        raw_train_features[key] = np.concatenate(raw_train_features[key])
        raw_val_features[key] = np.concatenate(raw_val_features[key])

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
        run_path = utils.io_utils.get_run(
            FLAGS.run_name, prefix=FLAGS.run_prefix, version=FLAGS.run_version)
        ckpt_path, _ = utils.io_utils.get_best_checkpoint(run_path)
    else:
        ckpt_path = None

    trainer = pl.Trainer(
        default_root_dir=os.path.join(FLAGS.run_prefix, FLAGS.run_name),
        accelerator="auto",
        devices=1,
        max_epochs=FLAGS.max_epochs,
        logger=CSVLogger(
            FLAGS.run_prefix, name=FLAGS.run_name, version=FLAGS.run_version),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                filename="{epoch}-{val_loss:.4f}", save_weights_only=False,
                mode="min", monitor="val_loss"),
            pl.callbacks.LearningRateMonitor("epoch"),
            pl.callbacks.early_stopping.EarlyStopping(
                monitor="val_loss", min_delta=0.00, patience=FLAGS.es_stopping,
                mode="min", verbose=True),
        ],
    )

    # Start training
    trainer.fit(
        model=model, train_dataloaders=train_loader,
        val_dataloaders=val_loader, ckpt_path=ckpt_path)

    LOGGER.info("Done!")

if __name__ == "__main__":
    main()
