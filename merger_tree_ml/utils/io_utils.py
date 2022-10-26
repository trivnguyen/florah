
import os
import h5py
import glob
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import TensorDataset

from ..envs import DEFAULT_RUN_PATH, DEFAULT_DATASET_PATH, DEFAULT_RAW_DATASET_PATH

def get_all_dataset(prefix=DEFAULT_DATASET_PATH):
    return glob.glob(os.path.join(prefix, "*"))

def get_all_run(prefix=DEFAULT_RUN_PATH):
    return glob.glob(os.path.join(prefix, "*"))

def get_dataset(name, prefix=DEFAULT_DATASET_PATH, train=True, is_dir=True):
    """ Get path to dataset directory """
    # check if prefix directory exists
    if not os.path.exists(prefix):
        raise FileNotFoundError(f"prefix directory {prefix} not found")

    # check if dataset exists in directory
    if is_dir:
        path = os.path.join(prefix, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"dataset {name} not found in directory {prefix}")
        if train:
            path = os.path.join(path, "training.h5")
        else:
            path = os.path.join(path, "validation.h5")
        return path
    else:
        path = os.path.join(prefix, name)
        if not os.path.exists(path):
            path = path + ".h5"
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"dataset {name} not found in directory {prefix}")
            return path
        return path

def get_run(name, version="best", prefix=DEFAULT_RUN_PATH):
    """ Get path to run directory """
    # check if prefix directory exists
    if not os.path.exists(prefix):
        raise FileNotFoundError(f"prefix directory {prefix} not found")
    # check if run directory exists
    path = os.path.join(prefix, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"run {name} not found in directory {prefix}")
    # check if version exists
    if version != "best":
        version_path = os.path.join(path, f"version_{version}")
        if not os.path.exists(version_path):
            raise FileNotFoundError(
                f"version {version} not found in run {name} with prefix {prefix}")
        return version_path
    else:
        return get_best_run(path)[0]

def get_best_run(run_path):
    """ Get the version path with the best checkpoint """
    # iterate over all version
    min_loss = 100000
    best_version_path = None
    for version in range(1000):
        run_version_path = os.path.join(run_path, f"version_{version}")
        if not os.path.exists(run_version_path):
            break
        best_checkpoint, best_version_loss = get_best_checkpoint(run_version_path)
        if best_version_loss < min_loss:
            min_loss = best_version_loss
            best_version_path = run_version_path
    return best_version_path, min_loss

def get_best_checkpoint(run_version_path):
    checkpoints = glob.glob(
        os.path.join(run_version_path, "checkpoints/epoch*.ckpt"))
    # get the loss of each checkpoint and return min loss and version
    min_loss = 100000
    best_checkpoint = None
    for ckpt in checkpoints:
        temp = Path(ckpt).stem
        loss = float(temp.split('=')[-1])
        if loss < min_loss:
            min_loss = loss
            best_checkpoint = ckpt
    return best_checkpoint, min_loss

def write_dataset(path, node_features, tree_features, ptr=None, headers={}):
    """ Write dataset into HDF5 file """
    if ptr is None:
        ptr = np.arange(len(list(node_features.values())[0]))

    default_headers ={
        "node_features": list(node_features.keys()),
        "tree_features": list(tree_features.keys()),
    }
    default_headers['all_features'] = (
        default_headers['node_features'] + default_headers['tree_features'])
    headers.update(default_headers)

    with h5py.File(path, 'w') as f:
        # write pointers
        f.create_dataset('ptr', data=ptr)

        # write node features
        for key in node_features:
            feat = np.concatenate(node_features[key])
            dset = f.create_dataset(key, data=feat)
            dset.attrs.update({'type': 0})

        # write tree features
        for key in tree_features:
            dset = f.create_dataset(key, data=tree_features[key])
            dset.attrs.update({'type': 1})

        # write headers
        f.attrs.update(headers)

def read_dataset(path, features_list=[]):
    """ Read dataset from path """
    with h5py.File(path, 'r') as f:
        # read dataset attributes
        headers = dict(f.attrs)
        if len(features_list) == 0:
            features_list = headers['all_features']

        # read pointer to each tree
        ptr = f['ptr'][:]

        # read node features
        node_features = {}
        for key in headers['node_features']:
            if key in features_list:
                feat = f[key][:]
                node_features[key] = [feat[ptr[i]:ptr[i+1]] for i  in range(len(ptr)-1)]

        # read tree features
        tree_features = {}
        for key in headers['tree_features']:
            if key in features_list:
                tree_features[key] = f[key][:]

    return node_features, tree_features, headers
