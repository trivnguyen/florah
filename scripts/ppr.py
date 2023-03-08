#!/usr/bin/env python
# coding: utf-8

import os
import h5py
import numpy as np
import pandas as pd
from merger_tree_ml import utils, physics

def squeeze_array(data):
    ptr = np.cumsum([len(d) for d in data])
    ptr = np.insert(ptr, 0, 0)
    new_data = np.concatenate(data)
    return new_data, ptr

def unsqueeze_array(data, ptr):
    new_data = [data[ptr[i]:ptr[i+1]] for i  in range(len(ptr)-1)]
    return new_data

def calc_props(data):
    """ Calculate derived properties """
    cosmo = physics.DEFAULT_COSMO
    Pk = physics.DEFAULT_Pk

    # calculate log mass
    mass, ptr = squeeze_array(data['mass'])
    data['log_mass'] = unsqueeze_array(np.log10(mass), ptr)

    # calculate DM concentration
    rvir, ptr = squeeze_array(data['rvir'])
    rs, _ = squeeze_array(data['rs'])
    cvir = rvir / rs
    cvir = unsqueeze_array(cvir, ptr)
    data['cvir'] = cvir

    # Calculate self-similar mass and time variable
    zred, ptr = squeeze_array(data['redshift'])
    zred[zred < 0] = 0

    # calculate negative mass variance S
    neg_svar_mass = physics.calc_Svar(mass, zred, cosmo=cosmo, P=Pk)
    neg_svar_mass = unsqueeze_array(-neg_svar_mass, ptr)
    sigma_r = Pk.sigma_r(rvir, kmin=1e-8, kmax=1e3)
    neg_svar_rvir = unsqueeze_array(-sigma_r**2, ptr)
    data['neg_svar_mass'] = neg_svar_mass
    data['neg_svar_rvir'] = neg_svar_rvir

    # calculate self-similar time variable omega
    omega = unsqueeze_array(physics.calc_omega(zred, cosmo=cosmo), ptr)
    data['omega'] = omega

    return data


if __name__ == "__main__":

    # Define parameters
    dataset_name = "GUREFT_BP"
    out_dataset_name = "ParamsA_omega/GUREFT_BP_large"
    prefix = utils.io_utils.DEFAULT_RAW_DATASET_PATH
    seq_start = 0
    min_seq_len = 20
    max_seq_len = 20
    min_subtree_len = 10
    step = 4
    seed = 10
    train_frac = 0.9
    node_props = ['log_mass', 'cvir', ]
    time_props = ['omega', ]
    default_headers = {
        'seq_start': seq_start,
        'min_seq_len': min_seq_len,
        'max_seq_len': max_seq_len,
        'min_subtree_len': min_subtree_len,
        'step': step,
        'node_props': node_props,
        'seed': seed if seed is not None else -1
    }

    # Read in raw dataset
    path = utils.io_utils.get_dataset(
        os.path.join(dataset_name, 'raw_data.h5'), prefix=prefix, is_dir=False)
    node_features, tree_features, headers = utils.io_utils.read_dataset(path)
    node_features = calc_props(node_features)
    num_samples = headers['num_trees']

    x_data = []
    t_data = []
    tree_id = []

    for i in range(num_samples):
        if (i % (num_samples // 100)) == 0:
            print(i)

        # get mass
        log_mass = np.log10(node_features['mass'][i])[::step]
        tree = np.stack([node_features[p][i] for p in node_props]).T
        time = np.stack([node_features[p][i] for p in time_props]).T

        stop = np.where(np.diff(log_mass) > 0)[0]
        stop = len(tree) if len(stop) == 0 else stop[0]
        stop = min(stop, max_seq_len+1)

        tree = tree[:stop * step: step]
        time = time[:stop * step: step]
        if len(tree) < min_seq_len + 1:
            continue

        for j in range(len(tree)):
            subtree = tree[j:]
            subtime = time[j:]
            if len(subtree) < min_subtree_len + 1:
                break

            # add to array
            x_data.append(subtree)
            t_data.append(subtime)
            tree_id.append(i)

    tree_id = np.array(tree_id)

    print(tree.shape, time.shape)

    # divide data into training and validation
    num_total = len(x_data)
    num_train = int(num_total * train_frac)

    np.random.seed(seed)
    shuffle = np.random.permutation(num_total)
    x_data = [x_data[i] for i in shuffle]
    t_data = [t_data[i] for i in shuffle]
    tree_id = tree_id[shuffle]

    train_x = x_data[:num_train]
    train_t = t_data[:num_train]
    train_tree_id = tree_id[:num_train]
    val_x = x_data[num_train:]
    val_t = t_data[num_train:]
    val_tree_id = tree_id[num_train:]

    # create pointer
    train_ptr = np.cumsum([len(feat) for feat in train_x])
    train_ptr = np.insert(train_ptr, 0, 0)
    val_ptr = np.cumsum([len(feat) for feat in val_x])
    val_ptr = np.insert(val_ptr, 0, 0)

    print(train_ptr)
    print(val_ptr)

    # create headers
    train_headers = default_headers.copy()
    train_headers['num_trees'] = len(train_x)
    val_headers = default_headers.copy()
    val_headers['num_trees'] = len(val_x)

    # add to dictionary
    train_features = {
        'x': list(train_x),
        't': list(train_t),
    }
    train_tree_features = {'tree_id': train_tree_id}
    val_features = {
        'x': list(val_x),
        't': list(val_t),
    }
    val_tree_features = {'tree_id': val_tree_id}

    # write training and validation dataset
    new_train_path = os.path.join(
        utils.io_utils.DEFAULT_DATASET_PATH, out_dataset_name, 'training.h5')
    new_val_path = os.path.join(
        utils.io_utils.DEFAULT_DATASET_PATH, out_dataset_name, 'validation.h5')

    os.makedirs(os.path.dirname(new_train_path), exist_ok=True)
    os.makedirs(os.path.dirname(new_val_path), exist_ok=True)

    utils.io_utils.write_dataset(
        new_train_path, train_features, tree_features=train_tree_features,
        ptr=train_ptr, headers=train_headers)
    utils.io_utils.write_dataset(
        new_val_path, val_features, tree_features=val_tree_features,
        ptr=val_ptr, headers=val_headers)
