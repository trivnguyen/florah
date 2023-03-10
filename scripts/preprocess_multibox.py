#!/usr/bin/env python
# coding: utf-8

import argparse
import os

import numpy as np
import pandas as pd
import yaml

from merger_tree_ml import utils, config
from merger_tree_ml.logger import logger

def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.parse_args('config', type=str, default='Config file')
    return parser.parse_args()

def main():
    FLAGS = parse_cmd()

    with open(FLAGS['config'], 'r') as file:
        # Load the contents of the file using PyYAML
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    dset_cfg = cfg['dataset']
    halo_cfg = cfg['halo']
    tree_cfg = cfg['tree']

    zmax_table = pd.read_csv(dset_cfg['zmax_table'])

    # Preprocess trees
    trees_id = []
    trees_box = []
    trees_preprocess = []
    branches_id = []
    for ibox, box in enumerate(dset_cfg['box']):
        logger.info('Read box {}: {}'.format(ibox, box))

        # read box and calculate derived node features
        box_prefix = dset_cfg.get('box_prefix', config.DEFAULT_RAW_DATASET_PATH)
        box_path = os.path.join(box_prefix, box, 'raw_data.h5')
        node_features = utils.io_utils.read_dataset(box_path)[0]
        node_features = utils.ppr_utils.calc_derived_props(node_features)

        # get minimum mass of halo
        Mhalo_min = halo_cfg['mass_dark'][ibox] * halo_cfg['num_min_halo']
        Mroot_min = halo_cfg['mass_dark'][ibox] * halo_cfg['num_min_root']

        # get redshift table
        ztable = utils.ppr_utils.get_ztable(box)

        # get maximum redshift per mass bin
        zmax_table_cut = zmax_table[zmax_table['box']==box]
        bins = zmax_table_cut['bin_edge'].values
        zmax_index = zmax_table_cut['index'].values

        # print out some info
        logger.info("Minimum root mass: {}".format(Mroot_min))
        logger.info("Minimum halo mass: {}".format(Mhalo_min))
        logger.info('Mass bins: {}'.format(bins))
        logger.info('Max redshift index: {}'.format(zmax_index))
        logger.info('Uniform: {}'.format(halo_cfg['uniform']))

        # iterate over all trees in box
        num_trees = len(node_features['mass'])

        # sample trees based on the initial mass
        if halo_cfg['uniform']:
            logm0 = np.array([np.log10(m[0]) for m in node_features['mass']])
            nbins =  int(np.ceil(
                (logm0.max() - logm0.min()) / halo_cfg['dlogm'] + 1))
            sampled_idx = utils.ppr_utils.sample_uniform(
                logm0, nbins, size=num_trees, replace=True)
        else:
            sampled_idx = np.arange(num_trees)

        for i in range(len(sampled_idx)):
            if i % (num_trees // 10) == 0:
                logger.info('Progress {}/{} halos: {} trees'.format(
                    i, num_trees, len(trees_preprocess)))

            # get node properties of each tree
            tree = np.stack(
                [node_features[p][sampled_idx[i]] for p in halo_cfg['node_props']],
                axis=1)

            # ignore tree with low root mask
            if tree[0, 0] < np.log10(Mroot_min):
                continue

            # identify first unresolved halo and truncate tree
            unre_first = np.where(tree[..., 0] <= np.log10(Mhalo_min))[0]
            if len(unre_first) > 0:
                tree = tree[:unre_first[0]]

            # get maximum length of tree
            # max redshift for all trees
            zmax_all = halo_cfg.get('redshift_max', 1000)
            max_tree_len_all = utils.ppr_utils.find_max_tree_len(
                ztable, zmax_all)

            # get max redshift based on mass bin
            bin_idx = np.digitize(tree[0, 0], bins)
            if (bin_idx == 0) or (bin_idx == len(bins)):
                max_tree_len = max_tree_len_all
            else:
                max_tree_len = min(max_tree_len_all, zmax_index[bin_idx-1])

            # truncate tree based on maximum length
            tree = tree[:max_tree_len]

            # sample subtrees
            for j in range(tree_cfg['num_trees']):
                ini = np.random.randint(
                    tree_cfg['min_ini'], tree_cfg['max_ini'] + 1)
                indices = utils.ppr_utils.sample_cumulative(
                    len(tree), tree_cfg['step_min'], tree_cfg['step_max'] + 1,
                    ini=ini)
                subtree = tree[indices]

                if tree_cfg['enforce_growth']:
                    stop = np.where(np.diff(subtree) > 0)[0]
                    stop = len(subtree) if len(stop) == 0 else stop[0]
                    subtree = subtree[:stop]
                subtree = subtree[:tree_cfg['max_length']]

                if (len(subtree) < tree_cfg['min_length']):
                    continue

                # add subtree to final list
                trees_preprocess.append(subtree)
                trees_box.append(ibox)
                trees_id.append(sampled_idx[i])

    trees_preprocess = np.array(trees_preprocess, dtype='object')


    trees_id = np.array(trees_id)
    trees_box = np.array(trees_box)

    # Create training and validation dataset
    num_total = len(trees_preprocess)
    num_train = int(num_total * dset_cfg['train_frac'])

    logger.info('Training samples: {}'.format(num_train))
    logger.info('Validation samples: {}'.format(num_total - num_train))

    # shuffle data
    if dset_cfg['shuffle']:
        shuffle = np.random.permutation(num_total)
        trees_preprocess = trees_preprocess[shuffle]
        trees_id = trees_id[shuffle]

    # divide data into training and validation
    trees_train =  trees_preprocess[:num_train]
    trees_val = trees_preprocess[num_train:]
    trees_id_train = trees_id[:num_train]
    trees_id_val = trees_id[num_train:]
    trees_box_train = trees_box[:num_train]
    trees_box_val = trees_box[num_train:]

    # divide each tree into halo properties and times
    it =  len(halo_cfg['node_props']) - halo_cfg['n_dim_time']
    x_train = np.array([x[:, :it] for x in trees_train], dtype='object')
    t_train = np.array([x[:, it:] for x in trees_train], dtype='object')
    x_val = np.array([x[:, :it] for x in trees_val], dtype='object')
    t_val = np.array([x[:, it:] for x in trees_val], dtype='object')

    # Write data
    # because each tree has a different length, create pointer to each tree
    train_ptr = np.cumsum([len(feat) for feat in x_train])
    train_ptr = np.insert(train_ptr, 0, 0)
    val_ptr = np.cumsum([len(feat) for feat in x_val])
    val_ptr = np.insert(val_ptr, 0, 0)

    # create headers
    default_headers = {}
    default_headers.update(dset_cfg)
    default_headers.update(halo_cfg)
    default_headers.update(tree_cfg)
    train_headers = default_headers.copy()
    train_headers['num_trees'] = len(x_train)
    val_headers = default_headers.copy()
    val_headers['num_trees'] = len(x_val)

    # add to dictionary
    train_features = {
        'x': list(x_train),
        't': list(t_train),
    }
    train_tree_features = {
        'tree_id': trees_id_train,
        'tree_box': trees_box_train
    }
    val_features = {
        'x': list(x_val),
        't': list(t_val),
    }
    val_tree_features = {
        'tree_id': trees_id_val,
        'tree_box': trees_box_val
    }

    # Create dataset directory and write training and validation  dataset
    out_prefix = dset_cfg.get('out_prefix', config.DEFAULT_DATASET_PATH)
    outdir = os.path.join(out_prefix, dset_cfg['out'])
    os.makedirs(outdir, exist_ok=True)

    logger.info('Writing data to {}'.format(outdir))

    utils.io_utils.write_dataset(
        os.path.join(outdir, 'training.h5'), train_features,
        tree_features=train_tree_features, ptr=train_ptr, headers=train_headers
    )
    utils.io_utils.write_dataset(
        os.path.join(outdir, 'validation.h5'), val_features,
        tree_features=val_tree_features, ptr=val_ptr, headers=val_headers
    )

if __name__ == "__main__":
    main()
