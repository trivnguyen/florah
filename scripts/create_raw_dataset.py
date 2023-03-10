#!/usr/bin/env python
# coding: utf-8

import os
import h5py
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import ytree
from florah import utils


def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("--box-name", required=True, type=str)
    parser.add_argument("--num-subbox-dim", default=1, required=False, type=int)
    parser.add_argument("--min-num-root", default=500, required=False, type=float)
    parser.add_argument("--ijob", required=False, type=int)
    return parser.parse_args()


def main(box_name, num_subbox_dim, min_num_root=500, ijob=None):

    # Define parameters
    basedir = os.path.join("/mnt/home/tnguyen/isotrees", box_name)
    outdir = os.path.join("/mnt/home/tnguyen/ceph/raw_dataset/", box_name)
    os.makedirs(outdir, exist_ok=True)

    node_props = ['mass', 'redshift', 'rvir', 'vrms', 'vmax', 'rs']
    tree_props = ['indices', 'mass_ini', 'tree_len']

    # Read meta data
    meta = pd.read_csv(
        '/mnt/home/tnguyen/merger_tree_ml/tables/meta.csv',
        sep=',', header=0)
    Mdm = float(meta['Mdm'][meta['name']==box_name])
    Mmin_root = Mdm * min_num_root

    print("Mass resolution: {}".format(Mdm))
    print("Minimum root mass: {}".format(Mmin_root))
    print("Minimum root number: {}".format(min_num_root))

    # Get all boxes
    tree_files = []
    for i in range(num_subbox_dim):
        for j in range(num_subbox_dim):
            for k in range(num_subbox_dim):
                path = os.path.join(
                    basedir, f"isotree_{i}_{j}_{k}.dat")
                if not os.path.exists(path):
                    print(f"{path} not exist.  skipping...")
                    continue
                tree_files.append(path)

    # Run over box or iterate over all boxes
    num_subboxes = len(tree_files)
    iterjob = [ijob, ] if ijob is not None else range(num_subboxes)

    for i in iterjob:
        # use ytree to read in trees
        tree_fn = tree_files[i]
        data = ytree.load(tree_fn)

        if "TNG" in box_name:
            data.add_alias_field("rvir", "Rvir")
        elif "VSMDPL" in box_name:
            data.add_alias_field("rvir", "Rvir")
            data.add_alias_field("mass", "Mvir")
        else:
            data.add_alias_field("mass", "mvir")

        output = os.path.join(outdir, Path(tree_fn).stem + '.h5')
        print(tree_fn, output)

        # selection by minimum root mass
        mass = data['mass'].value
        indices = np.where(mass > Mmin_root)[0]
        tree_list = list(data[indices])

        n_trees = len(tree_list)
        print(n_trees)

        node_features = {p : [] for p in node_props}
        for itree, tree in enumerate(tree_list):
            if itree % (n_trees//10) == 0:
                print(itree)
            for p in node_props:
                node_features[p].append(np.array(tree['prog', p]))

        for p in node_props:
            node_features[p] = np.array(node_features[p], dtype='object')

        # Compute tree features
        tree_features = {'indices': indices}
        tree_features['mass_ini'] = np.array([m[0] for m in node_features['mass']])
        tree_features['tree_len'] = np.array([len(m) for m in node_features['mass']])

        # Shuffle tree
        shuffle = np.random.permutation(n_trees)
        print(len(node_features[p]), n_trees)
        node_features = {p: node_features[p][shuffle] for p in node_props}
        tree_features = {p: tree_features[p][shuffle] for p in tree_props}

        # Write dataset
        ptr = np.cumsum(tree_features['tree_len'])
        ptr = np.insert(ptr, 0, 0)
        utils.io_utils.write_dataset(
            output, node_features, tree_features, ptr=ptr,
            headers=dict(
                num_trees=n_trees,
                Mdm=Mdm, Mmin_root=Mmin_root
            )
        )

if __name__ == "__main__":
    FLAGS = parse_cmd()
    main(**vars(FLAGS))
