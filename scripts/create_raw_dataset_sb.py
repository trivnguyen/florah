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

def get_ancestors(halo, node_props, branch_id=0, min_mass=0):
    """ Get full halo trees """

    features = {p: [np.array(halo['prog', p]), ] for p in node_props}
    branch_indices = [branch_id, ]

    for prog in list(halo['prog']):
        ancestors = list(prog.ancestors)

        for i, anc in enumerate(ancestors[1:]):
            if anc['mass'] >= min_mass:
                next_branch_id = branch_id + i + 1
                next_features, next_branch_indices = get_ancestors(
                    anc, node_props, next_branch_id, min_mass)
                for p in node_props:
                    features[p] += next_features[p]
                branch_indices += next_branch_indices

    return features, branch_indices

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
    outdir = os.path.join("/mnt/home/tnguyen/ceph/raw_dataset/", f"{box_name}_test")
    os.makedirs(outdir, exist_ok=True)

    node_props = ['mass', 'redshift', 'rvir', 'vrms', 'vmax', 'rs']

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

        node_features = {p: [] for p in node_props}
        tree_features = {'tree_id': [], 'branch_id': []}
        for itree, halo in enumerate(tree_list):
            # print out progress
            if itree % (n_trees//10) == 0:
                print(itree)
            # get tree and all ancestors
            halo_features, halo_branch_indices = get_ancestors(
                halo, node_props, min_mass=Mmin_root)
            for p in node_props:
                node_features[p] += halo_features[p]
            tree_features['tree_id'] += [indices[itree], ] * len(halo_branch_indices)
            tree_features['branch_id'] += halo_branch_indices
        tree_features['mass_ini'] = [m[0] for m in node_features['mass']]
        tree_features['tree_len'] = [len(m) for m in node_features['mass']]

        for p in node_props:
            node_features[p] = np.array(node_features[p], dtype='object')

        tree_props = list(tree_features.keys())
        for p in tree_props:
            tree_features[p] = np.array(tree_features[p])

        # Shuffle tree
        n_trees_final = len(tree_features['tree_len'])

        shuffle = np.random.permutation(n_trees_final)
        node_features = {p: node_features[p][shuffle] for p in node_props}
        tree_features = {p: tree_features[p][shuffle] for p in tree_props}

        # Write dataset
        ptr = np.cumsum(tree_features['tree_len'])
        ptr = np.insert(ptr, 0, 0)
        utils.io_utils.write_dataset(
            output, node_features, tree_features, ptr=ptr,
            headers=dict(
                num_trees=n_trees_final,
                Mdm=Mdm, Mmin_root=Mmin_root
            )
        )

if __name__ == "__main__":
    FLAGS = parse_cmd()
    main(**vars(FLAGS))

