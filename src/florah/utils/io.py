
import h5py
import numpy as np


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

def read_dataset(path, features_list=[], to_array=True):
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
                node_features[key] = [
                    feat[ptr[i]:ptr[i+1]] for i  in range(len(ptr)-1)]

        # read tree features
        tree_features = {}
        for key in headers['tree_features']:
            if key in features_list:
                tree_features[key] = f[key][:]

    if to_array:
        node_features = {
            p: np.array(v, dtype='object') for p, v in node_features.items()}

    return node_features, tree_features, headers
