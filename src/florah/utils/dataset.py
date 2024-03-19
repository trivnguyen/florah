
import os
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from . import io

def create_dataloader_from_path(
        dataset_path, preprocess, preprocess_kwargs=None, verbose=True, **kwargs):
    """ Create a data loader from a dataset
    Parameters
    ----------
    dataset_path: str
        Path to the dataset
    preprocess: callable
        Preprocess function from coordinates to torch_geometric.data.Data
    preprocess_kwargs: dict
        Keyword arguments for preprocess
    verbose: bool
        Whether to print out the dataset information
    kwargs: dict
        Keyword arguments for DataLoader

    Returns
    -------
    dataloader: torch_geometric.loader.DataLoader
    """
    # find dataset dataset_path, return None if not found
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if preprocess_kwargs is None:
        preprocess_kwargs = {}

    node_features, tree_features, headers = io.read_dataset(dataset_path)
    node_features = preprocess(node_features, **preprocess_kwargs)

    # print out dataset information
    if verbose:
        print(f"Dataset: {dataset_path}")
        print("Headers:")
        for header in headers:
            print(f"- {header}: {headers[header]}")

    return DataLoader(TensorDataset(*node_features), **kwargs)


    # # raw_train_features = {}
    # # for dset_name in FLAGS.dataset_names:
    # #     logger.info(f"Dataset path: {FLAGS.dataset_prefix}/{dset_name}")
    # #     train_path = utils.io_utils.get_dataset(
    # #         dset_name, FLAGS.dataset_prefix, train=True)
    # #     val_path = utils.io_utils.get_dataset(
    # #         dset_name, FLAGS.dataset_prefix, train=False)

    # #     dset_train_features  = utils.io_utils.read_dataset(train_path)[0]
    # #     dset_val_features = utils.io_utils.read_dataset(val_path)[0]

    # #     for key in dset_train_features:
    # #         if raw_train_features.get(key) is None:
    # #             raw_train_features[key] = []
    # #         if raw_val_features.get(key) is None:
    # #             raw_val_features[key] = []
    # #         raw_train_features[key].append(dset_train_features[key])
    # #         raw_val_features[key].append(dset_val_features[key])

    # for key in raw_train_features:
    #     raw_train_features[key] = np.concatenate(raw_train_features[key])
    #     raw_val_features[key] = np.concatenate(raw_val_features[key])
