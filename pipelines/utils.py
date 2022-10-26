
import numpy as np
from merger_tree_ml.utils import io_utils
from merger_tree_ml.models import attention_maf, recurrent_maf

PLOT_PREFIX = "/mnt/home/tnguyen/merger_tree_ml/pipelines_figures"

# model architecture
ALL_MODELS = {
    "AttentionMAF": attention_maf,
    "RecurrentMAF": recurrent_maf,
}
# default log-mass bins for each box
DEFAULT_BINS = {
    "GUREFT05": [7, 7.5, 8, 8.5],
    "GUREFT15": [8.5, 9, 9.5, 10],
    "GUREFT35": [9.5, 10, 10.5, 11],
    "GUREFT90": [10.4, 10.8, 11.2, 11.6],
    "GUREFT": [7, 8, 9, 10, 11],
}
# default log-mass bin widths for each box
DEFAULT_WIDTHS= {
    "GUREFT05": [0.25, 0.25, 0.25, 0.25],
    "GUREFT15": [0.25, 0.25, 0.25, 0.25],
    "GUREFT35": [0.25, 0.25, 0.25, 0.25],
    "GUREFT90": [0.2, 0.2, 0.2, 0.2],
    "GUREFT": [0.2, 0.2, 0.2, 0.2, 0.2],
}


def read_dataset(dataset_name, dataset_prefix):
    """ Read dataset for pipeline testing"""

    # read in training and validation data and concatenate
    train_data, _, headers = io_utils.read_dataset(
        io_utils.get_dataset(dataset_name, prefix=dataset_prefix, train=True))
    val_data, _, _ = io_utils.read_dataset(
        io_utils.get_dataset(dataset_name, prefix=dataset_prefix, train=False))
    data = {key: np.concatenate(
        [train_data[key], val_data[key]]) for key in train_data}

    # get time and trees info
    trees = data.get("x")
    times = data.get("t")
    trees = [np.hstack([trees[i], times[i]]) for i in range(len(trees))]

    # pad sequence to maximum length
    seq_len = np.array([len(trees[i]) for i in range(len(trees))])
    max_len = np.max(seq_len)
    trees_padded = np.zeros((len(trees), max_len, *trees[0].shape[1:]))
    for i in range(len(trees)):
        trees_padded[i, :seq_len[i]] = trees[i]

    return trees_padded, seq_len

def get_model_arch(model_arch):
     """ Return architecture given arch name """
     if model_arch not in ALL_MODELS:
         raise KeyError(
                 f"Unknown arch name \"{model_arch}\"."\
                 f"Available archs are: {str(ALL_MODELS.keys())}")
     return ALL_MODELS[model_arch]

