
from collections import OrderedDict

import numpy as np
import scipy.interpolate as interpolate
from merger_tree_ml import physics
from merger_tree_ml.utils import io_utils

import compute_statistics
import compute_mass_concentration
import sample_and_plot_trees
import self_similarity

ALL_PIPELINES = OrderedDict([
    ("sample_and_plot_trees", sample_and_plot_trees),
    ("compute_mass_concentration", compute_mass_concentration),
    ("compute_statistics", compute_statistics),
    ("self_similarity", self_similarity)
])

PLOT_PREFIX = "/mnt/home/tnguyen/merger_tree_ml/pipelines_figures"

# default log-mass bins for each box
DEFAULT_BINS = {
    "GUREFT05": [7, 7.5, 8, 8.5],
    "GUREFT15": [8.5, 9, 9.5, 10],
    "GUREFT35": [9.5, 10, 10.5, 11],
    "GUREFT90": [10.4, 10.8, 11.2, 11.6],
    "GUREFT": [7, 8, 9, 10, 11],
    "BP": [11, 12, 13, 14],
    "TNG50": [8.5, 9, 9.5, 10, 10.5],
    "TNG100": [9.5, 10, 10.5, 11, 11.5],
    "TNG300": [10, 11, 12, 13, 14],
}
# default log-mass bin widths for each box
DEFAULT_WIDTHS = {
    "GUREFT05": [0.25, 0.25, 0.25, 0.25],
    "GUREFT15": [0.25, 0.25, 0.25, 0.25],
    "GUREFT35": [0.25, 0.25, 0.25, 0.25],
    "GUREFT90": [0.2, 0.2, 0.2, 0.2],
    "GUREFT": [0.2, 0.2, 0.2, 0.2, 0.2],
    "BP": [0.2, 0.2, 0.2, 0.2],
    "TNG50": [0.2, 0.2, 0.2, 0.2, 0.2],
    "TNG100": [0.2, 0.2, 0.2, 0.2, 0.2],
    "TNG300": [0.2, 0.2, 0.2, 0.2, 0.2],
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

def create_interpolator(times, is_omega=True):
    cosmo = physics.DEFAULT_COSMO
    Pk = physics.DEFAULT_Pk

    if is_omega:
        redshift = physics.calc_redshift(times, cosmo=cosmo)
    else:
        redshift =  times

    mass_arr = np.logspace(-1, 20, 100)
    XX, YY = np.meshgrid(mass_arr, redshift)
    XX = XX.ravel()
    YY = YY.ravel()
    ZZ = -physics.calc_Svar(XX, YY, cosmo, Pk, kmin=1e-5, kmax=1e10)
    XX = np.log10(XX)


    def svar_to_mass(svar, z):
        if is_omega:
            z = physics.calc_redshift(z, cosmo=cosmo)
        return interpolate.griddata((ZZ, YY), XX, (svar, z))

    def mass_to_svar(mass, z):
        if is_omega:
            z = physics.calc_redshift(z, cosmo=cosmo)
        return interpolate.griddata((XX, YY), ZZ, (mass, z))

    return svar_to_mass, mass_to_svar


