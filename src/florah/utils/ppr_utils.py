
import os

import numpy as np

from ..config import DEFAULT_DATA_PATH

# Redshift functions
def get_ztable(box, return_scale_factor=False):
    """ Read in z table """
    if box == "BP":
        path = os.path.join(DEFAULT_DATA_PATH, "bp_redshifts.txt")
    elif box == "VSMDPL":
       path = os.path.join(DEFAULT_DATA_PATH, "vsmdpl_redshifts.txt")
    elif box in ("GUREFT", "GUREFT05", "GUREFT15", "GUREFT35", "GUREFT90"):
        path = os.path.join(DEFAULT_DATA_PATH, "gureft_redshifts.txt")
    z_table = np.genfromtxt(path, delimiter=',', unpack=True)[-1]
    if return_scale_factor:
        return 1 / (1 + z_table)
    return z_table

def find_max_tree_len(ztable, zmax):
    ''' Find the maximum tree length given redshift table and max redshift '''
    idx = np.searchsorted(ztable, zmax, side='left')
    if idx == 0:
        return None
    else:
        return idx

# Halo and tree functions
def squeeze_array(data):
    ptr = np.cumsum([len(d) for d in data])
    ptr = np.insert(ptr, 0, 0)
    new_data = np.concatenate(data)
    return new_data, ptr

def unsqueeze_array(data, ptr):
    new_data = [data[ptr[i]:ptr[i+1]] for i  in range(len(ptr)-1)]
    new_data = np.array(new_data, dtype='object')
    return new_data

def calc_derived_props(data):
    """ Calculate derived properties """
    # calculate log mass
    mass, ptr = squeeze_array(data['mass'])
    data['log_mass'] = unsqueeze_array(np.log10(mass), ptr)

    # calculate DM concentration
    rvir, ptr = squeeze_array(data['rvir'])
    rs, _ = squeeze_array(data['rs'])
    cvir = rvir / rs
    data['cvir'] = unsqueeze_array(cvir, ptr)
    data['log_cvir'] = unsqueeze_array(np.log10(cvir), ptr)

    # calculate scale factor
    zred, ptr = squeeze_array(data['redshift'])
    zred[zred < 0] = 0
    data['aexp'] = unsqueeze_array(1 / (1 + zred), ptr)

    return data

# Sampling functions
def sample_cumulative(N_max, step_min , step_max, ini=None):
    steps = np.random.randint(step_min, step_max, N_max)
    if ini is not None:
        steps = np.insert(steps, 0, ini)
    indices = np.cumsum(steps)
    indices = indices[indices < N_max]
    return indices

def sample_uniform(x, bins, size=1, replace=True):
    ''' Sample an array x uniformly and return the index of sampled x '''
    hist, bin_edges = np.histogram(x, bins=bins, density=True)

    # compute the inverse histogram probabilities
    inv_hist = np.where(
        hist > 0, 1.0 / (hist * np.diff(bin_edges)), 0)

    # assign probabilities to each tree
    bin_idx = np.digitize(x, bin_edges) - 1
    p = np.zeros_like(x)
    for i in range(len(bin_edges) - 1):
        p[bin_idx==i] = inv_hist[i]

    # normalize probabilities to 1
    p /= np.sum(p)

    # sample indices and return
    return np.random.choice(
        np.arange(len(x)), p=p, size=size, replace=replace)
