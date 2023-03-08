
import os

import numpy as np

from ..config import DEFAULT_DATA_PATH

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

def squeeze_array(data):
    ptr = np.cumsum([len(d) for d in data])
    ptr = np.insert(ptr, 0, 0)
    new_data = np.concatenate(data)
    return new_data, ptr

def unsqueeze_array(data, ptr):
    new_data = [data[ptr[i]:ptr[i+1]] for i  in range(len(ptr)-1)]
    new_data = np.array(new_data, dtype='object')
    return new_data

def sample_cumulative(N_max, step_min , step_max, ini=None):
    steps = np.random.randint(step_min, step_max, N_max)
    if ini is not None:
        steps = np.insert(steps, 0, ini)
    indices = np.cumsum(steps)
    indices = indices[indices < N_max]
    return indices


