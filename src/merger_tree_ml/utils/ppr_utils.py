
import numpy as np

from ..config import BP_TABLE_PATH, GUREFT_TABLE_PATH, VSMDPL_TABLE_PATH

def get_ztable(box, return_scale_factor=False):
    """ Read in z table """
    if 'BP' in box:
        z_table = np.genfromtxt(
            BP_TABLE_PATH, delimiter=',', unpack=True)[-1]
    elif 'GUREFT' in box:
        z_table = np.genfromtxt(
            GUREFT_TABLE_PATH, delimiter=',', unpack=True)[-1]
    elif 'VSMDPL' in box:
        z_table = np.genfromtxt(
            VSMDPL_TABLE_PATH, delimiter=',', unpack=True)[-1]
    else:
        return None

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


