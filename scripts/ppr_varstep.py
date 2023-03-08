import os
import h5py
import numpy as np
import pandas as pd
from merger_tree_ml import utils, physics


def squeeze_array(data):
    ptr = np.cumsum([len(d) for d in data])
    ptr = np.insert(ptr, 0, 0)
    new_data = np.concatenate(data)
    return new_data, ptr

def unsqueeze_array(data, ptr):
    new_data = [data[ptr[i]:ptr[i+1]] for i  in range(len(ptr)-1)]
    return new_data

def calc_props(data):
    """ Calculate derived properties """
    cosmo = physics.DEFAULT_COSMO
    Pk = physics.DEFAULT_Pk

    # calculate log mass
    mass, ptr = squeeze_array(data['mass'])
    data['log_mass'] = unsqueeze_array(np.log10(mass), ptr)

    # calculate DM concentration
    rvir, ptr = squeeze_array(data['rvir'])
    rs, _ = squeeze_array(data['rs'])
    cvir = rvir / rs
    cvir = unsqueeze_array(cvir, ptr)
    data['cvir'] = cvir

    # Calculate self-similar mass and time variable
    zred, ptr = squeeze_array(data['redshift'])
    zred[zred < 0] = 0

    # calculate negative mass variance S
    neg_svar_mass = physics.calc_Svar(mass, zred, cosmo=cosmo, P=Pk)
    neg_svar_mass = unsqueeze_array(-neg_svar_mass, ptr)
    sigma_r = Pk.sigma_r(rvir, kmin=1e-8, kmax=1e3)
    neg_svar_rvir = unsqueeze_array(-sigma_r**2, ptr)
    data['neg_svar_mass'] = neg_svar_mass
    data['neg_svar_rvir'] = neg_svar_rvir

    # calculate self-similar time variable omega
    omega = unsqueeze_array(physics.calc_omega(zred, cosmo=cosmo), ptr)
    data['omega'] = omega
    data['aexp'] = unsqueeze_array(1 / (1 + zred), ptr)

    return data


def sample_cumulative(x, step_min , step_max, ini=None):
    steps = np.random.randint(step_min, step_max, len(x))
    if ini is not None:
        steps = np.insert(steps, 0, ini)
    indices = np.cumsum(steps)
    indices = indices[indices < len(x)]
    return indices


if __name__ == "__main__":

    # Define parameters
    dataset_name = "BP"
    out_dataset_name = "CombinedBoxH/BP"
    prefix = utils.io_utils.DEFAULT_RAW_DATASET_PATH
    step_min_1 = 4
    step_max_1 = 7
    step_min_2 = 4
    step_max_2 = 7
    zred_min = 5
    zred_mid = 9
    zred_max = 9
    min_seq_len = 10
    max_seq_len = 10
    num_samples_per_tree = 10
    num_ini = 5
    step_ini = 4
    seed = 10
    train_frac = 0.9
    node_props = ['log_mass', 'cvir', ]
    time_props = ['aexp', ]
    default_headers = {
        'step_min_1': step_min_1,
        'step_max_1': step_max_1,
        'step_min_2': step_min_2,
        'step_max_2': step_max_2,
        'num_samples_per_tree': num_samples_per_tree,
        'num_ini': num_ini,
        'step_ini': step_ini,
        'zred_min': zred_min,
        'zred_mid': zred_mid,
        'zred_max': zred_max,
        'min_seq_len': min_seq_len,
        'max_seq_len': max_seq_len,
        'node_props': node_props,
        'time_props': time_props,
        'seed': seed if seed is not None else -1
    }

    path = utils.io_utils.get_dataset(
        os.path.join(dataset_name, 'raw_data.h5'), prefix=prefix, is_dir=False)
    node_features, tree_features, headers = utils.io_utils.read_dataset(path)
    node_features = calc_props(node_features)
    num_samples = headers['num_trees']

    x_data = []
    t_data = []
    tree_id = []

    print(num_samples)

    for i in range(num_samples):
        if (i % (num_samples // 10)) == 0:
            print(i, len(x_data))

        # get tree properties, log mass and redshift
        log_mass = np.log10(node_features['mass'][i])
        zred = node_features['redshift'][i]
        zred = zred[(zred_min <= zred) & (zred <= zred_max)]
        idx_mid = np.where(zred <= zred_mid)[0][-1] + 1
        tree = np.stack([node_features[p][i] for p in node_props]).T
        time = np.stack([node_features[p][i] for p in time_props]).T

        # sample tree indices
        for j in range(num_ini):
            for _ in range(num_samples_per_tree):
                indices1 = sample_cumulative(
                    zred[:idx_mid], step_min_1, step_max_1, ini=j * step_ini)
                indices2 = sample_cumulative(
                    zred[idx_mid:], step_min_2, step_max_2)
                indices = np.concatenate([indices1, indices2 +  idx_mid])

                tree_sample = tree[indices]
                time_sample = time[indices]
                log_mass_sample = log_mass[indices]

                stop = np.where(np.diff(log_mass_sample) > 0)[0]
                stop = len(tree_sample) if len(stop) == 0 else stop[0]
                stop = min(stop, max_seq_len + 1)

                if stop < min_seq_len + 1 - j:
                    continue

                tree_sample = tree_sample[:stop]
                time_sample = time_sample[:stop]

                # add to array
                x_data.append(tree_sample)
                t_data.append(time_sample)
                tree_id.append(i)

    tree_id = np.array(tree_id)

    # divide data into training and validation
    num_total = len(x_data)
    num_train = int(num_total * train_frac)

    np.random.seed(seed)
    shuffle = np.random.permutation(num_total)
    x_data = [x_data[i] for i in shuffle]
    t_data = [t_data[i] for i in shuffle]
    tree_id = tree_id[shuffle]

    train_x = x_data[:num_train]
    train_t = t_data[:num_train]
    train_tree_id = tree_id[:num_train]
    val_x = x_data[num_train:]
    val_t = t_data[num_train:]
    val_tree_id = tree_id[num_train:]

    # create pointer
    train_ptr = np.cumsum([len(feat) for feat in train_x])
    train_ptr = np.insert(train_ptr, 0, 0)
    val_ptr = np.cumsum([len(feat) for feat in val_x])
    val_ptr = np.insert(val_ptr, 0, 0)

    print(train_ptr)
    print(val_ptr)

    # create headers
    train_headers = default_headers.copy()
    train_headers['num_trees'] = len(train_x)
    val_headers = default_headers.copy()
    val_headers['num_trees'] = len(val_x)

    # add to dictionary
    train_features = {
        'x': list(train_x),
        't': list(train_t),
    }
    train_tree_features = {'tree_id': train_tree_id}
    val_features = {
        'x': list(val_x),
        't': list(val_t),
    }
    val_tree_features = {'tree_id': val_tree_id}

    # write training and validation dataset
    new_train_path = os.path.join(
        utils.io_utils.DEFAULT_DATASET_PATH, out_dataset_name, 'training.h5')
    new_val_path = os.path.join(
        utils.io_utils.DEFAULT_DATASET_PATH, out_dataset_name, 'validation.h5')

    os.makedirs(os.path.dirname(new_train_path), exist_ok=True)
    os.makedirs(os.path.dirname(new_val_path), exist_ok=True)

    utils.io_utils.write_dataset(
        new_train_path, train_features, tree_features=train_tree_features,
        ptr=train_ptr, headers=train_headers)
    utils.io_utils.write_dataset(
        new_val_path, val_features, tree_features=val_tree_features,
        ptr=val_ptr, headers=val_headers)

