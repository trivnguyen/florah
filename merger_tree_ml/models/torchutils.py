from typing import Optional, Union

import numpy as np
import torch
from torch import FloatTensor

def look_ahead_mask(seq_len: int) -> FloatTensor:
    """  Return mask which prevents future data to be seen """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask[mask.bool()] = -torch.inf
    return mask


def sample_trees_recurrent(
        model: torch.nn.Module, roots: np.ndarray, max_len: int,
        time: Optional[np.ndarray] = None, to_numpy: bool = True,
        device: Optional = None,
    ) -> Union[FloatTensor, np.ndarray]:
    """ Sample trees using recurrent MAF model """
    model.eval()
    num_trees, num_feat = roots.shape
    num_feat_nt = num_feat - 1 if time is not None else num_feat
    sub_dim = model.transform.sub_dim

    if device is None:
        device = torch.device('cpu' if torch.cuda.is_available() else 'cuda')

    with torch.no_grad():
        # initialize array with root features
        x = torch.zeros(
            (num_trees, max_len, num_feat), device=device, dtype=torch.float32)
        x[:, 0] = torch.tensor(roots, dtype=torch.float32)

        # add time information to features
        if time is not None:
            x[..., -1] = torch.tensor(
                time, dtype=torch.float32).repeat(num_trees, 1)

        # iteratively generate trees
        seq_len = torch.ones(num_trees)
        for itree in range(max_len - 1):

            # transform x and sample
            x_transform = model.transform.x_scaler(x)
            y = model.sample(x_transform, seq_len, num_samples=1)
            y = y.reshape(num_trees, itree + 1, -1)

            # inverse transform y add to x
            y_transform = model.transform.y_scaler.inverse(y)
            x[:, itree + 1, :num_feat_nt] = y_transform[:, itree]
            x[:, itree + 1, sub_dim] = (
                x[:, itree, sub_dim] - x[:, itree + 1, sub_dim])

            seq_len += 1

    if to_numpy:
        x = x.cpu().numpy()
    return x


def sample_trees_attention(
        model: torch.nn.Module, roots: np.ndarray, max_len: int,
        time: Optional[np.ndarray] = None, to_numpy: bool = True,
        device: Optional = None,
    ) -> Union[FloatTensor, np.ndarray]:
    model.eval()
    num_trees, num_feat = roots.shape
    num_feat_nt = num_feat - 1 if time is not None else num_feat
    sub_dim = model.transform.sub_dim
    attn_mask = look_ahead_mask(max_len).to(device)

    if device is None:
        device = torch.device('cpu' if torch.cuda.is_available() else 'cuda')

    with torch.no_grad():
        # initialize array with root features
        x = torch.zeros(
            (num_trees, max_len, num_feat), device=device, dtype=torch.float32)
        x[:, 0] = torch.tensor(roots, dtype=torch.float32)

        # add time information to features
        if time is not None:
            x[..., -1] = torch.tensor(
                time, dtype=torch.float32).repeat(num_trees, 1)

        for itree in range(max_len - 1):
            # transform x and sample
            x_transform = model.transform.x_scaler(x)
            y = model.sample(x_transform, attn_mask, num_samples=1)
            y = y.reshape(num_trees, max_len, -1)

            # inverse transform y add to x
            y_transform = model.transform.y_scaler.inverse(y)
            x[:, itree + 1, :num_feat_nt] = y_transform[:, itree]
            x[:, itree + 1, sub_dim] = (
                x[:, itree, sub_dim] - x[:, itree + 1, sub_dim])

    if to_numpy:
        x = x.cpu().numpy()
    return x
