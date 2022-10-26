from typing import Optional, List, Tuple

import torch
import numpy as np

class StandardScaler(torch.nn.Module):
    """ Standard scaling transformation """
    def __init__(
            self, n: int, with_loc: bool = True, 
            with_scale: bool = True, **kargs) -> None:
        """ 
        Parameters
        ----------
            n: int
                Number of channels
            with_loc: bool
                If True, apply transformation with mean. Default: True
            with_scale: bool
                If True, apply transformation with standard devitation. Default: True

        """
        super(StandardScaler, self).__init__()

        self.register_buffer("loc", torch.zeros(n))
        self.register_buffer("scale", torch.ones(n))
        self.with_loc = with_loc
        self.with_scale = with_scale
        self.n = n
        self.kargs = kargs

    def forward(self, *args, **kargs):
        """ Call transform """
        return self.transform(*args, **kargs)

    def fit(self, data):
        """ Fit data """
        # compute mean
        if self.with_loc:
            loc = torch.mean(data, **self.kargs)
            if not torch.equal(
                torch.tensor(loc.shape), torch.tensor(self.loc.shape)):
                raise ValueError("invalid shape: expected {}, got {}".format(
                    self.loc.shape, loc.shape))
            self.loc = loc

        # compute standard deviation
        if self.with_scale:
            scale = torch.std(data, **self.kargs)
            if not torch.equal(
                torch.tensor(scale.shape), torch.tensor(self.scale.shape)):
                raise ValueError("invalid shape: expected {}, got {}".format(
                    self.scale.shape, scale.shape))
            self.scale = scale

    def transform(self, data):
        """ Transform data """
        return (data - self.loc) / self.scale

    def fit_transform(self, data):
        """ Apply both fit and transform """
        self.fit(data)
        return self.transform(data)

    def inverse(self, data):
        """ Inverse transform data """
        return data * self.scale + self.loc


class Preprocess(torch.nn.Module):
    """ Preprocessing module for merger tree data """, Tuple
    def __init__(
            self, nx: int, ny: int, sub_dim: Optional[List[int]] = None,
            use_t: bool = True, use_dt: bool = False) -> None:
        """
        Parameters
        ----------
            nx: int
                Number of dimensions for input series X
            ny: int
                Number of dimensions for output series Y
            use_t: bool
                If True, add time to input X. Default: True
            use_dt: bool
                If True, add time difference to input X. Default: False
        """
        super(Preprocess, self).__init__()

        nx = nx + int(use_t) + int(use_dt)

        self.x_scaler = StandardScaler(nx, dim=[0, 1])
        self.y_scaler = StandardScaler(ny, dim=[0, 1])
        self.use_t = use_t
        self.use_dt = use_dt
        self.sub_dim = sub_dim if sub_dim is not None else []

    def forward(self, *args, **kargs):
        """ Apply transformation """
        return self.transform(*args, **kargs)

    def transform(self, data: dict, fit: bool = True) -> Tuple[()]:
        """ Apply transformation
        Parameters:
            data: dict
                Data dictionary with key "t" and "y"
            fit: bool
                If True, fit scaler to data and then transform. Default: True
        """
        # get input and output series from data
        # output is input shifted right
        trees = data.get("x")
        times = data.get("t")
        if self.use_t or self.use_dt:
            if self.use_t:
                x = [
                    np.hstack([
                        trees[i][:-1], times[i][:-1]]) for i in range(len(trees)
                    )]
            if self.use_dt:
                x = [
                    np.hstack([
                        trees[i][:-1],
                        times[i][1:] - times[i][:-1]
                        ]) for i in range(len(trees)
                    )]
        else:
            x = [trees[i][:-1] for i in range(len(trees))]
        y = [trees[i][1:] for i in range(len(trees))]

        # pad sequence to maximum length
        seq_len = np.array([len(x[i]) for i in range(len(x))])
        max_len = np.max(seq_len)
        x_padded = np.zeros((len(x), max_len, *x[0].shape[1:]))
        y_padded = np.zeros((len(y), max_len, *y[0].shape[1:]))
        mask = np.zeros((len(y), max_len), dtype=np.bool)
        for i in range(len(x)):
            x_padded[i, :seq_len[i]] = x[i]
            y_padded[i, :seq_len[i]] = y[i]
            mask[i, :seq_len[i]] = True

        y_padded[..., self.sub_dim] = (
            x_padded[..., self.sub_dim] - y_padded[..., self.sub_dim])

        # convert to tensor
        x_padded = torch.tensor(x_padded, dtype=torch.float32)
        y_padded = torch.tensor(y_padded, dtype=torch.float32)
        seq_len = torch.tensor(seq_len, device='cpu', dtype=torch.int64)
        mask = torch.tensor(mask, dtype=torch.bool)

        # apply scaler as the last step
        if fit:
            x_padded = self.x_scaler.fit_transform(x_padded)
            y_padded = self.y_scaler.fit_transform(y_padded)
        else:
            x_padded = self.x_scaler.transform(x_padded)
            y_padded = self.y_scaler.transform(y_padded)

        return x_padded, y_padded, seq_len, mask
