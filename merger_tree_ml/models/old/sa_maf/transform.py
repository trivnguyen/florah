
import torch
import numpy as np

class StandardScaler(torch.nn.Module):

    def __init__(self, n, with_loc=True, with_scale=True, **kargs):
        super().__init__()

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
            if not torch.equal(torch.tensor(loc.shape), torch.tensor(self.loc.shape)):
                raise ValueError("invalid shape")
            self.loc = loc

        # compute standard deviation
        if self.with_scale:
            scale = torch.std(data, **self.kargs)
            if not torch.equal(torch.tensor(scale.shape), torch.tensor(self.scale.shape)):
                raise ValueError("invalid shape")
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


class Transform(torch.nn.Module):

    def __init__(self, nx, ny, diff=True):
        super().__init__()
        self.x_scaler = StandardScaler(nx, dim=[0, 1])
        self.y_scaler = StandardScaler(ny, dim=[0, 1])
        self.diff = diff

    def forward(self, *args, **kargs):
        return self.transform(*args, **kargs)

    def transform(self, data, fit=True):
        """ Transform data """
        x = data["x"]
        y = data["y"]

        seq_len = np.array([len(x[i]) for i in range(len(x))])
        max_len = np.max(seq_len)

        x_padded = np.zeros((len(x), max_len, *x[0].shape[1:]))
        y_padded = np.zeros((len(y), max_len, *y[0].shape[1:]))
        mask = np.zeros((len(y), max_len), dtype=np.bool)

        for i in range(len(x)):
            x_padded[i, :seq_len[i]] = x[i]
            y_padded[i, :seq_len[i]] = y[i]
            mask[i, :seq_len[i]] = True

        if self.diff:
            y_padded[..., 0] = x_padded[..., 0] - y_padded[..., 0]

        # convert to tensor
        x_padded = torch.tensor(x_padded, dtype=torch.float32)
        y_padded = torch.tensor(y_padded, dtype=torch.float32)
        seq_len = torch.tensor(seq_len, dtype=torch.int64)
        mask = torch.tensor(mask, dtype=torch.bool)

        print(mask)

        # apply scaler as the last step
        if fit:
            x_padded = self.x_scaler.fit_transform(x_padded)
            y_padded = self.y_scaler.fit_transform(y_padded)
        else:
            x_padded = self.x_scaler.transform(x_padded)
            y_padded = self.y_scaler.transform(y_padded)

        return x_padded, y_padded, seq_len, mask

