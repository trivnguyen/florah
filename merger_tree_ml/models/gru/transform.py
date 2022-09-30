
import torch
import numpy as np

class StandardScaler(torch.nn.Module):

    def __init__(self, features, with_loc=True, with_scale=True, axis=None):
        super().__init__()

        self.register_buffer("loc", torch.zeros(features))
        self.register_buffer("scale", torch.ones(features))
        self.features = features
        self.with_loc = with_loc
        self.with_scale = with_scale
        self.axis = axis

    def forward(self, *args, **kargs):
        return self.transform(*args, **kargs)

    def fit(self, data):
        if self.with_loc:
            if self.axis is not None:
                self.loc = torch.mean(data, axis=self.axis)
            else:
                self.loc = torch.mean(data)

        if self.with_scale:
            if self.axis is not None:
                self.scale = torch.std(data, axis=self.axis)
            else:
                self.scale = torch.std(data)

    def transform(self, data):
        return (data - self.loc) / self.scale

    def inverse(self, data):
        return data * self.scale + self.loc


class Transform(torch.nn.Module):

    def __init__(self, x_features, y_features):
        super().__init__()
        self.x_scaler = StandardScaler(x_features, axis=0)
        self.y_scaler = StandardScaler(
            y_features, axis=0, with_loc=False)

    def forward(self, *args, **kargs):
        return self.transform(*args, **kargs)

    def transform(self, data, fit=True):
        """ Transform data """
        x = data["x"]
        y = data["y"]
        tree = np.hstack([x[..., np.newaxis], x[..., np.newaxis] - np.cumsum(y, -1)])
        x = tree[..., :-1]

        if isinstance(x, (list, tuple)):
            x = np.array(x)
        if isinstance(y, (list, tuple)):
            y = np.array(y)

        # convert to tensor
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        # apply scaler as the last step
        if fit:
            self.x_scaler.fit(x)
            self.y_scaler.fit(y)
        x = self.x_scaler(x)
        y = self.y_scaler(y)

        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)

        return x, y

