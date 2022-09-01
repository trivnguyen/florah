
import numpy as np
import torch

class ToTensor(object):

    dtype_dict = {
        'int32': torch.int32,
        'int64': torch.int64,
        'float32': torch.float32,
        'float64': torch.float64
    }

    def __init__(self, dtype):
        self.dtype = self.dtype_dict[str(dtype)]

    def __call__(self, data):
        new_data = []
        for d in data:
            d_ts = torch.tensor(d, dtype=self.dtype)
            new_data.append(d_ts)
        return new_data

    def __str__(self):
        return (f'{self.__class__.__name__}({self.dtype})')

    def __repr__(self) -> str:
        return self.__str__()


class TruncateTree(object):

    def __init__(self, max_length, pad_value=None):
        self.max_length = max_length
        self.pad_value = pad_value

    def __call__(self, data):
        new_data = []
        for d in data:
            if len(d) < self.max_length:
                # ignore tree if no value is given
                if self.pad_value is None:
                    continue
                pad = np.zeros(d.ndim*2, dtype=int)
                pad[d.ndim * 2 - 1] = self.max_length - len(d)
                d = torch.nn.functional.pad(d, tuple(pad), value=self.pad_value)
                new_data.append(d)
            else:
                new_data.append(d[:self.max_length])
        new_data = torch.stack(new_data)
        return new_data

    def __str__(self):
        return (f'{self.__class__.__name__}'\
                f'(max_length={self.max_length}, pad_value={self.pad_value})')

    def __repr__(self) -> str:
        return self.__str__()


class LinearInterpolateTree(object):

    def __init__(self, dim=0, max_iters=100):
        self.dim = dim
        self.max_iters = max_iters

    def __call__(self, data):

        new_data = torch.clone(data)
        for i, d in enumerate(new_data):
            x = d[..., self.dim]

            for _ in range(self.max_iters):
                bad = torch.where(torch.diff(x) > 0)[0]
                if len(bad) == 0:
                    break
                x[bad] = x[bad + 1] + 0.5 * (x[bad - 1] - x[bad + 1])
            new_data[i, ..., self.dim] = x

        return new_data

    def __str__(self):
        return (f'{self.__class__.__name__}(dim={self.dim}, max_iters=self.max_iters)')

    def __repr__(self) -> str:
        return self.__str__()


class StandardScaler(object):

    def __init__(self, mean=0, stdv=1, axis=None, data=None):
        self.axis = axis
        if data is not None:
            self.fit(data)
        else:
            self.mean = torch.tensor(mean, dtype=torch.float32)
            self.stdv = torch.tensor(stdv, dtype=torch.float32)

    def fit(self, data):
        if self.axis is not None:
            self.mean = torch.mean(data, axis=self.axis, keepdims=True)
            self.stdv = torch.std(data, axis=self.axis, keepdims=True)
        else:
            self.mean = torch.mean(data)
            self.stdv = torch.std(data)

    def transform(self, data, dim=None):
        if dim is not None:
            mean = self.mean[..., dim]
            stdv = self.stdv[..., dim]
            return (data - mean) / stdv
        return (data - self.mean) / self.stdv

    def inverse(self, data, dim=None):
        if dim is not None:
            mean = self.mean[..., dim]
            stdv = self.stdv[..., dim]
            return data * stdv + mean
        return data * self.stdv + self.mean

    def __call__(self, data):
        return self.transform(data)

    def __str__(self) -> str:
        return (f'{self.__class__.__name__}')

    def __repr__(self) -> str:
        return self.__str__()

class LogTransform(object):

    def __init__(self, dims):
        self.dims = dims

    def __call__(self, data):
        data[..., self.dims] = torch.log10(data[..., self.dims])
        return data

    def __str__(self) -> str:
        return (f'{self.__class__.__name__}({self.dims})')

    def __repr__(self) -> str:
        return self.__str__()


class Transform(object):

    transform_dict = {
        'ToTensor': ToTensor,
        'TruncateTree': TruncateTree,
        'LinearInterpolateTree': LinearInterpolateTree,
        'LogTransform': LogTransform,
        'StandardScaler': StandardScaler,
    }
    transform_default_kargs = {
        'ToTensor': {'dtype': 'float32'},
        'TruncateTree': {'max_length': 10, 'pad_value': None},
        'LinearInterpolateTree': {},
        'LogTransform': {"dims": None},
        'StandardScaler': {},
    }

    def __init__(self, transform_hparams):
        self.transforms = self._parse_hparams(transform_hparams)
        self.num_transforms = len(self.transforms)

    def _parse_hparams(self, transform_hparams):
        """ Parse Hparams dictionary """

        transforms = []
        order = transform_hparams['order']
        for i in range(len(order)):
            name = order[i]
            if name in self.transform_dict:
                transform = self.transform_dict[name]
            else:
                raise KeyError(
                    f"Unknown transform name \"{name}\"."\
                    f"Available transformation are: {str(self.transforms_dict.keys())}")
            default_transform_kargs = self.transform_default_kargs[name]
            default_transform_kargs.update(transform_hparams[name])
            transforms.append(transform(**default_transform_kargs))

        return transforms

    def __call__(self, data, train=True):
        for i in range(self.num_transforms):
            if isinstance(self.transforms[i], StandardScaler) and train:
                self.transforms[i].fit(data)
                data = self.transforms[i](data)
            else:
                data = self.transforms[i](data)
        return data

    def __getitem__(self, i):
        return self.transforms[i]

