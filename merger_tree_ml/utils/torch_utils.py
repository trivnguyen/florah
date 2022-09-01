
import torch

def _inverse_softplus_1(x):
    """ Inverse softplus function """
    return torch.expm1(x).log()

def _inverse_softplus_2(x):
    """ Inverse softplus function for large x"""
    return x + (1 - x.neg().exp()).log()

def inverse_softplus(x):
    big = x > torch.tensor(torch.finfo(x.dtype).max).log()
    return torch.where(
        big,
        _inverse_softplus_2(x.masked_fill(~big, 1.)),
        _inverse_softplus_1(x.masked_fill(big, 1.)),
    )

