
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from . import modules
from . import flows
from . import transforms
from . import grud

class DataModule(modules.MAFModule):
    """
    DataModule for Recurrent-MAF Decay model
    """
    arch_type = "RecurrentMAFDecay"
    def __init__(
            self, model_hparams: Optional[dict] = None,
            transform_hparams: Optional[dict] = None,
            optimizer_hparams: Optional[dict] = None
        ) -> None:
        super(DataModule, self).__init__(
            RecurrentMAFDecay, transforms.Preprocess, model_hparams,
            transform_hparams, optimizer_hparams)

class TimeEmbedding(torch.nn.Module):
    r""" Time embedding neural network.
    .. math::
        PE() =
    where :math:`d` is the embedding dimension
    """
    def __init__(
        self, embed_channels: int) -> None:
        """
        Parameters
        ----------
        embed_channels: int
            Number of embedded dimension
        """
        super(TimeEmbedding, self).__init__()

        self.embed_channels = embed_channels
        self.linear_embed = torch.nn.Linear(1, embed_channels)

    def forward(self, t: Tensor) -> Tensor:
        return torch.cos(self.linear_embed(t))


class RecurrentMAFDecay(torch.nn.Module):
    """ GRU with decay and MAF architecture for merger tree generation """

    def __init__(
            self, in_channels: int, out_channels: int,
            hidden_features: int = 64, num_layers: int = 1,
            time_embed_channels: Optional[int] = None,
            num_layers_flows: int = 1, hidden_features_flows: int = 64,
            num_blocks: int = 2, softplus: bool = False
        ) -> None:
        """
        Parameters
        ----------
            in_channels: int
                Number of input channels
            out_channels: int
                Number of output channels
            time_embed_channels: int
                Number of time embedding channels
            num_layers: int
                Number of recurrent hidden layers
            hidden_features: int
                Number of RNN hidden channels
            num_layers_flows: int
                number of MAF transformation
            hidden_features_flows: int
                Number of MAF hidden channels
            num_blocks: int
                Number of MADE blocks in each MAF transformation
            softplus: bool
                Deprecated.
        """
        super(RecurrentMAFDecay, self).__init__()

        # time embedding layers, currently set to torch.nn.Identity
        if time_embed_channels is None:
            self.embedding_net = torch.nn.Identity()
            in_channels = in_channels + 1
        else:
            self.embedding_net = TimeEmbedding(time_embed_channels)
            in_channels = in_channels + time_embed_channels

        # recurrent layers
        self.rnn = torch.nn.ModuleList()
        for i in range(num_layers):
            n_in = in_channels if i==0 else hidden_features
            n_out = hidden_features
            self.rnn.append(
                grud.GRUD(n_in, n_out, 1, batch_first=True))

        # MAF blocks
        self.maf_blocks = flows.build_maf(
            out_channels, hidden_features_flows, hidden_features,
            num_layers_flows, num_blocks)

        # activation
        self.activation = F.relu

    def forward(
            self, x: Tensor, t: Tensor, h0: Optional[Tuple] = None
        ) -> Tuple[Tensor, Tuple]:
        r"""
        Forward pass

        Parameters:
            x: Tensor (N_batch, L_padded, H_in)
                Input tensor where `N_batch` is the batch size, `L_padded` is
                the padded sequence length and `H_in` is the input dimension
            t: Tensor (N_batch, L_padded, H_in_t)
                Input time
            h0: Tuple of FloatTensor
                Tuple of initial hidden states to pass into each recurrent layer
        """

        hout = []  # list of output hidden states

        # time embedding and append into input array
        x = torch.cat([x, self.embedding_net(t)], dim=-1)

        # compute time difference (without embedding)
        t_delta = torch.diff(
            t, axis=1, prepend=torch.zeros(
                t.shape[0], 1, t.shape[2], dtype=t.dtype, device=t.device))

        # iterate over all recurrent layers
        for i in range(len(self.rnn)):
            x, h = self.rnn[i](x, t_delta, h0[i] if h0 is not None else None)
            if i != len(self.rnn) - 1:
                x = self.activation(x)
            hout.append(h)

        # return output sequence and hidden states
        return x, hout

    def log_prob(self, batch: Tuple[Tensor],
                 return_context: bool = False) -> Tensor:
        """ Calculate log-likelihood from batch """
        xt, y, seq_len, mask = batch

        # apply forward and return context
        context, _ = self(xt[...,:-1], xt[..., -2:-1])

        # reshape tensor before passing through MAF blocks
        context = context.flatten(0, 1)
        y = y.flatten(0, 1)
        mask = mask.flatten()

        # get log-likelihood P(y | context)
        log_prob = self._log_prob_from_context(y[mask], context=context[mask])

        if return_context:
            return log_prob, context
        else:
            return log_prob

    def sample(self, x: Tensor, t: Tensor, num_samples: int,
               return_context: bool = False) -> Tensor:
        """ Sample from batch """
        # forward pass and get context
        context, _ = self(x, t)
        context = context.flatten(0, 1)

        # sample and reshape
        y = self._sample_from_context(num_samples, context=context)
        return y

    def _sample_from_context(
            self, num_samples: int, context: Optional[Tensor] = None
        ) -> Tensor:
        """ Sample P(x | context) """
        return self.maf_blocks.sample(num_samples, context=context)

    def _log_prob_from_context(
            self, x: Tensor,
            context: Optional[Tensor] = None) -> Tensor:
        """ Return MAF log-likelihood P(x | context)"""
        return self.maf_blocks.log_prob(x, context=context)

