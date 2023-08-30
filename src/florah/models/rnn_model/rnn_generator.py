
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .. import base_modules, flows, transforms
from . import grud


class DataModule(base_modules.BaseFlowModule):
    """
    DataModule for Recurrent-MAF model
    """
    arch_type = "RecurrentMAF"
    def __init__(
            self, model_hparams: Optional[dict] = None,
            transform_hparams: Optional[dict] = None,
            optimizer_hparams: Optional[dict] = None
        ) -> None:
        super(DataModule, self).__init__(
            RecurrentMAF, transforms.Preprocess, model_hparams,
            transform_hparams, optimizer_hparams)


class TimeEmbedding(torch.nn.Module):
    r""" Time embedding neural network.
    .. math::
        PE() =
    where :math:`d` is the embedding dimension
    """
    def __init__(
        self, time_channels: int, embed_channels: int) -> None:
        """
        Parameters
        ----------
        time_channels: int
            Number of time input channels
        embed_channels: int
            Number of embedded dimension
        """
        super().__init__()

        self.embed_channels = embed_channels
        self.linear_embed = torch.nn.Linear(time_channels, embed_channels)

    def forward(self, t: Tensor) -> Tensor:
        return torch.cos(self.linear_embed(t))


class RecurrentMAF(torch.nn.Module):
    """
    Recurrent-MAF model for time series forecasting.

    Parameters
    ----------
    in_channels: int
        Number of input channels
    out_channels: int
        Number of output channels
    time_embed_channels: int
        Number of time embeeding channels
    rnn_name: str
        Type of the recurrent layer to use
    rnn_hparams: dict
        Dictionary with extra kargs for current layer
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

    Attributes
    ----------
    rnn_name: str
        Type of the recurrent layer to use
    rnn_layer: torch.nn.Module
        Recurrent layer
    embedding_net: torch.nn.Module
        Embedding layer
    rnn: torch.nn.ModuleList
        List of recurrent layers
    maf_blocks: torch.nn.ModuleList
        List of MAF blocks
    activation: torch.nn.Module
        Activation function
    """
    # Static dictionary with recurrent layers
    RNN_LAYERS = {
        'RNN': torch.nn.RNN,
        'GRU': torch.nn.GRU,
        'LSTM': torch.nn.LSTM,
        'GRUD': grud.GRUD
    }
    RNN_LAYERS_DEFAULT_ARGS = {
        'RNN': {},
        'GRU': {},
        'LSTM': {},
        'GRUD': {'delta_size': 1}
    }

    def __init__(
            self, in_channels: int, out_channels: int,
            hidden_features: int = 64, num_layers: int = 1,
            rnn_name: str = "GRU", rnn_hparams: Optional[dict] = None,
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
                Number of time embeeding channels
            rnn_name: str
                Type of the recurrent layer to use
            rnn_hparams: dict
                Dictionary with extra kargs for current layer
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
        super().__init__()

        # get recurrent layer type to use
        self.rnn_name = rnn_name
        if rnn_name in self.RNN_LAYERS:
            self.rnn_layer = self.RNN_LAYERS[rnn_name]
        else:
            raise KeyError(
                f"Unknown model name \"{rnn_name}\"."\
                f"Available models are: {str(self.RNN_LAYERS.keys())}")

        # time embedding layers, currently set to torch.nn.Identity
        if time_embed_channels is None:
            self.embedding_net = torch.nn.Identity()
            in_channels = in_channels + 2
        else:
            self.embedding_net = TimeEmbedding(2, time_embed_channels)
            in_channels = in_channels + time_embed_channels

        # create RNN layers
        self.rnn = torch.nn.ModuleList()
        default_rnn_hparams = self.RNN_LAYERS_DEFAULT_ARGS[rnn_name]
        if rnn_hparams is not None:
            default_rnn_hparams.update(rnn_hparams)
        for i in range(num_layers):
            n_in = in_channels if i==0 else hidden_features
            n_out = hidden_features
            self.rnn.append(
                self.rnn_layer(
                    n_in, n_out, batch_first=True, **default_rnn_hparams))

        # MAF blocks
        self.maf_blocks = flows.build_maf(
            out_channels, hidden_features_flows, hidden_features,
            num_layers_flows, num_blocks)

        # activation
        self.activation = F.relu

    def forward(
            self, x: Tensor, t_in: Tensor, t_out:  Tensor,
            h0: Optional[Tuple] = None) -> Tuple[Tensor, Tuple]:
        r"""
        Forward pass

        Parameters:
            x: Tensor (N_batch, L_padded, H_in)
                Input tensor where `N_batch` is the batch size, `L_padded` is
                the padded sequence length and `H_in` is the input dimension
            t_in: Tensor (N_batch, L_padded, H_in_t)
                Input time tensor
            t_out: Tensor (N_batch, L_padded, H_in_t)
                Output time tensor
            h0: Tuple of Tensor
                Tuple of initial hidden states to pass into each recurrent layer

        """
        hout = []  # list of output hidden states
        total_length = x.shape[1]

        # time embedding and append into input array
        t_embed = self.embedding_net(torch.cat([t_in, t_out], dim=-1))
        x = torch.cat([x, t_embed], dim=-1)

        # compute time difference (without embedding)
        if self.rnn_name == "GRUD":
            t_delta = torch.cat([
                torch.zeros(t_in.shape[0], 1, 1, dtype=t_in.dtype, device=t_in.device),
                t_out[:, :-1] - t_in[:, :-1]
            ], dim=1)
        else:
            t_delta = None

        # iterate over all recurrent layers
        for i in range(len(self.rnn)):
            if t_delta is not None:
                x, h = self.rnn[i](x, t_delta, h0[i] if h0 is not None else None)
            else:
                x, h = self.rnn[i](x, h0[i] if h0 is not None else None)
            if i != len(self.rnn) - 1:
                x = self.activation(x)
            hout.append(h)

        # return output sequence and hidden states
        return x, hout

    def log_prob(self, batch: Tuple[Tensor],
                 return_context: bool = False) -> Tensor:
        """ Calculate log-likelihood from batch """
        x, y, t, seq_len, mask = batch
        t_in = t[:, :-1]
        t_out = t[:, 1:]

        # apply forward and return context
        context, _ = self(x, t_in, t_out)

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

    def sample(self, x: Tensor, t_in: Tensor, t_out: Tensor, num_samples: int,
               return_context: bool = False) -> Tensor:
        """ Sample from batch """
        # forward pass and get context
        context, _ = self(x, t_in, t_out)
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

