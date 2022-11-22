from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import LongTensor, FloatTensor

from . import modules
from . import flows
from . import transforms

class DataModule(modules.MAFModule):
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


class RecurrentMAF(torch.nn.Module):
    """ Recurrent-MAF architecture for merger tree generation """
    layer_dict = {
        'RNN': torch.nn.RNN,
        'GRU': torch.nn.GRU,
        'LSTM': torch.nn.LSTM,
    }
    layer_default_kargs = {
        'RNN': {},
        'GRU': {},
        'LSTM': {},
    }

    def __init__(
            self,
            in_channels: int, out_channels: int, num_layers: int,
            hidden_features: int, rnn_name: str , rnn_hparams: dict,
            num_layers_flows: int, hidden_features_flows: int,
            num_blocks: int, softplus: bool = False
        ) -> None:
        """
        Parameters
        ----------
            in_channels: int
                Number of input channels
            out_channels: int
                Number of output channels
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
        super(RecurrentMAF, self).__init__()

        # get recurrent layer type to use
        if rnn_name in self.layer_dict:
            self.rnn_layer = self.layer_dict[rnn_name]
        else:
            raise KeyError(
                f"Unknown model name \"{rnn_name}\"."\
                f"Available models are: {str(self.layer_dict.keys())}")

        # create RNN layers
        self.rnn = torch.nn.ModuleList()
        default_rnn_hparams = self.layer_default_kargs[rnn_name]
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
            self, x: FloatTensor, seq_len: LongTensor,
            h0: Optional[Tuple] = None) -> Tuple[FloatTensor, Tuple]:
        r"""
        Forward pass

        Parameters:
            x: FloatTensor (N_batch, L_padded, H_in)
                Input tensor where `N_batch` is the batch size, `L_padded` is
                the padded sequence length and `H_in` is the input dimension
            seq_len: LongTensor (N_batch, )
                The pre-padded sequence length of each input tensor
            h0: Tuple of FloatTensor
                Tuple of initial hidden states to pass into each recurrent layer

        """
        hout = []  # list of output hidden states
        total_length = x.shape[1]

        # iterate over all recurrent layers
        for i in range(len(self.rnn)):
            # pack sequence, pass through recurrent layer, and unpack
            x = torch.nn.utils.rnn.pack_padded_sequence(
                x, seq_len.cpu(), batch_first=True,
                enforce_sorted=False)
            x, h = self.rnn[i](
                x, h0[i] if h0 is not None else None)
            x, _ = torch.nn.utils.rnn.pad_packed_sequence(
                x, batch_first=True, total_length=total_length)

            # activation
            if i != len(self.rnn) - 1:
                x = self.activation(x)
            hout.append(h)

        # return output sequence and hidden states
        return x, hout

    def log_prob(self, batch: Tuple[FloatTensor],
                 return_context: bool = False) -> FloatTensor:
        """ Calculate log-likelihood from batch """
        x, y, seq_len, mask = batch

        # apply forward and return context
        context, _ = self(x, seq_len)

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

    def sample(self, x: FloatTensor, seq_len: LongTensor, num_samples: int,
               return_context: bool = False) -> FloatTensor:
        """ Sample from batch """
        # forward pass and get context
        context, _ = self(x, seq_len)
        context = context.flatten(0, 1)

        # sample and reshape
        y = self._sample_from_context(num_samples, context=context)
        return y

    def _sample_from_context(
            self, num_samples: int, context: Optional[FloatTensor] = None
        ) -> FloatTensor:
        """ Sample P(x | context) """
        return self.maf_blocks.sample(num_samples, context=context)

    def _log_prob_from_context(
            self, x: FloatTensor,
            context: Optional[FloatTensor] = None) -> FloatTensor:
        """ Return MAF log-likelihood P(x | context)"""
        return self.maf_blocks.log_prob(x, context=context)
