from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, FloatTensor

from . import modules
from . import flows
from . import transforms
from . import grud

class DataModule(modules.BaseModule):
    """
    DataModule for Recurrent-MAF model
    """
    arch_type = "RecurrentBinaryClassifier"
    def __init__(
            self, model_hparams: Optional[dict] = None,
            transform_hparams: Optional[dict] = None,
            optimizer_hparams: Optional[dict] = None
        ) -> None:
        super(DataModule, self).__init__(
            RecurrentBinaryClassifier, transforms.PreprocessClassifier, 
            model_hparams, transform_hparams, optimizer_hparams)

    def training_step(self, batch, batch_idx) -> FloatTensor:
        batch_size = len(batch[0])
        x, y, t, seq_len, mask = batch
        t_in = t[:, :-1]
        t_out = t[:, 1:]
        yhat, _ = self(x, t_in, t_out)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(yhat[mask], y[mask])
        self.log('train_loss', loss, on_epoch=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx) -> FloatTensor:
        batch_size = len(batch[0])
        x, y, t, seq_len, mask = batch
        t_in = t[:, :-1]
        t_out = t[:, 1:]
        yhat, _ = self(x, t_in, t_out)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(yhat[mask], y[mask])
        self.log('val_loss', loss, on_epoch=True, batch_size=batch_size)
        return loss


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
        super(TimeEmbedding, self).__init__()

        self.embed_channels = embed_channels
        self.linear_embed = torch.nn.Linear(time_channels, embed_channels)

    def forward(self, t: Tensor) -> Tensor:
        return torch.cos(self.linear_embed(t))


class RecurrentBinaryClassifier(torch.nn.Module):
    """ Recurrent-architecture for binary classification on node-level label"""
    layer_dict = {
        'RNN': torch.nn.RNN,
        'GRU': torch.nn.GRU,
        'LSTM': torch.nn.LSTM,
        'GRUD': grud.GRUD
    }
    layer_default_kargs = {
        'RNN': {},
        'GRU': {},
        'LSTM': {},
        'GRUD': {'delta_size': 1}
    }

    def __init__(
            self, in_channels: int, out_channels: int,
            hidden_features: int = 64, num_layers: int = 1,
            rnn_name: str = "GRU", rnn_hparams: Optional[dict] = None,
            hidden_features_fc: int = 64, num_layers_fc: int = 1,
            time_embed_channels: Optional[int] = None,
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
        """
        super(RecurrentBinaryClassifier, self).__init__()

        # get recurrent layer type to use
        self.rnn_name = rnn_name
        if rnn_name in self.layer_dict:
            self.rnn_layer = self.layer_dict[rnn_name]
        else:
            raise KeyError(
                f"Unknown model name \"{rnn_name}\"."\
                f"Available models are: {str(self.layer_dict.keys())}")

        # time embedding layers, currently set to torch.nn.Identity
        if time_embed_channels is None:
            self.embedding_net = torch.nn.Identity()
            in_channels = in_channels + 2
        else:
            self.embedding_net = TimeEmbedding(2, time_embed_channels)
            in_channels = in_channels + time_embed_channels

        # create RNN layers
        self.rnn = torch.nn.ModuleList()
        default_rnn_hparams = self.layer_default_kargs[rnn_name]
        if rnn_hparams is not None:
            default_rnn_hparams.update(rnn_hparams)
        for i in range(num_layers):
            n_in = in_channels if i==0 else hidden_features
            n_out = hidden_features
            self.rnn.append(
                self.rnn_layer(
                    n_in, n_out, batch_first=True, **default_rnn_hparams))
        
        # create FC layers
        self.fc = torch.nn.ModuleList()
        for i in range(num_layers_fc):
            n_in = hidden_features if i==0 else hidden_features_fc
            n_out = hidden_features_fc
            self.fc.append(torch.nn.Linear(n_in, n_out))
        
        # create output layer
        self.out = torch.nn.Linear(hidden_features_fc, out_channels)

        # activation
        self.activation = F.relu

    def forward(
            self, x: Tensor, t_in: Tensor, t_out:  Tensor,
            h0: Optional[Tuple] = None, return_h: bool = True) -> Tuple[Tensor, Tuple]:
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
            return_h: bool
                If True, return hidden states. Default: False

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

        # iterate over all FC layers
        for i in range(len(self.fc)):
            x = self.fc[i](x)
            if i != len(self.fc) - 1:
                x = self.activation(x)
        
        # output layer
        x = self.out(x)

        if return_h:
            return x, hout
        return x

