
import torch
import torch.nn.functional as F

from nflows import distributions, transforms, flows
from ..bijectors import InverseSoftplus

class MergerTreeRNN(torch.nn.Module):
    """ RNN Model for merger tree generation """

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
        self, in_channels, out_channels, num_layers, hidden_features,
        rnn_name , rnn_hparams, num_layers_flows, hidden_features_flows,
        num_blocks, softplus=False):
        """
        Args:
        - in_channels: [int] number of input channels
        - out_channels: [int] number of output channels
        - rnn_name [str]: name of the RNN layer
        - rnn_hparams: [dict] dictionary with extra kargs for RNN layer
        - num_layers: [int] number of RNN hidden layers
        - hidden_channels: [int] number of RNN hidden channels
        - num_layers_flows: [int] number of MAF hidden layers
        - hidden_channels_flows: [int] number of MAF hidden channels
        - num_blocks: [int] number of MAF blocks
        - softplus: [bool] if true, add InverseSoftplus layer to MAF transformation
        """

        super().__init__()

        # Determine RNN layer type to use
        if rnn_name in self.layer_dict:
            self.rnn_layer = self.layer_dict[rnn_name]
        else:
            raise KeyError(
                f"Unknown model name \"{rnn_name}\"."\
                f"Available models are: {str(self.layer_dict.keys())}")

        # Create RNN layers
        self.rnn = torch.nn.ModuleList()
        default_rnn_hparams = self.layer_default_kargs[rnn_name]
        default_rnn_hparams.update(rnn_hparams)
        for i in range(num_layers):
            n_in = in_channels if i==0 else hidden_features
            n_out = hidden_features
            self.rnn.append(
                self.rnn_layer(n_in, n_out, batch_first=True, **default_rnn_hparams))

        # Create normalizing flow layers
        transform = []
        if softplus:
            transform.append(InverseSoftplus())
        transform.append(transforms.CompositeTransform(
            [
                transforms.CompositeTransform(
                    [
                        transforms.MaskedAffineAutoregressiveTransform(
                            features=out_channels,
                            hidden_features=hidden_features_flows,
                            context_features=hidden_features,
                            num_blocks=num_blocks,
                            use_residual_blocks=False,
                            random_mask=False,
                            activation=torch.tanh,
                            dropout_probability=0.0,
                            use_batch_norm=True,
                        ),
                        transforms.RandomPermutation(features=out_channels),
                    ]
                )
                for _ in range(num_layers_flows)
            ]
        ))
        transform = transforms.CompositeTransform(transform)
        distribution = distributions.StandardNormal((out_channels,))
        self.maf = flows.Flow(transform, distribution)

    def forward(self, x, seq_len, h0_list=None):
        """ Forward modeling"""
        h_out = []

        for i in range(len(self.rnn)):
            h0 = h0_list[i] if h0_list is not None else None
            x = torch.nn.utils.rnn.pack_padded_sequence(
                x, seq_len.cpu(), batch_first=True, enforce_sorted=False)
            x, h = self.rnn[i](x, h0)
            x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            if i != len(self.rnn) - 1:
                x = F.relu(x)
            h_out.append(h)

        return x, h_out

