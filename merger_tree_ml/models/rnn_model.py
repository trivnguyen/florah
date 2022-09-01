
import torch

from .flows import build_maf

class RNNModel(torch.nn.Module):

    rnn_dict = {
        'RNN': torch.nn.RNN,
        'GRU': torch.nn.GRU,
        'LSTM': torch.nn.LSTM
    }

    def __init__(
        self, in_channels, out_channels,
        num_layers, hidden_channels, num_layers_fc, hidden_channels_fc,
        num_transforms, hidden_channels_flows,
    ):
        super().__init__()

        # RNN layers
        self.rnn_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            n_in = in_channels if i == 0 else hidden_channels
            n_out = hidden_channels
            self.rnn_layers.append(self.rnn_layer(n_in, n_out))


        # Linear layer
        self.fc_layers = torch.nn.ModuleList()
        for i in range(num_layers_fc):
            n_in = hidden_channels if i == 0 else hidden_channels_fc
            n_out = hidden_channels_fc
            self.fc_layers.append(torch.nn.Linear(n_in, n_out))

        # Create MAF layers
        self.maf = build_maf(
            dim=out_channels, num_transforms=num_transforms,
            context_features=hidden_channels,
            hidden_features=hidden_channels_flows)

    def forward(self, x):
        for i in range(len(self.fc_layers)):
            x = self.fc_layers[i](x)
            if i != len(self.fc_layers) - 1:
                x = torch.nn.functional.relu(x)
        return x

