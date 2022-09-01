
import torch

from .flows import build_maf

class LinearModel(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels,
        num_layers, hidden_channels, num_transforms, hidden_channels_flows,
    ):
        super().__init__()

        # Linear layer
        self.fc_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            n_in = in_channels if i == 0 else hidden_channels
            n_out = hidden_channels
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

