
import torch
from nflows import distributions, transforms, flows
from ..bijectors import InverseSoftplus

class MergerTreeRNN(torch.nn.Module):

    def __init__(
        self, in_channels, out_channels, num_layers, hidden_features,
        num_layers_flows, hidden_features_flows, softplus=True):

        super().__init__()

        # Create RNN layers
        self.rnn = nn.ModuleList()
        for i in range(num_layers):
            n_in = in_channels if i==0 else hidden_features
            n_out = hidden_features
            self.rnn.append(nn.GRU(n_in, n_out, batch_first=True))

        # Create normalizing flow layers
        transform = []
        if softplus:
            transform.append(InverseSoftplus())
        transform.append(nflows.transforms.CompositeTransform(
            [
                nflows.transforms.CompositeTransform(
                    [
                        nflows.transforms.MaskedAffineAutoregressiveTransform(
                            features=out_channels,
                            hidden_features=hidden_features_flows,
                            context_features=hidden_features,
                            num_blocks=2,
                            use_residual_blocks=False,
                            random_mask=False,
                            activation=torch.tanh,
                            dropout_probability=0.0,
                            use_batch_norm=True,
                        ),
                        nflows.transforms.RandomPermutation(features=out_channels),
                    ]
                )
                for _ in range(num_layers_flows)
            ]
        ))
        transform = nflows.transforms.CompositeTransform(transform)
        distribution = nflows.distributions.StandardNormal((out_channels,))
        self.maf = nflows.flows.Flow(transform, distribution)

    def forward(self, x, h0_list=None):
        h_list = []
        for i in range(len(self.rnn)):
            h0 = h0_list[i] if h0_list is not None else None
            x, h = self.rnn[i](x, h0)
            if i != len(self.rnn) - 1:
                x = F.relu(x)
            h_list.append(h)
        return x, h_list
