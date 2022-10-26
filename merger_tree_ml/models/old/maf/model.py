
import torch
from nflows import distributions, transforms, flows
from ..bijectors import InverseSoftplus

class MergerTreeMAF(torch.nn.Module):

    def __init__(
        self, features, hidden_features, context_features,
        num_layers, num_blocks, softplus=True
    ):

        super().__init__()

        # Create normalizing flow
        transform = []
        if softplus:
            transform.append(InverseSoftplus())
        transform.append(
            transforms.CompositeTransform(
                [
                    transforms.CompositeTransform(
                        [
                            transforms.MaskedAffineAutoregressiveTransform(
                                features=features,
                                hidden_features=hidden_features,
                                context_features=context_features,
                                num_blocks=num_blocks,
                                use_residual_blocks=False,
                                random_mask=False,
                                activation=torch.tanh,
                                dropout_probability=0.0,
                                use_batch_norm=True,
                            ),
                        ]
                    )
                    for _ in range(num_layers)
                ]
            )
        )
        transform = transforms.CompositeTransform(transform)
        distribution = distributions.StandardNormal((features,))
        self.maf = flows.Flow(transform, distribution)

    def forward(self, x):
        return x

