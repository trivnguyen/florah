

from nflows import distributions, flows, transforms

from .utils import get_activation


def build_maf(
        features: int, hidden_features: int, context_features: int,
        num_layers: int, num_blocks: int, activation: str = 'tanh'
    ) -> flows.Flow:
    """ Build a MAF normalizing flow

    Parameters
    ----------
    features: int
        Number of features
    hidden_features: int
        Number of hidden features
    context_features: int
        Number of context features
    num_layers: int
        Number of layers
    num_blocks: int
        Number of blocks
    activation: str
        Name of the activation function

    Returns
    -------
    maf: flows.Flow
        MAF normalizing flow
    """
    transform = []
    transform.append(transforms.CompositeTransform(
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
                        activation=get_activation(activation),
                        dropout_probability=0.0,
                        use_batch_norm=True,
                    ),
                    transforms.RandomPermutation(features=features),
                ]
            )
            for _ in range(num_layers)
        ]
    ))
    transform = transforms.CompositeTransform(transform)
    distribution = distributions.StandardNormal((features,))
    maf = flows.Flow(transform, distribution)
    return maf
