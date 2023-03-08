from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from nflows.transforms import Transform
from nflows.utils import torchutils

class Softplus(Transform):
    r"""
    Elementwise bijector via the mapping :math:`\text{Softplus}(x) = \log(1 + \exp(x))`.
    """

    def forward(
            self, x: torch.Tensor,
            context: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Return y = Softplus[x] and det of Jacobian """
        y = F.softplus(x)
        ladj = self._log_abs_det_jacobian(x)
        ladj = torchutils.sum_except_batch(ladj, num_batch_dims=1)
        return y, ladj

    def inverse(
            self, y: torch.Tensor,
            context: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Return x = Softplus^(-1)[y] and det of Jacobian """
        x = self._softplus_inv(y)
        ladj = self._log_abs_det_jacobian(x)
        ladj = torchutils.sum_except_batch(ladj, num_batch_dims=1)
        return x, -ladj

    def _log_abs_det_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        return -F.softplus(-x)

    def _softplus_inv(self, y: torch.Tensor) ->torch.Tensor:
        return y + y.neg().expm1().neg().log()


class InverseSoftplus(Softplus):
    r"""
    Elementwise bijector via the mapping inverse Softplus
    """
    def forward(
            self, x: torch.Tensor,
            context: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        return super(InverseSoftplus, self).inverse(x, context)

    def inverse(
            self, y: torch.Tensor,
            context: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        return super(InverseSoftplus, self).forward(y, context)

