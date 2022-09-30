
import torch.nn.functional as F
from nflows.transforms import Transform
from nflows.utils import torchutils

class Softplus(Transform):
    r"""
    Elementwise bijector via the mapping :math:`\text{Softplus}(x) = \log(1 + \exp(x))`.
    """

    def forward(self, x, context=None):
        y = F.softplus(x)
        ladj = self._log_abs_det_jacobian(x)
        ladj = torchutils.sum_except_batch(ladj, num_batch_dims=1)
        return y, ladj

    def inverse(self, y, context=None):
        x = self._softplus_inv(y)
        ladj = self._log_abs_det_jacobian(x)
        ladj = torchutils.sum_except_batch(ladj, num_batch_dims=1)
        return x, -ladj

    def _log_abs_det_jacobian(self, x):
        return -F.softplus(-x)

    def _softplus_inv(self, y):
        return y + y.neg().expm1().neg().log()


class InverseSoftplus(Softplus):
    def forward(self, x, context=None):
        return super().inverse(x, context)

    def inverse(self, y, context=None):
        return super().forward(y, context)

