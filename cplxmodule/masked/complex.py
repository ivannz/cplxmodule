import torch

from ..layers import CplxLinear
from .base import BaseMasked, SparseWeightMixin

from ..cplx import cplx_linear


class CplxLinearMasked(SparseWeightMixin, BaseMasked, CplxLinear):
    def forward(self, input):
        return cplx_linear(input, self.weight_masked, self.bias)

    def _sparsity(self, threshold=None, hard=True):
        if not self.is_sparse:
            return []

        # get the mask
        mask = torch.gt(self.mask, 0) if hard else self.mask
        n_relevant = float(mask.sum().item())

        # handle broadcasted masks
        n, m = mask.shape

        # bypass CplxWeightMixin and get the parameter dict itself
        pd_weight = self.__getattr__("weight")
        if n == 1:
            n_relevant = pd_weight.shape[0] * n_relevant

        elif m == 1:
            n_relevant = n_relevant * pd_weight.shape[1]

        return [
            (id(pd_weight.real), pd_weight.real.numel() - n_relevant),
            (id(pd_weight.imag), pd_weight.imag.numel() - n_relevant),
        ]
