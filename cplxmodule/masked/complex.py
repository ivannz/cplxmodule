import torch

from ..cplx import cplx_linear
from ..layers import CplxLinear
from .base import BaseMasked, MaskedWeightMixin

from ..utils.stats import SparsityStats


class CplxLinearMasked(MaskedWeightMixin, CplxLinear,
                       BaseMasked, SparsityStats):
    def forward(self, input):
        return cplx_linear(input, self.weight_masked, self.bias)

    def sparsity(self, *, hard=True, **kwargs):
        weight = self.weight

        n_dropped = float(weight.real.numel())
        if self.is_sparse:
            mask = torch.gt(self.mask, 0) if hard else self.mask
            n_dropped -= float(mask.sum().item())

        return [(id(weight.real), n_dropped), (id(weight.imag), n_dropped), ]
