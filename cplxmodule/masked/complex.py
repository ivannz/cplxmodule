import torch

from ..cplx import cplx_linear, cplx_bilinear
from ..layers import CplxLinear, CplxBilinear
from .base import BaseMasked, MaskedWeightMixin

from ..utils.stats import SparsityStats


class CplxLinearMasked(MaskedWeightMixin, CplxLinear,
                       BaseMasked, SparsityStats):
    def forward(self, input):
        return cplx_linear(input, self.weight_masked, self.bias)

    def sparsity(self, *, hard=True, **kwargs):
        weight = self.weight

        if self.is_sparse:
            mask = torch.gt(self.mask, 0) if hard else self.mask
            n_dropped = float(weight.real.numel())
            n_dropped -= float(mask.sum().item())

        else:
            n_dropped = 0.

        return [(id(weight.real), n_dropped), (id(weight.imag), n_dropped), ]


class CplxBilinearMasked(MaskedWeightMixin, CplxBilinear,
                         BaseMasked, SparsityStats):
    def forward(self, input1, input2):
        return cplx_bilinear(input1, input2, self.weight_masked, self.bias)

    sparsity = CplxLinearMasked.sparsity
