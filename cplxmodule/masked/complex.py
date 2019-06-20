import torch

from ..layers import CplxLinear
from .base import BaseMasked, SparseWeightMixin

from ..cplx import cplx_linear


class CplxLinearMasked(SparseWeightMixin, BaseMasked, CplxLinear):
    def forward(self, input):
        return cplx_linear(input, self.weight_masked, self.bias)
