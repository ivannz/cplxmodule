import torch
import torch.nn.functional as F

from torch.nn import Linear
from .base import BaseMasked, MaskedWeightMixin

from ..utils.stats import SparsityStats


class LinearMasked(MaskedWeightMixin, Linear,
                   BaseMasked, SparsityStats):
    def forward(self, input):
        return F.linear(input, self.weight_masked, self.bias)

    def sparsity(self, *, hard=True, **kwargs):
        n_dropped = float(self.weight.numel())
        if self.is_sparse:
            mask = torch.gt(self.mask, 0) if hard else self.mask
            n_dropped -= float(mask.sum().item())

        return [(id(self.weight), n_dropped)]
