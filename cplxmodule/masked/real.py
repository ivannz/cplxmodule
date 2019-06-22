import torch
import torch.nn.functional as F

from torch.nn import Linear
from .base import BaseMasked, SparseWeightMixin


class LinearMasked(SparseWeightMixin, Linear, BaseMasked):
    def forward(self, input):
        return F.linear(input, self.weight_masked, self.bias)

    def _sparsity(self, threshold=None, hard=True):
        if not self.is_sparse:
            return []

        # get the mask
        mask = torch.gt(self.mask, 0) if hard else self.mask
        n_relevant = float(mask.sum().item())

        # handle broadcasted masks
        n, m = mask.shape
        if n == 1:
            n_relevant = self.weight.shape[0] * n_relevant

        elif m == 1:
            n_relevant = n_relevant * self.weight.shape[1]

        return [(id(self.weight), self.weight.numel() - n_relevant)]
