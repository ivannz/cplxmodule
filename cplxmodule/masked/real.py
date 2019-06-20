import torch
import torch.nn.functional as F

from torch.nn import Linear
from .base import BaseMasked, SparseWeightMixin


class LinearMasked(SparseWeightMixin, BaseMasked, Linear):
    def forward(self, input):
        return F.linear(input, self.weight_masked, self.bias)
