import torch
import torch.nn.functional as F

from torch.nn import Linear, Conv2d, Bilinear
from .base import BaseMasked, MaskedWeightMixin

from ..utils.stats import SparsityStats


class LinearMasked(MaskedWeightMixin, Linear,
                   BaseMasked, SparsityStats):
    def forward(self, input):
        return F.linear(input, self.weight_masked, self.bias)

    def sparsity(self, *, hard=True, **kwargs):
        if self.is_sparse:
            mask = torch.gt(self.mask, 0) if hard else self.mask
            n_dropped = float(self.weight.numel())
            n_dropped -= float(mask.sum().item())

        else:
            n_dropped = 0.

        return [(id(self.weight), n_dropped)]


class Conv2dMasked(MaskedWeightMixin, Conv2d, torch.nn.modules.conv._ConvNd,
                   BaseMasked, SparsityStats):
    def forward(self, input):
        return F.conv2d(input, self.weight_masked, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

    sparsity = LinearMasked.sparsity


class BilinearMasked(MaskedWeightMixin, Bilinear,
                     BaseMasked, SparsityStats):
    def forward(self, input1, input2):
        return F.bilinear(input1, input2, self.weight_masked, self.bias)

    sparsity = LinearMasked.sparsity
