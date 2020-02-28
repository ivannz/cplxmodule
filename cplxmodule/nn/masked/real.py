import torch
import torch.nn.functional as F

from torch.nn import Linear, Conv1d, Conv2d, Bilinear
from .base import BaseMasked, MaskedWeightMixin

from ..utils.sparsity import SparsityStats


class _BaseRealMixin(BaseMasked, SparsityStats):
    def sparsity(self, *, hard=True, **kwargs):
        if self.is_sparse:
            mask = torch.gt(self.mask, 0) if hard else self.mask
            n_dropped = float(self.weight.numel())
            n_dropped -= float(mask.sum().item())

        else:
            n_dropped = 0.

        return [(id(self.weight), n_dropped)]


class LinearMasked(MaskedWeightMixin, Linear, _BaseRealMixin):
    def forward(self, input):
        return F.linear(input, self.weight_masked, self.bias)


class Conv1dMasked(MaskedWeightMixin, Conv1d, _BaseRealMixin):
    def forward(self, input):
        return F.conv1d(input, self.weight_masked, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


class Conv2dMasked(MaskedWeightMixin, Conv2d, _BaseRealMixin):
    def forward(self, input):
        return F.conv2d(input, self.weight_masked, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


class BilinearMasked(MaskedWeightMixin, Bilinear, _BaseRealMixin):
    def forward(self, input1, input2):
        return F.bilinear(input1, input2, self.weight_masked, self.bias)
