import torch
import torch.nn.functional as F

from torch.nn import Linear, Conv1d, Conv2d, Conv3d, Bilinear
from .base import BaseMasked, MaskedWeightMixin

from ..utils.sparsity import SparsityStats


class _BaseRealMixin(MaskedWeightMixin, BaseMasked, SparsityStats):
    __sparsity_ignore__ = ("mask",)

    def sparsity(self, *, hard=True, **kwargs):
        if self.is_sparse:
            mask = torch.gt(self.mask, 0) if hard else self.mask
            n_dropped = float(self.weight.numel())
            n_dropped -= float(mask.sum().item())

        else:
            n_dropped = 0.

        return [(id(self.weight), n_dropped)]


class LinearMasked(Linear, _BaseRealMixin):
    def forward(self, input):
        return F.linear(input, self.weight_masked, self.bias)


class Conv1dMasked(Conv1d, _BaseRealMixin):
    def forward(self, input):
        return F.conv1d(input, self.weight_masked, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


class Conv2dMasked(Conv2d, _BaseRealMixin):
    def forward(self, input):
        return F.conv2d(input, self.weight_masked, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


class Conv3dMasked(Conv3d, _BaseRealMixin):
    def forward(self, input):
        return F.conv3d(input, self.weight_masked, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


class BilinearMasked(Bilinear, _BaseRealMixin):
    def forward(self, input1, input2):
        return F.bilinear(input1, input2, self.weight_masked, self.bias)
