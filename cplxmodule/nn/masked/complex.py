import torch

from ... import cplx
from ..modules.linear import CplxLinear, CplxBilinear

from ..modules.conv import CplxConv1d, CplxConv2d, CplxConv3d

from .base import BaseMasked, MaskedWeightMixin

from ..utils.sparsity import SparsityStats


class _BaseCplxMixin(MaskedWeightMixin, BaseMasked, SparsityStats):
    __sparsity_ignore__ = ("mask",)

    def sparsity(self, *, hard=True, **kwargs):
        weight = self.weight

        if self.is_sparse:
            mask = torch.gt(self.mask, 0) if hard else self.mask
            n_dropped = float(weight.real.numel())
            n_dropped -= float(mask.sum().item())

        else:
            n_dropped = 0.

        return [(id(weight.real), n_dropped), (id(weight.imag), n_dropped), ]


class CplxLinearMasked(CplxLinear, _BaseCplxMixin):
    def forward(self, input):
        return cplx.linear(input, self.weight_masked, self.bias)


class CplxBilinearMasked(CplxBilinear, _BaseCplxMixin):
    def forward(self, input1, input2):
        return cplx.bilinear(input1, input2, self.weight_masked, self.bias)


class CplxConv1dMasked(CplxConv1d, _BaseCplxMixin):
    def forward(self, input):
        return cplx.conv1d(input, self.weight_masked, self.bias,
                           self.stride, self.padding, self.dilation,
                           self.groups, self.padding_mode)


class CplxConv2dMasked(CplxConv2d, _BaseCplxMixin):
    def forward(self, input):
        return cplx.conv2d(input, self.weight_masked, self.bias,
                           self.stride, self.padding, self.dilation,
                           self.groups, self.padding_mode)


class CplxConv3dMasked(CplxConv3d, _BaseCplxMixin):
    def forward(self, input):
        return cplx.conv3d(input, self.weight_masked, self.bias,
                           self.stride, self.padding, self.dilation,
                           self.groups, self.padding_mode)
