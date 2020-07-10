import torch
import torch.nn

import torch.nn.functional as F

from .... import cplx

from ...modules.linear import CplxLinear, CplxBilinear
from ...modules.conv import CplxConv1d, CplxConv2d, CplxConv3d


class GaussianMixin:
    r"""Trait class with log-alpha property for variational dropout.

    Attributes
    ----------
    log_alpha : computed torch.Tensor, read-only
        Log-variance of the multiplicative scaling noise. Computed as a log
        of the ratio of the variance of the weight to the squared absolute
        value of the weight. The higher the log-alpha the less relevant the
        parameter is.
    """
    def reset_variational_parameters(self):
        self.log_sigma2.data.uniform_(-10, -10)  # wtf?

    @property
    def log_alpha(self):
        r"""Get $\log \alpha$ from $(\theta, \sigma^2)$ parameterization."""
        # $\alpha = \tfrac{\sigma^2}{\theta \bar{\theta}}$
        return self.log_sigma2 - 2 * torch.log(abs(self.weight) + 1e-12)


class CplxLinearGaussian(GaussianMixin, CplxLinear):
    """Complex-valued linear layer with variational dropout."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)

        self.log_sigma2 = torch.nn.Parameter(torch.Tensor(*self.weight.shape))
        self.reset_variational_parameters()

    def forward(self, input):
        # $\mu = \theta x$ in $\mathbb{C}$
        mu = super().forward(input)
        if not self.training:
            return mu

        # \gamma = \sigma^2 (x \odot \bar{x})
        s2 = F.linear(input.real * input.real + input.imag * input.imag,
                      torch.exp(self.log_sigma2), None)

        return mu + cplx.randn_like(s2) * torch.sqrt(torch.clamp(s2, 1e-8))


class CplxBilinearGaussian(GaussianMixin, CplxBilinear):
    """Complex-valued bilinear layer with variational dropout."""

    def __init__(self, in1_features, in2_features, out_features, bias=True,
                 conjugate=True):
        super().__init__(in1_features, in2_features, out_features,
                         bias=bias, conjugate=conjugate)

        self.log_sigma2 = torch.nn.Parameter(torch.Tensor(*self.weight.shape))
        self.reset_variational_parameters()

    def forward(self, input1, input2):
        mu = super().forward(input1, input2)
        if not self.training:
            return mu

        s2 = F.bilinear(input1.real * input1.real + input1.imag * input1.imag,
                        input2.real * input2.real + input2.imag * input2.imag,
                        torch.exp(self.log_sigma2), None)

        return mu + cplx.randn_like(s2) * torch.sqrt(torch.clamp(s2, 1e-8))


class CplxConvNdGaussianMixin(GaussianMixin):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups,
                         bias=bias, padding_mode=padding_mode)

        if self.padding_mode != "zeros":
            raise ValueError(f"Only `zeros` padding mode is supported. "
                             f"Got `{self.padding_mode}`.")

        self.log_sigma2 = torch.nn.Parameter(torch.Tensor(*self.weight.shape))
        self.reset_variational_parameters()

    def _forward_impl(self, input, conv):
        mu = super().forward(input)
        if not self.training:
            return mu

        s2 = conv(input.real * input.real + input.imag * input.imag,
                  torch.exp(self.log_sigma2), None, self.stride,
                  self.padding, self.dilation, self.groups)

        return mu + cplx.randn_like(s2) * torch.sqrt(torch.clamp(s2, 1e-8))


class CplxConv1dGaussian(CplxConvNdGaussianMixin, CplxConv1d):
    """1D complex-valued convolution layer with variational dropout."""

    def forward(self, input):
        return self._forward_impl(input, F.conv1d)


class CplxConv2dGaussian(CplxConvNdGaussianMixin, CplxConv2d):
    """2D complex-valued convolution layer with variational dropout."""

    def forward(self, input):
        return self._forward_impl(input, F.conv2d)


class CplxConv3dGaussian(CplxConvNdGaussianMixin, CplxConv3d):
    """3D complex-valued convolution layer with variational dropout."""

    def forward(self, input):
        return self._forward_impl(input, F.conv3d)
