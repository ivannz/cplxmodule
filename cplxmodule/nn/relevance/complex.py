import warnings

import torch
import torch.nn
import scipy
import scipy.special

import torch.nn.functional as F

from numpy import euler_gamma

from .base import BaseARD

from ..modules.linear import CplxLinear, CplxBilinear
from ..modules.conv import CplxConv1d, CplxConv2d, CplxConv3d
from ... import cplx

from ..utils.sparsity import SparsityStats


class ExpiFunction(torch.autograd.Function):
    r"""Pythonic differentiable port of scipy's Exponential Integral Ei.
    $$
        Ei
            \colon \mathbb{R} \to \mathbb{R} \cup \{\pm \infty\}
            \colon x \mapsto \int_{-\infty}^x \tfrac{e^t}{t} dt
        \,. $$

    Notes
    -----
    This may potentially introduce a memory transfer and compute bottleneck
    during the forward pass due to CPU-GPU device switch. Backward pass does
    not suffer from this issue and is computed on-device.
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)

        x_cpu = x.data.cpu().numpy()
        output = scipy.special.expi(x_cpu, dtype=x_cpu.dtype)
        return torch.from_numpy(output).to(x.device)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[-1]
        return grad_output * torch.exp(x) / x


torch_expi = ExpiFunction.apply


class _BaseRelevanceCplx(BaseARD, SparsityStats):
    r"""Base class with kl-divergence penalty of the variational dropout.

    Details
    -------
    This module assumes the standard loss-minimization framework. Hence
    instead of -ve KL divergence for ELBO and log-likelihood maximization,
    this property computes and returns the divergence as is, which implies
    minimization of minus log-likelihood (and, thus, minus ELBO).

    Attributes
    ----------
    penalty : computed torch.Tensor, read-only
        The Kullback-Leibler divergence between the mean field approximate
        complex variational posterior of the weights and the scale-free
        log-uniform complex prior:
        $$
            KL(\mathcal{CN}(w\mid \theta, \alpha \theta \bar{\theta}, 0) \|
                    \tfrac1{\lvert w \rvert^2})
                = 2 \mathbb{E}_{\xi \sim \mathcal{CN}(1, \alpha, 0)}
                    \log{\lvert \xi \rvert}
                  + C - \log \alpha
                = C - \log \alpha - Ei( - \tfrac1{\alpha})
            \,, $$
        where $Ei(x) = \int_{-\infty}^x e^t t^{-1} dt$ is the exponential
        integral. Unlike real-valued variational dropout, this KL divergence
        does not need an approximation, since it can be computed exactly via
        a special function. $Ei(x)$ behaves well on the -ve values, and near
        $0-$. The constant $C$ is fixed to Euler's gamma, so that the divergence
        is +ve.

    log_alpha : computed torch.Tensor, read-only
        Log-variance of the multiplicative scaling noise. Computed as a log
        of the ratio of the variance of the weight to the squared absolute
        value of the weight. The higher the log-alpha the less relevant the
        parameter is.
    """
    __sparsity_ignore__ = ("log_sigma2",)

    def reset_variational_parameters(self):
        self.log_sigma2.data.uniform_(-10, -10)  # wtf?

    @property
    def log_alpha(self):
        r"""Get $\log \alpha$ from $(\theta, \sigma^2)$ parameterization."""
        # $\alpha = \tfrac{\sigma^2}{\theta \bar{\theta}}$
        return self.log_sigma2 - 2 * torch.log(abs(self.weight) + 1e-12)

    @property
    def penalty(self):
        """Exact complex KL divergence."""
        n_log_alpha = - self.log_alpha
        return euler_gamma + n_log_alpha - torch_expi(- torch.exp(n_log_alpha))

    def relevance(self, *, threshold, **kwargs):
        """Get the relevance mask based on the threshold."""
        with torch.no_grad():
            return torch.le(self.log_alpha, threshold).to(self.log_alpha)

    def sparsity(self, *, threshold, **kwargs):
        relevance = self.relevance(threshold=threshold)

        weight = self.weight
        n_dropped = float(weight.real.numel()) - float(relevance.sum().item())
        return [(id(weight.real), n_dropped), (id(weight.imag), n_dropped)]


class CplxLinearVD(CplxLinear, _BaseRelevanceCplx):
    """Complex-valued linear layer with variational dropout."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)

        self.log_sigma2 = torch.nn.Parameter(
            torch.Tensor(out_features, in_features))
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


class CplxBilinearVD(CplxBilinear, _BaseRelevanceCplx):
    """Complex-valued bilinear layer with variational dropout."""

    def __init__(self, in1_features, in2_features, out_features, bias=True,
                 conjugate=True):
        super().__init__(in1_features, in2_features, out_features,
                         bias=bias, conjugate=conjugate)

        self.log_sigma2 = torch.nn.Parameter(
            torch.Tensor(out_features, in1_features, in2_features))
        self.reset_variational_parameters()

    def forward(self, input1, input2):
        mu = super().forward(input1, input2)
        if not self.training:
            return mu

        s2 = F.bilinear(input1.real * input1.real + input1.imag * input1.imag,
                        input2.real * input2.real + input2.imag * input2.imag,
                        torch.exp(self.log_sigma2), None)

        return mu + cplx.randn_like(s2) * torch.sqrt(torch.clamp(s2, 1e-8))


class CplxConv1dVD(CplxConv1d, _BaseRelevanceCplx):
    """1D complex-valued convolution layer with variational dropout."""

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

    def forward(self, input):
        mu = super().forward(input)
        if not self.training:
            return mu

        s2 = F.conv1d(input.real * input.real + input.imag * input.imag,
                      torch.exp(self.log_sigma2), None, self.stride,
                      self.padding, self.dilation, self.groups)

        return mu + cplx.randn_like(s2) * torch.sqrt(torch.clamp(s2, 1e-8))


class CplxConv2dVD(CplxConv2d, _BaseRelevanceCplx):
    """2D complex-valued convolution layer with variational dropout."""

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

    def forward(self, input):
        mu = super().forward(input)
        if not self.training:
            return mu

        s2 = F.conv2d(input.real * input.real + input.imag * input.imag,
                      torch.exp(self.log_sigma2), None, self.stride,
                      self.padding, self.dilation, self.groups)

        return mu + cplx.randn_like(s2) * torch.sqrt(torch.clamp(s2, 1e-8))


class CplxConv3dVD(CplxConv3d, _BaseRelevanceCplx):
    """3D complex-valued convolution layer with variational dropout."""

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

    def forward(self, input):
        mu = super().forward(input)
        if not self.training:
            return mu

        s2 = F.conv3d(input.real * input.real + input.imag * input.imag,
                      torch.exp(self.log_sigma2), None, self.stride,
                      self.padding, self.dilation, self.groups)

        return mu + cplx.randn_like(s2) * torch.sqrt(torch.clamp(s2, 1e-8))


class CplxLinearARD(object):
    def __new__(cls, in_features, out_features, bias=True):
        warnings.warn("Importing complex-valued Automatic Relevance Determination"
                      " layers (ARD) from `cplxmodule.nn.relevance.complex` has"
                      " been deprecated due to misleading name. Starting with"
                      " version `1.0` the `.real` submodule will export complex-"
                      "valued Variational Dropout (VD) layers only. Please"
                      " import ARD layers from `relevance.ard`.",
                      FutureWarning)

        return CplxLinearVD(in_features, out_features, bias)


class CplxBilinearARD(object):
    def __new__(cls, in1_features, in2_features, out_features, bias=True,
                conjugate=True):
        warnings.warn("Importing complex-valued Automatic Relevance Determination"
                      " layers (ARD) from `cplxmodule.nn.relevance.complex` has"
                      " been deprecated due to misleading name. Starting with"
                      " version `1.0` the `.real` submodule will export complex-"
                      "valued Variational Dropout (VD) layers only. Please"
                      " import ARD layers from `relevance.ard`.",
                      FutureWarning)

        return CplxBilinearVD(in1_features, in2_features, out_features, bias,
                              conjugate)


class CplxConv1dARD(object):
    def __new__(cls, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1,
                bias=True, padding_mode='zeros'):
        warnings.warn("Importing complex-valued Automatic Relevance Determination"
                      " layers (ARD) from `cplxmodule.nn.relevance.complex` has"
                      " been deprecated due to misleading name. Starting with"
                      " version `1.0` the `.real` submodule will export complex-"
                      "valued Variational Dropout (VD) layers only. Please"
                      " import ARD layers from `relevance.ard`.",
                      FutureWarning)

        return CplxConv1dVD(in_channels, out_channels, kernel_size, stride,
                            padding, dilation, groups, bias, padding_mode)


class CplxConv2dARD(object):
    def __new__(cls, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1,
                bias=True, padding_mode='zeros'):
        warnings.warn("Importing complex-valued Automatic Relevance Determination"
                      " layers (ARD) from `cplxmodule.nn.relevance.complex` has"
                      " been deprecated due to misleading name. Starting with"
                      " version `1.0` the `.real` submodule will export complex-"
                      "valued Variational Dropout (VD) layers only. Please"
                      " import ARD layers from `relevance.ard`.",
                      FutureWarning)

        return CplxConv2dVD(in_channels, out_channels, kernel_size, stride,
                            padding, dilation, groups, bias, padding_mode)
