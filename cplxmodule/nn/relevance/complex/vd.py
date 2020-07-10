import warnings

import torch
import torch.nn
import scipy
import scipy.special

from numpy import euler_gamma

from ..base import BaseARD
from ...utils.sparsity import SparsityStats

from .base import CplxLinearGaussian, CplxBilinearGaussian
from .base import CplxConv1dGaussian, CplxConv2dGaussian, CplxConv3dGaussian


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


class RelevanceMixin(SparsityStats):
    __sparsity_ignore__ = ("log_sigma2",)

    def relevance(self, *, threshold, **kwargs):
        """Get the relevance mask based on the threshold."""
        with torch.no_grad():
            return torch.le(self.log_alpha, threshold).to(self.log_alpha)

    def sparsity(self, *, threshold, **kwargs):
        relevance = self.relevance(threshold=threshold)

        weight = self.weight
        n_dropped = float(weight.real.numel()) - float(relevance.sum().item())
        return [(id(weight.real), n_dropped), (id(weight.imag), n_dropped)]


class CplxVDMixin:
    r"""Trait class with kl-divergence penalty of the cplx variational dropout.

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
    """

    @property
    def penalty(self):
        """Exact complex KL divergence."""
        n_log_alpha = - self.log_alpha
        return euler_gamma + n_log_alpha - torch_expi(- torch.exp(n_log_alpha))


class CplxLinearVD(CplxVDMixin, RelevanceMixin, CplxLinearGaussian, BaseARD):
    """Complex-valued linear layer with variational dropout."""
    pass


class CplxBilinearVD(CplxVDMixin, RelevanceMixin, CplxBilinearGaussian, BaseARD):
    """Complex-valued bilinear layer with variational dropout."""
    pass


class CplxConv1dVD(CplxVDMixin, RelevanceMixin, CplxConv1dGaussian, BaseARD):
    """1D complex-valued convolution layer with variational dropout."""
    pass


class CplxConv2dVD(CplxVDMixin, RelevanceMixin, CplxConv2dGaussian, BaseARD):
    """2D complex-valued convolution layer with variational dropout."""
    pass


class CplxConv3dVD(CplxVDMixin, RelevanceMixin, CplxConv3dGaussian, BaseARD):
    """3D complex-valued convolution layer with variational dropout."""
    pass


class CplxLinearARD(object):
    def __new__(cls, in_features, out_features, bias=True):
        warnings.warn("Importing complex-valued Automatic Relevance Determination"
                      " layers (ARD) from `cplxmodule.nn.relevance.complex` has"
                      " been deprecated due to misleading name. Starting with"
                      " version `2021` the `.complex` submodule will export complex-"
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
                      " version `2021` the `.complex` submodule will export complex-"
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
                      " version `2021` the `.complex` submodule will export complex-"
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
                      " version `2021` the `.complex` submodule will export complex-"
                      "valued Variational Dropout (VD) layers only. Please"
                      " import ARD layers from `relevance.ard`.",
                      FutureWarning)

        return CplxConv2dVD(in_channels, out_channels, kernel_size, stride,
                            padding, dilation, groups, bias, padding_mode)
