import warnings

import torch
import torch.nn

import torch.nn.functional as F

from ..base import BaseARD
from ...utils.sparsity import SparsityStats

from .base import LinearGaussian, BilinearGaussian
from .base import Conv1dGaussian, Conv2dGaussian, Conv3dGaussian


class RelevanceMixin(SparsityStats):
    __sparsity_ignore__ = ("log_sigma2",)

    def relevance(self, *, threshold, **kwargs):
        """Get the relevance mask based on the threshold."""
        with torch.no_grad():
            return torch.le(self.log_alpha, threshold).to(self.log_alpha)

    def sparsity(self, *, threshold, **kwargs):
        relevance = self.relevance(threshold=threshold)
        n_relevant = float(relevance.sum().item())
        return [(id(self.weight), self.weight.numel() - n_relevant)]


class RealVDMixin:
    r"""Trait class with kl-divergence penalty of the variational dropout.

    Details
    -------
    This uses the ideas and formulae of Kingma et al. and Molchanov et al.
    This module assumes the standard loss-minimization framework. Hence
    instead of -ve KL divergence for ELBO and log-likelihood maximization,
    this property computes and returns the divergence as is, which implies
    minimization of minus log-likelihood (and, thus, minus ELBO).

    Attributes
    ----------
    penalty : computed torch.Tensor, read-only
        The Kullback-Leibler divergence between the mean field approximate
        variational posterior of the weights and the scale-free log-uniform
        prior:
        $$
            KL(\mathcal{N}(w\mid \theta, \alpha \theta^2) \|
                    \tfrac1{\lvert w \rvert})
                = \mathbb{E}_{\xi \sim \mathcal{N}(1, \alpha)}
                    \log{\lvert \xi \rvert}
                - \tfrac12 \log \alpha + C
            \,. $$
    """

    @property
    def penalty(self):
        r"""Sofplus-sigmoid approximation of the Kl divergence from
        arxiv:1701.05369:
        $$
            \alpha \mapsto
                \tfrac12 \log (1 + e^{-\log \alpha}) - C
                - k_1 \sigma(k_2 + k_3 \log \alpha)
            \,, $$
        with $C$ chosen to be $- k_1$. Note that $x \mapsto \log(1 + e^x)$
        is known as `softplus` and in fact needs different compute paths
        depending on the sign of $x$, much like the stable method for the
        `log-sum-exp`:
        $$
            x \mapsto
                \log(1 + e^{-\lvert x\rvert}) + \max{\{x, 0\}}
            \,. $$
        See the paper eq. (14) (mind the overall negative sign) or the
        accompanying notebook for the MC estimation of the constants:
        `k1, k2, k3 = 0.63576, 1.87320, 1.48695`
        """
        n_log_alpha = - self.log_alpha
        sigmoid = torch.sigmoid(1.48695 * n_log_alpha - 1.87320)
        return F.softplus(n_log_alpha) / 2 + 0.63576 * sigmoid


class LinearVD(RealVDMixin, RelevanceMixin, LinearGaussian, BaseARD):
    """Linear layer with variational dropout.

    Details
    -------
    See `torch.nn.Linear` for reference on the dimensions and parameters.
    """
    pass


class BilinearVD(RealVDMixin, RelevanceMixin, BilinearGaussian, BaseARD):
    """Bilinear layer with variational dropout.

    Details
    -------
    See `torch.nn.Bilinear` for reference on the dimensions and parameters.
    """
    pass


class Conv1dVD(RealVDMixin, RelevanceMixin, Conv1dGaussian, BaseARD):
    """1D convolution layer with variational dropout.

    Details
    -------
    See `torch.nn.Conv1d` for reference on the dimensions and parameters. See
    `cplxmodule.nn.relevance.ConvNdGaussianMixin` for details about the
    implementation of the reparameterization trick.
    """
    pass


class Conv2dVD(RealVDMixin, RelevanceMixin, Conv2dGaussian, BaseARD):
    """2D convolution layer with variational dropout.

    Details
    -------
    See `torch.nn.Conv2d` for reference on the dimensions and parameters. See
    `cplxmodule.nn.relevance.ConvNdGaussianMixin` for details about the
    implementation of the reparameterization trick.
    """
    pass


class Conv3dVD(RealVDMixin, RelevanceMixin, Conv3dGaussian, BaseARD):
    """3D convolution layer with variational dropout.

    Details
    -------
    See `torch.nn.Conv3d` for reference on the dimensions and parameters. See
    `cplxmodule.nn.relevance.ConvNdGaussianMixin` for details about the
    implementation of the reparameterization trick.
    """
    pass


class LinearARD(object):
    def __new__(cls, in_features, out_features, bias=True):
        # Due to incorrect naming real-valued Automatic Relevance Determination
        # (ARD) layers from `.real` are and have always been Variational Dropout
        # layers (VD). The only difference is in the prior: ARD uses Gaussian
        # prior with adaptive precision, while VD uses log-uniform prior.
        # While empirical evidence suggests that they both sprasify similary,
        # and yield close compression levels, VD layers tended to have higher
        # arithmetic complexity due to the pentaly term.
        warnings.warn("Importing real-valued Automatic Relevance Determination"
                      " layers (ARD) from `cplxmodule.nn.relevance.real` has"
                      " been deprecated due to misleading name. Starting with"
                      " version `2021` the `.real` submodule will export real-"
                      "valued Variational Dropout (VD) layers only. Please"
                      " import ARD layers from `relevance.ard`.",
                      FutureWarning)

        return LinearVD(in_features, out_features, bias)


class Conv1dARD(object):
    def __new__(cls, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1,
                bias=True, padding_mode='zeros'):
        warnings.warn("Importing real-valued Automatic Relevance Determination"
                      " layers (ARD) from `cplxmodule.nn.relevance.real` has"
                      " been deprecated due to misleading name. Starting with"
                      " version `2021` the `.real` submodule will export real-"
                      "valued Variational Dropout (VD) layers only. Please"
                      " import ARD layers from `relevance.ard`.",
                      FutureWarning)

        return Conv1dVD(in_channels, out_channels, kernel_size, stride,
                        padding, dilation, groups, bias, padding_mode)


class Conv2dARD(object):
    def __new__(cls, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1,
                bias=True, padding_mode='zeros'):
        warnings.warn("Importing real-valued Automatic Relevance Determination"
                      " layers (ARD) from `cplxmodule.nn.relevance.real` has"
                      " been deprecated due to misleading name. Starting with"
                      " version `2021` the `.real` submodule will export real-"
                      "valued Variational Dropout (VD) layers only. Please"
                      " import ARD layers from `relevance.ard`.",
                      FutureWarning)

        return Conv2dVD(in_channels, out_channels, kernel_size, stride,
                        padding, dilation, groups, bias, padding_mode)


class BilinearARD(object):
    def __new__(cls, in1_features, in2_features, out_features, bias=True):
        warnings.warn("Importing real-valued Automatic Relevance Determination"
                      " layers (ARD) from `cplxmodule.nn.relevance.real` has"
                      " been deprecated due to misleading name. Starting with"
                      " version `2021` the `.real` submodule will export real-"
                      "valued Variational Dropout (VD) layers only. Please"
                      " import ARD layers from `relevance.ard`.",
                      FutureWarning)

        return BilinearVD(in1_features, in2_features, out_features, bias)
