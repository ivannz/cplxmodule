import warnings

import torch
import torch.nn

import torch.nn.functional as F

from .base import BaseARD
from ..utils.sparsity import SparsityStats


class _BaseRelevanceReal(BaseARD, SparsityStats):
    r"""Base class with kl-divergence penalty of the variational dropout.

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

    log_alpha : computed torch.Tensor, read-only
        Log-variance of the multiplicative scaling noise. Computed as a log
        of the ratio of the variance of the weight to the squared absolute
        value of the weight. The higher the log-alpha the less relevant the
        parameter is.
    """

    __sparsity_ignore__ = ("log_sigma2",)

    def reset_variational_parameters(self):
        # initially everything is relevant
        self.log_sigma2.data.uniform_(-10, -10)

    @property
    def log_alpha(self):
        r"""Get $\log \alpha$ from $(\theta, \sigma^2)$ parameterization."""
        return self.log_sigma2 - 2 * torch.log(abs(self.weight) + 1e-12)

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

    def relevance(self, *, threshold, **kwargs):
        """Get the relevance mask based on the threshold."""
        with torch.no_grad():
            return torch.le(self.log_alpha, threshold).to(self.log_alpha)

    def sparsity(self, *, threshold, **kwargs):
        relevance = self.relevance(threshold=threshold)
        n_relevant = float(relevance.sum().item())
        return [(id(self.weight), self.weight.numel() - n_relevant)]


class LinearVD(torch.nn.Linear, _BaseRelevanceReal):
    """Linear layer with variational dropout.

    Details
    -------
    See `torch.nn.Linear` for reference on the dimensions and parameters.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)

        self.log_sigma2 = torch.nn.Parameter(torch.Tensor(*self.weight.shape))
        self.reset_variational_parameters()

    def forward(self, input):
        mu = super().forward(input)
        if not self.training:
            return mu

        s2 = F.linear(input * input, torch.exp(self.log_sigma2), None)
        return mu + torch.randn_like(s2) * torch.sqrt(torch.clamp(s2, 1e-8))


class Conv1dVD(torch.nn.Conv1d, _BaseRelevanceReal):
    """1D convolution layer with variational dropout.

    Details
    -------
    See `torch.nn.Conv1d` for reference on the dimensions and parameters. See
    `cplxmodule.nn.relevance.Conv2dVD` for details about the implementation of
    the reparameterization trick.
    """

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
        """Forward pass of the SGVB method for a 1d convolutional layer.

        Details
        -------
        See `.forward` of Conv2dVD layer.
        """
        mu = super().forward(input)
        if not self.training:
            return mu

        s2 = F.conv1d(input * input, torch.exp(self.log_sigma2), None,
                      self.stride, self.padding, self.dilation, self.groups)
        return mu + torch.randn_like(s2) * torch.sqrt(torch.clamp(s2, 1e-8))


class Conv2dVD(torch.nn.Conv2d, _BaseRelevanceReal):
    """2D convolution layer with variational dropout.

    Details
    -------
    See `torch.nn.Conv2d` for reference on the dimensions and parameters.
    """

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
        r"""Forward pass of the SGVB method for a 2d convolutional layer.

        Details
        -------
        A convolution can be represented as matrix-vector product of the doubly
        block-circulant embedding (Toeplitz) of the kernel and the unravelled
        input. As such, it is an implicit linear layer with block structured
        weight matrix, but unlike it, the local reparameterization trick has
        a little caveat. If the kernel itself is assumed to have the specified
        variational distribution, then the outputs will be spatially correlated
        due to the same weight block being reused at each location:
        $$
            cov(y_{f\beta}, y_{k\omega})
                = \delta_{f=k} \sum_{c \alpha}
                    \sigma^2_{fc \alpha}
                    x_{c i_\beta(\alpha)}
                    x_{c i_\omega(\alpha)}
            \,, $$
        where $i_\beta(\alpha)$ is the location in $x$ for the output location
        $\beta$ and kernel offset $\alpha$ (depends on stride and dilation).
        In contrast, if instead the Toeplitz embedding blocks are assumed iid
        draws from the variational distribution, then covariance becomes
        $$
            cov(y_{f\beta}, y_{k\omega})
                = \delta_{f\beta = k\omega} \sum_{c \alpha}
                    \sigma^2_{fc \alpha}
                    \lvert x_{c i_\omega(\alpha)} \rvert^2
            \,. $$
        Molchanov et al. (2017) implicitly assume that kernels is are iid draws
        from the variational distribution for different spatial locations. This
        effectively zeroes the spatial cross-correlation in the output, reduces
        the variance of the gradient in SGVB method.
        """
        mu = super().forward(input)
        if not self.training:
            return mu

        s2 = F.conv2d(input * input, torch.exp(self.log_sigma2), None,
                      self.stride, self.padding, self.dilation, self.groups)
        return mu + torch.randn_like(s2) * torch.sqrt(torch.clamp(s2, 1e-8))


class Conv3dVD(torch.nn.Conv3d, _BaseRelevanceReal):
    """3D convolution layer with variational dropout.

    Details
    -------
    See `torch.nn.Conv3d` for reference on the dimensions and parameters. See
    `cplxmodule.nn.relevance.Conv2dVD` for details about the implementation of
    the reparameterization trick.
    """

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
        """Forward pass of the SGVB method for a 1d convolutional layer.

        Details
        -------
        See `.forward` of Conv2dVD layer.
        """
        mu = super().forward(input)
        if not self.training:
            return mu

        s2 = F.conv3d(input * input, torch.exp(self.log_sigma2), None,
                      self.stride, self.padding, self.dilation, self.groups)
        return mu + torch.randn_like(s2) * torch.sqrt(torch.clamp(s2, 1e-8))


class BilinearVD(torch.nn.Bilinear, _BaseRelevanceReal):
    """Bilinear layer with variational dropout.

    Details
    -------
    See `torch.nn.Bilinear` for reference on the dimensions and parameters.
    """

    def __init__(self, in1_features, in2_features, out_features, bias=True):
        super().__init__(in1_features, in2_features, out_features, bias=bias)

        self.log_sigma2 = torch.nn.Parameter(torch.Tensor(*self.weight.shape))
        self.reset_variational_parameters()

    def forward(self, input1, input2):
        """Forward pass of the SGVB method for a bilinear layer.

        Straightforward generalization of the local reparameterization trick.
        """
        mu = super().forward(input1, input2)
        if not self.training:
            return mu

        s2 = F.bilinear(input1 * input1, input2 * input2,
                        torch.exp(self.log_sigma2), None)

        # .normal reports a grad-fn, but weirdly does not pass grads!
        return mu + torch.randn_like(s2) * torch.sqrt(torch.clamp(s2, 1e-8))


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
                      " version `1.0` the `.real` submodule will export real-"
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
                      " version `1.0` the `.real` submodule will export real-"
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
                      " version `1.0` the `.real` submodule will export real-"
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
                      " version `1.0` the `.real` submodule will export real-"
                      "valued Variational Dropout (VD) layers only. Please"
                      " import ARD layers from `relevance.ard`.",
                      FutureWarning)

        return BilinearVD(in1_features, in2_features, out_features, bias)
