import torch
import torch.nn

import torch.nn.functional as F

from math import sqrt
from numpy import euler_gamma

from .base import BaseARD

from .utils import kldiv_approx, torch_expi, ExpiFunction

from ..layers import CplxLinear
from ..cplx import Cplx


def cplx_nkldiv_exact(log_alpha, reduction="mean"):
    r"""
    Exact negative complex KL divergence
    $$
        - KL(\mathcal{CN}(w\mid \theta, \alpha \theta \bar{\theta}, 0) \|
                \tfrac1{\lvert w \rvert^2})
            = \log \alpha
              - 2 \mathbb{E}_{\xi \sim \mathcal{CN}(1, \alpha, 0)}
                \log{\lvert \xi \rvert} + C
            = \log \alpha + Ei( - \tfrac1{\alpha}) + C
        \,, $$
    where $Ei(x) = \int_{-\infty}^x e^t t^{-1} dt$ is the exponential integral.
    """
    if reduction is not None and reduction not in ("mean", "sum"):
        raise ValueError("""`reduction` must be either `None`, "sum" """
                         """or "mean".""")

    # Ei behaves well on the -ve values, and near 0-.
    kl_div = log_alpha + torch_expi(- torch.exp(- log_alpha)) - euler_gamma

    if reduction == "mean":
        return kl_div.mean()

    elif reduction == "sum":
        return kl_div.sum()

    return kl_div


class CplxLinearARD(CplxLinear, BaseARD):
    def __init__(self, in_features, out_features, bias=True, reduction="mean"):
        super().__init__(in_features, out_features, bias=bias)
        self.reduction = reduction

        self.log_sigma2 = torch.nn.Parameter(
            torch.Tensor(out_features, in_features))
        self.reset_variational_parameters()

    def reset_variational_parameters(self):
        self.log_sigma2.data.uniform_(-10, -10)  # wtf?

    @property
    def log_alpha(self):
        r"""Get $\log \alpha$ from $(\theta, \sigma^2)$ parameterization."""
        # $\alpha = \tfrac{\sigma^2}{\theta \bar{\theta}}$
        abs_weight = abs(Cplx(self.weight.real, self.weight.imag))
        return self.log_sigma2 - 2 * torch.log(abs_weight + 1e-12)

    @property
    def penalty(self):
        r"""Compute the variational penalty term."""
        # neg KL divergence must be maximized, hence the -ve sign.
        return -cplx_nkldiv_exact(self.log_alpha, reduction=self.reduction)

    def get_sparsity_mask(self, threshold):
        r"""Get the dropout mask based on the log-relevance."""
        with torch.no_grad():
            return torch.ge(self.log_alpha, threshold)

    def num_zeros(self, threshold=1.0):
        return 2 * self.get_sparsity_mask(threshold).sum().item()

    def forward(self, input):
        # $\mu = \theta x$ in $\mathbb{C}$
        mu = super().forward(input)
        # mu = cplx_linear(input, Cplx(**self.weight), self.bias)
        if not self.training:
            return mu

        # \gamma = \sigma^2 (x \odot \bar{x})
        s2 = F.linear(input.real * input.real + input.imag * input.imag,
                      torch.exp(self.log_sigma2), None)

        # generate complex gaussian noise with proper scale
        noise = Cplx(*map(torch.rand_like, (s2, s2))) / sqrt(2)
        return mu + noise * torch.sqrt(s2 + 1e-20)


def cplx_nkldiv_apprx(log_alpha, reduction="mean"):
    r"""
    Sofplus-sigmoid approximation of the negative complex KL divergence.
    $$
        - KL(\mathcal{CN}(w\mid \theta, \alpha \theta \bar{\theta}, 0) \|
                \tfrac1{\lvert w \rvert^2})
            = \log \alpha
              - 2 \mathbb{E}_{\xi \sim \mathcal{CN}(1, \alpha, 0)}
                \log{\lvert \xi \rvert} + C
        \,. $$
    For coef estimation and derivation c.f. the supplementary notebook.
    """
    coef = 0.57811, 1.46018, 1.36562, 1.  # 0.57811265, 1.4601848, 1.36561527
    return kldiv_approx(log_alpha, coef, reduction)


class CplxLinearARDApprox(CplxLinearARD):
    @property
    def penalty(self):
        r"""Compute the variational penalty term."""
        # neg KL divergence must be maximized, hence the -ve sign.
        return -cplx_nkldiv_apprx(self.log_alpha, reduction=self.reduction)


class BogusExpiFunction(ExpiFunction):
    """The Dummy Expi function, that compute bogus values on the forward pass,
    but correct values on the backwards pass, provided there is no downstream
    dependence on its forward-pass output.
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.zeros_like(x)


bogus_expi = BogusExpiFunction.apply


class CplxLinearARDBogus(CplxLinearARD):
    @property
    def penalty(self):
        r"""-ve KL-div with bogus forward output, but correct gradient."""
        log_alpha = self.log_alpha
        kl_div = log_alpha + bogus_expi(- torch.exp(- log_alpha))

        if self.reduction == "mean":
            return -kl_div.mean()

        elif self.reduction == "sum":
            return -kl_div.sum()

        return -kl_div
