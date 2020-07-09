import math

import torch
import torch.nn.functional as F

from ..... import cplx

from ....modules.linear import CplxLinear
from ....utils.sparsity import SparsityStats
from ...base import BaseARD


class CplxLinearPolARD(CplxLinear, BaseARD, SparsityStats):
    __sparsity_ignore__ = ("log_sigma2", "log_eta", "log_phi")

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)

        self.log_sigma2 = torch.nn.Parameter(
            torch.Tensor(out_features, in_features))
        self.log_eta = torch.nn.Parameter(
            torch.Tensor(out_features, in_features))
        self.log_phi = torch.nn.Parameter(
            torch.Tensor(out_features, in_features))

        self.reset_variational_parameters()

    def reset_variational_parameters(self):
        self.log_sigma2.data.uniform_(-10, -10)
        self.log_eta.data.uniform_(0., 0.)
        self.log_phi.data.uniform_(-10., +10.)

    @property
    def penalty(self):
        r"""Exact complex KL divergence.
        $$
            KL(q\|\pi^*)
                = \log(1+\tfrac1{\alpha}) - \tfrac12 \log(1-\eta^2)
                = \log(1+\tfrac1{\alpha}) + \log(1+e^{-x}) + x /2 - \log 2
            \,, $$
        for $\eta = 2 * \sigma(x) - 1$.
        """
        kl_div = F.softplus(- self.log_alpha) - math.log(2)
        return kl_div + self.log_eta / 2 + F.softplus(-self.log_eta)

    @property
    def log_alpha(self):
        r"""Get $\log \alpha$ from $(\theta, \sigma^2)$ parameterization."""
        # $\alpha = \tfrac{\sigma^2}{\theta \bar{\theta}}$
        return self.log_sigma2 - 2 * torch.log(abs(self.weight) + 1e-12)

    def relevance(self, *, threshold, **kwargs):
        """Get the relevance mask based on the threshold."""
        with torch.no_grad():
            return torch.le(self.log_alpha, threshold).to(self.log_alpha)

    def sparsity(self, *, threshold, **kwargs):
        relevance = self.relevance(threshold=threshold)

        weight = self.weight
        n_dropped = float(weight.real.numel()) - float(relevance.sum().item())
        return [(id(weight.real), n_dropped), (id(weight.imag), n_dropped)]

    def forward(self, input):
        '''See handwritten notes.'''
        # $\mu = \theta x$ in $\mathbb{C}$
        mu = super().forward(input)
        if not self.training:
            return mu

        # \sigma^2 = \Sigma^2 (x \odot \bar{x})
        var = torch.exp(self.log_sigma2)
        s2 = torch.clamp(F.linear(input.real * input.real +
                                  input.imag * input.imag, var, None), 1e-8)

        vareta = torch.tanh(self.log_eta / 2) * var

        # there has to be some trig idenity for cos(atan) and sin(atan)!
        # sin, cos = vert / hypot, horz / hypot. 'atan' is 'vert', 'horz' in '1.'
        # phi = torch.atan(self.log_phi)
        # unit = cplx.Cplx(torch.cos(phi), torch.sin(phi))
        root = torch.sqrt(1 + self.log_phi * self.log_phi)
        unit = cplx.Cplx(1. / root, self.log_phi / root)

        xi = cplx.linear_cat(input * input, vareta * unit, None) / s2

        eps1, eps2 = torch.randn_like(s2), torch.randn_like(s2)

        rho = torch.sqrt(torch.clamp(1 - xi.real * xi.real - xi.imag * xi.imag, 1e-8))

        std = 0.5 * torch.sqrt(s2 / (1 + rho))
        return mu + cplx.Cplx(
            eps1 * (1 + xi.real + rho) + xi.imag * eps2,
            eps1 * xi.imag + (1 - xi.real + rho) * eps2,
        ) * std
