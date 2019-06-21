import torch
import torch.nn

import torch.nn.functional as F

from .base import BaseARD


class LinearARD(torch.nn.Linear, BaseARD):
    r"""Linear layer with automatic relevance detection.

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
    __ard_ignore__ = ("log_sigma2",)

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)

        self.log_sigma2 = torch.nn.Parameter(torch.Tensor(*self.weight.shape))
        self.reset_variational_parameters()

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
        arxiv:1701.05369 :
        $$
            \alpha \mapsto
                k_4 \log (1 + e^{-\log \alpha}) - C
                - k_1 \sigma(k_2 + k_3 \log \alpha)
            \,, $$
        with $C = - k_1$ and $k_4 = 0.5$. Note that $x \mapsto \log(1 + e^x)$
        is known as `softplus` and in fact needs different compute paths
        depending on the sign of $x$, much like the stable method for the
        `log-sum-exp`:
        $$
            x \mapsto
                \log(1+e^{-\lvert x\rvert}) + \max{\{x, 0\}}
            \,. $$
        """
        log_alpha = self.log_alpha
        k1, k2, k3 = 0.63576, 1.87320, 1.48695
        sigmoid = torch.sigmoid(k2 + k3 * log_alpha)
        return F.softplus(- log_alpha) / 2 - k1 * sigmoid + k1

    def forward(self, input):
        mu = super().forward(input)
        if not self.training:
            return mu

        s2 = F.linear(input * input, torch.exp(self.log_sigma2), None)
        return mu + torch.randn_like(s2) * torch.sqrt(s2 + 1e-20)

    def relevance(self, threshold, hard=None):
        r"""Get the relevance mask based on the threshold."""
        with torch.no_grad():
            return torch.le(self.log_alpha, threshold).to(self.log_alpha)

    def _sparsity(self, threshold, hard=None):
        n_relevant = float(self.relevance(threshold).sum().item())
        return [(id(self.weight), self.weight.numel() - n_relevant)]
