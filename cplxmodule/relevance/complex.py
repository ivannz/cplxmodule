import torch
import torch.nn
import scipy
import scipy.special

import torch.nn.functional as F

from math import sqrt
from numpy import euler_gamma

from .base import BaseARD

from ..layers import CplxLinear
from ..cplx import Cplx


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


class CplxLinearARD(CplxLinear, BaseARD):
    r"""Complex valued linear layer with automatic relevance detection.

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
    __ard_ignore__ = ("log_sigma2",)

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)

        self.log_sigma2 = torch.nn.Parameter(
            torch.Tensor(out_features, in_features))
        self.reset_variational_parameters()

    def reset_variational_parameters(self):
        self.log_sigma2.data.uniform_(-10, -10)  # wtf?

    @property
    def log_alpha(self):
        r"""Get $\log \alpha$ from $(\theta, \sigma^2)$ parameterization."""
        # $\alpha = \tfrac{\sigma^2}{\theta \bar{\theta}}$
        return self.log_sigma2 - 2 * torch.log(abs(self.weight) + 1e-12)

    @property
    def penalty(self):
        r"""Exact complex KL divergence."""
        log_alpha = self.log_alpha
        return euler_gamma - log_alpha - torch_expi(- torch.exp(- log_alpha))

    def forward(self, input):
        # $\mu = \theta x$ in $\mathbb{C}$
        mu = super().forward(input)
        if not self.training:
            return mu

        # \gamma = \sigma^2 (x \odot \bar{x})
        s2 = F.linear(input.real * input.real + input.imag * input.imag,
                      torch.exp(self.log_sigma2), None)

        # generate complex Gaussian noise with proper scale
        noise = Cplx(*map(torch.randn_like, (s2, s2))) / sqrt(2)
        return mu + noise * torch.sqrt(s2 + 1e-20)

    def relevance(self, threshold, hard=None):
        r"""Get the relevance mask based on the threshold."""
        with torch.no_grad():
            return torch.le(self.log_alpha, threshold).to(self.log_alpha)

    def _sparsity(self, threshold, hard=None):
        n_relevant = float(self.relevance(threshold).sum().item())
        weight = self.weight
        return [
            (id(weight.real), weight.real.numel() - n_relevant),
            (id(weight.imag), weight.imag.numel() - n_relevant),
        ]


class CplxLinearARDApprox(CplxLinearARD):
    @property
    def penalty(self):
        r"""Sofplus-sigmoid approximation of the complex KL divergence.
        $$
            \alpha \mapsto
                \log (1 + e^{-\log \alpha}) - C
                - k_1 \sigma(k_2 + k_3 \log \alpha)
            \,, $$
        with $C$ chosen as $- k_1$. Note that $x \mapsto \log(1 + e^x)$
        is known as `softplus` and in fact needs different compute paths
        depending on the sign of $x$, much like the stable method for the
        `log-sum-exp`:
        $$
            x \mapsto
                \log(1+e^{-\lvert x\rvert}) + \max{\{x, 0\}}
            \,. $$

        See the accompanying notebook for the MC estimation of the k1-k3
        constants: `k1, k2, k3 = 0.57810091, 1.45926293, 1.36525956`
        """
        n_log_alpha = - self.log_alpha
        sigmoid = torch.sigmoid(1.36526 * n_log_alpha - 1.45926)
        return F.softplus(n_log_alpha) + 0.57810 * sigmoid


class BogusExpiFunction(ExpiFunction):
    """The Dummy Expi function, that computes bogus values on the forward pass,
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
        r"""KL-div with bogus forward output, but correct gradient."""
        log_alpha = self.log_alpha
        return - log_alpha - bogus_expi(- torch.exp(- log_alpha))
