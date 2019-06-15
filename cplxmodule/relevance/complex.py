import warnings

import torch
import torch.nn

import torch.nn.functional as F

from math import sqrt
from numpy import euler_gamma

from .base import BaseARD, SparseModeMixin

from .utils import kldiv_approx, torch_expi, ExpiFunction
from .utils import torch_sparse_cplx_linear, torch_sparse_tensor
from .utils import parameter_to_buffer, buffer_to_parameter

from ..layers import CplxLinear, CplxParameter
from ..cplx import Cplx, cplx_linear


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


class CplxLinearARD(CplxLinear, BaseARD, SparseModeMixin):
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
        abs_weight = abs(Cplx(self.weight.real, self.weight.imag))
        return self.log_sigma2 - 2 * torch.log(abs_weight + 1e-12)

    @property
    def penalty(self):
        r"""Compute the variational penalty term."""
        # neg KL divergence must be maximized, hence the -ve sign.
        return -cplx_nkldiv_exact(self.log_alpha, reduction="mean")

    def get_sparsity_mask(self, threshold):
        r"""Get the dropout mask based on the log-relevance."""
        with torch.no_grad():
            return torch.ge(self.log_alpha, threshold)

    def num_zeros(self, threshold=1.0):
        return 2 * self.get_sparsity_mask(threshold).sum().item()

    def forward(self, input):
        if self.is_sparse:
            return self.forward_sparse(input)

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

    def forward_sparse(self, input):
        nonzero_, weight_ = self.nonzero_, self.weight_
        bias = Cplx(**self.bias) if self.bias is not None else None
        if self.sparsity_mode_ == "dense":
            return cplx_linear(input, Cplx(**weight_) * nonzero_, bias)

        elif self.sparsity_mode_ == "sparse":
            shape = self.weight.real.shape
            weight_ = Cplx(
                torch_sparse_tensor(nonzero_, weight_.real, shape),
                torch_sparse_tensor(nonzero_, weight_.imag, shape))
            return torch_sparse_cplx_linear(input, weight_, bias)

        raise RuntimeError(f"Unrecognized sparsity mode. "
                           f"Got `{self.sparsity_mode_}`")

    def sparsify(self, mask, mode="dense"):
        if not hasattr(self, "sparsity_mode_"):
            self.sparsity_mode_ = None

        if mode is not None and mode not in ("dense", "sparse"):
            raise ValueError(f"`mode` must be either 'dense', 'sparse' "
                             f"or `None`. Got '{mode}'.")

        if mode == "sparse":
            warnings.warn("mode 'sparse' will likely be discontinued "
                          "and later deprecated.", DeprecationWarning)

        if mask is not None and (mask.dtype not in (torch.bool, torch.uint8)
           or mask.shape != self.weight.real.shape):
            raise RuntimeError(f"`mask` must be None or a binary matrix "
                               f"{self.weight.real.shape}. Got '{mask.shape}'.")

        if mask is None:
            mode = None

        # None -> sparse/dense : mutate par-to-buf
        if not self.is_sparse and mode is not None:
            weight = Cplx(**self.weight)
            if mode == "sparse":
                # truly sparse mode: using torch sparse tensor
                mask = mask.detach().to(weight.real.device)
                weight_ = weight.detach()[mask].apply(torch.clone)
                nonzero_ = mask.nonzero().t()

            elif mode == "dense":
                # simulated sparse mode: using dense matrices with hard zeros
                nonzero_ = mask.detach().to(weight.real)
                weight_ = weight.detach() * nonzero_

            # .register_parameter() doesn't register parameter containers.
            self.weight_ = CplxParameter(weight_)
            self.register_buffer("nonzero_", nonzero_)

            # lastly, mutate the original parameter into a no-grad buffer
            parameter_to_buffer(self, "weight")
            parameter_to_buffer(self, "log_sigma2")

        # sparse/dense -> None : mutate buf-to-par
        elif self.is_sparse and mode is None:
            # some copying on new learnt weights could take place here.
            pass

            del self.nonzero_, self.weight_
            buffer_to_parameter(self, "weight")
            buffer_to_parameter(self, "log_sigma2")

        # sparse / dense -> dense / sparse : re-mutatation
        elif self.is_sparse and mode is not None:
            # sparse -> sparse or dense -> dense : check mask
            if self.sparsity_mode_ == mode:
                # binary masks or nonzero indices are exactly equal : nothing
                if mode == "sparse":
                    nonzero_ = mask.nonzero().t()
                    if torch.equal(nonzero_.to(self.nonzero_), self.nonzero_):
                        return self

                elif mode == "dense":
                    if torch.equal(mask.to(self.nonzero_), self.nonzero_):
                        return self

            # sparse -> dense or dense -> sparse : re-mutate
            else:
                pass

            # perform "sparse/dense -> None -> sparse/dense" : discards data
            self.sparsify(mask, mode=None)
            return self.sparsify(mask, mode=mode)

        # None -> None : nothing
        elif not self.is_sparse and mode is None:
            pass

        self.sparsity_mode_ = mode
        return self


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
        return -cplx_nkldiv_apprx(self.log_alpha, reduction="mean")


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
        log_alpha = self.log_alpha
        kl_div = log_alpha + bogus_expi(- torch.exp(- log_alpha))
        return -kl_div.mean()
