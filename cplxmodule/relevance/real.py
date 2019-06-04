import torch
import torch.nn

import torch.nn.functional as F

from .base import BaseLinearARD

from .utils import kldiv_approx
from .utils import torch_sparse_linear, torch_sparse_tensor


def real_nkldiv_apprx(log_alpha, reduction="mean"):
    r"""
    Approximation of the negative Kl divergence from arxiv:1701.05369.
    $$
        - KL(\mathcal{N}(w\mid \theta, \alpha \theta^2) \|
                \tfrac1{\lvert w \rvert})
            = \tfrac12 \log \alpha
              - \mathbb{E}_{\xi \sim \mathcal{N}(1, \alpha)}
                \log{\lvert \xi \rvert} + C
        \,. $$
    """
    coef = 0.63576, 1.87320, 1.48695, 0.5
    return kldiv_approx(log_alpha, coef, reduction)


class LinearARD(torch.nn.Linear, BaseLinearARD):
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
        return self.log_sigma2 - 2 * torch.log(abs(self.weight) + 1e-12)

    @property
    def penalty(self):
        r"""Compute the variational penalty term."""
        # neg KL divergence must be maximized, hence the -ve sign.
        return - real_nkldiv_apprx(self.log_alpha, reduction="mean")

    def forward(self, input):
        if not self.training and self.is_sparse:
            return self.forward_sparse(input)

        mu = super().forward(input)
        # mu = F.linear(input, self.weight, self.bias)
        if not self.training:
            return mu
        # end if

        s2 = F.linear(input * input, torch.exp(self.log_sigma2), None)
        return mu + torch.randn_like(s2) * torch.sqrt(s2 + 1e-20)

    def forward_sparse(self, input):
        weight = self.sparse_weight_
        if self.sparsity_mode_ == "dense":
            return F.linear(input, weight, self.bias)

        return torch_sparse_linear(input, weight, self.bias)

    def sparsify(self, threshold=1.0, mode="dense"):
        if mode is not None and mode not in ("dense", "sparse"):
            raise ValueError(f"""`mode` must be either 'dense', 'sparse' or """
                             f"""`None` (got '{mode}').""")

        if mode is not None and self.training:
            raise RuntimeError("Cannot sparsify model while training.")

        self.sparsity_mode_ = mode
        if mode is not None:
            mask = ~self.get_sparsity_mask(threshold)

            if mode == "sparse":
                weight = torch_sparse_tensor(
                    mask.nonzero().t(), self.weight[mask], self.weight.shape)

            elif mode == "dense":
                zero = torch.tensor(0.).to(self.weight)
                weight = torch.where(mask, self.weight, zero)

            self.register_buffer("sparse_weight_", weight)

        else:
            if hasattr(self, "sparse_weight_"):
                del self.sparse_weight_

        # end if

        return self

    def num_zeros(self, threshold=1.0):
        return self.get_sparsity_mask(threshold).sum().item()
