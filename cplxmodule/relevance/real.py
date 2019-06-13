import torch
import torch.nn

import torch.nn.functional as F

from .base import BaseLinearARD

from .utils import kldiv_approx
from .utils import torch_sparse_linear, torch_sparse_tensor

from .utils import parameter_to_buffer, buffer_to_parameter


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
        if self.is_sparse:
            return self.forward_sparse(input)

        mu = super().forward(input)
        # mu = F.linear(input, self.weight, self.bias)
        if not self.training:
            return mu
        # end if

        s2 = F.linear(input * input, torch.exp(self.log_sigma2), None)
        return mu + torch.randn_like(s2) * torch.sqrt(s2 + 1e-20)

    def forward_sparse(self, input):
        nonzero, weight = self.nonzero_, self.weight_
        if self.sparsity_mode_ == "dense":
            return F.linear(input, weight * nonzero, self.bias)

        else:
            weight_ = torch_sparse_tensor(nonzero, weight, self.weight.shape)
            return torch_sparse_linear(input, weight_, self.bias)

    def sparsify(self, threshold=1.0, mode="dense"):
        if mode is not None and mode not in ("dense", "sparse"):
            raise ValueError(f"""`mode` must be either 'dense', 'sparse' or """
                             f"""`None` (got '{mode}').""")

        if self.is_sparse and self.sparsity_mode_ != mode:
            # reinstate dropout mode and discard runtime data on mode change
            del self.nonzero_, self.weight_
            buffer_to_parameter(self, "log_sigma2")
            buffer_to_parameter(self, "weight")

        if mode is not None:
            # switch off variatonal dropout and create runtime sparse data
            parameter_to_buffer(self, "log_sigma2")
            parameter_to_buffer(self, "weight")

            mask = ~self.get_sparsity_mask(threshold)
            if mode == "sparse":
                # truly sparse mode
                weight = self.weight.data[mask].clone()
                self.register_buffer("nonzero_", mask.nonzero().t())

            elif mode == "dense":
                # smiluated sparse mode
                mask = mask.data.to(self.weight)
                weight = self.weight.data * mask
                self.register_buffer("nonzero_", mask)

            # make weight into a buffer (load_state dict doesn't care
            #  about param/buffer distinction!)
            self.register_parameter("weight_", torch.nn.Parameter(weight))
        # end if

        self.sparsity_mode_ = mode

        return self

    def num_zeros(self, threshold=1.0):
        return self.get_sparsity_mask(threshold).sum().item()
