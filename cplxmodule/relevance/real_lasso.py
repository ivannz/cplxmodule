import torch

from ..relevance.base import BaseARD


class LinearLASSO(torch.nn.Linear, BaseARD):
    def __init__(self, in_features, out_features, bias=True, reduction="mean"):
        super().__init__(in_features, out_features, bias=bias)
        self.reduction = reduction

    @property
    def penalty(self):
        w_norm = abs(self.weight)
        if self.reduction == "mean":
            return w_norm.mean()

        elif self.reduction == "sum":
            return w_norm.sum()

        return w_norm

    def get_sparsity_mask(self, threshold):
        with torch.no_grad():
            # the mask is $\tau \mapsto \lvert w_{ij} \rvert \leq \tau$
            return torch.le(torch.log(abs(self.weight) + 1e-20), threshold)

    def num_zeros(self, threshold):
        return self.get_sparsity_mask(threshold).sum().item()
