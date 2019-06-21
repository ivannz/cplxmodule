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

    def relevance(self, threshold, hard=None):
        with torch.no_grad():
            # the mask is $\tau \mapsto \lvert w_{ij} \rvert \geq \tau$
            return torch.ge(torch.log(abs(self.weight) + 1e-20), threshold)

    def _sparsity(self, threshold, hard=None):
        n_relevant = float(self.relevance(threshold).sum().item())
        return [(id(self.weight), self.weight.numel() - n_relevant)]
