import torch

from ...base import BaseARD
from ....utils.sparsity import SparsityStats


class LinearLASSO(torch.nn.Linear, BaseARD, SparsityStats):
    @property
    def penalty(self):
        return abs(self.weight)

    def relevance(self, *, threshold, **kwargs):
        with torch.no_grad():
            # the mask is $\tau \mapsto \lvert w_{ij} \rvert \geq \tau$
            return torch.ge(torch.log(abs(self.weight) + 1e-20), threshold)

    def sparsity(self, *, threshold, **kwargs):
        n_relevant = float(self.relevance(threshold=threshold).sum().item())
        return [(id(self.weight), self.weight.numel() - n_relevant)]
