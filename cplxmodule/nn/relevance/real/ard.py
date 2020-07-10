import torch
import torch.nn.functional as F

from .vd import LinearVD, BilinearVD
from .vd import Conv1dVD, Conv2dVD, Conv3dVD


class RealARDMixin():
    @property
    def penalty(self):
        r"""Penalty from arxiv:1811.00596.

        Naming
        ------
        In fact this is the true ARD setup, and the all layers in other
        files are in fact ordinary Variational Dropout. The difference
        is in the prior.

        Notes
        -----
        Computes the KL-divergence of $q_\theta(W)$ from $p(W; \tau)$ in
        the (true) ARD setup, suggested in arxiv:1811.00596. Uses mean
        field gaussian variational approximation $q_\theta(W)$ against
        the assumed proper (mean field) prior $p(W; \tau)$ gaussian with
        elementwise precision parameter $\tau$ optimised in the ELBO.
        $$
            \mathop{KL}\bigl(q_\theta(W) \| p(W)\bigr)
                = \mathbb{E}_{W \sim q_\theta}
                    \log \tfrac{q_\theta(W)}{p(W)}
                = \frac12 \sum_{ij} \bigl(
                    \tau_{ij} \sigma^2_{ij} + \tau_{ij} \mu_{ij}^2
                    - \log \sigma^2_{ij} - \log \tau_{ij} - 1
                \bigr)
            \,, $$
        at $\tau_{ij} = (\sigma^2_{ij} + \mu_{ij}^2)^{-1}$.
        """

        # `softplus` is $x \mapsto \log(1 + e^x)$
        return 0.5 * F.softplus(- self.log_alpha)


class LinearARD(RealARDMixin, LinearVD):
    """Linear layer with automatic relevance determination."""
    pass


class Conv1dARD(RealARDMixin, Conv1dVD):
    """1D convolution layer with automatic relevance determination."""
    pass


class Conv2dARD(RealARDMixin, Conv2dVD):
    """2D convolution layer with automatic relevance determination."""
    pass


class Conv3dARD(RealARDMixin, Conv3dVD):
    """3D convolution layer with automatic relevance determination."""
    pass


class BilinearARD(RealARDMixin, BilinearVD):
    """Bilinear layer with automatic relevance determination."""
    pass
