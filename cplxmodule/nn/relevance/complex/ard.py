import torch
import torch.nn.functional as F

from .vd import CplxLinearVD, CplxBilinearVD
from .vd import CplxConv1dVD, CplxConv2dVD, CplxConv3dVD


class CplxARDMixin():
    @property
    def penalty(self):
        r"""Empricial Bayes penalty for complex layer with complex gaussian vi.

        Naming
        ------
        See `LinearARD`

        Notes
        -----
        Computes the KL-divergence of $q_\theta(W)$ from the emprical Bayes
        prior $\pi(W; \tau)$ given by a circular symmetric complex gaussian
        with optimal precision parameter $\tau$.
        $$
            \mathop{KL}\bigl(q_\theta(W) \| \pi(W; \tau_{ij} \bigr)
                = \mathbb{E}_{W \sim q_\theta}
                    \log \tfrac{q_\theta(W)}{\pi(W; \tau_{ij})}
                = \sum_{ij} \bigl(
                    \tau_{ij} \sigma^2_{ij}
                    + \tau_{ij} \lvert \mu_{ij} \rvert^2
                    - \log \sigma^2_{ij} \tau_{ij}
                    - 1
                \bigr)
            \,, $$
        at $\tau_{ij} = (\sigma^2_{ij} + \mu_{ij}^2)^{-1}$.

        Note the absence of $\tfrac12$!
        """

        # `softplus` is $x \mapsto \log(1 + e^x)$
        return F.softplus(- self.log_alpha)


class CplxLinearARD(CplxARDMixin, CplxLinearVD):
    """Complex-valued linear layer with automatic relevance
    determination.
    """
    pass


class CplxBilinearARD(CplxARDMixin, CplxBilinearVD):
    """Complex-valued bilinear layer with automatic relevance
    determination.
    """
    pass


class CplxConv1dARD(CplxARDMixin, CplxConv1dVD):
    """1D complex-valued convolution layer with automatic relevance
    determination.
    """
    pass


class CplxConv2dARD(CplxARDMixin, CplxConv2dVD):
    """2D complex-valued convolution layer with automatic relevance
    determination.
    """
    pass


class CplxConv3dARD(CplxARDMixin, CplxConv3dVD):
    """3D complex-valued convolution layer with automatic relevance
    determination.
    """
    pass
