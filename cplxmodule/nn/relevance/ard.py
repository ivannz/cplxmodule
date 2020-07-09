import torch
import torch.nn.functional as F

from .real import LinearVD
from .real import Conv1dVD
from .real import Conv2dVD
from .real import Conv3dVD
from .real import BilinearVD

from .complex import CplxLinearVD
from .complex import CplxBilinearVD
from .complex import CplxConv1dVD
from .complex import CplxConv2dVD
from .complex import CplxConv3dVD


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
