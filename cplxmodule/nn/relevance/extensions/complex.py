"""Naming convention is thus:
VDApprox -- approximate KL-div from improper pow2 prior for complex VD
VDScaleFree -- exact KL-div (improper pow1 prior) for complex VD
VDBogus -- improper pow2 prior, but with fake forward outputs for complex VD
"""
import torch
import torch.nn.functional as F

from ..complex import CplxLinearVD
from ..complex import CplxBilinearVD
from ..complex import CplxConv1dVD
from ..complex import CplxConv2dVD
from ..complex import CplxConv3dVD

from ..complex import ExpiFunction, torch_expi


class CplxVDScaleFreeMixin():
    @property
    def penalty(self):
        r"""The Kullback-Leibler divergence between the mean field approximate
        complex variational posterior of the weights and the scale-free
        log-uniform complex prior:
        $$
            KL(\mathcal{CN}(w\mid \theta, \alpha \theta \bar{\theta}, 0) \|
                    \tfrac1{\lvert w \rvert})
                = \mathbb{E}_{\xi \sim \mathcal{CN}(1, \alpha, 0)}
                    \log{\lvert \xi \rvert}
                  + \log \lvert \theta \rvert
                  + C - \log \alpha \lvert \theta \rvert^2
                = - \log \lvert \theta \rvert - \log \alpha
                  + C - \tfrac12 Ei( - \tfrac1{\alpha})
            \,, $$
        where $Ei(x) = \int_{-\infty}^x e^t t^{-1} dt$ is the exponential
        integral. Unlike real-valued variational dropout, this KL divergence
        does not need an approximation, since it can be computed exactly via
        a special function. $Ei(x)$ behaves well on the -ve values, and near
        $0-$. The constant $C$ is fixed to half of Euler's gamma, so that the
        divergence is +ve.
        """
        log_abs_w = torch.log(abs(self.weight) + 1e-12)
        n_log_alpha = 2 * log_abs_w - self.log_sigma2
        ei = torch_expi(- torch.exp(n_log_alpha))
        return log_abs_w - self.log_sigma2 - 0.5 * ei


class CplxLinearVDScaleFree(CplxVDScaleFreeMixin, CplxLinearVD):
    """Complex-valued linear layer with scale-free prior."""
    pass


class CplxBilinearVDScaleFree(CplxVDScaleFreeMixin, CplxBilinearVD):
    """Complex-valued bilinear layer with scale-free prior."""
    pass


class CplxConv1dVDScaleFree(CplxVDScaleFreeMixin, CplxConv1dVD):
    """1D complex-valued convolution layer with scale-free prior."""
    pass


class CplxConv2dVDScaleFree(CplxVDScaleFreeMixin, CplxConv2dVD):
    """2D complex-valued convolution layer with scale-free prior."""
    pass


class CplxConv3dVDScaleFree(CplxVDScaleFreeMixin, CplxConv3dVD):
    """3D complex-valued convolution layer with scale-free prior."""
    pass


class CplxVDApproxMixin():
    @property
    def penalty(self):
        r"""Softplus-sigmoid approximation of the complex KL divergence.
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


class CplxLinearVDApprox(CplxVDApproxMixin, CplxLinearVD):
    """Complex-valued linear layer with approximate
    var-dropout penalty.
    """
    pass


class CplxBilinearVDApprox(CplxVDApproxMixin, CplxBilinearVD):
    """Complex-valued bilinear layer with approximate
    var-dropout penalty.
    """
    pass


class CplxConv1dVDApprox(CplxVDApproxMixin, CplxConv1dVD):
    """1D complex-valued convolution layer with approximate
    var-dropout penalty.
    """
    pass


class CplxConv2dVDApprox(CplxVDApproxMixin, CplxConv2dVD):
    """2D complex-valued convolution layer with approximate
    var-dropout penalty.
    """
    pass


class CplxConv3dVDApprox(CplxVDApproxMixin, CplxConv3dVD):
    """3D complex-valued convolution layer with approximate
    var-dropout penalty.
    """
    pass


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


class CplxVDBogusMixin():
    @property
    def penalty(self):
        r"""KL-div with bogus forward output, but correct gradient."""
        log_alpha = self.log_alpha
        return - log_alpha - bogus_expi(- torch.exp(- log_alpha))


class CplxLinearVDBogus(CplxVDBogusMixin, CplxLinearVD):
    """Complex-valued linear layer with correct var dropout penalty
    gradient, but bogus penalty values.
    """
    pass


class CplxBilinearVDBogus(CplxVDBogusMixin, CplxBilinearVD):
    """Complex-valued bilinear layer with correct var dropout penalty
    gradient, but bogus penalty values.
    """
    pass


class CplxConv1dVDBogus(CplxVDBogusMixin, CplxConv1dVD):
    """1D complex-valued convolution layer with correct var dropout penalty
    gradient, but bogus penalty values.
    """
    pass


class CplxConv2dVDBogus(CplxVDBogusMixin, CplxConv2dVD):
    """2D complex-valued convolution layer with correct var dropout penalty
    gradient, but bogus penalty values.
    """
    pass


class CplxConv3dVDBogus(CplxVDBogusMixin, CplxConv3dVD):
    """3D complex-valued convolution layer with correct var dropout penalty
    gradient, but bogus penalty values.
    """
    pass
