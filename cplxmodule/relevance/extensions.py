import torch
import torch.nn.functional as F

from .real import LinearARD as LinearVD
from .complex import CplxLinearARD as CplxLinearVD
from .complex import ExpiFunction


class LinearARD(LinearVD):
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


class CplxLinearVDApprox(CplxLinearVD):
    @property
    def penalty(self):
        r"""Sofplus-sigmoid approximation of the complex KL divergence.
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


class CplxLinearVDBogus(CplxLinearVD):
    @property
    def penalty(self):
        r"""KL-div with bogus forward output, but correct gradient."""
        log_alpha = self.log_alpha
        return - log_alpha - bogus_expi(- torch.exp(- log_alpha))
