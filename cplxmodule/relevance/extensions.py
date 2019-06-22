import torch
import torch.nn.functional as F

from .complex import ExpiFunction, CplxLinearARD


class CplxLinearARDApprox(CplxLinearARD):
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


class CplxLinearARDBogus(CplxLinearARD):
    @property
    def penalty(self):
        r"""KL-div with bogus forward output, but correct gradient."""
        log_alpha = self.log_alpha
        return - log_alpha - bogus_expi(- torch.exp(- log_alpha))
