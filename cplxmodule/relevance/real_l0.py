import math
import torch
import torch.nn

import torch.nn.functional as F
from torch.nn import Parameter

from .base import BaseARD, SparseModeMixin
from .utils import parameter_to_buffer, buffer_to_parameter


class LinearL0ARD(torch.nn.Linear, BaseARD, SparseModeMixin):
    """L0 regularized linear layer according to [1]_.

    Details
    -------
    This implementation use -ve log-alpha parametrization in order to keep the
    layer's parameters interpretation aligned with the interpretation in
    variational dropout layer of Kingma et al (2015) and Molchanov et al (2017)
    (see also `cplxmodule.relevance.LinearARD`).

    This implementation follows the ICLR2018 closely, specifically it uses the
    equations 10-13, but ignores the caveat just before section 3. Instead, it
    uses the same sample of the gate $z$ for the whole minitbatch, as mentioned
    just before section 4.1, which could lead to much "larger variance in the
    gradients" w.r.t weights (Kingma et al. 2015).

    References
    ----------
    .. [1] Louizos, C., Welling M., Kingma, D. P. (2018). Learning Sparse
           Neural Networks through L0 Regularization. ICLR 2018
           https://arxiv.org/abs/1712.01312.pdf

    .. [2] Gale, T., Elsen, E., Hooker, S. (2019). The State of Sparsity in
           Deep Neural Networks. Arxiv preprint arXiv:1902.09574
           https://arxiv.org/abs/1902.09574.pdf

    .. [3] Maddison, C. J., Mnih, A., Teh, Y. W. (2017). The Concrete
           Distribution: a Continuous Relaxation of discrete Random Variables.
           ICLR 2017
           https://arxiv.org/pdf/1611.00712.pdf
    """
    beta, gamma, zeta = .25, -0.25, 1.25
    # beta, gamma, zeta = 0.66, -0.1, 1.1

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)

        self.log_alpha = Parameter(torch.Tensor(*self.weight.shape))

        self.reset_variational_parameters()

    def reset_variational_parameters(self):
        # assume everything is important (but do not saturate too much)
        # self.log_alpha.data.uniform_(-0.45, -0.45)
        self.log_alpha.data.uniform_(-0.45, -0.45)

    @property
    def penalty(self):
        shift = self.beta * math.log(- self.gamma / self.zeta)

        # the penalty has this expression, due to -ve log-alpha c.f. eq. (12)
        return 1 - torch.sigmoid(self.log_alpha + shift).mean()

    def get_sparsity_mask(self, threshold):
        r"""Get the dropout mask based on the confidence level $\tau \in (0, 1)$:
        $$
            \Pr(\lvert w_i \rvert > 0)
                \leq \Pr(z_i \neq 0)
                = 1 - \sigma\bigl(
                    \log\alpha + \beta \log \tfrac{-\gamma}{\zeta}
                \bigr)
                \leq \tau
            \,. $$
        For $\tau=0.25$ and $\beta=0.66$ we have `threshold=2.96`.
        """
        # threshold = math.log(1 - tau) - math.log(tau)
        threshold -= self.beta * math.log(- self.gamma / self.zeta)
        with torch.no_grad():
            return torch.ge(self.log_alpha, threshold)

    def num_zeros(self, threshold=1.0):
        return self.get_sparsity_mask(threshold).sum().item()

    def forward(self, input):
        if self.is_sparse:
            return self.forward_sparse(input)

        if self.training:
            # a single mask sample for the whole batch!
            u = torch.rand_like(self.log_alpha)

            # The distribution of $\log \tfrac{u}{1-u}$ for $u\sim U(0, 1)$ is
            # sigmoid: $\{u \leq \sigma(x)\} = \{\log \tfrac{u}{1-u} \leq x\}$.
            logit = torch.log(u) - torch.log(1 - u)
        else:
            logit = 0.

        return F.linear(input, self.weight * self.gate(logit), self.bias)

    def gate(self, logit):
        r"""Implements the binary concrete hard-sigmoid transformation:
        $$
            F
            \colon \mathbb{R} \to \mathbb{R}
            \colon x \to g \bigl(
                    \ell_{\zeta, \gamma}(
                        \sigma_{\beta^{-1}}(x - \log \alpha)
                    )\bigr)
            \,, $$
        where $g(x) = \min\{1, \max\{0, x\}\}$ is the hard-sigmoid, $\gamma <
        0 < \zeta$ are the stretch parameters, $\beta$ is the temperature and
        $\sigma(z) = (1+e^{-z})^{-1}$.
        """
        # on inference in eq. (13) beta is fixed at 1.0, but not in their code
        s = torch.sigmoid((logit - self.log_alpha) / self.beta)
        return torch.clamp((self.zeta - self.gamma) * s + self.gamma, 0, 1)

    def sparsify(self, tau, mode="dense"):
        # None -> sparse/dense : mutate par-to-buf
        if not self.is_sparse and mode is not None:
            # switch off l0 dropout and create runtime sparse data
            parameter_to_buffer(self, "log_alpha")

        # sparse/dense -> None : mutate buf-to-par
        elif self.is_sparse and mode is None:
            # reinstate l0 dropout mode and discard runtime data on mode change
            buffer_to_parameter(self, "log_alpha")

        return super().sparsify(tau, mode)
