import math
import torch
import torch.nn

import torch.nn.functional as F

from .base import BaseARD


class LinearL0ARD(torch.nn.Linear, BaseARD):
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

    def __init__(self, in_features, out_features, bias=True,
                 reduction="mean", group=None):
        super().__init__(in_features, out_features, bias=bias)
        self.reduction = reduction

        # weight is out_features, in_features
        if group == "input":
            # thats' what they do in l0_layers.py#L99
            shape = 1, in_features

        elif group == "output":
            # sparsifying outout s is worse than inputs
            shape = out_features, 1

        else:
            # thay do this in l0_layers.py#L107
            shape = out_features, in_features

        self.log_alpha = torch.nn.Parameter(torch.Tensor(*shape))

        self.reset_variational_parameters()

    def reset_variational_parameters(self):
        # assume everything is important (but do not saturate too much)
        # self.log_alpha.data.uniform_(-0.45, -0.45)
        self.log_alpha.data.uniform_(-0.45, -0.45)

    @property
    def penalty(self):
        shift = self.beta * math.log(- self.gamma / self.zeta)

        # compute P(z=0) (no minus, due to -ve log-alpha, c.f. eq. (12)).
        p_zeq0 = torch.sigmoid(self.log_alpha + shift)

        if self.reduction == "mean":
            return 1 - p_zeq0.mean()

        elif self.reduction == "sum":
            return p_zeq0.numel() - p_zeq0.sum()

        return 1 - p_zeq0

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
        mask = self.get_sparsity_mask(threshold)
        n, m = mask.shape
        n_zer = mask.sum().item()
        if n == 1:
            return self.weight.shape[0] * n_zer

        elif m == 1:
            return n_zer * self.weight.shape[1]

        return n_zer

    def forward(self, input):
        # draw uniform rv for the hard-concrete
        n, m = self.log_alpha.shape
        if not self.training:
            # eval : let `u` be its mean
            u = torch.tensor(0.5).to(input)
        elif n == 1 or m == 1:
            # this is a relatively "small" batch of unform rv.
            u = torch.rand(*input.shape[:-1], n, m)
        else:
            # one unform sample for the whole batch! Very high gradient var.
            u = torch.rand_like(self.log_alpha)
            # u = torch.rand_like(*input.shape[:-1], n, m)

        # compute the mask (broadcasting applies!)
        mask = self.gate(torch.log(u) - torch.log(1 - u))
        # The distribution of $\log \tfrac{u}{1-u}$ for $u\sim U(0, 1)$ is
        # sigmoid: $\{u \leq \sigma(x)\} = \{\log \tfrac{u}{1-u} \leq x\}$.

        if n == 1:
            # group = input : "premultiply" `y = W (x \cdot mask) + b`
            output = F.linear(input * mask.squeeze(-2), self.weight)
        elif m == 1:
            # group = output : "postmultiply" `y = (W x) \cdot mask + b`
            output = F.linear(input, self.weight) * mask.squeeze(-1)
        else:
            # group = None : "elementwise" `y = (W \cdot z) x + b`
            output = F.linear(input, self.weight * mask)
            # torch.matmul(input.unsqueeze(-2), mask * self.weight).squeeze(-2)

        if self.bias is not None:
            output += self.bias

        return output

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
