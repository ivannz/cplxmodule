import warnings

import math
import torch
import torch.nn

import torch.nn.functional as F

from ...base import BaseARD
from ....utils.sparsity import SparsityStats


class LinearL0(torch.nn.Linear, BaseARD, SparsityStats):
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
    __sparsity_ignore__ = ("log_alpha",)

    beta, gamma, zeta = 0.66, -0.1, 1.1

    def __init__(self, in_features, out_features, bias=True, group=None):
        super().__init__(in_features, out_features, bias=bias)

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
        # assume everything is unimportant (but do not desaturate too much)
        # $\log\alpha \approx \log p - \log(1-p)$, for dropout rate $p=0.90$
        self.log_alpha.data.uniform_(-2.197, -2.197)

    @property
    def penalty(self):
        r"""Penalty the probability of a nonzero gate $z$:
        $$
            \Pr(\lvert w_i \rvert > 0)
                \leq \Pr(z_i \neq 0)
                = 1 - \sigma\bigl(
                    \log\alpha + \beta \log \tfrac{-\gamma}{\zeta}
                \bigr)
            \,, $$
        where $\sigma(x) = (1+e^{-x})^{-1}$, which also satisfies the
        realtion $1 - \sigma(x) = \sigma(-x)$.
        """
        # compute P(z\neq 0) (minus, due to -ve log-alpha, c.f. eq. (12)).
        shift = - self.beta * math.log(- self.gamma / self.zeta)
        return torch.sigmoid(shift - self.log_alpha)

    def forward(self, input):
        # draw uniform rv for the hard-concrete
        n, m = self.log_alpha.shape
        if self.training:
            if n == 1 or m == 1:
                # this is a relatively "small" batch of unform rv.
                u = torch.rand(*input.shape[:-1], n, m,
                               dtype=input.dtype, device=input.device)
            else:
                # one unform sample for the whole batch! Very high gradient var.
                u = torch.rand_like(self.log_alpha)
                # u = torch.rand_like(*input.shape[:-1], n, m)

            # compute the mask (broadcasting applies!)
            # The distribution of $\log \tfrac{u}{1-u}$ for $u\sim U(0, 1)$ is
            # sigmoid: $\{u \leq \sigma(x)\} = \{\log \tfrac{u}{1-u} \leq x\}$.
            mask = self.gate(torch.log(u) - torch.log(1 - u))
        else:
            # eval : let `u` be its mean
            mask = self.gate(None)
        # end if

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

    def gate(self, logit=None):
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

        On train
        https://github.com/AMLab-Amsterdam/L0_regularization/blob/master/l0_layers.py#L64

        On eval
        https://github.com/AMLab-Amsterdam/L0_regularization/blob/master/l0_layers.py#L103
        """
        if logit is not None:
            s = torch.sigmoid((logit - self.log_alpha) / self.beta)

        else:
            # on inference in eq. (13) beta is fixed at 1.0, but not in their code
            s = torch.sigmoid(- self.log_alpha)
            # It seems that this beta is very important! But why?

        return torch.clamp((self.zeta - self.gamma) * s + self.gamma, 0, 1)

    def relevance(self, *, hard, **kwargs):
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
        with torch.no_grad():
            velue = torch.gt(self.gate(None), 0) if hard else self.gate(None)
            return velue.to(self.weight).expand_as(self.weight)

    def sparsity(self, *, hard, **kwargs):
        n_relevant = float(self.relevance(hard=hard).sum().item())
        return [(id(self.weight), self.weight.numel() - n_relevant)]


class LinearL0ARD(torch.nn.Linear, BaseARD, SparsityStats):
    def __new__(self, in_features, out_features, bias=True, group=None):
        warnings.warn("L0 layer learns probabilities of each parameter being"
                      " nonzero and has little relation to Variational methods."
                      " It was a serious oversight to name it as such. Thus"
                      " starting with version `1.0` importing this layer under"
                      " the incorrect name will be deprecated, and the name"
                      " will be reverted to `LinearL0`.",
                      FutureWarning)

        return LinearL0(in_features, out_features, bias, group)
