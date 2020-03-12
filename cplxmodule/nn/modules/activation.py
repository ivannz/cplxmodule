import torch
import torch.nn

from .base import CplxToCplx, BaseCplxToReal
from ... import cplx


class CplxModReLU(CplxToCplx):
    r"""Applies soft thresholding to the complex modulus:
    $$
        F
        \colon \mathbb{C} \to \mathbb{C}
        \colon z \mapsto (\lvert z \rvert - \tau)_+
                         \tfrac{z}{\lvert z \rvert}
        \,, $$
    with $\tau \in \mathbb{R}$. The if threshold=None then it
    becomes a learnable parameter.
    """
    def __init__(self, threshold=0.5):
        super().__init__()
        if not isinstance(threshold, float):
            threshold = torch.nn.Parameter(torch.rand(1) * 0.25)
        self.threshold = threshold

    def forward(self, input):
        return cplx.modrelu(input, self.threshold)


class CplxAdaptiveModReLU(CplxToCplx):
    r"""Applies soft thresholding to the complex modulus:
    $$
        F
        \colon \mathbb{C}^d \to \mathbb{C}^d
        \colon z \mapsto (\lvert z_j \rvert - \tau_j)_+
                        \tfrac{z_j}{\lvert z_j \rvert}
        \,, $$
    with $\tau_j \in \mathbb{R}$ being the $j$-th learnable threshold. Torch's
    broadcasting rules apply and the passed dimensions must conform with the
    upstream input. `CplxChanneledModReLU(1)` learns a common threshold for all
    features of the $d$-dim complex vector, and `CplxChanneledModReLU(d)` lets
    each dimension have its own threshold.
    """
    def __init__(self, *dim):
        super().__init__()
        self.dim = dim if dim else (1,)
        self.threshold = torch.nn.Parameter(torch.randn(*self.dim) * 0.02)

    def forward(self, input):
        return cplx.modrelu(input, self.threshold)

    def __repr__(self):
        body = repr(self.dim)[1:-1] if len(self.dim) > 1 else repr(self.dim[0])
        return f"{self.__class__.__name__}({body})"


class CplxModulus(BaseCplxToReal):
    def forward(self, input):
        return abs(input)


class CplxAngle(BaseCplxToReal):
    def forward(self, input):
        return input.angle
