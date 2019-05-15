import torch
import torch.nn
import torch.nn.functional as F

from .cplx import cplx_exp, cplx_log, cplx_modrelu

from .layers import CplxToCplx, CplxToReal

from .utils import torch_module


class CplxActivation(CplxToCplx):
    r"""
    Applies the function elementwise passing positional and keyword arguments.
    """
    def __init__(self, f, *a, **k):
        super().__init__()
        self.f, self.a, self.k = f, a, k

    def forward(self, input):
        return input.apply(self.f, *self.a, **self.k)


class CplxModReLU(CplxToCplx):
    r"""
    Applies soft thresholding to the complex modulus:
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
        return cplx_modrelu(input, self.threshold)


class CplxModulus(CplxToReal):
    def forward(self, input):
        return abs(input)


class CplxAngle(CplxToReal):
    def forward(self, input):
        return input.angle


class CplxIdentity(CplxToReal):
    def forward(self, input):
        return input


CplxExp = torch_module(cplx_exp, (CplxToCplx,), "CplxExp")

CplxLog = torch_module(cplx_log, (CplxToCplx,), "CplxLog")
