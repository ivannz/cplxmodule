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

    def extra_repr(self):
        body = "."
        if len(self.a) > 1:
            body += f", {repr(self.a)[1:-1]}"
        elif self.a:
            body += f", {repr(self.a[0])}"

        if self.k:
            vals = map(repr, self.k.values())
            body += ", " + ", ".join(map("=".join, zip(self.k, vals)))

        return f"{self.f.__name__}({body})"


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


class CplxAdaptiveModReLU(CplxToCplx):
    r"""
    Applies soft thresholding to the complex modulus:
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
        return cplx_modrelu(input, self.threshold)

    def __repr__(self):
        body = repr(self.dim)[1:-1] if len(self.dim) > 1 else repr(self.dim[0])
        return f"{self.__class__.__name__}({body})"


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
