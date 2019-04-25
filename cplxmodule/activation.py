import torch
import torch.nn

import torch.nn.functional as F

from .base import CplxToCplx, CplxToReal

from .cplx import cplx_modulus, cplx_angle
from .cplx import cplx_exp, cplx_log

from .cplx import cplx_apply, cplx_modrelu
from .cplx import cplx_identity

from .utils import torch_module


class CplxActivation(CplxToCplx):
    r"""
    Applies the function elementwise passing positional and keyword arguments.
    """
    def __init__(self, f, *a, **k):
        super().__init__()
        self.f, self.a, self.k = f, a, k

    def forward(self, input):
        return cplx_apply(input, self.f, *self.a, **self.k)


class CplxModReLU(CplxToCplx):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, input):
        return cplx_modrelu(input, self.threshold)


CplxModulus = torch_module(cplx_modulus, (CplxToReal,), "CplxModulus")

CplxAngle = torch_module(cplx_angle, (CplxToReal,), "CplxAngle")

CplxIdentity = torch_module(cplx_identity, (CplxToCplx,), "CplxIdentity")

CplxExp = torch_module(cplx_exp, (CplxToCplx,), "CplxExp")

CplxLog = torch_module(cplx_log, (CplxToCplx,), "CplxLog")
