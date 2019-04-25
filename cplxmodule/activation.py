import torch
import torch.nn

import torch.nn.functional as F

from .base import CplxToCplx, CplxToReal

from .cplx import cplx_modulus, cplx_angle
from .cplx import cplx_exp, cplx_log

from .cplx import cplx_apply, cplx_modrelu


class CplxActivation(CplxToCplx):
    r"""
    Applies the function elementwise passing positional and keyword arguments.
    """
    def __init__(self, f, *a, **k):
        super().__init__()
        self.f, self.a, self.k = f, a, k

    def forward(self, input):
        return cplx_apply(input, self.f, *self.a, **self.k)


class CplxModulus(CplxToReal):
    def forward(self, input):
        return cplx_modulus(input)


class CplxAngle(CplxToReal):
    def forward(self, input):
        return cplx_angle(input)


class CplxIdentity(CplxToCplx):
    def forward(self, input):
        return input


class CplxModReLU(CplxToCplx):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, input):
        return cplx_modrelu(input, self.threshold)


class CplxExp(CplxToCplx):
    def forward(self, input):
        return cplx_exp(input)


class CplxLog(CplxToCplx):
    def forward(self, input):
        return cplx_log(input)
