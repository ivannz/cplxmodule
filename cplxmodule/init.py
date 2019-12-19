import math

import torch
import numpy as np

from torch.nn import init
from functools import wraps

from .cplx import Cplx
from .layers import CplxParameter


@wraps(init.kaiming_normal_, assigned=("__name__", "__doc__", "__annotations__"))
def cplx_kaiming_normal_(tensor, a=0., mode='fan_in', nonlinearity='leaky_relu'):
    assert isinstance(tensor, (Cplx, CplxParameter))

    a = math.sqrt(1 + 2 * a * a)
    init.kaiming_normal_(tensor.real, a=a, mode=mode, nonlinearity=nonlinearity)
    init.kaiming_normal_(tensor.imag, a=a, mode=mode, nonlinearity=nonlinearity)


@wraps(init.xavier_normal_, assigned=("__name__", "__doc__", "__annotations__"))
def cplx_xavier_normal_(tensor, gain=1.0):
    assert isinstance(tensor, (Cplx, CplxParameter))

    init.xavier_normal_(tensor.real, gain=gain/math.sqrt(2))
    init.xavier_normal_(tensor.imag, gain=gain/math.sqrt(2))
