import torch
import torch.nn

from .base import CplxToCplx
from .base import CplxToReal
from .base import RealToCplx


def is_from_cplx(module):
    if isinstance(module, (CplxToCplx, CplxToReal)):
        return True

    if isinstance(module, torch.nn.Sequential):
        return is_from_cplx(module[0])

    return False


def is_to_cplx(module):
    if isinstance(module, (CplxToCplx, RealToCplx)):
        return True

    if isinstance(module, torch.nn.Sequential):
        return is_to_cplx(module[-1])

    return False


def is_cplx_to_cplx(module):
    return is_from_cplx(module) and is_to_cplx(module)
