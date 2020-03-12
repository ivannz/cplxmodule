import math

import torch
import numpy as np

from torch.nn import init
from functools import wraps

from ..cplx import Cplx


def get_fans(cplxtensor):
    """Almost verbatim copy of `init._calculate_fan_in_and_fan_out`"""
    ndim = cplxtensor.dim()
    if ndim < 2:
        raise ValueError("Fan in and fan out can not be computed "
                         "for tensor with fewer than 2 dimensions.")

    n_fmaps_output, n_fmaps_input, *rest = cplxtensor.shape
    if ndim == 2:
        fan_in, fan_out = n_fmaps_output, n_fmaps_input

    else:
        receptive_field_size = np.prod((1, *rest))
        fan_in = n_fmaps_input * receptive_field_size
        fan_out = n_fmaps_output * receptive_field_size

    return fan_in, fan_out


@wraps(init.kaiming_normal_, assigned=("__name__", "__doc__", "__annotations__"))
def cplx_kaiming_normal_(tensor, a=0., mode='fan_in', nonlinearity='leaky_relu'):
    assert isinstance(tensor, Cplx)

    a = math.sqrt(1 + 2 * a * a)
    init.kaiming_normal_(tensor.real, a=a, mode=mode, nonlinearity=nonlinearity)
    init.kaiming_normal_(tensor.imag, a=a, mode=mode, nonlinearity=nonlinearity)


@wraps(init.xavier_normal_, assigned=("__name__", "__doc__", "__annotations__"))
def cplx_xavier_normal_(tensor, gain=1.0):
    assert isinstance(tensor, Cplx)

    init.xavier_normal_(tensor.real, gain=gain/math.sqrt(2))
    init.xavier_normal_(tensor.imag, gain=gain/math.sqrt(2))


@wraps(init.kaiming_uniform_, assigned=("__name__", "__doc__", "__annotations__"))
def cplx_kaiming_uniform_(tensor, a=0., mode='fan_in', nonlinearity='leaky_relu'):
    assert isinstance(tensor, Cplx)

    a = math.sqrt(1 + 2 * a * a)
    init.kaiming_uniform_(tensor.real, a=a, mode=mode, nonlinearity=nonlinearity)
    init.kaiming_uniform_(tensor.imag, a=a, mode=mode, nonlinearity=nonlinearity)


@wraps(init.xavier_uniform_, assigned=("__name__", "__doc__", "__annotations__"))
def cplx_xavier_uniform_(tensor, gain=1.0):
    assert isinstance(tensor, Cplx)

    init.xavier_uniform_(tensor.real, gain=gain/math.sqrt(2))
    init.xavier_uniform_(tensor.imag, gain=gain/math.sqrt(2))


def cplx_trabelsi_standard_(cplx, kind="glorot"):
    """Standard complex initialization proposed in Trabelsi et al. (2018)."""
    kind = kind.lower()
    assert kind in ("glorot", "xavier", "kaiming", "he")

    fan_in, fan_out = get_fans(cplx)
    if kind == "glorot" or kind == "xavier":
        scale = 1 / math.sqrt(fan_in + fan_out)
    else:
        scale = 1 / math.sqrt(fan_in)

    # Rayleigh(\sigma / \sqrt2) x uniform[-\pi, +\pi] on p. 7
    rho = np.random.rayleigh(scale, size=cplx.shape)
    theta = np.random.uniform(-np.pi, +np.pi, size=cplx.shape)

    # eq. (8) on p. 6
    with torch.no_grad():
        cplx.real.copy_(torch.from_numpy(np.cos(theta) * rho))
        cplx.imag.copy_(torch.from_numpy(np.sin(theta) * rho))

    return cplx


def cplx_trabelsi_independent_(cplx, kind="glorot"):
    """Orthogonal complex initialization proposed in Trabelsi et al. (2018)."""
    kind = kind.lower()
    assert kind in ("glorot", "xavier", "kaiming", "he")

    ndim = cplx.dim()
    if ndim == 2:
        shape = cplx.shape

    else:
        shape = np.prod(cplx.shape[:2]), np.prod(cplx.shape[2:])

    # generate a semi- unitary (orthogonal) matrix from a random matrix
    # M = U V is semi-unitary: V^H U^H U V = I_k
    Z = np.random.randn(*shape) + 1j * np.random.randn(*shape)
    u, _, vh = np.linalg.svd(Z, compute_uv=True, full_matrices=False)
    k = min(*shape)
    M = np.dot(u[:, :k], vh[:, :k].conjugate().T)

    fan_in, fan_out = get_fans(cplx)
    if kind == "glorot" or kind == "xavier":
        scale = 1 / math.sqrt(fan_in + fan_out)
    else:
        scale = 1 / math.sqrt(fan_in)

    M /= M.std() / scale
    M = M.reshape(cplx.shape)
    with torch.no_grad():
        cplx.real.copy_(torch.from_numpy(M.real))
        cplx.imag.copy_(torch.from_numpy(M.imag))

    return cplx


def cplx_uniform_independent_(cplx, a=0., b=1.):
    torch.nn.init.uniform_(cplx.real, a, b)
    torch.nn.init.uniform_(cplx.imag, a, b)

    return cplx
