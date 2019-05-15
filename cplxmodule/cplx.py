import torch

import torch.nn.functional as F

from .base import Cplx


def cplx_modulus(input):
    return abs(input)


def cplx_angle(input):
    return input.angle


def cplx_apply(input, f, *a, **k):
    return input.apply(f, *a, **k)


def cplx_conj(input):
    return input.conj


def cplx_mul(input0, input1):
    return input0 * input1


def cplx_add(input0, input1):
    r"""Multiply the complex tensors in re-im pairs."""
    return input0 + input1


def cplx_identity(input):
    r"""Return the complex tensor in re-im pair."""
    return input


def cplx_exp(input):
    r"""Compute the exponential of the complex tensor in re-im pair."""
    scale = torch.exp(input.real)
    return Cplx(scale * torch.cos(input.imag),
                scale * torch.sin(input.imag))


def cplx_log(input):
    r"""Compute the logarithm of the complex tensor in re-im pair."""
    r, theta = cplx_modulus(input), cplx_angle(input)
    return Cplx(torch.log(r), theta)


def cplx_sinh(input):
    r"""Compute the hyperbolic sine of the complex tensor in re-im pair."""
    return Cplx(torch.sinh(input.real) * torch.cos(input.imag),
                torch.cosh(input.real) * torch.sin(input.imag))


def cplx_cosh(input):
    r"""Compute the hyperbolic sine of the complex tensor in re-im pair."""
    return Cplx(torch.cosh(input.real) * torch.cos(input.imag),
                torch.sinh(input.real) * torch.sin(input.imag))


def cplx_modrelu(input, threshold=0.5):
    r"""Compute the modulus relu of the complex tensor in re-im pair."""
    # scale = (1 - \trfac{b}{|z|})_+
    mod = torch.clamp(cplx_modulus(input), min=1e-5)
    scale = torch.relu(mod - threshold) / mod
    return Cplx(scale * input.real, scale * input.imag)


def cplx_phaseshift(input, phi=0.0):
    r"""
    Apply phase shift to the complex tensor in re-im pair.
    $$
        F
        \colon \mathbb{C} \to \mathbb{C}
        \colon z \mapsto z e^{i\phi}
                = u cos \phi - v sin \phi
                    + i (u sin \phi + v cos \phi)
        \,, $$
    with $\phi$ in radians.
    """
    return cplx_mul(input, Cplx(torch.cos(phi), torch.sin(phi)))


def cplx_linear(input, weight, bias=None):
    r"""Applies a complex linear transformation to the incoming complex
    data: :math:`y = x A^T + b`.
    """
    # W = U + i V,  z = u + i v, c = \Re c + i \Im c
    #  W z + c = (U + i V) (u + i v) + \Re c + i \Im c
    #          = (U u + \Re c - V v) + i (V u + \Im c + U v)
    re = F.linear(input.real, weight.real, None) \
        - F.linear(input.imag, weight.imag, None)
    im = F.linear(input.real, weight.imag, None) \
        + F.linear(input.imag, weight.real, None)

    output = Cplx(re, im)
    if isinstance(bias, Cplx):
        output += bias

    return output


def cplx_conv1d(input, weight, bias=None, stride=1,
                padding=0, dilation=1, groups=1):
    r"""Applies a complex 1d convolution to the incoming complex
    tensor `B x c_in x L`: :math:`y = x \star W + b`.
    """
    # W = U + i V,  z = u + i v, c = \Re c + i \Im c
    bias = bias if isinstance(bias, Cplx) else Cplx(None, None)

    real = F.conv1d(input.real, weight.real, bias.real,
                    stride, padding, dilation, groups) \
        - F.conv1d(input.imag, weight.imag, None,
                   stride, padding, dilation, groups)
    imag = F.conv1d(input.real, weight.imag, bias.imag,
                    stride, padding, dilation, groups) \
        + F.conv1d(input.imag, weight.real, None,
                   stride, padding, dilation, groups)

    return Cplx(real, imag)
