import torch


def cplx_modulus(input):
    r"""
    Compute the modulus of the complex tensor in re-im pair:
    $$
        F
        \colon \mathbb{C}^{\ldots \times d}
                \to \mathbb{R}_+^{\ldots \times d}
        \colon u + i v \mapsto \lvert u + i v \rvert
        \,. $$
    """
    input = torch.stack(input, dim=0)
    return torch.norm(input, p=2, dim=0, keepdim=False)


def cplx_angle(input):
    r"""
    Compute the angle of the complex tensor in re-im pair.
    $$
        F
        \colon \mathbb{C}^{\ldots \times d}
                \to \mathbb{R}^{\ldots \times d}
        \colon \underbrace{u + i v}_{r e^{i\phi}} \mapsto \phi
                = \arctan \tfrac{v}{u}
        \,. $$
    """
    re, im = input
    return torch.atan2(im, re)


def cplx_conj(input):
    r"""Compute the conjugate of the complex tensor in re-im pair."""
    re, im = input
    return re, -im


def cplx_mul(input0, input1):
    r"""Multiply the complex tensors in re-im pairs."""
    re0, im0 = input0
    re1, im1 = input1
    return re0 * re1 - im0 * im1, re0 * im1 + im0 * re1


def cplx_apply(input, f, *a, **k):
    r"""Applies the function elementwise to the complex tensor in re-im pair."""
    re, im = input
    return f(re, *a, **k), f(im, *a, **k)


def cplx_exp(input):
    r"""Compute the exponential of the complex tensor in re-im pair."""
    re, im = input
    r, u, v = torch.exp(re), torch.cos(im), torch.sin(im)
    return r * u, r * v


def cplx_log(input):
    r"""Compute the logarithm of the complex tensor in re-im pair."""
    r, theta = cplx_modulus(input), cplx_angle(input)
    return torch.log(r), theta


def cplx_sinh(input):
    r"""Compute the hyperbolic sine of the complex tensor in re-im pair."""
    re, im = input
    return torch.sinh(re) * torch.cos(im), torch.cosh(re) * torch.sin(im)


def cplx_cosh(input):
    r"""Compute the hyperbolic sine of the complex tensor in re-im pair."""
    re, im = input
    return torch.cosh(re) * torch.cos(im), torch.sinh(re) * torch.sin(im)
