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


def cplx_identity(input):
    r"""Return the complex tensor in re-im pair."""
    return input


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


def cplx_modrelu(input, threshold=0.5):
    r"""Compute the modulus relu of the complex tensor in re-im pair."""

    # gain = (1 - \trfac{b}{|z|})_+
    mod = torch.clamp(cplx_modulus(input), min=1e-5)
    gain = torch.relu(mod - threshold) / mod

    re, im = input
    return gain * re, gain * im


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
    return cplx_mul(input, (torch.cos(phi), torch.sin(phi)))


def cplx_linear(input, weight, bias=None):
    r"""Applies a complex linear transformation to the incoming complex
    data: :math:`y = x A^T + b`.
    """
    # W = U + i V,  z = u + i v, c = \Re c + i \Im c
    x_re, x_im = input
    w_re, w_im = weight
    b_re, b_im = bias if bias is not None else (None, None)

    #  W z + c = (U + i V) (u + i v) + \Re c + i \Im c
    #          = (U u + \Re c - V v) + i (V u + \Im c + U v)
    u = F.linear(x_re, w_re, b_re) - F.linear(x_im, w_im, None)
    v = F.linear(x_re, w_im, b_im) + F.linear(x_im, w_re, None)

    return u, v
