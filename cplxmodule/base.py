import torch
import torch.nn

from numbers import Complex
from collections import namedtuple


BaseCplx = namedtuple("BaseCplx", ["real", "imag"])


class Cplx(BaseCplx, Complex):
    """A very limited container for complex tensors."""
    def __new__(cls, real=None, imag=None):
        if isinstance(real, cls):
            return real
        return super().__new__(cls, real, imag)

    @property
    def real(self):
        return super().__getitem__(0)

    @property
    def imag(self):
        return super().__getitem__(1)

    def __getitem__(self, key):
        if self:
            return Cplx(self.real[key], self.imag[key])
        return self

    @property
    def shape(self):
        return self.real.shape

    def __len__(self):
        return self.shape[0]

    def __bool__(self):
        return self.real is not None or self.imag is not None

    @property
    def conj(self):
        return Cplx(self.real, -self.imag)

    def conjugate(self):
        r"""Get the conjugate of the complex tensor tensor."""
        return self.conj

    def __pos__(self):
        return self

    def __neg__(self):
        return Cplx(-self.real, -self.imag)

    def __add__(u, v):
        if not isinstance(v, Cplx):
            return Cplx(u.real + v, u.imag + v)

        return Cplx(u.real + v.real, u.imag + v.imag)

    def __sub__(u, v):
        if not isinstance(v, Cplx):
            return Cplx(u.real - v, u.imag - v)

        return Cplx(u.real - v.real, u.imag - v.imag)

    def __mul__(u, v):
        if not isinstance(v, Cplx):
            return Cplx(u.real * v, u.imag * v)

        re = u.real * v.real - u.imag * v.imag
        im = u.imag * v.real + u.real * v.imag
        return Cplx(re, im)

    def __truediv__(u, v):
        if not isinstance(v, Cplx):
            return Cplx(u.real / v, u.imag / v)

        scale = abs(v)
        return (u / scale) * (v.conj / scale)

    __radd__ = __add__
    __rsub__ = __sub__
    # __rmul__ = __mul__

    def __rmul__(u, v):
        return Cplx(v, 0.) * u

    def __rtruediv__(u, v):
        return Cplx(v, 0.) / u

    def __abs__(self):
        r"""
        Compute the modulus of the complex tensor in re-im pair:
        $$
            F
            \colon \mathbb{C}^{\ldots \times d}
                    \to \mathbb{R}_+^{\ldots \times d}
            \colon u + i v \mapsto \lvert u + i v \rvert
            \,. $$
        """
        input = torch.stack([self.real, self.imag], dim=0)
        return torch.norm(input, p=2, dim=0, keepdim=False)

    @property
    def angle(self):
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
        return torch.atan2(self.imag, self.real)

    def apply(self, f, *a, **k):
        r"""Applies the function elementwise to the complex tensor in re-im pair."""
        return Cplx(f(self.real, *a, **k), f(self.imag, *a, **k))


def real_to_cplx(input, copy=True):
    """Map real tensor input `... x [D * 2]` to a pair (re, im) with dim `... x D`."""
    *head, n_features = input.shape
    assert (n_features & 1) == 0

    if copy:
        return Cplx(input[..., 0::2].clone(), input[..., 1::2].clone())

    input = input.reshape(*head, -1, 2)
    return Cplx(input[..., 0], input[..., 1])


def cplx_to_real(input, flatten=True):
    """Interleave the complex re-im pair into a real tensor."""
    # re, im = input
    input = torch.stack([input.real, input.imag], dim=-1)
    if flatten:
        return input.flatten(-2)
    return input


class RealToCplx(torch.nn.Module):
    r"""
    A layer that splits an interleaved real tensor with even number in the last
    dim to a complex tensor represented by a pair of real and imaginary tensors
    of the same size. Preserves the all dimensions but the last, which is halved.
    $$
        F
        \colon \mathbb{R}^{\ldots \times [d\times 2]}
                \to \mathbb{C}^{\ldots \times d}
        \colon x \mapsto \bigr(
            x_{2k} + i x_{2k+1}
        \bigl)_{k=0}^{d-1}
        \,. $$
    """
    def forward(self, input):
        return real_to_cplx(input)


class CplxToReal(torch.nn.Module):
    r"""
    A layer that interleaves the complex tensor represented by a pair of real
    and imaginary tensors into a larger real tensor along the last dimension.
    $$
        F
        \colon \mathbb{C}^{\ldots \times d}
                \to \mathbb{R}^{\ldots \times [d \times 2]}
        \colon u + i v \mapsto \bigl(u_\omega, v_\omega\bigr)_{\omega}
        \,. $$
    """
    def __init__(self, flatten=True):
        super().__init__()
        self.flatten = flatten

    def forward(self, input):
        return cplx_to_real(input, self.flatten)


class CplxToCplx(torch.nn.Module):
    pass
