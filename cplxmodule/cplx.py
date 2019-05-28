import torch
import torch.nn.functional as F

from .utils import complex_view


class Cplx(tuple):
    """A type partially implementing complex valued tensors in torch."""
    def __new__(cls, real, imag=None):
        if isinstance(real, cls):
            return real

        if isinstance(real, complex):
            # Silently ignore imag if real is complex
            real, imag = torch.tensor(real.real), torch.tensor(real.imag)

        elif isinstance(real, float):
            if imag is None:
                imag = 0.0

            elif not isinstance(imag, float):
                raise TypeError("""Imaginary part must be float.""")

            real, imag = torch.tensor(real), torch.tensor(imag)

        elif not isinstance(real, torch.Tensor):
            raise TypeError("""Real part must be torch.Tensor.""")

        if imag is None:
            imag = torch.zeros_like(real)

        elif not isinstance(imag, torch.Tensor):
            raise TypeError("""Imaginary part must be torch.Tensor.""")

        if real.shape != imag.shape:
            raise ValueError("""Real and imaginary parts have """
                             """mistmatching shape.""")

        return super().__new__(cls, (real, imag))

    @property
    def real(self):
        return super().__getitem__(0)

    @property
    def imag(self):
        return super().__getitem__(1)

    def __getitem__(self, key):
        return Cplx(self.real[key], self.imag[key])

    def __iter__(self):
        return map(Cplx, self.real, self.imag)

    def __contains__(self, value):
        return value in self.real or value in self.imag

    def __reversed__(self):
        return Cplx(reversed(self.real), reversed(self.imag))

    @property
    def conj(self):
        return Cplx(self.real, -self.imag)

    def conjugate(self):
        return self.conj

    def __pos__(self):
        return self

    def __neg__(self):
        return Cplx(-self.real, -self.imag)

    def __add__(u, v):
        if not isinstance(v, (Cplx, complex)):
            return Cplx(u.real + v, u.imag)
        return Cplx(u.real + v.real, u.imag + v.imag)

    __radd__ = __add__

    def __sub__(u, v):
        if not isinstance(v, (Cplx, complex)):
            return Cplx(u.real - v, u.imag)
        return Cplx(u.real - v.real, u.imag - v.imag)

    def __rsub__(u, v):
        return -u + v

    def __mul__(u, v):
        if not isinstance(v, (Cplx, complex)):
            return Cplx(u.real * v, u.imag * v)

        # (a + ib) (u + iv) = au - bv + i(av + bu)
        # (a+u)(b+v) = ab + uv + (av + ub)
        # (a-v)(b+u) = ab - uv + (au - vb)
        return Cplx(u.real * v.real - u.imag * v.imag,
                    u.imag * v.real + u.real * v.imag)

    __rmul__ = __mul__

    def __truediv__(u, v):
        if not isinstance(v, (Cplx, complex)):
            return Cplx(u.real / v, u.imag / v)

        denom = v.real * v.real + v.imag * v.imag
        return u * (v.conjugate() / denom)

    def __rtruediv__(u, v):
        # v / u and v is not Cplx
        denom = u.real * u.real + u.imag * u.imag
        return (u.conjugate() / denom) * v

    def __matmul__(u, v):
        if not isinstance(v, Cplx):
            return Cplx(torch.matmul(u.real, v), torch.matmul(u.imag, v))

        re = torch.matmul(u.real, v.real) - torch.matmul(u.imag, v.imag)
        im = torch.matmul(u.imag, v.real) + torch.matmul(u.real, v.imag)
        return Cplx(re, im)

    def __abs__(self):
        r"""Compute the complex modulus:
        $$
            \mathbb{C}^{\ldots \times d}
                \to \mathbb{R}_+^{\ldots \times d}
            \colon u + i v \mapsto \lvert u + i v \rvert
            \,. $$
        """
        input = torch.stack([self.real, self.imag], dim=0)
        return torch.norm(input, p=2, dim=0, keepdim=False)

    @property
    def angle(self):
        r"""Compute the complex argument:
        $$
            \mathbb{C}^{\ldots \times d}
                \to \mathbb{R}^{\ldots \times d}
            \colon \underbrace{u + i v}_{r e^{i\phi}} \mapsto \phi
                    = \arctan \tfrac{v}{u}
            \,. $$
        """
        return torch.atan2(self.imag, self.real)

    def apply(self, f, *a, **k):
        r"""Applies the function to real and imaginary parts."""
        return Cplx(f(self.real, *a, **k), f(self.imag, *a, **k))

    @property
    def shape(self, *shape):
        r"""Returns the shape of the complex tensor."""
        return self.real.shape

    def __len__(self):
        return self.shape[0]

    def t(self):
        return Cplx(self.real.t(), self.imag.t())

    def h(self):
        return self.conj.t()  # Cplx(self.real.t(), -self.imag.t())

    def reshape(self, *shape):
        r"""Reshape the complex tensor (both real and imaginary parts)."""
        shape = shape[0] if shape and isinstance(shape[0], tuple) else shape
        return Cplx(self.real.reshape(*shape), self.imag.reshape(*shape))

    def item(self):
        return float(self.real) + 1j * float(self.imag)

    @staticmethod
    def from_numpy(numpy):
        re = torch.from_numpy(numpy.real)
        im = torch.from_numpy(numpy.imag)
        return Cplx(re, im)

    def numpy(self):
        return self.real.numpy() + 1j * self.imag.numpy()

    def __repr__(self):
        return f"{self.__class__.__name__}(\n" \
               f"  real={self.real},\n  imag={self.imag}\n)"

    def __bool__(self):
        return self.real is not None or self.imag is not None

    def detach(self):
        return Cplx(self.real.detach(), self.imag.detach())

    def requires_grad_(self, requires_grad=True):
        return Cplx(self.real.requires_grad_(requires_grad),
                    self.imag.requires_grad_(requires_grad))

    @property
    def grad(self):
        """EXPERIMETNAL"""
        re, im = self.real.grad, self.imag.grad
        return None if re is None or im is None else Cplx(re, im)

    def cuda(self, device=None, non_blocking=False):
        re = self.real.cuda(device=device, non_blocking=non_blocking)
        im = self.imag.cuda(device=device, non_blocking=non_blocking)
        return Cplx(re, im)

    def cpu(self):
        return Cplx(self.real.cpu(), self.imag.cpu())

    def to(self, *args, **kwargs):
        return Cplx(self.real.to(*args, **kwargs),
                    self.imag.to(*args, **kwargs))


def real_to_cplx(input, copy=True, dim=-1):
    """Map real tensor input `... x [D * 2]` to a pair (re, im) with dim `... x D`."""
    real, imag = complex_view(input, dim, squeeze=False)
    return Cplx(real.clone(), imag.clone()) if copy else Cplx(real, imag)


def cplx_to_real(input, flatten=True):
    """Interleave the complex re-im pair into a real tensor."""
    # re, im = input
    input = torch.stack([input.real, input.imag], dim=-1)
    return input.flatten(-2) if flatten else input


def cplx_exp(input):
    r"""Compute the exponential of the complex tensor in re-im pair."""
    scale = torch.exp(input.real)
    return Cplx(scale * torch.cos(input.imag),
                scale * torch.sin(input.imag))


def cplx_log(input):
    r"""Compute the logarithm of the complex tensor in re-im pair."""
    return Cplx(torch.log(abs(input)), input.angle)


def cplx_sin(input):
    r"""Compute the sine of the complex tensor in re-im pair."""
    return Cplx(torch.sin(input.real) * torch.cosh(input.imag),
                torch.cos(input.real) * torch.sinh(input.imag))


def cplx_cos(input):
    r"""Compute the cosine of the complex tensor in re-im pair."""
    return Cplx(torch.cos(input.real) * torch.cosh(input.imag),
                - torch.sin(input.real) * torch.sinh(input.imag))


def cplx_tan(input):
    r"""Compute the tangent of the complex tensor in re-im pair."""
    return cplx_sin(input) / cplx_cos(input)


def cplx_sinh(input):
    r"""Compute the hyperbolic sine of the complex tensor in re-im pair.

    sinh(z) = - j sin(j z)
    """
    return Cplx(torch.sinh(input.real) * torch.cos(input.imag),
                torch.cosh(input.real) * torch.sin(input.imag))


def cplx_cosh(input):
    r"""Compute the hyperbolic cosine of the complex tensor in re-im pair.

    cosh(z) = cos(j z)
    """
    return Cplx(torch.cosh(input.real) * torch.cos(input.imag),
                torch.sinh(input.real) * torch.sin(input.imag))


def cplx_tanh(input):
    r"""Compute the hyperbolic tangent of the complex tensor in re-im pair.

    tanh(z) = j tan(z)
    """
    return cplx_sinh(input) / cplx_cosh(input)


def cplx_modrelu(input, threshold=0.5):
    r"""Compute the modulus relu of the complex tensor in re-im pair."""
    # scale = (1 - \trfac{b}{|z|})_+
    modulus = torch.clamp(abs(input), min=1e-5)
    return input * torch.relu(1. - threshold / modulus)


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
    return input * Cplx(torch.cos(phi), torch.sin(phi))


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
    if bias is not None:
        output += bias

    return output


def cplx_conv1d(input, weight, bias=None, stride=1, padding=0,
                dilation=1, groups=1, padding_mode="zeros"):
    r"""Applies a complex 1d convolution to the incoming complex
    tensor `B x c_in x L`: :math:`y = x \star W + b`.
    """
    if padding_mode == 'circular':
        expanded_padding = ((padding + 1) // 2, padding // 2)
        input = input.apply(F.pad, expanded_padding, mode="circular")
        padding = 0

    re = F.conv1d(input.real, weight.real, None,
                  stride, padding, dilation, groups) \
        - F.conv1d(input.imag, weight.imag, None,
                   stride, padding, dilation, groups)
    im = F.conv1d(input.real, weight.imag, None,
                  stride, padding, dilation, groups) \
        + F.conv1d(input.imag, weight.real, None,
                   stride, padding, dilation, groups)

    output = Cplx(re, im)
    if bias is not None:
        output += bias.apply(torch.unsqueeze, -1)

    return output
