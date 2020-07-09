import warnings

import torch
import torch.nn.functional as F

from math import sqrt
from .utils import complex_view, fix_dim


class Cplx(object):
    r"""A type partially implementing complex valued tensors in torch.

    Details
    -------
    Creates a complex tensor object from the real and imaginary torch tensors,
    or pythonic floats and complex numbers. This is a container-wrapper which
    does not copy the supplied torch tensors on creation.
    """
    __slots__ = ("__real", "__imag")

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

        self = super().__new__(cls)
        self.__real, self.__imag = real, imag
        return self

    @property
    def real(self):
        r"""Real part of the complex tensor."""
        return self.__real

    @property
    def imag(self):
        r"""Imaginary part of the complex tensor."""
        return self.__imag

    def __getitem__(self, key):
        r"""Index the complex tensor."""
        return type(self)(self.__real[key], self.__imag[key])

    def __setitem__(self, key, value):
        r"""Alter the complex tensor at index inplace."""
        if not isinstance(value, (Cplx, complex)):
            self.__real[key], self.__imag[key] = value, value
        else:
            self.__real[key], self.__imag[key] = value.real, value.imag

    def __iter__(self):
        r"""Iterate over the zero-th dimension of the complex tensor."""
        return map(type(self), self.__real, self.__imag)

    def __reversed__(self):
        r"""Reverse the complex tensor along the zero-th dimension."""
        return type(self)(reversed(self.__real), reversed(self.__imag))

    def clone(self):
        r"""Clone a complex tensor."""
        return type(self)(self.__real.clone(), self.__imag.clone())

    @property
    def conj(self):
        r"""The complex conjugate of the complex tensor."""
        return type(self)(self.__real, -self.__imag)

    def conjugate(self):
        r"""The complex conjugate of the complex tensor."""
        return self.conj

    def __pos__(self):
        r"""Return the complex tensor as is."""
        return self

    def __neg__(self):
        r"""Flip the sign of the complex tensor."""
        return type(self)(-self.__real, -self.__imag)

    def __add__(u, v):
        r"""Sum of complex tensors."""
        if not isinstance(v, (Cplx, complex)):
            return type(u)(u.__real + v, u.__imag)
        return type(u)(u.__real + v.real, u.__imag + v.imag)

    __radd__ = __add__
    __iadd__ = __add__

    def __sub__(u, v):
        r"""Difference of complex tensors."""
        if not isinstance(v, (Cplx, complex)):
            return type(u)(u.__real - v, u.__imag)
        return type(u)(u.__real - v.real, u.__imag - v.imag)

    def __rsub__(u, v):
        r"""Difference of complex tensors."""
        return -u + v

    __isub__ = __sub__

    def __mul__(u, v):
        r"""Elementwise product of complex tensors."""
        if not isinstance(v, (Cplx, complex)):
            return type(u)(u.__real * v, u.__imag * v)

        # (a + ib) (u + iv) = au - bv + i(av + bu)
        # (a+u)(b+v) = ab + uv + (av + ub)
        # (a-v)(b+u) = ab - uv + (au - vb)
        return type(u)(u.__real * v.real - u.__imag * v.imag,
                       u.__imag * v.real + u.__real * v.imag)

    __rmul__ = __mul__
    __imul__ = __mul__

    def __truediv__(u, v):
        r"""Elementwise division of complex tensors."""
        if not isinstance(v, (Cplx, complex)):
            return type(u)(u.__real / v, u.__imag / v)

        denom = v.real * v.real + v.imag * v.imag
        return u * (v.conjugate() / denom)

    def __rtruediv__(u, v):
        r"""Elementwise division of something by a complex tensor."""
        # v / u and v is not Cplx
        denom = u.__real * u.__real + u.__imag * u.__imag
        return (u.conjugate() / denom) * v

    __itruediv__ = __truediv__

    def __matmul__(u, v):
        r"""Complex matrix-matrix product of complex tensors."""
        if not isinstance(v, Cplx):
            return type(u)(torch.matmul(u.__real, v), torch.matmul(u.__imag, v))

        re = torch.matmul(u.__real, v.__real) - torch.matmul(u.__imag, v.__imag)
        im = torch.matmul(u.__imag, v.__real) + torch.matmul(u.__real, v.__imag)
        return type(u)(re, im)

    def __rmatmul__(u, v):
        r"""Matrix multiplication by a complex tensor from the right."""
        # v @ u and v is not Cplx
        return type(u)(torch.matmul(v, u.__real), torch.matmul(v, u.__imag))

    __imatmul__ = __matmul__

    def __abs__(self):
        r"""Compute the complex modulus:
        $$
            \mathbb{C}^{\ldots \times d}
                \to \mathbb{R}_+^{\ldots \times d}
            \colon u + i v \mapsto \lvert u + i v \rvert
            \,. $$
        """
        input = torch.stack([self.__real, self.__imag], dim=0)
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
        return torch.atan2(self.__imag, self.__real)

    def apply(self, f, *a, **k):
        r"""Applies the function to real and imaginary parts."""
        return type(self)(f(self.__real, *a, **k), f(self.__imag, *a, **k))

    @property
    def shape(self):
        r"""Returns the shape of the complex tensor."""
        return self.__real.shape

    def __len__(self):
        r"""The size of the zero-th dimension of the complex tensor."""
        return self.shape[0]

    def t(self):
        r"""The transpose of a 2d compelx tensor."""
        return type(self)(self.__real.t(), self.__imag.t())

    def h(self):
        r"""The Hermitian transpose of a 2d compelx tensor."""
        return self.conj.t()  # Cplx(self.__real.t(), -self.__imag.t())

    def reshape(self, *shape):
        r"""Reshape the complex tensor."""
        shape = shape[0] if shape and isinstance(shape[0], tuple) else shape
        return type(self)(self.__real.reshape(*shape), self.__imag.reshape(*shape))

    def item(self):
        r"""The scalar value of zero-dim complex tensor."""
        return float(self.__real) + 1j * float(self.__imag)

    @classmethod
    def from_numpy(cls, numpy):
        r"""Create a complex tensor from numpy array."""
        re = torch.from_numpy(numpy.real)
        im = torch.from_numpy(numpy.imag)
        return cls(re, im)

    def numpy(self):
        r"""Export a complex tensor as complex numpy array."""
        return self.__real.numpy() + 1j * self.__imag.numpy()

    def __repr__(self):
        return f"{self.__class__.__name__}(\n" \
               f"  real={self.__real},\n  imag={self.__imag}\n)"

    def detach(self):
        r"""Return a copy of the complex tensor detached from autograd graph."""
        return type(self)(self.__real.detach(), self.__imag.detach())

    def requires_grad_(self, requires_grad=True):
        r"""Toggle the gradient of real and imaginary parts."""
        return type(self)(self.__real.requires_grad_(requires_grad),
                          self.__imag.requires_grad_(requires_grad))

    @property
    def grad(self):
        r"""Collect the accumulated gradinet of the complex tensor."""
        re, im = self.__real.grad, self.__imag.grad
        return None if re is None or im is None else type(self)(re, im)

    def cuda(self, device=None, non_blocking=False):
        r"""Move the complex tensor to a CUDA device."""
        re = self.__real.cuda(device=device, non_blocking=non_blocking)
        im = self.__imag.cuda(device=device, non_blocking=non_blocking)
        return type(self)(re, im)

    def cpu(self):
        r"""Move the complex tensor to CPU."""
        return type(self)(self.__real.cpu(), self.__imag.cpu())

    def to(self, *args, **kwargs):
        r"""Move / typecast the complex tensor."""
        return type(self)(self.__real.to(*args, **kwargs),
                          self.__imag.to(*args, **kwargs))

    @property
    def device(self):
        r"""The hosting device of the complex tensor."""
        return self.__real.device

    @property
    def dtype(self):
        r"""The base dtype of the complex tensor."""
        return self.__real.dtype

    def dim(self):
        r"""The number of dimensions in the complex tensor."""
        return len(self.shape)

    def permute(self, *dims):
        r"""Shuffle the dimensions of the complex tensor."""
        return type(self)(self.__real.permute(*dims),
                          self.__imag.permute(*dims))

    def transpose(self, dim0, dim1):
        r"""Transpose the specified dimensions of the complex tensor."""
        return type(self)(self.__real.transpose(dim0, dim1),
                          self.__imag.transpose(dim0, dim1))

    def is_complex(self):
        r"""Test if the tensor indeed represents a complex number."""
        return True

    @classmethod
    def empty(cls, *sizes, dtype=None, device=None, requires_grad=False):
        r"""Create an empty complex tensor."""
        re = torch.empty(*sizes, dtype=dtype, device=device,
                         requires_grad=requires_grad)
        return cls(re, torch.empty_like(re, requires_grad=requires_grad))

    @classmethod
    def zeros(cls, *sizes, dtype=None, device=None, requires_grad=False):
        r"""Create an empty complex tensor."""
        re = torch.zeros(*sizes, dtype=dtype, device=device,
                         requires_grad=requires_grad)
        return cls(re, torch.zeros_like(re, requires_grad=requires_grad))

    @classmethod
    def ones(cls, *sizes, dtype=None, device=None, requires_grad=False):
        r"""Create an empty complex tensor."""
        re = torch.ones(*sizes, dtype=dtype, device=device,
                        requires_grad=requires_grad)
        return cls(re, torch.zeros_like(re, requires_grad=requires_grad))


def cat(tensors, dim):
    tensors = [*map(Cplx, tensors)]
    return Cplx(torch.cat([z.real for z in tensors], dim=dim),
                torch.cat([z.imag for z in tensors], dim=dim))


def split(input, split_size_or_sections, dim=0):
    """see documentation for `torch.split`"""
    return tuple(Cplx(re, im) for re, im in zip(
        torch.split(input.real, split_size_or_sections, dim),
        torch.split(input.imag, split_size_or_sections, dim),
    ))


def chunk(input, chunks, dim=0):
    """see documentation for `torch.chunk`"""
    return tuple(Cplx(re, im) for re, im in zip(
        torch.chunk(input.real, chunks, dim),
        torch.chunk(input.imag, chunks, dim),
    ))


def stack(tensors, dim):
    tensors = [*map(Cplx, tensors)]
    return Cplx(torch.stack([z.real for z in tensors], dim=dim),
                torch.stack([z.imag for z in tensors], dim=dim))


def unbind(input, dim=0):
    """see documentation for `torch.unbind`"""
    return tuple(Cplx(re, im) for re, im in zip(
        torch.unbind(input.real, dim),
        torch.unbind(input.imag, dim),
    ))


def take(input, index):
    """see documentation for `torch.take`"""
    return Cplx(torch.take(input.real, index),
                torch.take(input.imag, index))


def narrow(input, dim, start, length):
    """see documentation for `torch.narrow`"""
    return Cplx(torch.narrow(input.real, dim, start, length),
                torch.narrow(input.imag, dim, start, length))


def squeeze(input, dim=None):
    """see documentation for `torch.squeeze`"""
    return Cplx(torch.squeeze(input.real, dim),
                torch.squeeze(input.imag, dim))


def unsqueeze(input, dim):
    """see documentation for `torch.unsqueeze`"""
    return Cplx(torch.unsqueeze(input.real, dim),
                torch.unsqueeze(input.imag, dim))


def from_interleaved_real(input, copy=True, dim=-1):
    """Map real tensor input `... x [D * 2]` to a pair (re, im) with dim `... x D`."""
    output = Cplx(*complex_view(input, dim, squeeze=False))
    return output.clone() if copy else output


from_real = from_interleaved_real


def from_concatenated_real(input, copy=True, dim=-1):
    """Map real tensor input `... x [2 * D]` to a pair (re, im) with dim `... x D`."""
    output = Cplx(*torch.chunk(input, 2, dim=dim))
    return output.clone() if copy else output


def to_interleaved_real(input, flatten=True, dim=-1):
    """Interleave the complex re-im pair into a real tensor."""
    dim = 1 + fix_dim(dim, input.dim())
    input = torch.stack([input.real, input.imag], dim=dim)
    return input.flatten(dim-1, dim) if flatten else input


to_real = to_interleaved_real


def to_concatenated_real(input, flatten=None, dim=-1):
    """Map real tensor input `... x [2 * D]` to a pair (re, im) with dim `... x D`."""
    assert flatten is None
    return torch.cat([input.real, input.imag], dim=dim)


def exp(input):
    r"""Compute the exponential of the complex tensor in re-im pair."""
    scale = torch.exp(input.real)
    return Cplx(scale * torch.cos(input.imag),
                scale * torch.sin(input.imag))


def log(input):
    r"""Compute the logarithm of the complex tensor in re-im pair."""
    return Cplx(torch.log(abs(input)), input.angle)


def sin(input):
    r"""Compute the sine of the complex tensor in re-im pair."""
    return Cplx(torch.sin(input.real) * torch.cosh(input.imag),
                torch.cos(input.real) * torch.sinh(input.imag))


def cos(input):
    r"""Compute the cosine of the complex tensor in re-im pair."""
    return Cplx(torch.cos(input.real) * torch.cosh(input.imag),
                - torch.sin(input.real) * torch.sinh(input.imag))


def tan(input):
    r"""Compute the tangent of the complex tensor in re-im pair."""
    return sin(input) / cos(input)


def sinh(input):
    r"""Compute the hyperbolic sine of the complex tensor in re-im pair.

    sinh(z) = - j sin(j z)
    """
    return Cplx(torch.sinh(input.real) * torch.cos(input.imag),
                torch.cosh(input.real) * torch.sin(input.imag))


def cosh(input):
    r"""Compute the hyperbolic cosine of the complex tensor in re-im pair.

    cosh(z) = cos(j z)
    """
    return Cplx(torch.cosh(input.real) * torch.cos(input.imag),
                torch.sinh(input.real) * torch.sin(input.imag))


def tanh(input):
    r"""Compute the hyperbolic tangent of the complex tensor in re-im pair.

    tanh(z) = j tan(z)
    """
    return sinh(input) / cosh(input)


def randn(*size, dtype=None, device=None, requires_grad=False):
    """Generate standard complex Gaussian noise."""
    normal = torch.randn(2, *size, dtype=dtype, layout=torch.strided,
                         device=device, requires_grad=False) / sqrt(2)
    z = Cplx(normal[0], normal[1])
    return z.requires_grad_(True) if requires_grad else z


def randn_like(input, dtype=None, device=None, requires_grad=False):
    """Returns a tensor with the same size as `input` that is filled with
    standard comlpex Gaussian random numbers.
    """
    return randn(*input.size(),
                 dtype=input.dtype if dtype is None else dtype,
                 device=input.device if device is None else device,
                 requires_grad=requires_grad)


def modrelu(input, threshold=0.5):
    r"""Compute the modulus relu of the complex tensor in re-im pair."""
    # scale = (1 - \trfac{b}{|z|})_+
    modulus = torch.clamp(abs(input), min=1e-5)
    return input * torch.relu(1. - threshold / modulus)


def phaseshift(input, phi=0.0):
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


def linear_naive(input, weight, bias=None):
    r"""Applies a complex linear transformation to the incoming complex
    data: :math:`y = x A^T + b`.
    """
    # W = U + i V,  z = u + i v, c = \Re c + i \Im c
    #  W z + c = (U + i V) (u + i v) + \Re c + i \Im c
    #          = (U u + \Re c - V v) + i (V u + \Im c + U v)
    re = F.linear(input.real, weight.real) \
        - F.linear(input.imag, weight.imag)
    im = F.linear(input.real, weight.imag) \
        + F.linear(input.imag, weight.real)

    output = Cplx(re, im)
    if bias is not None:
        output += bias

    return output


def linear_cat(input, weight, bias=None):
    # [n_out, n_in] -> [2 * n_out, 2 * n_in] : [[U, V], [-V, U]]
    ww = torch.cat([
        torch.cat([ weight.real, weight.imag], dim=0),
        torch.cat([-weight.imag, weight.real], dim=0)
    ], dim=1)

    xx = to_concatenated_real(input, dim=-1)  # [..., 2 * n_in]
    output = from_concatenated_real(F.linear(xx, ww, None))
    if bias is not None:
        output += bias

    return output


def linear_3m(input, weight, bias=None):
    r"""Applies a complex linear transformation to the incoming complex
    data: :math:`y = x A^T + b`.

    Strassen's 3M
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.118.1356&rep=rep1&type=pdf

    This method
    https://cnx.org/contents/4kChocHM@6/Efficient-FFT-Algorithm-and-Programming-Tricks
    """
    # W = U + i V,  z = u + i v, c = \Re c + i \Im c
    #  W z + c = (U + i V) (u + i v) + \Re c + i \Im c
    #          = (U u + \Re c - V v) + i (V u + \Im c + U v)
    K1 = F.linear(input.real + input.imag,  weight.real)
    K2 = F.linear(input.real, weight.imag - weight.real)
    K3 = F.linear(input.imag, weight.real + weight.imag)

    output = Cplx(K1 - K3, K1 + K2)
    if bias is not None:
        output += bias

    return output


# use naive multiplication by default
linear = linear_naive


def symmetric_circular_padding(input, padding):
    # `F.pad` works only for 3d, 4d, and 5d inputs
    assert input.dim() > 2
    if isinstance(padding, int):
        padding = (input.dim() - 2) * [padding]

    assert isinstance(padding, (tuple, list))
    assert len(padding) + 2 == input.dim()

    expanded_padding = []
    for pad in padding:
        expanded_padding.extend(((pad + 1) // 2, pad // 2))

    return input.apply(F.pad, tuple(expanded_padding), mode="circular")


def convnd_naive(conv, input, weight, stride=1,
                 padding=0, dilation=1, groups=1):

    re = conv(input.real, weight.real, None,
              stride, padding, dilation, groups) \
        - conv(input.imag, weight.imag, None,
               stride, padding, dilation, groups)
    im = conv(input.real, weight.imag, None,
              stride, padding, dilation, groups) \
        + conv(input.imag, weight.real, None,
               stride, padding, dilation, groups)

    return Cplx(re, im)


def convnd_quick(conv, input, weight, stride=1,
                 padding=0, dilation=1):
    n_out = weight.shape[0]
    ww = torch.cat([weight.real, weight.imag], dim=0)
    wr = conv(input.real, ww, None, stride, padding, dilation, 1)
    wi = conv(input.imag, ww, None, stride, padding, dilation, 1)

    rwr, iwr = wr[:, :n_out], wr[:, n_out:]
    rwi, iwi = wi[:, :n_out], wi[:, n_out:]
    return Cplx(rwr - iwi, iwr + rwi)


def convnd(conv, input, weight, bias=None, stride=1,
           padding=0, dilation=1, groups=1, padding_mode="zeros"):
    r"""Applies a complex n-d convolution to the incoming complex
    tensor `B x c_in x L_1 x ... L_n`: :math:`y = x \star W + b`.
    """
    if padding_mode == 'circular':
        input = symmetric_circular_padding(input, padding)
        padding = 0

    if groups == 1:
        # ungroupped convolution can be done a little bit faster
        output = convnd_quick(conv, input, weight, stride,
                              padding, dilation)
    else:
        output = convnd_naive(conv, input, weight, stride,
                              padding, dilation, groups)

    if bias is not None:
        broadcast = (input.dim() - 2) * [1]
        output += bias.reshape(-1, *broadcast)

    return output


def conv1d(input, weight, bias=None, stride=1, padding=0,
           dilation=1, groups=1, padding_mode="zeros"):
    r"""Applies a complex 1d convolution to the incoming complex
    tensor `B x c_in x L`: :math:`y = x \star W + b`.
    """

    return convnd(F.conv1d, input, weight, bias, stride,
                  padding, dilation, groups, padding_mode)


def conv2d(input, weight, bias=None, stride=1, padding=0,
           dilation=1, groups=1, padding_mode="zeros"):
    r"""Applies a complex 2d convolution to the incoming complex
    tensor `B x c_in x H x W`: :math:`y = x \star W + b`.
    """

    return convnd(F.conv2d, input, weight, bias, stride,
                  padding, dilation, groups, padding_mode)


def conv3d(input, weight, bias=None, stride=1, padding=0,
           dilation=1, groups=1, padding_mode="zeros"):
    r"""Applies a complex 3d convolution to the incoming complex
    tensor `B x c_in x H x W x D`: :math:`y = x \star W + b`.
    """

    return convnd(F.conv3d, input, weight, bias, stride,
                  padding, dilation, groups, padding_mode)


def einsum(equation, *tensors):
    """2-tensor einstein summation."""
    if not tensors:
        raise RuntimeError("""`einsum()` requires """
                           """at least one tensor.""")

    cplx1, *tensors = tensors
    if not tensors:
        # no complex multiplication with only one tensor
        return Cplx(torch.einsum(equation, cplx1.real),
                    torch.einsum(equation, cplx1.imag))

    cplx2, *tensors = tensors
    if not tensors:
        # Einsum is complex bilinear -- the logic is same here.
        re = torch.einsum(equation, cplx1.real, cplx2.real) \
            - torch.einsum(equation, cplx1.imag, cplx2.imag)

        im = torch.einsum(equation, cplx1.real, cplx2.imag) \
            + torch.einsum(equation, cplx1.imag, cplx2.real)

        return Cplx(re, im)

    raise RuntimeError(f"""`einsum()` does not support more """
                       f"""than 2 tensors. Got {2 + len(tensors)}.""")


def bilinear_naive(input1, input2, weight, bias=None, conjugate=True):
    r"""Applies a complex bilinear transformation to the incoming complex
    data: :math:`y = x^(T/H) W z + b`.
    """

    n_out = weight.shape[0]

    ww = torch.cat([weight.real, weight.imag], dim=0)
    a, b = input1.real, input1.imag
    u, v = input2.real, input2.imag
    au, av = F.bilinear(a, u, ww, bias=None), F.bilinear(a, v, ww, bias=None)
    bu, bv = F.bilinear(b, u, ww, bias=None), F.bilinear(b, v, ww, bias=None)

    if conjugate:
        pp, qq = au + bv, av - bu
    else:
        pp, qq = au - bv, av + bu

    repp, impp = pp[..., :n_out], pp[..., n_out:]
    reqq, imqq = qq[..., :n_out], qq[..., n_out:]

    output = Cplx(repp - imqq, impp + reqq)
    if bias is not None:
        output += bias

    return output


bilinear = bilinear_naive


def bilinear_cat(input1, input2, weight, bias=None, conjugate=True):
    # [n_out, n_in1, n_in2] -> [2 * n_out, 2 * n_in1, 2 * n_in2]
    U, V = weight.real, weight.imag

    UV = torch.cat([U, -V], dim=2)
    VU = torch.cat([V,  U], dim=2)
    if conjugate:
        ww = torch.cat([
            torch.cat([UV,  VU], dim=1),
            torch.cat([VU, -UV], dim=1)
        ], dim=0)
    else:
        ww = torch.cat([
            torch.cat([UV, -VU], dim=1),
            torch.cat([VU,  UV], dim=1)
        ], dim=0)

    x1 = to_concatenated_real(input1, dim=-1)
    x2 = to_concatenated_real(input2, dim=-1)

    output = from_concatenated_real(F.bilinear(x1, x2, ww, None))
    if bias is not None:
        output += bias

    return output
