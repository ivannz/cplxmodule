import math

import torch
import torch.nn
import torch.nn.functional as F

from functools import lru_cache

from torch.nn import Parameter

from . import init
from .. import cplx
from ..cplx import Cplx


class CplxParameter(torch.nn.ParameterDict):
    """Torch-friendly container for complex-valued parameter."""
    def __init__(self, cplx):
        if not isinstance(cplx, Cplx):
            raise TypeError(f"""`{type(self).__name__}` accepts only """
                            f"""Cplx tensors.""")

        super().__init__({
            "real": Parameter(cplx.real),
            "imag": Parameter(cplx.imag),
        })

        # save reference to the underlying cplx data
        self._cplx = cplx

    def extra_repr(self):
        return repr(tuple(self._cplx.shape))[1:-1]

    @property
    def data(self):
        return self._cplx


class CplxParameterAccessor():
    """Cosmetic complex parameter accessor.

    Details
    -------
    This works both for the default `forward()` inherited from Linear,
    and for what the user expects to see when they request weight from
    the layer (masked zero values).

    Warning
    -------
    This hacky property works only because torch.nn.Module implements
    its own special attribute access mechanism via `__getattr__`. This
    is why `SparseWeightMixin` in .masked couldn't work with 'weight'
    as a read-only @property.
    """
    def __getattr__(self, name):
        # default attr lookup straight to parent's __getattr__
        attr = super().__getattr__(name)
        if not isinstance(attr, CplxParameter):  # automatically handles None
            return attr

        # Cplx() is a light weight container for mutable real-imag parts.
        #  Can we cache this? What if creating `Cplx` is costly?
        return Cplx(attr.real, attr.imag)


class BaseRealToCplx(torch.nn.Module):
    pass


class InterleavedRealToCplx(BaseRealToCplx):
    r"""A layer that splits an interleaved real tensor with even number in the
    last dim to a complex tensor represented by a pair of real and imaginary
    tensors of the same size. Preserves the all dimensions but the last, which
    is halved.
    $$
        F
        \colon \mathbb{R}^{\ldots \times [d\times 2]}
                \to \mathbb{C}^{\ldots \times d}
        \colon x \mapsto \bigr(
            x_{2k} + i x_{2k+1}
        \bigl)_{k=0}^{d-1}
        \,. $$
    """
    def __init__(self, copy=False, dim=-1):
        super().__init__()
        self.copy, self.dim = copy, dim

    def forward(self, input):
        return cplx.from_interleaved_real(input, self.copy, self.dim)


RealToCplx = InterleavedRealToCplx


class ConcatenatedRealToCplx(BaseRealToCplx):
    r"""Convert float tensor in concatenated format, i.e. real followed by
    imag, to a Cplx tensor. Preserves all dimensions except for the one
    specified, which is halved.
    $$
        F
        \colon \mathbb{R}^{\ldots \times [d\times 2]}
                \to \mathbb{C}^{\ldots \times d}
        \colon x \mapsto \bigr(
            x_{k} + i x_{d + k}
        \bigl)_{k=0}^{d-1}
        \,. $$
    """
    def __init__(self, copy=False, dim=-1):
        super().__init__()
        self.copy, self.dim = copy, dim

    def forward(self, input):
        return cplx.from_concatenated_real(input, self.copy, self.dim)


class AsTypeCplx(BaseRealToCplx):
    r"""A layer that differentibaly casts the real tensor into a complex tensor.
    $$
        F
        \colon \mathbb{R}^{\ldots \times d}
                \to \mathbb{C}^{\ldots \times d}
        \colon x \mapsto x + 0 i
        \,. $$
    """
    def forward(self, input):
        return Cplx(input)


class BaseCplxToReal(torch.nn.Module):
    pass


class CplxToInterleavedReal(BaseCplxToReal):
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
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return cplx.to_interleaved_real(input, True, self.dim)


CplxToReal = CplxToInterleavedReal


class CplxToConcatenatedReal(BaseCplxToReal):
    r"""
    A layer that concatenates the real and imaginary parts of a complex tensor
    into a larger real tensor along the last dimension.
    $$
        F
        \colon \mathbb{C}^{\ldots \times d}
                \to \mathbb{R}^{\ldots \times [2 \times d]}
        \colon u + i v \mapsto \bigl(u_\omega, v_\omega\bigr)_{\omega}
        \,. $$
    """
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return cplx.to_concatenated_real(input, None, self.dim)


class _CplxToCplxMeta(type):
    """Meta class for bracketed creation of componentwise operations."""
    @lru_cache(maxsize=None)
    def __getitem__(self, Base):
        # make sure that base is not an instance, and that no
        #  nested wrapping takes place.
        assert isinstance(Base, type) and issubclass(Base, torch.nn.Module)
        if issubclass(Base, (CplxToCplx, BaseRealToCplx)):
            return Base

        if Base is torch.nn.Module:
            return CplxToCplx

        class template(Base, CplxToCplx):
            def forward(self, input):
                """Apply to real and imaginary parts independently."""
                return input.apply(super().forward)

        name = "Cplx" + Base.__name__
        template.__name__ = template.__qualname__ = name
        return template


class CplxToCplx(CplxParameterAccessor, torch.nn.Module,
                 metaclass=_CplxToCplxMeta):
    pass


def is_from_cplx(module):
    if isinstance(module, (CplxToCplx, BaseCplxToReal)):
        return True

    if isinstance(module, torch.nn.Sequential):
        return is_from_cplx(module[0])

    if isinstance(module, type):
        return issubclass(module, (CplxToCplx, BaseCplxToReal))

    return False


def is_to_cplx(module):
    if isinstance(module, (CplxToCplx, BaseRealToCplx)):
        return True

    if isinstance(module, torch.nn.Sequential):
        return is_to_cplx(module[-1])

    if isinstance(module, type):
        return issubclass(module, (CplxToCplx, BaseRealToCplx))

    return False


def is_cplx_to_cplx(module):
    return is_from_cplx(module) and is_to_cplx(module)


class CplxLinear(CplxToCplx):
    r"""
    Complex linear transform:
    $$
        F
        \colon \mathbb{C}^{\ldots \times d_0}
                \to \mathbb{C}^{\ldots \times d_1}
        \colon u + i v \mapsto W_\mathrm{re} (u + i v) + i W_\mathrm{im} (u + i v)
                = (W_\mathrm{re} u - W_\mathrm{im} v)
                    + i (W_\mathrm{im} u + W_\mathrm{re} v)
        \,. $$
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = CplxParameter(Cplx.empty(out_features, in_features))

        if bias:
            self.bias = CplxParameter(Cplx.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        weight, bias = self.weight, self.bias  # inplace acessors
        init.cplx_kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = init.get_fans(weight)
            bound = 1 / math.sqrt(fan_in)
            init.cplx_uniform_independent_(bias, -bound, bound)

    def forward(self, input):
        return cplx.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class CplxDropout(torch.nn.Dropout2d, CplxToCplx):
    r"""
    Complex 1d dropout layer: simultaneous dropout on both real and
    imaginary parts.

    See torch.nn.Dropout1d for reference on the input dimensions and arguments.
    """
    def forward(self, input):
        *head, n_last = input.shape

        # shape -> [*shape, 2] : re-im are feature maps!
        tensor = torch.stack([input.real, input.imag], dim=-1)
        output = super().forward(tensor.reshape(-1, 1, 2))

        # [-1, 1, 2] -> [*head, n_last * 2]
        output = output.reshape(*head, -1)

        # [*head, n_last * 2] -> [*head, n_last]
        return cplx.from_interleaved_real(output, False, -1)


CplxDropout1d = CplxDropout


class CplxAvgPool1d(torch.nn.AvgPool1d, CplxToCplx):
    r"""
    Complex 1d average pooling layer: simultaneously pools both real
    and imaginary parts.

    See torch.nn.AvgPool1d for reference on the input dimensions and arguments.
    """
    def forward(self, input):
        # apply parent.forward to re and im parts
        return input.apply(super().forward)


class CplxPhaseShift(CplxToCplx):
    r"""
    A learnable complex phase shift
    $$
        F
        \colon \mathbb{C}^{\ldots \times C \times d}
                \to \mathbb{C}^{\ldots \times C \times d}
        \colon z \mapsto z_{\ldots kj} e^{i \theta_{kj}}
        \,, $$
    where $\theta_k$ is the phase shift of the $k$-the channel in radians.
    Torch's broadcasting rules apply and the passed dimensions conform with
    the upstream input. For example, `CplxPhaseShift(C, 1)` shifts each $d$-dim
    complex vector by the phase of its channel, and `CplxPhaseShift(d)` shifts
    each complex feature in all channels by the same phase. Finally calling
    with CplxPhaseShift(1) shifts the inputs by the single common phase.
    """
    def __init__(self, *dim):
        super().__init__()
        self.phi = Parameter(torch.randn(*dim) * 0.02)

    def forward(self, input):
        return cplx.phaseshift(input, self.phi)


class CplxBilinear(CplxToCplx):
    r"""Complex bilinear transform"""
    def __init__(self, in1_features, in2_features, out_features, bias=True,
                 conjugate=True):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features

        self.weight = CplxParameter(Cplx.empty(
            out_features, in1_features, in2_features))

        if bias:
            self.bias = CplxParameter(Cplx.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self.conjugate = conjugate

        self.reset_parameters()

    def reset_parameters(self):
        init.cplx_kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init.get_fans(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.cplx_uniform_independent_(self.bias, -bound, bound)

    def forward(self, input1, input2):
        return cplx.bilinear(input1, input2, self.weight,
                             self.bias, self.conjugate)

    def extra_repr(self):
        fmt = """in1_features={}, in2_features={}, out_features={}, """
        fmt += """bias={}, conjugate={}"""
        return fmt.format(
            self.in1_features, self.in2_features, self.out_features,
            self.bias is not None, self.conjugate)


class CplxReal(BaseCplxToReal):
    def forward(self, input):
        return input.real


class CplxImag(BaseCplxToReal):
    def forward(self, input):
        return input.imag
