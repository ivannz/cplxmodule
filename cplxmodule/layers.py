import math

import torch
import torch.nn
import torch.nn.functional as F

from torch.nn import Parameter

from .cplx import Cplx, real_to_cplx, cplx_to_real
from .cplx import cplx_linear
from .cplx import cplx_phaseshift


class CplxParameter(torch.nn.ParameterDict):
    def __init__(self, cplx):
        if not isinstance(cplx, Cplx):
            raise TypeError(f"""`{type(self).__name__}` accepts only """
                            f"""Cplx tensors.""")

        super().__init__({
            "real": Parameter(cplx.real),
            "imag": Parameter(cplx.imag),
        })


def is_from_cplx(module):
    if isinstance(module, (CplxToCplx, CplxToReal)):
        return True

    if isinstance(module, torch.nn.Sequential):
        return is_from_cplx(module[0])

    if isinstance(module, type):
        return issubclass(module, (CplxToCplx, CplxToReal))

    return False


def is_to_cplx(module):
    if isinstance(module, (CplxToCplx, RealToCplx)):
        return True

    if isinstance(module, torch.nn.Sequential):
        return is_to_cplx(module[-1])

    if isinstance(module, type):
        return issubclass(module, (CplxToCplx, RealToCplx))

    return False


def is_cplx_to_cplx(module):
    return is_from_cplx(module) and is_to_cplx(module)


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
    def __init__(self, copy=False):
        super().__init__()
        self.copy = copy

    def forward(self, input):
        return real_to_cplx(input, copy=self.copy)


class AsTypeCplx(RealToCplx):
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


class CplxWeightMixin(torch.nn.Module):
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
    @property
    def weight(self):
        # bypass default attr lookup straight to own __getattr__
        weight = self.__getattr__("weight")

        # can we cache this? Cause what if creating `Cplx` is costly?
        return Cplx(weight.real, weight.imag)

    @property
    def bias(self):
        bias = self.__getattr__("bias")
        if bias is None:
            return None
        return Cplx(bias.real, bias.imag)


class CplxToCplx(torch.nn.Module):
    pass


class CplxLinear(CplxWeightMixin, CplxToCplx):
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
        torch.nn.init.kaiming_uniform_(weight.real, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(weight.imag, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight.real)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(bias.real, -bound, bound)
            torch.nn.init.uniform_(bias.imag, -bound, bound)

    def forward(self, input):
        return cplx_linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class CplxDropout1d(torch.nn.Dropout2d, CplxToCplx):
    r"""
    Complex 1d dropout layer: simultaneous dropout on both real and
    imaginary parts.

    See torch.nn.Dropout1d for reference on the input dimensions and arguments.
    """
    def forward(self, input):
        output = super().forward(cplx_to_real(input, flatten=False))
        return real_to_cplx(output.flatten(-2))


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
        return cplx_phaseshift(input, self.phi)
