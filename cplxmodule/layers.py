import math

import torch
import torch.nn
import torch.nn.functional as F

from torch.nn import Parameter

from .cplx import Cplx, real_to_cplx, cplx_to_real
from .cplx import cplx_linear, cplx_conv1d
from .cplx import cplx_phaseshift


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

        self.weight = torch.nn.ParameterDict({
            "real": Parameter(torch.Tensor(out_features, in_features)),
            "imag": Parameter(torch.Tensor(out_features, in_features)),
        })

        if bias:
            self.bias = torch.nn.ParameterDict({
                "real": Parameter(torch.Tensor(out_features)),
                "imag": Parameter(torch.Tensor(out_features)),
            })
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight.real, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.weight.imag, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight.real)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias.real, -bound, bound)
            torch.nn.init.uniform_(self.bias.imag, -bound, bound)

    def forward(self, input):
        return cplx_linear(input, Cplx(**self.weight), self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class CplxConv1d(CplxToCplx):
    r"""
    Complex 1D convolution:
    $$
        F
        \colon \mathbb{C}^{B \times c_{in} \times L}
                \to \mathbb{C}^{B \times c_{out} \times L'}
        \colon u + i v \mapsto (W_\mathrm{re} \star u - W_\mathrm{im} \star v)
                                + i (W_\mathrm{im} \star u + W_\mathrm{re} \star v)
        \,. $$

    See torch.nn.Conv1d for reference on the input dimensions.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super().__init__()

        self.re = torch.nn.Conv1d(in_channels, out_channels, kernel_size,
                                  stride=stride, padding=padding,
                                  dilation=dilation, groups=groups,
                                  bias=bias)
        self.im = torch.nn.Conv1d(in_channels, out_channels, kernel_size,
                                  stride=stride, padding=padding,
                                  dilation=dilation, groups=groups,
                                  bias=bias)

    def forward(self, input):
        """Complex tensor (re-im) `B x c_in x L`"""
        weight = Cplx(self.re.weight, self.im.weight)
        bias = Cplx(self.re.bias, self.im.bias)

        return cplx_conv1d(input, weight, bias, self.re.stride,
                           self.re.padding, self.re.dilation, self.re.groups)


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
