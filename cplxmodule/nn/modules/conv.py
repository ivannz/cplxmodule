import math
import torch

from torch.nn.modules.utils import _single, _pair, _triple

from .base import CplxToCplx, CplxParameter
from .. import init
from ... import cplx


class CplxConvNd(CplxToCplx):
    r"""An almost verbatim copy of `_ConvNd` from torch/nn/modules/conv.py"""
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias, padding_mode):
        super().__init__()

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")

        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size, self.stride = kernel_size, stride
        self.padding, self.dilation = padding, dilation

        self.groups, self.padding_mode = groups, padding_mode
        self.groups, self.padding_mode = groups, padding_mode

        self.weight = CplxParameter(cplx.Cplx.empty(
            out_channels, in_channels // groups, *kernel_size))

        if bias:
            self.bias = CplxParameter(cplx.Cplx.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        init.cplx_kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init.get_fans(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.cplx_uniform_independent_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ("{in_channels}, {out_channels}, kernel_size={kernel_size}"
             ", stride={stride}")

        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"

        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"

        if self.groups != 1:
            s += ", groups={groups}"

        if self.bias is None:
            s += ", bias=False"

        if self.padding_mode != "zeros":
            s += ", padding_mode='{padding_mode}'"

        return s.format(**self.__dict__)


class CplxConv1d(CplxConvNd):
    r"""Complex 1D convolution:
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
                 bias=True,
                 padding_mode="zeros"):
        super().__init__(
            in_channels, out_channels, _single(kernel_size), _single(stride),
            _single(padding), _single(dilation), groups, bias, padding_mode)

    def forward(self, input):
        return cplx.conv1d(input, self.weight, self.bias,
                           self.stride[0], self.padding[0], self.dilation[0],
                           self.groups, self.padding_mode)


class CplxConv2d(CplxConvNd):
    r"""Complex 2D convolution:
    $$
        F
        \colon \mathbb{C}^{B \times c_{in} \times L}
                \to \mathbb{C}^{B \times c_{out} \times L'}
        \colon u + i v \mapsto (W_\mathrm{re} \star u - W_\mathrm{im} \star v)
                                + i (W_\mathrm{im} \star u + W_\mathrm{re} \star v)
        \,. $$

    See torch.nn.Conv2d for reference on the input dimensions.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode="zeros"):
        super().__init__(
            in_channels, out_channels, _pair(kernel_size), _pair(stride),
            _pair(padding), _pair(dilation), groups, bias, padding_mode)

    def forward(self, input):
        return cplx.conv2d(input, self.weight, self.bias,
                           self.stride, self.padding, self.dilation,
                           self.groups, self.padding_mode)


class CplxConv3d(CplxConvNd):
    r"""Complex 3D convolution:
    $$
        F
        \colon \mathbb{C}^{B \times c_{in} \times L}
                \to \mathbb{C}^{B \times c_{out} \times L'}
        \colon u + i v \mapsto (W_\mathrm{re} \star u - W_\mathrm{im} \star v)
                                + i (W_\mathrm{im} \star u + W_\mathrm{re} \star v)
        \,. $$

    See torch.nn.Conv2d for reference on the input dimensions.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode="zeros"):
        super().__init__(
            in_channels, out_channels, _triple(kernel_size), _triple(stride),
            _triple(padding), _triple(dilation), groups, bias, padding_mode)

    def forward(self, input):
        return cplx.conv3d(input, self.weight, self.bias,
                           self.stride, self.padding, self.dilation,
                           self.groups, self.padding_mode)
