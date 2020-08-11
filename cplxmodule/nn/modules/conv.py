import math
import torch

from torch.nn.modules.utils import _single, _pair, _triple

from .base import CplxToCplx, CplxParameter
from .. import init
from ... import cplx


class CplxConvNd(CplxToCplx):
    r"""An almost verbatim copy of `_ConvNd` from torch/nn/modules/conv.py"""
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups,
                 bias, padding_mode):
        super().__init__()

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")

        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size, self.stride = kernel_size, stride
        self.padding, self.dilation = padding, dilation
        self.transposed = transposed
        self.output_padding = output_padding

        self.groups, self.padding_mode = groups, padding_mode
        self.groups, self.padding_mode = groups, padding_mode

        if transposed:
            self.weight = CplxParameter(cplx.Cplx.empty(
                in_channels, in_channels // groups, *kernel_size))
        else:
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

        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"

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
            _single(padding), _single(dilation), False, _single(0), groups,
            bias, padding_mode)

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
            _pair(padding), _pair(dilation), False, _pair(0), groups,
            bias, padding_mode)

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
            _triple(padding), _triple(dilation), False, _triple(0), groups,
            bias, padding_mode)

    def forward(self, input):
        return cplx.conv3d(input, self.weight, self.bias,
                           self.stride, self.padding, self.dilation,
                           self.groups, self.padding_mode)


class CplxConvTransposeNd(CplxConvNd):
    r"""An almost verbatim copy of `_ConvTransposeNd` from torch/nn/modules/conv.py"""
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups,
                 bias, padding_mode):
        if padding_mode not in ('zeros', 'circular'):
            raise ValueError('Only "zeros" or "circular" padding mode are supported for {}'\
                             .format(self.__class__.__name__))

        super().__init__(
            in_channels, out_channels, _triple(kernel_size), _triple(stride),
            _triple(padding), _triple(dilation), transposed, output_padding,
            groups, bias, padding_mode)

    def _output_padding(self, input, output_size, stride, padding, kernel_size):
        # type: (Tensor, Optional[List[int]], List[int], List[int], List[int]) -> List[int]
        if output_size is None:
            ret = _single(self.output_padding)  # converting to list if was not already
        else:
            k = input.dim() - 2
            if len(output_size) == k + 2:
                output_size = output_size[2:]
            if len(output_size) != k:
                raise ValueError(
                    "output_size must have {} or {} elements (got {})"
                    .format(k, k + 2, len(output_size)))

            min_sizes = []
            max_sizes = []
            for d in range(k):
                dim_size = ((input.size(d + 2) - 1) * stride[d] -
                            2 * padding[d] + kernel_size[d])
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError((
                        "requested an output size of {}, but valid sizes range "
                        "from {} to {} (for an input of {})").format(
                            output_size, min_sizes, max_sizes, input.size()[2:]))

            res = []
            for d in range(k):
                res.append(output_size[d] - min_sizes[d])

            ret = res
        return ret


class CplxConvTranspose1d(CplxConvTransposeNd):
    r"""Complex 1D transposed convolution.

    See torch.nn.ConvTranspose1d for reference on the input dimensions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, output_padding=0, groups=1,
                 bias=None, padding_mode="zeros"):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        output_padding = _single(output_padding)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, True, output_padding, groups,
                         bias, padding_mode)

    def forward(self, input, output_size=None):
        if self.padding_mode not in ('zeros', 'circular'):
            raise ValueError(
                'Only `zeros` or `circular` padding mode are supported for CplxConvTranspose1d'
            )
        output_padding = self._output_padding(input, output_size, self.stride,
                                              self.padding, self.kernel_size)

        return cplx.conv_transpose1d(input, self.weight, self.bias, self.stride,
                                     self.padding, output_padding, self.groups,
                                     self.dilation, self.padding_mode)


class CplxConvTranspose2d(CplxConvTransposeNd):
    r"""Complex 2D transposed convolution.

    See torch.nn.ConvTranspose2d for reference on the input dimensions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, output_padding=0, groups=1,
                 bias=None, padding_mode="zeros"):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, True, output_padding, groups,
                         bias, padding_mode)

    def forward(self, input, output_size=None):
        if self.padding_mode not in ('zeros', 'circular'):
            raise ValueError(
                'Only `zeros` or `circular` padding mode are supported for CplxConvTranspose2d'
            )
        output_padding = self._output_padding(input, output_size, self.stride,
                                              self.padding, self.kernel_size)

        return cplx.conv_transpose2d(input, self.weight, self.bias, self.stride,
                                     self.padding, output_padding, self.groups,
                                     self.dilation, self.padding_mode)


class CplxConvTranspose3d(CplxConvTransposeNd):
    r"""Complex 3D transposed convolution.

    See torch.nn.ConvTranspose3d for reference on the input dimensions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, output_padding=0, groups=1,
                 bias=None, padding_mode="zeros"):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        output_padding = _triple(output_padding)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, True, output_padding, groups,
                         bias, padding_mode)

    def forward(self, input, output_size=None):
        if self.padding_mode not in ('zeros', 'circular'):
            raise ValueError(
                'Only `zeros` or `circular` padding mode are supported for CplxConvTranspose2d'
            )
        output_padding = self._output_padding(input, output_size, self.stride,
                                              self.padding, self.kernel_size)

        return cplx.conv_transpose3d(input, self.weight, self.bias, self.stride,
                                     self.padding, output_padding, self.groups,
                                     self.dilation, self.padding_mode)
