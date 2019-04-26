import torch
import torch.nn

import torch.nn.functional as F

from .base import CplxToCplx
from .base import real_to_cplx
from .base import cplx_to_real

from .cplx import cplx_phaseshift


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

        self.re = torch.nn.Linear(in_features, out_features, bias=bias)
        self.im = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        re, im = input

        # W = U + i V,  z = u + i v, c = c_re + i c_im:
        #  W z + c = (U + i V) (u + i v) + c_re + i c_im
        #          = (U u + c_re - V v) + i (V u + c_im + U v)
        u = self.re(re) - F.linear(im, self.im.weight, None)
        v = self.im(re) + F.linear(im, self.re.weight, None)

        return u, v


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
        re, im = input
        u = self.re(re) - F.conv1d(im, self.im.weight, None, self.im.stride,
                                   self.im.padding, self.im.dilation, self.im.groups)
        v = self.im(re) + F.conv1d(im, self.re.weight, None, self.re.stride,
                                   self.re.padding, self.re.dilation, self.re.groups)
        return u, v


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
        return tuple(map(super().forward, input))


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
        self.phi = torch.nn.Parameter(torch.randn(*dim) * 0.02)

    def forward(self, input):
        return cplx_phaseshift(input, self.phi)
