import math
import torch

from .base import CplxToCplx, CplxParameter
from .base import BaseCplxToReal
from .. import init
from ... import cplx


class CplxIdentity(torch.nn.Identity, CplxToCplx):
    pass


class CplxReal(BaseCplxToReal):
    def forward(self, input):
        return input.real


class CplxImag(BaseCplxToReal):
    def forward(self, input):
        return input.imag


class CplxLinear(CplxToCplx):
    r"""Complex linear transform:
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

        self.weight = CplxParameter(cplx.Cplx.empty(out_features, in_features))

        if bias:
            self.bias = CplxParameter(cplx.Cplx.empty(out_features))
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


class CplxBilinear(CplxToCplx):
    r"""Complex bilinear transform:
    $$
        F
        \colon \mathbb{C}^{\ldots \times d_0}
                    \times  \mathbb{C}^{\ldots \times d_1}
                \to \mathbb{C}^{\ldots \times d_2}
        \colon (u, v) \mapsto (u^\top A_j v)_{j=1}^{d_2}
        \,. $$
    """
    def __init__(self, in1_features, in2_features, out_features, bias=True,
                 conjugate=True):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features

        self.weight = CplxParameter(cplx.Cplx.empty(
            out_features, in1_features, in2_features))

        if bias:
            self.bias = CplxParameter(cplx.Cplx.empty(out_features))
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


class CplxPhaseShift(CplxToCplx):
    r"""A learnable complex phase shift
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
        return cplx.phaseshift(input, self.phi)
