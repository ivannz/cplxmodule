from .base import BaseRealToCplx
from .base import BaseCplxToReal

from ... import cplx


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
        return cplx.Cplx(input)
