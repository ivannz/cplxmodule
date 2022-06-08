from .base import BaseRealToCplx
from .base import BaseCplxToReal

from ... import cplx


class InterleavedRealToCplx(BaseRealToCplx):
    r"""Reinterpret the last dimension as interleaved real and imaginary
    components of a complex tensor. The input tensor must have even number
    in the last dimension, and the output has all dimensions preserved but
    the last, which is halved and not squeezed.
    $$
        F
        \colon \mathbb{R}^{\ldots \times [d \times 2]}
                \to \mathbb{C}^{\ldots \times d}
        \colon x \mapsto \bigr(
            x_{2k} + i x_{2k+1}
        \bigl)_{k=0}^{d-1}
        \,. $$

    Inverts `CplxToInterleavedReal`.
    """

    def __init__(self, copy=False, dim=-1):
        super().__init__()
        self.copy, self.dim = copy, dim

    def forward(self, input):
        return cplx.from_interleaved_real(input, self.copy, self.dim)


class ConcatenatedRealToCplx(BaseRealToCplx):
    r"""Interpret the last dimension as a concatenation of real part and then
    imaginary component. Preserves all dimensions except for the last, which
    is halved and not squeezed.
    $$
        F
        \colon \mathbb{R}^{\ldots \times [2 \times d]}
                \to \mathbb{C}^{\ldots \times d}
        \colon x \mapsto \bigr(
            x_{k} + i x_{d + k}
        \bigl)_{k=0}^{d-1}
        \,. $$

    Inverts `CplxToConcatenatedReal`.
    """

    def __init__(self, copy=False, dim=-1):
        super().__init__()
        self.copy, self.dim = copy, dim

    def forward(self, input):
        return cplx.from_concatenated_real(input, self.copy, self.dim)


class CplxToInterleavedReal(BaseCplxToReal):
    r"""Represent a Cplx tensor in the interleaved format along the last
    dimension: in consecutive pairs of real and imaginary parts
    $$
        F
        \colon \mathbb{C}^{\ldots \times d}
                \to \mathbb{R}^{\ldots \times [d \times 2]}
        \colon u + i v \mapsto \bigl(u_\omega, v_\omega\bigr)_{\omega}
        \,. $$

    Inverts `InterleavedRealToCplx`.
    """

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return cplx.to_interleaved_real(input, True, self.dim)


class CplxToConcatenatedReal(BaseCplxToReal):
    r"""Represent a Cplx tensor in concatenated format along the last
    dimension: the whole real component followed by the whole imaginary part
    $$
        F
        \colon \mathbb{C}^{\ldots \times d}
                \to \mathbb{R}^{\ldots \times [2 \times d]}
        \colon u + i v \mapsto \bigl(u_\omega, v_\omega \bigr)_{\omega}
        \,. $$

    Inverts `ConcatenatedRealToCplx`.
    """

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return cplx.to_concatenated_real(input, None, self.dim)


class AsTypeCplx(BaseRealToCplx):
    r"""Interpret the tensor as a Cplx tensor having zero imaginary part
    (embeds $\mathbb{R} \hookrightarrow \mathbb{C}$):
    $$
        F
        \colon \mathbb{R}^{\ldots \times d}
                \to \mathbb{C}^{\ldots \times d}
        \colon x \mapsto x + 0 i
        \,. $$

    Inverts `nn.linear.CplxReal`.
    """

    def forward(self, input):
        return cplx.Cplx(input)


class TensorToCplx(BaseRealToCplx):
    r"""Interpret a tensor with the last dimension of size exactly 2, which
    represents the real and imaginary components of a complex tensor. All
    dimensions preserved but the last, which is dropped.
    $$
        F
        \colon \mathbb{R}^{\ldots \times 2}
                \to \mathbb{C}^{\ldots}
        \colon x \mapsto x_{\ldots 0} + i x_{\ldots 1}
        \,. $$

    Inverts `CplxToTensor`.
    """

    def forward(self, input):
        """input must be a , and may have
        arbitrary number of.
        """
        assert input.shape[-1] == 2
        return cplx.Cplx(input[..., 0], input[..., 1])


class CplxToTensor(BaseCplxToReal):
    r"""Represent a Cplx tensor in torch's complex tensor format with a new
    last dimension of size exactly 2, representing the real and imaginary
    components of complex numbers.
    $$
        F
        \colon \mathbb{C}^{\ldots}
                \to \mathbb{R}^{\ldots \times 2}
        \colon u + i v \mapsto \bigl(u_\omega, v_\omega \bigr)_{\omega}
        \,. $$

    Inverts `TensorToCplx`.
    """

    def forward(self, input):
        return cplx.to_interleaved_real(input, False, -1)
