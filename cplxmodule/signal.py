import torch

from .cplx import Cplx, cplx_cat
from .layers import CplxToCplx

from .layers import is_cplx_to_cplx


class CplxMultichannelGainLayer(CplxToCplx):
    r"""
    Complex modulus multichannel gain layer:
    $$
        F
        \colon \mathbb{C}^n \to \mathbb{C}^{C \times n}
        \colon z \mapsto \Bigl( g_j(\lvert z \rvert) \odot z \Bigr)_{j=1}^C
        \,, $$
    where $g_j\colon \mathbb{R}^n \to \mathbb{K}^n$ is the gain network of the
    $j$-th channel, $\mathbb{K} = \mathbb{R}$ or $\mathbb{C}$.

    The layer takes in `... x n_in` complex input and applies complex modulus
    gain defined via a real-to-real or real-to-complex gain network, which maps
    `... x n_in` to `... x [C * n_in]` or `... x C x n_in`.

    The output dimension of the gain function must be a multiple of $n$, i.e.
    be able to be reshaped into `C \times n`.
    """
    def __init__(self, gain, flatten=False, squared=False):
        super().__init__()

        self.gain = gain
        self.flatten, self.squared = flatten, squared

    def forward(self, input):
        # compute the modulus gain
        modulus = abs(input)
        gain = self.gain(modulus**2 if self.squared else modulus)

        *head, n_features = input.shape
        try:
            gain = gain.reshape(*head, -1, n_features)

        except RuntimeError as e:
            tail = tuple(gain.shape[len(head):])
            raise ValueError(f"""`gain` {tail} is incompatible with the """
                             f"""input complex tensor {n_features}.""") from e

        # apply the gain
        output = input.apply(torch.unsqueeze, -2) * gain
        if self.flatten:
            return output.reshape(*head, -1)

        return output


class CplxProjectionGainLayer(CplxMultichannelGainLayer):
    r"""Complex modulus gain layer.

    Parameters
    ----------
    gain : torch.nn.Module
        The gain network, which takes real values and returns the multipliers.
    projection : CplxToCplx module
        The complex-valued projection operator.

    Details
    -------
    Let $\mathbb{K} = \mathbb{R}$ or $\mathbb{C}$. The $\mathbb{K}$-hump-gain
    dense net has the following funcitonal form:
    $$
        F
        \colon \mathbb{C}^n \to \mathbb{C}^m
        \colon z \mapsto L \biggl(
            \Bigl( g_j(\lvert z \rvert) \odot z \Bigr)_{j=0}^C
        \biggr)
        \,, $$
    with $g_j\colon \mathbb{R}^n \to \mathbb{K}^n$ being the gain networks
    of each of $C$ channels, and $L \colon \mathbb{C}^{(1+C)\times n} \to
    \mathbb{C}^{m}$ -- a linear map. By design $g_0$ is `pass-through`, i.e.
    $g_0(\cdot) = \mathbf{1} \in \mathbb{K}^n$. The linear operator $L$
    operates on the amplified concatenated complex input, but can be set to
    pass-through mode, which makes this layer into `CplxMultichannelGainLayer`
    with `flatten=True`.
    """
    def __init__(self, gain, projection=None, squared=False):
        super().__init__(gain, flatten=True, squared=squared)

        assert projection is None or is_cplx_to_cplx(projection)
        self.projection = projection

    def forward(self, input):
        channels = super().forward(input)

        # concatenate and project (if necessary)
        output = cplx_cat([channels, input], dim=-1)

        return output if self.projection is None else self.projection(output)
