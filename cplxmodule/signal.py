import torch

from .cplx import Cplx
from .layers import CplxToCplx

from .layers import is_cplx_to_cplx


class CplxMultichannelGainLayer(CplxToCplx):
    r"""
    Complex modulus multichannel gain layer:
    $$
        F
        \colon \mathbb{C}^d \to \mathbb{C}^{C \times d}
        \colon z \mapsto \Bigl(z \odot G_i(\lvert z \rvert)\Bigr)_{i=1}^C
        \,, $$
    where $G_i \colon [0, +\infty)^d \to \mathbb{C}^d$ is the complex modulus
    gain function of the $i$-th channel.

    The layer takes in `... x n_in` complex input and applies complex modulus
    gain defined via a real-to-real gain network, which maps `... x n_in` to
    `... x [C * n_in]` or `... x C x n_in`.

    The output of the gain function must be a mutliple of $d$, i.e. be able
    to be reshaped into `C \times d`.
    """
    def __init__(self, gain, flatten=False):
        super().__init__()

        self.gain = gain
        self.flatten = flatten

    def forward(self, input):
        """"""
        # compute the modulus gain
        gain = self.gain(abs(input))

        # reshape gain `... x C x n_in` and input = (re, im) `... x 1 x n_in`
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
    r"""
    Complex modulus gain layer:
    $$
        F
        \colon \mathbb{C}^d \to \mathbb{C}^d
        \colon z \mapsto A \Bigl(
                z ~\big \|~ \Bigl(z \odot G_i(\lvert z \rvert)\Bigr)_{i=1}^C
            \Bigr)
        \,, $$
    where $G_i \colon [0, +\infty)^d \to \mathbb{R}^d$ is the $i$-th complex
    modulus gain function and $A\colon \mathbb{C}^{[(1 + C) \times d]} \to
    \mathbb{C}^d$ is the projection operator, which operates on the amplified
    concatenated complex input.
    """
    def __init__(self, gain, projection):
        super().__init__(gain, flatten=True)

        assert is_cplx_to_cplx(projection)
        self.projection = projection

    def forward(self, input):
        channels = super().forward(input)

        # concatenate and project
        output = Cplx(
            torch.cat([channels.real, input.real], dim=-1),
            torch.cat([channels.imag, input.imag], dim=-1))

        return self.projection(output)
