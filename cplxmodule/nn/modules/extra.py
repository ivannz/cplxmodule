import torch

from .base import CplxToCplx
from ... import cplx


class CplxDropout(torch.nn.Dropout2d, CplxToCplx):
    r"""Complex 1d dropout layer: simultaneous dropout on both real and
    imaginary parts.

    See torch.nn.Dropout1d for reference on the input dimensions and arguments.
    """
    def forward(self, input):
        *head, n_last = input.shape

        # shape -> [*shape, 2] : re-im are feature maps!
        tensor = torch.stack([input.real, input.imag], dim=-1)
        output = super().forward(tensor.reshape(-1, 1, 2))

        # [-1, 1, 2] -> [*head, n_last * 2]
        output = output.reshape(*head, -1)

        # [*head, n_last * 2] -> [*head, n_last]
        return cplx.from_interleaved_real(output, False, -1)
