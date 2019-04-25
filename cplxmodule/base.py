import torch
import torch.nn


def real_to_cplx(input, copy=True):
    """Map real tensor input `... x [D * 2]` to a pair (re, im) with dim `... x D`."""
    *head, n_features = input.shape
    assert (n_features & 1) == 0

    if copy:
        return input[..., 0::2].clone(), input[..., 1::2].clone()

    input = input.reshape(*head, -1, 2)
    return input[..., 0], input[..., 1]


def cplx_to_real(input, flatten=True):
    """Interleave the complex re-im pair into a real tensor."""
    # re, im = input
    input = torch.stack(input, dim=-1)
    if flatten:
        return input.flatten(-2)
    return input


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
    def forward(self, input):
        return cplx_to_real(input)


class CplxToCplx(torch.nn.Module):
    pass
