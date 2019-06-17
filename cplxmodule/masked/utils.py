import torch
import torch.sparse

from ..cplx import Cplx


def torch_sparse_tensor(indices, data, shape):
    if data.dtype is torch.float:
        return torch.sparse.FloatTensor(indices, data, shape)

    elif data.dtype is torch.double:
        return torch.sparse.DoubleTensor(indices, data, shape)

    raise TypeError(f"""Unsupported dtype `{data.dtype}`""")


def torch_sparse_linear(input, weight, bias=None):
    *head, n_features = input.shape
    x = input.reshape(-1, n_features)

    out = torch.sparse.mm(weight, x.t()).t()
    out = out.reshape(*head, weight.shape[0])

    if bias is not None:
        out += bias

    return out


def torch_sparse_cplx_linear(input, weight, bias=None):
    #  W z + c = (U + i V) (u + i v) + \Re c + i \Im c
    #          = (U u + \Re c - V v) + i (V u + \Im c + U v)
    real = torch_sparse_linear(input.real, weight.real, None) \
        - torch_sparse_linear(input.imag, weight.imag, None)
    imag = torch_sparse_linear(input.real, weight.imag, None) \
        + torch_sparse_linear(input.imag, weight.real, None)

    output = Cplx(real, imag)
    if isinstance(bias, Cplx):
        output += bias

    return output
