import pytest

import torch
import numpy as np

from cplxmodule import Cplx
from cplxmodule.nn import init


def cplx_allclose_numpy(input, other):
    other = np.asarray(other)
    return (
        torch.allclose(input.real, torch.from_numpy(other.real))
        and torch.allclose(input.imag, torch.from_numpy(other.imag))
    )


@pytest.mark.parametrize('initializer', [
    init.cplx_kaiming_normal_,
    init.cplx_xavier_normal_,
    init.cplx_kaiming_uniform_,
    init.cplx_xavier_uniform_,
    init.cplx_trabelsi_standard_,
    init.cplx_trabelsi_independent_,
    init.cplx_uniform_independent_,
])
def test_initializer(initializer):
    initializer(Cplx.empty(500, 1250))
    initializer(Cplx.empty(1250, 500))
    initializer(Cplx.empty(32, 64, 3, 3))
    # with pytest.raises(ValueError, match="Fan in and fan out can not be computed"):
    #     initializer(Cplx.empty(32))


def test_cplx_trabelsi_independent_():
    # weight from an embeeding linear layer
    weight = Cplx.empty(500, 1250, dtype=torch.double)
    init.cplx_trabelsi_independent_(weight)

    mat = weight @ weight.conj.t()
    assert cplx_allclose_numpy(mat, np.diag(mat.real.diagonal()))

    # weight from a bottleneck linear layer
    weight = Cplx.empty(1250, 500, dtype=torch.double)
    init.cplx_trabelsi_independent_(weight)

    mat = weight.conj.t() @ weight
    assert cplx_allclose_numpy(mat, np.diag(mat.real.diagonal()))

    # weight from a generic 2d convolution
    weight = Cplx.empty(32, 64, 3, 3, dtype=torch.double)
    init.cplx_trabelsi_independent_(weight)

    weight = weight.reshape(weight.shape[:2].numel(), -1)

    mat = weight.conj.t() @ weight
    assert cplx_allclose_numpy(mat, np.diag(mat.real.diagonal()))

    # weight from a really small 2d convolution
    weight = Cplx.empty(3, 7, 5, 5, dtype=torch.double)
    init.cplx_trabelsi_independent_(weight)

    weight = weight.reshape(weight.shape[:2].numel(), -1)

    mat = weight @ weight.conj.t()
    assert cplx_allclose_numpy(mat, np.diag(mat.real.diagonal()))
