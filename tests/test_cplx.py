import pytest

import torch

import numpy as np
from numpy.testing import assert_allclose


from cplxmodule import cplx


def assert_allclose_cplx(npy, cplx):
    # assert np.allclose(npy.real, cplx.real) and \
    #         np.allclose(npy.imag, cplx.imag)
    assert_allclose(cplx.real, npy.real)
    assert_allclose(cplx.imag, npy.imag)


@pytest.fixture
def random_state():
    return np.random.RandomState(None)  # (1249563438)


def test_arithmetic_unary(random_state):
    a = random_state.randn(10, 20, 5) + 1j * random_state.randn(10, 20, 5)
    p = cplx.Cplx(torch.from_numpy(a.real), torch.from_numpy(a.imag))

    assert_allclose_cplx(a, p)
    assert_allclose(abs(a), abs(p))
    assert_allclose(np.angle(a), p.angle)
    assert_allclose_cplx(a.conj(), p.conj)
    assert_allclose_cplx(+a, +p)
    assert_allclose_cplx(-a, -p)


def test_arithmetic_binary(random_state):
    # prepare data
    a = random_state.randn(10, 20, 5) + 1j * random_state.randn(10, 20, 5)
    b = random_state.randn(10, 20, 5) + 1j * random_state.randn(10, 20, 5)
    c = random_state.randn(10, 20, 5)

    p = cplx.Cplx(torch.from_numpy(a.real), torch.from_numpy(a.imag))
    q = cplx.Cplx(torch.from_numpy(b.real), torch.from_numpy(b.imag))
    r = torch.from_numpy(c)

    # test against numpy
    assert_allclose_cplx(a + b, p + q)  # __add__ cplx-cplx
    assert_allclose_cplx(a - b, p - q)  # __sub__ cplx-cplx
    assert_allclose_cplx(a * b, p * q)  # __mul__ cplx-cplx
    assert_allclose_cplx(a / b, p / q)  # __div__ cplx-cplx

    assert_allclose_cplx(b + c, q + r)  # __add__ cplx-other
    assert_allclose_cplx(b - c, q - r)  # __sub__ cplx-other
    assert_allclose_cplx(b * c, q * r)  # __mul__ cplx-other
    assert_allclose_cplx(b / c, q / r)  # __div__ cplx-other

    # okay with pythonic real numbers (int, float)
    assert_allclose_cplx(3.1415 + b, 3.1415 + q)  # __radd__ other-cplx
    assert_allclose_cplx(3.1415 - b, 3.1415 - q)  # __rsub__ other-cplx
    assert_allclose_cplx(3.1415 * b, 3.1415 * q)  # __rmul__ other-cplx
    assert_allclose_cplx(3.1415 / b, 3.1415 / q)  # __rdiv__ other-cplx

    assert_allclose_cplx(int(10) + b, int(10) + q)  # __radd__ other-cplx
    assert_allclose_cplx(int(10) - b, int(10) - q)  # __rsub__ other-cplx
    assert_allclose_cplx(int(10) * b, int(10) * q)  # __rmul__ other-cplx
    assert_allclose_cplx(int(10) / b, int(10) / q)  # __rdiv__ other-cplx

    # _r*__ with more complex types raises TypeError
    with pytest.raises(TypeError, match=r".*be Tensor, not Cplx.*"):
        assert_allclose_cplx(c + b, r + q)  # __radd__ other-cplx
        assert_allclose_cplx(c - b, r - q)  # __rsub__ other-cplx
        assert_allclose_cplx(c * b, r * q)  # __rmul__ other-cplx
        assert_allclose_cplx(c / b, r / q)  # __rdiv__ other-cplx


def test_algebraic_functions(random_state):
    a = random_state.randn(10, 20, 5) + 1j * random_state.randn(10, 20, 5)
    p = cplx.Cplx(torch.from_numpy(a.real), torch.from_numpy(a.imag))

    assert_allclose_cplx(np.exp(a), cplx.cplx_exp(p))
    assert_allclose_cplx(np.log(a), cplx.cplx_log(p))

    assert_allclose_cplx(np.sin(a), cplx.cplx_sin(p))
    assert_allclose_cplx(np.cos(a), cplx.cplx_cos(p))
    assert_allclose_cplx(np.tan(a), cplx.cplx_tan(p))

    assert_allclose_cplx(np.sinh(a), cplx.cplx_sinh(p))
    assert_allclose_cplx(np.cosh(a), cplx.cplx_cosh(p))
    assert_allclose_cplx(np.tanh(a), cplx.cplx_tanh(p))


def test_linear_transform(random_state):
    a = random_state.randn(5, 5, 200) + 1j * random_state.randn(5, 5, 200)
    L = random_state.randn(321, 200) + 1j * random_state.randn(321, 200)
    b = random_state.randn(321) + 1j * random_state.randn(321)

    p = cplx.Cplx(torch.from_numpy(a.real), torch.from_numpy(a.imag))
    U = cplx.Cplx(torch.from_numpy(L.real), torch.from_numpy(L.imag))
    q = cplx.Cplx(torch.from_numpy(b.real), torch.from_numpy(b.imag))

    assert_allclose_cplx(np.dot(a, L.T) + b, cplx.cplx_linear(p, U, q))


def test_type_conversion(random_state):
    a = random_state.randn(5, 5, 200) + 1j * random_state.randn(5, 5, 200)
    b = np.stack([a.real, a.imag], axis=-1).reshape(*a.shape[:-1], -1)

    p = cplx.Cplx(torch.from_numpy(a.real), torch.from_numpy(a.imag))
    q = cplx.real_to_cplx(torch.from_numpy(b))

    # from cplx to double-real
    assert_allclose(b, cplx.cplx_to_real(p))
    assert_allclose(b, cplx.cplx_to_real(q))

    # from double-real to cplx
    assert_allclose_cplx(p, q)
    assert_allclose_cplx(a, q)
