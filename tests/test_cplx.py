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


def test_creation(random_state):
    a = random_state.randn(5, 5, 200) + 1j * random_state.randn(5, 5, 200)
    p = cplx.Cplx(torch.from_numpy(a.real), torch.from_numpy(a.imag))

    assert len(a) == len(p)
    assert_allclose_cplx(a, p)

    a = random_state.randn(5, 5, 200) + 0j
    p = cplx.Cplx(torch.from_numpy(a.real))

    assert len(a) == len(p)
    assert_allclose_cplx(a, p)

    cplx.Cplx(0.0)
    cplx.Cplx(-1 + 1j)

    with pytest.raises(TypeError):
        cplx.Cplx(0)

    with pytest.raises(TypeError):
        cplx.Cplx(0, None)

    with pytest.raises(TypeError):
        cplx.Cplx(torch.from_numpy(a.real), 0)

    with pytest.raises(ValueError):
        cplx.Cplx(torch.ones(11, 10), torch.ones(10, 11))

    p = cplx.Cplx.empty(10, 12, 31, dtype=torch.float64)
    assert p.real.dtype == p.imag.dtype
    assert p.real.requires_grad == p.imag.requires_grad
    assert p.real.dtype == torch.float64
    assert not p.real.requires_grad

    p = cplx.Cplx.empty(10, 12, 31, requires_grad=True)
    assert p.real.dtype == p.imag.dtype
    assert p.real.requires_grad == p.imag.requires_grad
    assert p.real.dtype == torch.float32
    assert p.real.requires_grad

    p = cplx.Cplx.zeros(10, 12, 31)
    assert_allclose(p.numpy(), np.zeros(p.shape, dtype=np.complex64))

    p = cplx.Cplx.ones(10, 12, 31)
    assert_allclose(p.numpy(), np.ones(p.shape, dtype=np.complex64))


def test_type_tofrom_numpy(random_state):
    a = random_state.randn(10, 32, 64) + 1j * random_state.randn(10, 32, 64)
    b = random_state.randn(10, 64, 40) + 1j * random_state.randn(10, 64, 40)

    p = cplx.Cplx(torch.from_numpy(a.real), torch.from_numpy(a.imag))
    q = cplx.Cplx(torch.from_numpy(b.real), torch.from_numpy(b.imag))

    assert_allclose_cplx(p, cplx.Cplx.from_numpy(a))
    assert_allclose_cplx(q, cplx.Cplx.from_numpy(b))

    assert_allclose_cplx(a, p.numpy())
    assert_allclose_cplx(b, q.numpy())


def test_arithmetic_unary(random_state):
    a = random_state.randn(10, 20, 5) + 1j * random_state.randn(10, 20, 5)
    p = cplx.Cplx.from_numpy(a)

    assert_allclose_cplx(a, p)
    assert_allclose(abs(a), abs(p))
    assert_allclose(np.angle(a), p.angle)
    assert_allclose_cplx(a.conjugate(), p.conjugate())
    assert_allclose_cplx(a.conj(), p.conj)
    assert_allclose_cplx(+a, +p)
    assert_allclose_cplx(-a, -p)


def test_arithmetic_binary(random_state):
    # prepare data
    a = random_state.randn(10, 20, 5) + 1j * random_state.randn(10, 20, 5)
    b = random_state.randn(10, 20, 5) + 1j * random_state.randn(10, 20, 5)
    c = random_state.randn(10, 20, 5)

    p, q = cplx.Cplx.from_numpy(a), cplx.Cplx.from_numpy(b)
    r = torch.from_numpy(c)

    # test against numpy
    assert_allclose_cplx(a + b, p + q)  # __add__ cplx-cplx
    assert_allclose_cplx(a - b, p - q)  # __sub__ cplx-cplx
    assert_allclose_cplx(a * b, p * q)  # __mul__ cplx-cplx
    assert_allclose_cplx(a / b, p / q)  # __div__ cplx-cplx

    # okay with pythonic integer, real and complex constants
    for z in [int(10), float(3.1415), 1e-3 + 1e3j, -10j]:
        assert_allclose_cplx(b + z, q + z)  # __add__ cplx-other
        assert_allclose_cplx(b - z, q - z)  # __sub__ cplx-other
        assert_allclose_cplx(b * z, q * z)  # __mul__ cplx-other
        assert_allclose_cplx(b / z, q / z)  # __div__ cplx-other

        assert_allclose_cplx(z + b, z + q)  # __radd__ other-cplx
        assert_allclose_cplx(z - b, z - q)  # __rsub__ other-cplx
        assert_allclose_cplx(z * b, z * q)  # __rmul__ other-cplx
        assert_allclose_cplx(z / b, z / q)  # __rdiv__ other-cplx

    assert_allclose_cplx(b + c, q + r)  # __add__ cplx-other
    assert_allclose_cplx(b - c, q - r)  # __sub__ cplx-other
    assert_allclose_cplx(b * c, q * r)  # __mul__ cplx-other
    assert_allclose_cplx(b / c, q / r)  # __div__ cplx-other

    # _r*__ with types like torch.Tensor raises TypeError
    with pytest.raises(TypeError, match=r".*be Tensor, not Cplx.*"):
        assert_allclose_cplx(c + b, r + q)  # __radd__ other-cplx

    with pytest.raises(TypeError, match=r".*be Tensor, not Cplx.*"):
        assert_allclose_cplx(c - b, r - q)  # __rsub__ other-cplx

    with pytest.raises(TypeError, match=r".*be Tensor, not Cplx.*"):
        assert_allclose_cplx(c * b, r * q)  # __rmul__ other-cplx

    with pytest.raises(TypeError, match=r".*be Tensor, not Cplx.*"):
        assert_allclose_cplx(c / b, r / q)  # __rdiv__ other-cplx


def test_arithmetic_inplace(random_state):
    a = random_state.randn(10, 20, 5) + 1j * random_state.randn(10, 20, 5)
    p = cplx.Cplx.from_numpy(a)

    n = cplx.Cplx.zeros(*a.shape, dtype=p.real.dtype, device=p.real.device)
    m = np.zeros_like(a)

    # test inplace __i*__
    n += p; m += a
    assert_allclose_cplx(m, n)

    n *= p; m *= a
    assert_allclose_cplx(m, n)

    n -= p; m -= a
    assert_allclose_cplx(m, n)

    n /= p; m /= a
    assert_allclose_cplx(m, n)

    with pytest.raises(RuntimeError, match=r"The expanded size of the tensor"):
        n[1:] @= p[0].t()

    assert_allclose_cplx(m[0, :5] @ a[0, 0], n[0, :5] @ p[0, 0])


def test_algebraic_functions(random_state):
    a = random_state.randn(10, 20, 5) + 1j * random_state.randn(10, 20, 5)
    p = cplx.Cplx.from_numpy(a)

    assert_allclose_cplx(np.exp(a), cplx.cplx_exp(p))
    assert_allclose_cplx(np.log(a), cplx.cplx_log(p))

    assert_allclose_cplx(np.sin(a), cplx.cplx_sin(p))
    assert_allclose_cplx(np.cos(a), cplx.cplx_cos(p))
    assert_allclose_cplx(np.tan(a), cplx.cplx_tan(p))

    assert_allclose_cplx(np.sinh(a), cplx.cplx_sinh(p))
    assert_allclose_cplx(np.cosh(a), cplx.cplx_cosh(p))
    assert_allclose_cplx(np.tanh(a), cplx.cplx_tanh(p))


def test_slicing(random_state):
    a = random_state.randn(10, 20, 5) + 1j * random_state.randn(10, 20, 5)
    p = cplx.Cplx.from_numpy(a)

    for i in range(a.shape[0]):
        assert_allclose_cplx(a[i], p[i])

    for i in range(a.shape[1]):
        assert_allclose_cplx(a[::2, i], p[::2, i])

    for i in range(a.shape[1]):
        assert_allclose_cplx(a[1::3, i], p[1::3, i])

    for i in range(a.shape[2]):
        assert_allclose_cplx(a[..., i], p[..., i])

    with pytest.raises(IndexError):
        p[10], p[2, ..., -10]


def test_iteration(random_state):
    a = random_state.randn(10, 20, 5) + 1j * random_state.randn(10, 20, 5)
    p = cplx.Cplx.from_numpy(a)

    for u, v in zip(a, p):
        assert_allclose_cplx(u, v)

    for u, v in zip(reversed(a), reversed(p)):
        assert_allclose_cplx(u, v)

    assert_allclose_cplx(a[::-1], reversed(p))

    for u, v in zip(a[-1], p[-1]):
        assert_allclose_cplx(u, v)

    for u, v in zip(a[..., -1], p[..., -1]):
        assert_allclose_cplx(u, v)


def test_immutability(random_state):
    a = random_state.randn(10, 20, 5) + 1j * random_state.randn(10, 20, 5)
    p = cplx.Cplx.from_numpy(a)

    with pytest.raises(AttributeError, match=r"can't set attribute"):
        p.real = p.imag

    with pytest.raises(AttributeError, match=r"can't set attribute"):
        p.real += p.imag


def test_linear_matmul(random_state):
    a = random_state.randn(10, 32, 64) + 1j * random_state.randn(10, 32, 64)
    b = random_state.randn(10, 64, 40) + 1j * random_state.randn(10, 64, 40)

    p, q = cplx.Cplx.from_numpy(a), cplx.Cplx.from_numpy(b)

    for i in range(len(a)):
        assert_allclose_cplx(a[i] @ b[i], p[i] @ q[i])

    assert_allclose_cplx(a @ b, p @ q)


def test_linear_transform(random_state):
    a = random_state.randn(5, 5, 200) + 1j * random_state.randn(5, 5, 200)
    L = random_state.randn(321, 200) + 1j * random_state.randn(321, 200)
    b = random_state.randn(321) + 1j * random_state.randn(321)

    p = cplx.Cplx.from_numpy(a)
    U = cplx.Cplx.from_numpy(L)
    q = cplx.Cplx.from_numpy(b)

    assert_allclose_cplx(np.dot(a, L.T), cplx.cplx_linear(p, U, None))
    assert_allclose_cplx(np.dot(a, L.T) + b, cplx.cplx_linear(p, U, q))


def test_type_conversion(random_state):
    a = random_state.randn(5, 5, 200) + 1j * random_state.randn(5, 5, 200)
    b = np.stack([a.real, a.imag], axis=-1).reshape(*a.shape[:-1], -1)

    p = cplx.Cplx.from_numpy(a)
    q = cplx.real_to_cplx(torch.from_numpy(b))

    # from cplx to double-real
    assert_allclose(b, cplx.cplx_to_real(p))
    assert_allclose(b, cplx.cplx_to_real(q))

    # from double-real to cplx
    assert_allclose_cplx(p, q)
    assert_allclose_cplx(a, q)

    assert cplx.Cplx(-1 + 1j).item() == -1 + 1j

    with pytest.raises(ValueError, match="one element tensors"):
        p.item()

    assert a[0, 0, 0] == p[0, 0, 0].item()


def test_enisum(random_state):
    a = random_state.randn(10, 32, 64) + 1j * random_state.randn(10, 32, 64)
    b = random_state.randn(10, 64, 32) + 1j * random_state.randn(10, 64, 32)
    c = random_state.randn(10, 10, 10) + 1j * random_state.randn(10, 10, 10)

    p, q, r = map(cplx.Cplx.from_numpy, (a, b, c))

    assert_allclose(cplx.cplx_einsum("ijk", r).numpy(), np.einsum("ijk", c))

    equations = ["iij", "iji", "jii", "iii"]
    for eq in equations:
        assert_allclose(cplx.cplx_einsum(eq, r).numpy(), np.einsum(eq, c))
        with pytest.raises(RuntimeError, match="dimension does not match"):
            cplx.cplx_einsum(eq, p)

    equations = [
        "ijk, ikj", "ijk, ikj -> ij", "ijk, ikj -> k",
        "ijk, lkj", "ijk, lkj -> li", "ijk, lkj -> lji",
        "ijk, lkp",
        ]
    for eq in equations:
        assert_allclose(cplx.cplx_einsum(eq, p, q).numpy(),
                        np.einsum(eq, a, b))

    with pytest.raises(RuntimeError, match="does not support more"):
        cplx.cplx_einsum("...", p, q, r)
