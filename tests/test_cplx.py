import pytest

import torch
import torch.nn.functional as F

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

    # _r*__ with types like torch.Tensor raised TypeError in pytroch<1.4
    assert_allclose_cplx(c + b, r + q)  # __radd__ other-cplx
    assert_allclose_cplx(c - b, r - q)  # __rsub__ other-cplx
    assert_allclose_cplx(c * b, r * q)  # __rmul__ other-cplx
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

    assert_allclose_cplx(np.exp(a), cplx.exp(p))
    assert_allclose_cplx(np.log(a), cplx.log(p))

    assert_allclose_cplx(np.sin(a), cplx.sin(p))
    assert_allclose_cplx(np.cos(a), cplx.cos(p))
    assert_allclose_cplx(np.tan(a), cplx.tan(p))

    assert_allclose_cplx(np.sinh(a), cplx.sinh(p))
    assert_allclose_cplx(np.cosh(a), cplx.cosh(p))
    assert_allclose_cplx(np.tanh(a), cplx.tanh(p))


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

    base = np.dot(a, L.T)
    assert_allclose_cplx(base, cplx.linear(p, U, None))
    assert_allclose_cplx(base, cplx.linear_naive(p, U, None))
    assert_allclose_cplx(base, cplx.linear_cat(p, U, None))
    assert_allclose_cplx(base, cplx.linear_3m(p, U, None))

    assert_allclose_cplx(base + b, cplx.linear(p, U, q))
    assert_allclose_cplx(base + b, cplx.linear_naive(p, U, q))
    assert_allclose_cplx(base + b, cplx.linear_cat(p, U, q))
    assert_allclose_cplx(base + b, cplx.linear_3m(p, U, q))


def test_conv1d_transform(random_state):
    # torch's real convolutions correspond to numpy `correlate`
    from scipy.signal import correlate

    x = random_state.randn(5, 12, 31)
    w = random_state.randn(1, 12, 7)

    # check pure R
    tt = F.conv1d(torch.from_numpy(x), torch.from_numpy(w)).numpy()
    nn = correlate(x, w, mode="valid")
    assert np.allclose(tt, nn)

    # check R - C embedding
    nn = correlate(x + 0j, w + 0j, mode="valid").real
    assert np.allclose(tt, nn)

    cx, cw = map(cplx.Cplx.from_numpy, [x, w])
    cc = cplx.convnd_naive(F.conv1d, cx, cw).numpy().real
    assert np.allclose(cc, nn)

    # check pure C: `correlate` uses conjugation
    #  https://docs.scipy.org/doc/numpy/reference/generated/numpy.correlate.html
    x = random_state.randn(5, 12, 31) + 1j * random_state.randn(5, 12, 31)
    w = random_state.randn(1, 12, 7) + 1j * random_state.randn(1, 12, 7)

    nn = correlate(x, w.conj(), mode="valid")

    cx, cw = map(cplx.Cplx.from_numpy, [x, w])
    cc = cplx.convnd_naive(F.conv1d, cx, cw).numpy()
    assert np.allclose(cc, nn)

    cc = cplx.convnd_quick(F.conv1d, cx, cw).numpy()
    assert np.allclose(cc, nn)

    a = random_state.randn(5, 12, 200) + 1j * random_state.randn(5, 12, 200)
    L = random_state.randn(14, 12, 7) + 1j * random_state.randn(14, 12, 7)

    p, U = map(cplx.Cplx.from_numpy, [a, L])
    assert_allclose_cplx(cplx.convnd_naive(F.conv1d, p, U),
                         cplx.convnd_quick(F.conv1d, p, U))

    assert_allclose_cplx(cplx.convnd_naive(F.conv1d, p, U, stride=5),
                         cplx.convnd_quick(F.conv1d, p, U, stride=5))

    assert_allclose_cplx(cplx.convnd_naive(F.conv1d, p, U, padding=2),
                         cplx.convnd_quick(F.conv1d, p, U, padding=2))

    assert_allclose_cplx(cplx.convnd_naive(F.conv1d, p, U, dilation=3),
                         cplx.convnd_quick(F.conv1d, p, U, dilation=3))


def test_conv2d_transform(random_state):
    # torch's real convolutions correspond to numpy `correlate`
    from scipy.signal import correlate

    x = random_state.randn(5, 12, 31, 47)
    w = random_state.randn(1, 12, 7, 11)

    # check pure R
    tt = F.conv2d(torch.from_numpy(x), torch.from_numpy(w)).numpy()
    nn = correlate(x, w, mode="valid")
    assert np.allclose(tt, nn)

    # check R - C embedding
    nn = correlate(x + 0j, w + 0j, mode="valid").real
    assert np.allclose(tt, nn)

    cx, cw = map(cplx.Cplx.from_numpy, [x, w])
    cc = cplx.convnd_naive(F.conv2d, cx, cw).numpy().real
    assert np.allclose(cc, nn)

    # check pure C: `correlate` uses conjugation
    #  https://docs.scipy.org/doc/numpy/reference/generated/numpy.correlate.html
    x = random_state.randn(5, 12, 31, 47) + 1j * random_state.randn(5, 12, 31, 47)
    w = random_state.randn(1, 12, 7, 11) + 1j * random_state.randn(1, 12, 7, 11)

    nn = correlate(x, w.conj(), mode="valid")

    cx, cw = map(cplx.Cplx.from_numpy, [x, w])
    cc = cplx.convnd_naive(F.conv2d, cx, cw).numpy()
    assert np.allclose(cc, nn)

    cc = cplx.convnd_quick(F.conv2d, cx, cw).numpy()
    assert np.allclose(cc, nn)

    a = random_state.randn(5, 12, 41, 39) + 1j * random_state.randn(5, 12, 41, 39)
    L = random_state.randn(14, 12, 7, 6) + 1j * random_state.randn(14, 12, 7, 6)

    p, U = map(cplx.Cplx.from_numpy, [a, L])
    assert_allclose_cplx(cplx.convnd_naive(F.conv2d, p, U),
                         cplx.convnd_quick(F.conv2d, p, U))

    assert_allclose_cplx(cplx.convnd_naive(F.conv2d, p, U, stride=5),
                         cplx.convnd_quick(F.conv2d, p, U, stride=5))

    assert_allclose_cplx(cplx.convnd_naive(F.conv2d, p, U, padding=2),
                         cplx.convnd_quick(F.conv2d, p, U, padding=2))

    assert_allclose_cplx(cplx.convnd_naive(F.conv2d, p, U, dilation=3),
                         cplx.convnd_quick(F.conv2d, p, U, dilation=3))


def test_conv3d_transform(random_state):
    # torch's real convolutions correspond to numpy `correlate`
    from scipy.signal import correlate

    x = random_state.randn(5, 12, 13, 21, 17)
    w = random_state.randn(1, 12, 7, 11, 5)

    # check pure R
    tt = F.conv3d(torch.from_numpy(x), torch.from_numpy(w)).numpy()
    nn = correlate(x, w, mode="valid")
    assert np.allclose(tt, nn)

    # check R - C embedding
    nn = correlate(x + 0j, w + 0j, mode="valid").real
    assert np.allclose(tt, nn)

    cx, cw = map(cplx.Cplx.from_numpy, [x, w])
    cc = cplx.convnd_naive(F.conv3d, cx, cw).numpy().real
    assert np.allclose(cc, nn)

    # check pure C: `correlate` uses conjugation
    #  https://docs.scipy.org/doc/numpy/reference/generated/numpy.correlate.html
    x = random_state.randn(5, 12, 13, 21, 17) + 1j * random_state.randn(5, 12, 13, 21, 17)
    w = random_state.randn(1, 12, 7, 11, 5) + 1j * random_state.randn(1, 12, 7, 11, 5)

    nn = correlate(x, w.conj(), mode="valid")

    cx, cw = map(cplx.Cplx.from_numpy, [x, w])
    cc = cplx.convnd_naive(F.conv3d, cx, cw).numpy()
    assert np.allclose(cc, nn)

    cc = cplx.convnd_quick(F.conv3d, cx, cw).numpy()
    assert np.allclose(cc, nn)

    a = random_state.randn(5, 12, 14, 19, 27) + 1j * random_state.randn(5, 12, 14, 19, 27)
    L = random_state.randn(14, 12, 3, 4, 5) + 1j * random_state.randn(14, 12, 3, 4, 5)

    p, U = map(cplx.Cplx.from_numpy, [a, L])
    assert_allclose_cplx(cplx.convnd_naive(F.conv3d, p, U),
                         cplx.convnd_quick(F.conv3d, p, U))

    assert_allclose_cplx(cplx.convnd_naive(F.conv3d, p, U, stride=5),
                         cplx.convnd_quick(F.conv3d, p, U, stride=5))

    assert_allclose_cplx(cplx.convnd_naive(F.conv3d, p, U, padding=2),
                         cplx.convnd_quick(F.conv3d, p, U, padding=2))

    assert_allclose_cplx(cplx.convnd_naive(F.conv3d, p, U, dilation=3),
                         cplx.convnd_quick(F.conv3d, p, U, dilation=3))


def test_bilinear_transform(random_state):
    a = random_state.randn(5, 5, 41) + 1j * random_state.randn(5, 5, 41)
    z = random_state.randn(5, 5, 23) + 1j * random_state.randn(5, 5, 23)
    L = random_state.randn(321, 41, 23) + 1j * random_state.randn(321, 41, 23)
    b = random_state.randn(321) + 1j * random_state.randn(321)

    p, r = cplx.Cplx.from_numpy(a), cplx.Cplx.from_numpy(z)
    U = cplx.Cplx.from_numpy(L)
    q = cplx.Cplx.from_numpy(b)

    base = np.einsum("bsi, bsj, fij ->bsf", a.conj(), z, L)
    assert_allclose_cplx(base, cplx.bilinear(p, r, U, None, conjugate=True))
    assert_allclose_cplx(base, cplx.bilinear_naive(p, r, U, None, conjugate=True))
    assert_allclose_cplx(base, cplx.bilinear_cat(p, r, U, None, conjugate=True))

    assert_allclose_cplx(base + b, cplx.bilinear(p, r, U, q, conjugate=True))
    assert_allclose_cplx(base + b, cplx.bilinear_naive(p, r, U, q, conjugate=True))
    assert_allclose_cplx(base + b, cplx.bilinear_cat(p, r, U, q, conjugate=True))

    base = np.einsum("bsi, bsj, fij ->bsf", a, z, L)
    assert_allclose_cplx(base, cplx.bilinear(p, r, U, None, conjugate=False))
    assert_allclose_cplx(base, cplx.bilinear_naive(p, r, U, None, conjugate=False))
    assert_allclose_cplx(base, cplx.bilinear_cat(p, r, U, None, conjugate=False))

    assert_allclose_cplx(base + b, cplx.bilinear(p, r, U, q, conjugate=False))
    assert_allclose_cplx(base + b, cplx.bilinear_naive(p, r, U, q, conjugate=False))
    assert_allclose_cplx(base + b, cplx.bilinear_cat(p, r, U, q, conjugate=False))


def test_type_conversion(random_state):
    a = random_state.randn(5, 5, 200) + 1j * random_state.randn(5, 5, 200)
    b = np.stack([a.real, a.imag], axis=-1).reshape(*a.shape[:-1], -1)

    p = cplx.Cplx.from_numpy(a)
    q = cplx.from_real(torch.from_numpy(b))

    # from cplx to double-real (interleaved)
    assert_allclose(b, cplx.to_real(p))
    assert_allclose(b, cplx.to_real(q))

    # from double-real to cplx
    assert_allclose_cplx(p, q)
    assert_allclose_cplx(a, q)

    assert cplx.Cplx(-1 + 1j).item() == -1 + 1j

    with pytest.raises(ValueError, match="one element tensors"):
        p.item()

    assert a[0, 0, 0] == p[0, 0, 0].item()

    # concatenated to cplx
    for dim in [0, 1, 2]:
        stacked = torch.cat([torch.from_numpy(a.real),
                             torch.from_numpy(a.imag)], dim=dim)

        q = cplx.from_concatenated_real(stacked, dim=dim)
        assert_allclose_cplx(a, q)

    # cplx to concatenated
    for dim in [0, 1, 2]:
        q = cplx.to_concatenated_real(cplx.Cplx.from_numpy(a), dim=dim)

        stacked = np.concatenate([a.real, a.imag], axis=dim)
        assert_allclose(q.numpy(), stacked)

    # cplx to interleaved
    for dim in [0, 1, 2]:
        q = cplx.from_interleaved_real(
                cplx.to_interleaved_real(
                    cplx.Cplx.from_numpy(a),
                    flatten=True, dim=dim
                ), dim=dim)

        assert_allclose(q.numpy(), a)


def test_enisum(random_state):
    a = random_state.randn(10, 32, 64) + 1j * random_state.randn(10, 32, 64)
    b = random_state.randn(10, 64, 32) + 1j * random_state.randn(10, 64, 32)
    c = random_state.randn(10, 10, 10) + 1j * random_state.randn(10, 10, 10)

    p, q, r = map(cplx.Cplx.from_numpy, (a, b, c))

    assert_allclose(cplx.einsum("ijk", r).numpy(), np.einsum("ijk", c))

    equations = ["iij", "iji", "jii", "iii"]
    for eq in equations:
        assert_allclose(cplx.einsum(eq, r).numpy(), np.einsum(eq, c))
        with pytest.raises(RuntimeError, match="dimension does not match"):
            cplx.einsum(eq, p)

    equations = [
        "ijk, ikj", "ijk, ikj -> ij", "ijk, ikj -> k",
        "ijk, lkj", "ijk, lkj -> li", "ijk, lkj -> lji",
        "ijk, lkp",
        ]
    for eq in equations:
        assert_allclose(cplx.einsum(eq, p, q).numpy(),
                        np.einsum(eq, a, b))

    with pytest.raises(RuntimeError, match="does not support more"):
        cplx.einsum("...", p, q, r)


def test_cat_stack(random_state):
    with pytest.raises(RuntimeError, match="a non-empty"):
        cplx.stack([], dim=0)

    np_tensors = 10 * [
        random_state.randn(5, 3, 7) + 1j * random_state.randn(5, 3, 7)
    ]
    tr_tensors = [*map(cplx.Cplx.from_numpy, np_tensors)]

    for n in [0, 1, 2]:
        assert_allclose(cplx.cat(tr_tensors, dim=n).numpy(),
                        np.concatenate(np_tensors, axis=n))

    for n in [0, 1, 2, 3]:
        assert_allclose(cplx.stack(tr_tensors, dim=n).numpy(),
                        np.stack(np_tensors, axis=n))

    np_tensors = [
        random_state.randn(3, 7) + 1j * random_state.randn(3, 7),
        random_state.randn(5, 7) + 1j * random_state.randn(5, 7),
    ]

    for n in [0, 1, 2]:
        with pytest.raises(RuntimeError, match="Sizes of tensors must match"):
            cplx.stack(map(cplx.Cplx.from_numpy, np_tensors), dim=n)

    with pytest.raises(RuntimeError, match="Sizes of tensors must match"):
        cplx.cat(map(cplx.Cplx.from_numpy, np_tensors), dim=1)


@pytest.mark.skip(reason="not implemented")
def test_splitting(random_state):
    # chunk, split, unbind
    assert False


@pytest.mark.skip(reason="not implemented")
def test_mutating(random_state):
    # squeeze, unsqueeze, reshape, t, transpose, permute
    assert False


@pytest.mark.skip(reason="not implemented")
def test_indexing(random_state):
    # take, narrow
    assert False
