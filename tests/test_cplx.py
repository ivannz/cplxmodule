import pytest
import copy

import torch
import torch.nn.functional as F

import numpy as np

from cplxmodule import cplx


def cplx_allclose(input, other):
    return torch.allclose(input.real, other.real) and \
           torch.allclose(input.imag, other.imag)


def cplx_allclose_numpy(input, other):
    other = np.asarray(other)
    return (
        torch.allclose(input.real, torch.from_numpy(other.real))
        and torch.allclose(input.imag, torch.from_numpy(other.imag))
    )


@pytest.fixture
def random_state():
    return np.random.RandomState(None)  # (1249563438)


def test_creation(random_state):
    a = random_state.randn(5, 5, 200) + 1j * random_state.randn(5, 5, 200)
    p = cplx.Cplx(torch.from_numpy(a.real), torch.from_numpy(a.imag))

    assert len(a) == len(p)
    assert np.allclose(p.numpy(), a)

    a = random_state.randn(5, 5, 200) + 0j
    p = cplx.Cplx(torch.from_numpy(a.real))

    assert len(a) == len(p)
    assert np.allclose(p.numpy(), a)

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
    assert np.allclose(p.numpy(), np.zeros(p.shape))

    p = cplx.Cplx.ones(10, 12, 31)
    assert np.allclose(p.numpy(), np.ones(p.shape))


def test_type_tofrom_numpy(random_state):
    a = random_state.randn(10, 32, 64) + 1j * random_state.randn(10, 32, 64)
    b = random_state.randn(10, 64, 40) + 1j * random_state.randn(10, 64, 40)

    p = cplx.Cplx(torch.from_numpy(a.real), torch.from_numpy(a.imag))
    q = cplx.Cplx(torch.from_numpy(b.real), torch.from_numpy(b.imag))

    assert cplx_allclose(cplx.Cplx.from_numpy(a), p)
    assert cplx_allclose(cplx.Cplx.from_numpy(b), q)

    assert np.allclose(p.numpy(), a)
    assert np.allclose(q.numpy(), b)


def test_arithmetic_unary(random_state):
    a = random_state.randn(10, 20, 5) + 1j * random_state.randn(10, 20, 5)
    p = cplx.Cplx.from_numpy(a)

    assert cplx_allclose_numpy(p, a)
    assert np.allclose(abs(p), abs(a))
    assert np.allclose(p.angle, np.angle(a))
    assert cplx_allclose_numpy(p.conjugate(), a.conjugate())
    assert cplx_allclose_numpy(p.conj, a.conj())
    assert cplx_allclose_numpy(+p, +a)
    assert cplx_allclose_numpy(-p, -a)


def test_arithmetic_binary(random_state):
    # prepare data
    a = random_state.randn(10, 20, 5) + 1j * random_state.randn(10, 20, 5)
    b = random_state.randn(10, 20, 5) + 1j * random_state.randn(10, 20, 5)
    c = random_state.randn(10, 20, 5)

    p, q = cplx.Cplx.from_numpy(a), cplx.Cplx.from_numpy(b)
    r = torch.from_numpy(c)

    # test against numpy
    assert cplx_allclose_numpy(p + q, a + b)  # __add__ cplx-cplx
    assert cplx_allclose_numpy(p - q, a - b)  # __sub__ cplx-cplx
    assert cplx_allclose_numpy(p * q, a * b)  # __mul__ cplx-cplx
    assert cplx_allclose_numpy(p / q, a / b)  # __div__ cplx-cplx

    # okay with pythonic integer, real and complex constants
    for z in [int(10), float(3.1415), 1e-3 + 1e3j, -10j]:
        assert cplx_allclose_numpy(q + z, b + z)  # __add__ cplx-other
        assert cplx_allclose_numpy(q - z, b - z)  # __sub__ cplx-other
        assert cplx_allclose_numpy(q * z, b * z)  # __mul__ cplx-other
        assert cplx_allclose_numpy(q / z, b / z)  # __div__ cplx-other

        assert cplx_allclose_numpy(z + q, z + b)  # __radd__ other-cplx
        assert cplx_allclose_numpy(z - q, z - b)  # __rsub__ other-cplx
        assert cplx_allclose_numpy(z * q, z * b)  # __rmul__ other-cplx
        assert cplx_allclose_numpy(z / q, z / b)  # __rdiv__ other-cplx

    assert cplx_allclose_numpy(q + r, b + c)  # __add__ cplx-other
    assert cplx_allclose_numpy(q - r, b - c)  # __sub__ cplx-other
    assert cplx_allclose_numpy(q * r, b * c)  # __mul__ cplx-other
    assert cplx_allclose_numpy(q / r, b / c)  # __div__ cplx-other

    # _r*__ with types like torch.Tensor raised TypeError in pytroch<1.4
    assert cplx_allclose_numpy(r + q, c + b)  # __radd__ other-cplx
    assert cplx_allclose_numpy(r - q, c - b)  # __rsub__ other-cplx
    assert cplx_allclose_numpy(r * q, c * b)  # __rmul__ other-cplx
    assert cplx_allclose_numpy(r / q, c / b)  # __rdiv__ other-cplx


def test_arithmetic_inplace(random_state):
    a = random_state.randn(10, 20, 5) + 1j * random_state.randn(10, 20, 5)
    p = cplx.Cplx.from_numpy(a)

    n = cplx.Cplx.zeros(*a.shape, dtype=p.real.dtype, device=p.real.device)
    m = np.zeros_like(a)

    # test inplace __i*__
    n += p ; m += a
    assert cplx_allclose_numpy(n, m)

    n *= p ; m *= a
    assert cplx_allclose_numpy(n, m)

    n -= p ; m -= a
    assert cplx_allclose_numpy(n, m)

    n /= p ; m /= a
    assert cplx_allclose_numpy(n, m)

    with pytest.raises(RuntimeError, match=r"The expanded size of the tensor"):
        n[1:] @= p[0].t()

    assert cplx_allclose_numpy(n[0, :5] @ p[0, 0], m[0, :5] @ a[0, 0])


def test_algebraic_functions(random_state):
    a = random_state.randn(10, 20, 5) + 1j * random_state.randn(10, 20, 5)
    p = cplx.Cplx.from_numpy(a)

    assert cplx_allclose_numpy(cplx.exp(p), np.exp(a))
    assert cplx_allclose_numpy(cplx.log(p), np.log(a))

    assert cplx_allclose_numpy(cplx.sin(p), np.sin(a))
    assert cplx_allclose_numpy(cplx.cos(p), np.cos(a))
    assert cplx_allclose_numpy(cplx.tan(p), np.tan(a))

    assert cplx_allclose_numpy(cplx.sinh(p), np.sinh(a))
    assert cplx_allclose_numpy(cplx.cosh(p), np.cosh(a))
    assert cplx_allclose_numpy(cplx.tanh(p), np.tanh(a))


def test_slicing(random_state):
    a = random_state.randn(10, 20, 5) + 1j * random_state.randn(10, 20, 5)
    p = cplx.Cplx.from_numpy(a)

    for i in range(a.shape[0]):
        assert cplx_allclose_numpy(p[i], a[i])

    for i in range(a.shape[1]):
        assert cplx_allclose_numpy(p[::2, i], a[::2, i])

    for i in range(a.shape[1]):
        assert cplx_allclose_numpy(p[1::3, i], a[1::3, i])

    for i in range(a.shape[2]):
        assert cplx_allclose_numpy(p[..., i], a[..., i])

    with pytest.raises(IndexError):
        p[10], p[2, ..., -10]


def test_iteration(random_state):
    a = random_state.randn(10, 20, 5) + 1j * random_state.randn(10, 20, 5)
    p = cplx.Cplx.from_numpy(a)

    for u, v in zip(a, p):
        assert cplx_allclose_numpy(v, u)

    for u, v in zip(reversed(a), reversed(p)):
        assert cplx_allclose_numpy(v, u)

    assert cplx_allclose_numpy(reversed(p), a[::-1].copy())

    for u, v in zip(a[-1], p[-1]):
        assert cplx_allclose_numpy(v, u)

    for u, v in zip(a[..., -1], p[..., -1]):
        assert cplx_allclose_numpy(v, u)


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
        assert cplx_allclose_numpy(p[i] @ q[i], a[i] @ b[i])

    assert cplx_allclose_numpy(p @ q, a @ b)


def test_linear_transform(random_state):
    a = random_state.randn(5, 5, 200) + 1j * random_state.randn(5, 5, 200)
    L = random_state.randn(321, 200) + 1j * random_state.randn(321, 200)
    b = random_state.randn(321) + 1j * random_state.randn(321)

    p = cplx.Cplx.from_numpy(a)
    U = cplx.Cplx.from_numpy(L)
    q = cplx.Cplx.from_numpy(b)

    base = np.dot(a, L.T)
    assert cplx_allclose_numpy(cplx.linear(p, U, None), base)
    assert cplx_allclose_numpy(cplx.linear_naive(p, U, None), base)
    assert cplx_allclose_numpy(cplx.linear_cat(p, U, None), base)
    assert cplx_allclose_numpy(cplx.linear_3m(p, U, None), base)

    assert cplx_allclose_numpy(cplx.linear(p, U, q), base + b)
    assert cplx_allclose_numpy(cplx.linear_naive(p, U, q), base + b)
    assert cplx_allclose_numpy(cplx.linear_cat(p, U, q), base + b)
    assert cplx_allclose_numpy(cplx.linear_3m(p, U, q), base + b)


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
    cc = cplx.convnd_naive(F.conv1d, cx, cw).real
    assert torch.allclose(cc, torch.from_numpy(nn))

    # check pure C: `correlate` uses conjugation
    #  https://docs.scipy.org/doc/numpy/reference/generated/numpy.correlate.html
    x = random_state.randn(5, 12, 31) + 1j * random_state.randn(5, 12, 31)
    w = random_state.randn(1, 12, 7) + 1j * random_state.randn(1, 12, 7)

    nn = correlate(x, w.conj(), mode="valid")

    cx, cw = map(cplx.Cplx.from_numpy, [x, w])
    cc = cplx.convnd_naive(F.conv1d, cx, cw)
    assert cplx_allclose_numpy(cc, nn)

    cc = cplx.convnd_quick(F.conv1d, cx, cw)
    assert cplx_allclose_numpy(cc, nn)

    cc = cplx.convnd_3m(F.conv1d, cx, cw)
    assert cplx_allclose_numpy(cc, nn)

    a = random_state.randn(5, 12, 200) + 1j * random_state.randn(5, 12, 200)
    L = random_state.randn(14, 12, 7) + 1j * random_state.randn(14, 12, 7)

    p, U = map(cplx.Cplx.from_numpy, [a, L])
    assert cplx_allclose(cplx.convnd_quick(F.conv1d, p, U),
                         cplx.convnd_naive(F.conv1d, p, U))

    assert cplx_allclose(cplx.convnd_quick(F.conv1d, p, U, stride=5),
                         cplx.convnd_naive(F.conv1d, p, U, stride=5))

    assert cplx_allclose(cplx.convnd_quick(F.conv1d, p, U, padding=2),
                         cplx.convnd_naive(F.conv1d, p, U, padding=2))

    assert cplx_allclose(cplx.convnd_quick(F.conv1d, p, U, dilation=3),
                         cplx.convnd_naive(F.conv1d, p, U, dilation=3))

    assert cplx_allclose(cplx.convnd_3m(F.conv1d, p, U),
                         cplx.convnd_naive(F.conv1d, p, U))

    assert cplx_allclose(cplx.convnd_3m(F.conv1d, p, U, stride=5),
                         cplx.convnd_naive(F.conv1d, p, U, stride=5))

    assert cplx_allclose(cplx.convnd_3m(F.conv1d, p, U, padding=2),
                         cplx.convnd_naive(F.conv1d, p, U, padding=2))

    assert cplx_allclose(cplx.convnd_3m(F.conv1d, p, U, dilation=3),
                         cplx.convnd_naive(F.conv1d, p, U, dilation=3))


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
    cc = cplx.convnd_naive(F.conv2d, cx, cw).real
    assert torch.allclose(cc, torch.from_numpy(nn))

    # check pure C: `correlate` uses conjugation
    #  https://docs.scipy.org/doc/numpy/reference/generated/numpy.correlate.html
    x = random_state.randn(5, 12, 31, 47) + 1j * random_state.randn(5, 12, 31, 47)
    w = random_state.randn(1, 12, 7, 11) + 1j * random_state.randn(1, 12, 7, 11)

    nn = correlate(x, w.conj(), mode="valid")

    cx, cw = map(cplx.Cplx.from_numpy, [x, w])
    cc = cplx.convnd_naive(F.conv2d, cx, cw)
    assert cplx_allclose_numpy(cc, nn)

    cc = cplx.convnd_quick(F.conv2d, cx, cw)
    assert cplx_allclose_numpy(cc, nn)

    cc = cplx.convnd_3m(F.conv2d, cx, cw)
    assert cplx_allclose_numpy(cc, nn)

    a = random_state.randn(5, 12, 41, 39) + 1j * random_state.randn(5, 12, 41, 39)
    L = random_state.randn(14, 12, 7, 6) + 1j * random_state.randn(14, 12, 7, 6)

    p, U = map(cplx.Cplx.from_numpy, [a, L])
    assert cplx_allclose(cplx.convnd_quick(F.conv2d, p, U),
                         cplx.convnd_naive(F.conv2d, p, U))

    assert cplx_allclose(cplx.convnd_quick(F.conv2d, p, U, stride=5),
                         cplx.convnd_naive(F.conv2d, p, U, stride=5))

    assert cplx_allclose(cplx.convnd_quick(F.conv2d, p, U, padding=2),
                         cplx.convnd_naive(F.conv2d, p, U, padding=2))

    assert cplx_allclose(cplx.convnd_quick(F.conv2d, p, U, dilation=3),
                         cplx.convnd_naive(F.conv2d, p, U, dilation=3))

    assert cplx_allclose(cplx.convnd_3m(F.conv2d, p, U),
                         cplx.convnd_naive(F.conv2d, p, U))

    assert cplx_allclose(cplx.convnd_3m(F.conv2d, p, U, stride=5),
                         cplx.convnd_naive(F.conv2d, p, U, stride=5))

    assert cplx_allclose(cplx.convnd_3m(F.conv2d, p, U, padding=2),
                         cplx.convnd_naive(F.conv2d, p, U, padding=2))

    assert cplx_allclose(cplx.convnd_3m(F.conv2d, p, U, dilation=3),
                         cplx.convnd_naive(F.conv2d, p, U, dilation=3))


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
    cc = cplx.convnd_naive(F.conv3d, cx, cw).real
    assert torch.allclose(cc, torch.from_numpy(nn))

    # check pure C: `correlate` uses conjugation
    #  https://docs.scipy.org/doc/numpy/reference/generated/numpy.correlate.html
    x = random_state.randn(5, 12, 13, 21, 17) + 1j * random_state.randn(5, 12, 13, 21, 17)
    w = random_state.randn(1, 12, 7, 11, 5) + 1j * random_state.randn(1, 12, 7, 11, 5)

    nn = correlate(x, w.conj(), mode="valid")

    cx, cw = map(cplx.Cplx.from_numpy, [x, w])
    cc = cplx.convnd_naive(F.conv3d, cx, cw)
    assert cplx_allclose_numpy(cc, nn)

    cc = cplx.convnd_quick(F.conv3d, cx, cw)
    assert cplx_allclose_numpy(cc, nn)

    cc = cplx.convnd_3m(F.conv3d, cx, cw)
    assert cplx_allclose_numpy(cc, nn)

    a = random_state.randn(5, 12, 14, 19, 27) + 1j * random_state.randn(5, 12, 14, 19, 27)
    L = random_state.randn(14, 12, 3, 4, 5) + 1j * random_state.randn(14, 12, 3, 4, 5)

    p, U = map(cplx.Cplx.from_numpy, [a, L])
    assert cplx_allclose(cplx.convnd_quick(F.conv3d, p, U),
                         cplx.convnd_naive(F.conv3d, p, U))

    assert cplx_allclose(cplx.convnd_quick(F.conv3d, p, U, stride=5),
                         cplx.convnd_naive(F.conv3d, p, U, stride=5))

    assert cplx_allclose(cplx.convnd_quick(F.conv3d, p, U, padding=2),
                         cplx.convnd_naive(F.conv3d, p, U, padding=2))

    assert cplx_allclose(cplx.convnd_quick(F.conv3d, p, U, dilation=3),
                         cplx.convnd_naive(F.conv3d, p, U, dilation=3))

    assert cplx_allclose(cplx.convnd_3m(F.conv3d, p, U),
                         cplx.convnd_naive(F.conv3d, p, U))

    assert cplx_allclose(cplx.convnd_3m(F.conv3d, p, U, stride=5),
                         cplx.convnd_naive(F.conv3d, p, U, stride=5))

    assert cplx_allclose(cplx.convnd_3m(F.conv3d, p, U, padding=2),
                         cplx.convnd_naive(F.conv3d, p, U, padding=2))

    assert cplx_allclose(cplx.convnd_3m(F.conv3d, p, U, dilation=3),
                         cplx.convnd_naive(F.conv3d, p, U, dilation=3))


def test_bilinear_transform(random_state):
    a = random_state.randn(5, 5, 41) + 1j * random_state.randn(5, 5, 41)
    z = random_state.randn(5, 5, 23) + 1j * random_state.randn(5, 5, 23)
    L = random_state.randn(321, 41, 23) + 1j * random_state.randn(321, 41, 23)
    b = random_state.randn(321) + 1j * random_state.randn(321)

    p, r = cplx.Cplx.from_numpy(a), cplx.Cplx.from_numpy(z)
    U = cplx.Cplx.from_numpy(L)
    q = cplx.Cplx.from_numpy(b)

    base = np.einsum("bsi, bsj, fij ->bsf", a.conj(), z, L)
    assert cplx_allclose_numpy(
        cplx.bilinear(p, r, U, None, conjugate=True),
        base)
    assert cplx_allclose_numpy(
        cplx.bilinear_naive(p, r, U, None, conjugate=True),
        base)
    assert cplx_allclose_numpy(
        cplx.bilinear_cat(p, r, U, None, conjugate=True),
        base)

    assert cplx_allclose_numpy(
        cplx.bilinear(p, r, U, q, conjugate=True),
        base + b)
    assert cplx_allclose_numpy(
        cplx.bilinear_naive(p, r, U, q, conjugate=True),
        base + b)
    assert cplx_allclose_numpy(
        cplx.bilinear_cat(p, r, U, q, conjugate=True),
        base + b)

    base = np.einsum("bsi, bsj, fij ->bsf", a, z, L)
    assert cplx_allclose_numpy(
        cplx.bilinear(p, r, U, None, conjugate=False),
        base)
    assert cplx_allclose_numpy(
        cplx.bilinear_naive(p, r, U, None, conjugate=False),
        base)
    assert cplx_allclose_numpy(
        cplx.bilinear_cat(p, r, U, None, conjugate=False),
        base)

    assert cplx_allclose_numpy(
        cplx.bilinear(p, r, U, q, conjugate=False),
        base + b)
    assert cplx_allclose_numpy(
        cplx.bilinear_naive(p, r, U, q, conjugate=False),
        base + b)
    assert cplx_allclose_numpy(
        cplx.bilinear_cat(p, r, U, q, conjugate=False),
        base + b)


def test_type_conversion(random_state):
    a = random_state.randn(5, 5, 200) + 1j * random_state.randn(5, 5, 200)
    b = np.stack([a.real, a.imag], axis=-1).reshape(*a.shape[:-1], -1)

    p = cplx.Cplx.from_numpy(a)
    q = cplx.from_real(torch.from_numpy(b))

    # from cplx to double-real (interleaved)
    assert torch.allclose(cplx.to_real(p), torch.from_numpy(b))
    assert torch.allclose(cplx.to_real(q), torch.from_numpy(b))

    # from double-real to cplx
    assert cplx_allclose(q, p)
    assert cplx_allclose_numpy(q, a)

    assert cplx.Cplx(-1 + 1j).item() == -1 + 1j

    with pytest.raises(ValueError, match="one element tensors"):
        p.item()

    assert a[0, 0, 0] == p[0, 0, 0].item()

    # concatenated to cplx
    for dim in [0, 1, 2]:
        stacked = torch.cat([torch.from_numpy(a.real),
                             torch.from_numpy(a.imag)], dim=dim)

        q = cplx.from_concatenated_real(stacked, dim=dim)
        assert cplx_allclose_numpy(q, a)

    # cplx to concatenated
    for dim in [0, 1, 2]:
        q = cplx.to_concatenated_real(cplx.Cplx.from_numpy(a), dim=dim)

        stacked = np.concatenate([a.real, a.imag], axis=dim)
        assert torch.allclose(q, torch.from_numpy(stacked))

    # cplx to interleaved
    for dim in [0, 1, 2]:
        q = cplx.from_interleaved_real(
                cplx.to_interleaved_real(
                    cplx.Cplx.from_numpy(a),
                    flatten=True, dim=dim
                ), dim=dim)

        assert cplx_allclose_numpy(q, a)


def test_enisum(random_state):
    a = random_state.randn(10, 32, 64) + 1j * random_state.randn(10, 32, 64)
    b = random_state.randn(10, 64, 32) + 1j * random_state.randn(10, 64, 32)
    c = random_state.randn(10, 10, 10) + 1j * random_state.randn(10, 10, 10)

    p, q, r = map(cplx.Cplx.from_numpy, (a, b, c))

    assert cplx_allclose_numpy(cplx.einsum("ijk", r), np.einsum("ijk", c))

    equations = ["iij", "iji", "jii", "iii"]
    for eq in equations:
        assert cplx_allclose_numpy(cplx.einsum(eq, r), np.einsum(eq, c))
        with pytest.raises(RuntimeError, match="but the sizes don't match"):
            cplx.einsum(eq, p)

    equations = [
        "ijk, ikj", "ijk, ikj -> ij", "ijk, ikj -> k",
        "ijk, lkj", "ijk, lkj -> li", "ijk, lkj -> lji",
        "ijk, lkp",
        ]
    for eq in equations:
        assert cplx_allclose_numpy(cplx.einsum(eq, p, q),
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
        assert cplx_allclose_numpy(
            cplx.cat(tr_tensors, dim=n),
            np.concatenate(np_tensors, axis=n))

    for n in [0, 1, 2, 3]:
        assert cplx_allclose_numpy(
            cplx.stack(tr_tensors, dim=n),
            np.stack(np_tensors, axis=n))

    np_tensors = [
        random_state.randn(3, 7) + 1j * random_state.randn(3, 7),
        random_state.randn(5, 7) + 1j * random_state.randn(5, 7),
    ]

    for n in [0, 1, 2]:
        with pytest.raises(RuntimeError, match="each tensor to be equal size"):
            cplx.stack(map(cplx.Cplx.from_numpy, np_tensors), dim=n)

    with pytest.raises(RuntimeError, match="Sizes of tensors must match"):
        cplx.cat(map(cplx.Cplx.from_numpy, np_tensors), dim=1)


def test_view(random_state):
    a = random_state.randn(10, 32, 64) + 1j * random_state.randn(10, 32, 64)

    p = cplx.Cplx.from_numpy(a)
    p_real = torch.from_numpy(a.real)
    p_imag = torch.from_numpy(a.imag)

    # Should be equivalent to reshape if in contiguous memory
    assert cplx_allclose(p.view(5, 64, 64), p.reshape(5, 64, 64))
    assert cplx_allclose(p.view(5, 64, 64),
                         cplx.Cplx.from_numpy(a.reshape(5, 64, 64)))

    # Real and Imaginary parts should remain the same
    assert cplx_allclose_numpy(p.view(5, 64, 64), a.reshape(5, 64, 64))
    assert torch.allclose(p.view(5, 64, 64).real, p_real.view(5, 64, 64))
    assert torch.allclose(p.view(5, 64, 64).imag, p_imag.view(5, 64, 64))

    # Should fail if not contiguous in memory
    with pytest.raises(RuntimeError, match="Use .reshape"):
        p.permute(2, 0, 1).view(5, 64, 64)

    # Should fail if new size does not match
    size = p.shape[0] * p.shape[1] * p.shape[2]
    with pytest.raises(RuntimeError, match="invalid for input of size {}".format(size)):
        p.view(4, 64, 64)


@pytest.mark.parametrize('shape', [
    [10, 12, 31], [7, 3], [17], [2, 5, 6, 3, 7, 2, 4]
])
def test_tensor_shape_dim_size(random_state, shape):
    # size, shape, and dim properties
    p = cplx.Cplx.empty(shape)

    assert p.dim() == len(shape) and p.shape == torch.Size(shape)
    assert all(p.size(i) == n for i, n in enumerate(shape))

    with pytest.raises(TypeError, match="invalid combination of arguments"):
        p.size(1, 2)

    with pytest.raises(RuntimeError, match="look up dimensions by name"):
        p.size(None)


def test_scalar_shape_dim_size(random_state):
    # size, shape, and dim properties
    p = cplx.Cplx(1+1j)
    assert p.dim() == 0 and p.shape == p.size() == torch.Size([])

    with pytest.raises(IndexError, match="tensor has no dimensions"):
        p.size(1)

    with pytest.raises(TypeError, match="invalid combination of arguments"):
        p.size(1, 2)

    with pytest.raises(RuntimeError, match="look up dimensions by name"):
        p.size(None)


def test_deepcopy(random_state):
    """Test shallow and deep copy support."""
    a = random_state.randn(5, 5, 200) + 1j * random_state.randn(5, 5, 200)
    p = cplx.Cplx.from_numpy(a)

    q = copy.copy(p)
    assert np.allclose(q.numpy(), p.numpy())
    assert q.real is p.real and q.imag is p.imag

    q = copy.deepcopy(p)
    assert np.allclose(q.numpy(), p.numpy())
    assert q.real is not p.real and q.imag is not p.imag


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
