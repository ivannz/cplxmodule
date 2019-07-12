import torch
import pytest
import numpy as np

from numpy.testing import assert_allclose


@pytest.fixture
def random_state():
    return np.random.RandomState(None)  # (1249563438)


def test_complex_fft(random_state):
    shape, axis = (2, 3, 256, 2, 2), 2

    np_x = random_state.randn(*shape) + 1j * random_state.randn(*shape)
    tr_x = torch.tensor(np.stack([np_x.real, np_x.imag], axis=-1))

    np_fft = np.fft.fft(np_x, axis=axis)
    tr_fft = torch.fft(tr_x.transpose(axis, -2),
                       signal_ndim=1).transpose(axis, -2)

    assert_allclose(tr_fft[..., 0].numpy(), np_fft.real)
    assert_allclose(tr_fft[..., 1].numpy(), np_fft.imag)


def test_hamming_window(random_state=None):
    from scipy.signal.windows import hamming

    n_window = 1024

    np_window = hamming(n_window, False).astype(np.float64)
    tr_window = torch.hamming_window(n_window, periodic=True,
                                     dtype=torch.float64)

    assert_allclose(tr_window.numpy(), np_window)

    np_window = hamming(n_window, True).astype(np.float64)
    tr_window = torch.hamming_window(n_window, periodic=False,
                                     dtype=torch.float64)

    assert_allclose(tr_window.numpy(), np_window)


def test_pwelch(random_state):
    from cplxmodule.utils.spectrum import pwelch
    from scipy.signal import welch

    # https://www.mathworks.com/help/signal/ref/pwelch.html#btulskp-6
    fs = 1000.
    tt = np.r_[:5 * fs - 1] / fs

    shape = 2, len(tt)

    epsilon = random_state.randn(*shape) + 1j * random_state.randn(*shape)
    np_x = np.cos(2 * np.pi * 100 * tt)[np.newaxis] + epsilon * 0.01

    tr_x = torch.tensor(np.stack([np_x.real, np_x.imag], axis=-1))
    tr_x.requires_grad = False

    tr_window = torch.hamming_window(500, periodic=False, dtype=tr_x.dtype)

    tr_ff, tr_px = pwelch(tr_x, 1, tr_window, fs=fs,
                          scaling="density", n_overlap=300)
    np_ff, np_px = welch(np_x, fs=fs, axis=-1, window=tr_window.numpy(),
                         nfft=None, nperseg=None, scaling="density",
                         noverlap=300, detrend=False, return_onesided=False)

    assert_allclose(tr_px.numpy(), np_px)
    assert_allclose(tr_ff.numpy(), np_ff)

    tr_ff, tr_px = pwelch(tr_x, 1, tr_window, fs=fs,
                          scaling="spectrum", n_overlap=499)
    np_ff, np_px = welch(np_x, fs=fs, axis=-1, window=tr_window.numpy(),
                         nfft=None, nperseg=None, scaling="spectrum",
                         noverlap=499, detrend=False, return_onesided=False)

    assert_allclose(tr_px.numpy(), np_px)
    assert_allclose(tr_ff.numpy(), np_ff)
