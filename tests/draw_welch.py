import torch
import numpy as np
import matplotlib.pyplot as plt

from cplxmodule.spectrum import pwelch, fftshift
from scipy.signal import welch

random_state = np.random.RandomState(479321083)

# https://www.mathworks.com/help/signal/ref/pwelch.html#btulskp-6
fs = 1000.
tt = np.r_[:5 * fs - 1] / fs

np_x = 1j * random_state.randn(2, len(tt)) * .01
np_x += random_state.randn(2, len(tt)) * .01

np_x += np.cos(2 * np.pi * 100 * tt)[np.newaxis]
np_x = np_x.astype(np.complex)

tr_x = torch.tensor(np_x.view(np.float).reshape(*np_x.shape, 2))
tr_x.requires_grad = False

tr_window = torch.hamming_window(500, periodic=False, dtype=tr_x.dtype)

tr_ff, tr_px = pwelch(tr_x, 1, tr_window, fs=fs,
                      scaling="density", n_overlap=300)
np_ff, np_px = welch(np_x, fs=fs, axis=-1, window=tr_window.numpy(),
                     nfft=None, nperseg=None, scaling="density",
                     noverlap=300, detrend=False, return_onesided=False)

plt.semilogy(np.fft.fftshift(np_ff), np.fft.fftshift(np_px[0]),
             label="scipy", alpha=0.5)
plt.semilogy(fftshift(tr_ff).numpy(), fftshift(tr_px[0]).numpy().T,
             label="torch", alpha=0.5)
plt.legend()

plt.show()
