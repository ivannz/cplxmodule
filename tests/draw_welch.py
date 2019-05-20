import torch
import numpy as np
import matplotlib.pyplot as plt

from cplxmodule.utils.spectrum import pwelch, fftshift
from scipy.signal import welch


from matplotlib.ticker import EngFormatter

random_state = np.random.RandomState(479321083)

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

fig = plt.figure(figsize=(14, 5))

ax = fig.add_subplot(111, ylabel="Decibels (dB / Hz)", xlabel="Frequency")
ax.xaxis.set_major_formatter(EngFormatter(unit="Hz"))

ax.semilogy(np.fft.fftshift(np_ff), np.fft.fftshift(np_px[0]),
            label="scipy", alpha=0.5)

ax.semilogy(fftshift(tr_ff).numpy(), fftshift(tr_px[0]).numpy().T,
            label="torch", alpha=0.5)

ax.axvspan(90, 110, alpha=0.25, color="green")
ax.axvline(100, lw=2, linestyle=":", color="black", alpha=0.5)

ax.legend()

plt.show()
