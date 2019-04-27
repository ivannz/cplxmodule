import torch
import numpy as np


def window_view(x, dim, size, stride):
    r"""
    Similar to `torch.unfold()`, but the window dimensions of size `size`
    is placed right after `dim`, and not appended.
    """

    # correct the dim
    dim = (x.dim() + dim) if dim < 0 else dim
    assert 0 <= dim < x.dim()

    if x.shape[dim] < size:
        raise ValueError(f"""`x` at dim {dim} is too short ({x.shape[dim]}) """
                         f"""for this window size ({size}).""")

    # compute new shape and strides
    shape, strides = x.size(), x.stride()
    strided_size = ((shape[dim] - size + 1) + stride - 1) // stride

    # number of full strides in dim
    shape = shape[:dim] + (strided_size, size) + shape[dim+1:]
    strides = strides[:dim] + (strides[dim] * stride,
                               strides[dim]) + strides[dim+1:]

    # differentiable strided view
    return torch.as_strided(x, shape, strides)


def pwelch(x, dim, window, fs=1., scaling="density", n_overlap=None):
    dim = (x.dim() + dim) if dim < 0 else dim
    assert 0 <= dim < x.dim() - 1

    n_window = len(window)
    if n_overlap is None:
        n_overlap = n_window // 2
    assert n_window > n_overlap

    # 1. make windowed view and dim-shuffle to the last but one dim
    x_window = window_view(x, dim, n_window, n_window - n_overlap)
    x_window = torch.transpose(x_window, dim + 1, -2)

    # 2. Apply window and compute the 1d-fft with `arbitrary number
    #   of leading batch dimensions`
    xw = torch.mul(x_window, window.unsqueeze(-1))
    fft = torch.fft(xw, signal_ndim=1, normalized=False)

    # 3. undo the dim-shuffle on the fft result
    fft = torch.transpose(fft, -2, dim + 1)

    # 4. compute the power spectrum with the proper scaling
    if scaling == 'density':
        scale = fs * torch.sum(window**2)
    elif scaling == 'spectrum':
        scale = torch.sum(window)**2
    # used to have `/ x_window.shape[dim]`
    pwr = torch.sum(fft**2, dim=-1).mean(dim=dim) / scale

    # 5. get the frequencies
    freq = np.fft.fftfreq(n_window, 1. / fs)
    return torch.tensor(freq, dtype=x.dtype), pwr


def fftshift(x, dim=-1):
    dim = (x.dim() + dim) if dim < 0 else dim
    assert 0 <= dim < x.dim()

    return torch.roll(x, x.shape[dim] // 2, dim)


def bandwidth_power(x, fs, bands, dim=-2, n_overlap=None, nperseg=None):
    dim = (x.dim() + dim) if dim < 0 else dim
    assert 0 <= dim < x.dim() - 1
    if nperseg is None:
        nperseg = x.shape[dim]

    # 1. Welch
    window = torch.hamming_window(nperseg, periodic=False, dtype=x.dtype)
    ff, px = pwelch(x, dim, window, fs=fs, scaling="spectrum",
                    n_overlap=n_overlap)

    ff, px = fftshift(ff), fftshift(px, dim=dim)
    if not bands:
        # this needs to return a tensor that has the expected shape
        return ff, px, None

    # 2. Compute power within each band
    channel, df = [], 1. / (nperseg * fs)
    for lo, hi in bands:
        index = torch.nonzero(ff.gt(lo) & ff.lt(hi))[:, 0]
        power = torch.index_select(px, dim, index)
        channel.append(power.sum(dim=dim))  # * df)
    # end for

    # 3. get the power in decibel
    return ff, px, 10 * torch.log10(torch.stack(channel, dim=-1))


def acpr_calc(signal, sample_rate, mcf, acf, mcb, acb, nperseg=None, dim=-2):
    r"""Calculate ACPR metric using pytorch

    Arguments
    ---------
    signal : tensor size = (..., T, 2)
        The complex signal to compute the per-channel power for.

    sample_rate : float
        The sampling frequency per unit time (Hz).

    mcf : float
        The frequency of the main channel in Hz.

    acf : arraylike (n_channels,)
        The frequencies of the adjacent channels in Hz.

    mcb : float
        The bandwidth of the main channel in Hz.

    acb : numeric or arraylike (n_channels,)
        The bandwidths of the adjacent channels in Hz.

    nperseg : int, or None
        The length of the window to use for power spectrum estimation.

    dim : int
        The dimension of over which the signal is recorded.

    Returns
    -------
    main_channel_power : tensor

    adjacent_channel_power : tensor
    """
    if isinstance(acb, (int, float)):
        acb = [acb] * len(acf)

    # form the bands
    bands = [(-0.5 * mcb + mcf, +0.5 * mcb + mcf)]
    for f, b in zip(acf, acb):
        bands.append((-0.5 * b + f + mcf, +0.5 * b + f + mcf))

    # compute the power in each band
    ff, px, channel = bandwidth_power(signal, sample_rate, bands,
                                      dim=dim, nperseg=nperseg,
                                      n_overlap=0)

    return channel[..., [0]], channel[..., 1:]


# ## TESTS

def test_window(random_state=None):
    from scipy.signal.windows import hamming

    n_window = 1024

    np_window = hamming(n_window, False).astype(np.float64)
    tr_window = torch.hamming_window(n_window, periodic=True, dtype=torch.float64)

    assert np.allclose(np_window, tr_window.numpy())

    np_window = hamming(n_window, True).astype(np.float64)
    tr_window = torch.hamming_window(n_window, periodic=False, dtype=torch.float64)

    assert np.allclose(np_window, tr_window.numpy())


def test_fft(random_state):
    shape, axis = (2, 3, 256, 2, 2), 2
    np_x = random_state.randn(*shape) + 1j * random_state.randn(*shape)
    tr_x = torch.tensor(np_x.view(np.float).reshape(*np_x.shape, 2))

    assert np.allclose(tr_x[..., 0].numpy(), np_x.real)
    assert np.allclose(tr_x[..., 1].numpy(), np_x.imag)

    np_fft = np.fft.fft(np_x, axis=axis)
    tr_fft = torch.fft(tr_x.transpose(axis, -2),
                       signal_ndim=1).transpose(axis, -2)

    assert np.allclose(tr_fft[..., 0].numpy(), np_fft.real)
    assert np.allclose(tr_fft[..., 1].numpy(), np_fft.imag)


def test_view(random_state):
    np_x = random_state.randn(2, 3, 1024, 2, 2)
    tr_x = torch.tensor(np_x)

    dim, size, stride = -3, 5, 2
    dim = (tr_x.dim() + dim) if dim < 0 else dim

    tr_x_view = window_view(tr_x, dim, size, stride)
    for i in range(tr_x_view.shape[dim]):
        slice_ = np.r_[i * stride:i * stride + size]
        a = tr_x_view.index_select(dim, torch.tensor(i)).squeeze(dim).numpy()
        b = tr_x.index_select(dim, torch.tensor(slice_)).numpy()
        assert np.allclose(a, b)


def test_welch(random_state):
    import matplotlib.pyplot as plt
    from scipy.signal import welch

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

    assert np.allclose(tr_px.numpy(), np_px)
    assert np.allclose(tr_ff.numpy(), np_ff)

    plt.semilogy(np.fft.fftshift(np_ff), np.fft.fftshift(np_px[0]),
                 label="scipy")
    plt.semilogy(fftshift(tr_ff).numpy(), fftshift(tr_px[0]).numpy().T,
                 label="torch")
    plt.legend()

    plt.show()


if __name__ == '__main__':

    random_state = np.random.RandomState(479321083)

    test_view(random_state)
    test_fft(random_state)
    test_window(random_state)
    test_welch(random_state)
