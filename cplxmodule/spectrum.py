import torch
import numpy as np


def fix_dim(dim, n_dim):
    r"""For the given dimensionality make sure the `axis` is nonnegative."""
    axis = (n_dim + dim) if dim < 0 else dim
    if not 0 <= axis < n_dim:
        raise ValueError(f"""Dimension {dim} is out of range for {n_dim}.""")

    return axis


def window_view(x, dim, size, stride, at=None):
    r"""Returns a sliding window view into the tensor.

    Similar to `torch.unfold()`, but the window dimensions of size `size`
    is placed right after `dim` (by default), and not appended.

    Arguments
    ---------
    x : torch.tensor
        Time series of measurement values
    dim : int
        Axis along which the periodogram is computed, i.e. ``dim=-1``.
    size : int
        The size of the sliding windows.
    stride : int
        The step between two sliding windows.
    at : int, optional
        The dimension at which to put the slice of each window.

    Returns
    -------
    x_view : torch.tensor
        The view into a sliding window. The returned tensor is not, in
        general, contiguous in memory.
    """
    if size <= 0:
        raise ValueError(f"""`size` must be a positive integer.""")

    if stride < 0:
        raise ValueError(f"""`stride` must be a nonnegative integer.""")

    dim = fix_dim(dim, x.dim())
    if x.shape[dim] < size:
        raise ValueError(f"""`x` at dim {dim} is too short ({x.shape[dim]}) """
                         f"""for this window size ({size}).""")

    if at is None:
        at = dim + 1
    at = fix_dim(at, x.dim() + 1)

    # compute new shape and strides
    shape, strides = list(x.size()), list(x.stride())
    strided_size = ((shape[dim] - size + 1) + stride - 1) // stride

    # new shape and stride structure
    shape_view = shape[:dim] + [strided_size] + shape[dim+1:]
    shape_view.insert(at, size)

    strides_view = strides[:dim] + [strides[dim] * stride] + strides[dim+1:]
    strides_view.insert(at, strides[dim])

    # differentiable strided view
    return torch.as_strided(x, shape_view, strides_view)


def pwelch(x, dim, window, fs=1., scaling="density", n_overlap=None):
    r"""Estimate power spectral density using Welch's method.

    Arguments
    ---------
    x : torch.tensor
        Time series of measurement values
    dim : int
        Axis along which the periodogram is computed, i.e. ``dim=-1``.
    window : array_like
        The weights to be used directly as the window and its length
        determines the length of the FFT used.
    fs : float, optional
        Sampling frequency of the `x` time series. Defaults to 1.0.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the power spectral density ('density')
        where `Pxx` has units of V**2/Hz and computing the power
        spectrum ('spectrum') where `Pxx` has units of V**2, if `x`
        is measured in V and `fs` is measured in Hz. Defaults to
        'density'.
    n_noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``n_noverlap = len(window) // 2``. Defaults to `None`.

    Returns
    -------
    f : torch.tensor
        The 1d tensor of sample frequencies.
    Pxx : torch.tensor
        Power spectral density or power spectrum of `x`.

    Compatibility
    -------------
    This torch implementation is designed to be compatible with `welch`
    from `scipy.signal` with the following fixed parameters:
    ``nfft=None, nperseg=None, detrend=False, return_onesided=False``.

    See Also
    --------
    scipy.signal.welch: the reference for this function.
    """
    if scaling not in ("density", "spectrum"):
        raise ValueError(f"""Unrecognized `scaling` value {scaling}""")

    if x.shape[-1] != 2:
        raise TypeError("""The last dimension of the input must be 2:"""
                        """x[..., 0] is real and x[..., 1] is imaginary.""")

    dim = fix_dim(dim, x.dim())
    if not 0 <= dim < x.dim() - 1:
        raise ValueError("""The last dimension of the input cannot contain """
                         """the signal.""")

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
    if scaling == "density":
        scale = fs * torch.sum(window**2)
    elif scaling == "spectrum":
        scale = torch.sum(window)**2

    # used to have `/ x_window.shape[dim]`
    Pxx = torch.sum(fft**2, dim=-1).mean(dim=dim) / scale

    # 5. get the frequencies
    freq = np.fft.fftfreq(n_window, 1. / fs)
    freq = torch.tensor(freq, dtype=x.dtype, device=x.device)
    return freq, Pxx


def fftshift(x, dim=-1):
    r"""Shift the zero-frequency component to the center of the spectrum.

    Parameters
    ----------
    x : torch.tensor
        Input tensor.
    dim : int, optional
        Dimension over which to shift.  Default is the last dimension.

    Returns
    -------
    y : torch.tensor
        The shifted tensor.

    Compatibility
    -------------
    This torch implementation is designed to be compatible with `fftshift`
    from `numpy.fft`.

    See Also
    --------
    numpy.fft.fftshift: the reference for this function.
    """
    dim = fix_dim(dim, x.dim())
    return torch.roll(x, x.shape[dim] // 2, dim)


def bandwidth_power(x, fs, bands, dim=-2, n_overlap=None,
                    nperseg=None, scaling="density"):
    r"""Compute the total power of a batch of signals in each band.

    Uses Welch's method (see `scipy.signal.welch`) with Hamming window
    to estimate the power spectrum.

    Arguments
    ---------
    x : torch.tensor
        Time series of measurement values
    fs : float
        Sampling frequency of the `x` time series.
    bands : iterable of tuples
        An iterable object of tuples, each containing the frequency band
        in a tuple `(lo, hi)` for the lower and upper end of the band
        respectively.
    dim : int
        Axis along which the periodogram is computed, i.e. ``dim=-2``.
    n_noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``n_noverlap = len(window) // 2``. Defaults to `None`.
    nperseg : int, optional
        Length of each segment to use for spectrum estimation. Defaults to
        None, in which case is set equal to the length of the signal in `x`.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the power spectral density ('density')
        where `Pxx` has units of V**2/Hz and computing the power
        spectrum ('spectrum') where `Pxx` has units of V**2, if `x`
        is measured in V and `fs` is measured in Hz. Defaults to
        'density'.

    Returns
    -------
    f : torch.tensor
        The 1d tensor of sample frequencies.
    Pxx : torch.tensor
        Power spectral density or power spectrum of `x`.
    band_pwr : tensor
        The tensor of shape `(... x len(bands))` with the per-band power
        distribution.
    """
    dim = fix_dim(dim, x.dim())
    if nperseg is None:
        nperseg = x.shape[dim]

    # 1. Welch
    window = torch.hamming_window(nperseg, periodic=False,
                                  dtype=x.dtype, device=x.device)
    ff, px = pwelch(x, dim, window, fs=fs, scaling=scaling,
                    n_overlap=n_overlap)

    ff, px = fftshift(ff), fftshift(px, dim=dim)
    if not bands:
        # this needs to return a tensor that has the expected shape
        return ff, px, torch.empty(*px.shape[:dim], *px.shape[dim+1:], 0,
                                   dtype=px.dtype, device=px.device)

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
                                      n_overlap=0, scaling="spectrum")

    return channel[..., [0]], channel[..., 1:]


# ## TESTS

def test_window(random_state=None):
    from scipy.signal.windows import hamming

    n_window = 1024

    np_window = hamming(n_window, False).astype(np.float64)
    tr_window = torch.hamming_window(n_window, periodic=True,
                                     dtype=torch.float64)

    assert np.allclose(np_window, tr_window.numpy())

    np_window = hamming(n_window, True).astype(np.float64)
    tr_window = torch.hamming_window(n_window, periodic=False,
                                     dtype=torch.float64)

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

    assert np.allclose(window_view(tr_x, dim, size, stride, at=-1).numpy(),
                       tr_x.unfold(dim, size, stride).numpy())


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
