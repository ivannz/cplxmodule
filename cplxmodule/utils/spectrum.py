import torch
import numpy as np

from .views import fix_dim, window_view


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


def acpr_calc(signal, sample_rate, mcf, mcb, acf=None, acb=None,
              nperseg=None, dim=-2):
    r"""Get the total power in the main and adjacent frequqency bands.

    Arguments
    ---------
    signal : tensor size = (..., T, 2)
        The complex signal to compute the per-channel power for.
    sample_rate : float
        The sampling frequency per unit time (Hz).
    mcf : float
        The frequency of the main channel in Hz.
    mcb : float
        The bandwidth of the main channel in Hz.
    acf : arraylike (n_channels,) or None
        The frequencies of adjacent channels in Hz.
    acb : numeric or arraylike (n_channels,)
        The bandwidths of the adjacent channels in Hz.
    nperseg : int, or None
        The length of the window to use for power spectrum estimation.
    dim : int
        The dimension of over which the signal is recorded.

    Returns
    -------
    main_channel_power : tensor size = (..., 1)
        The total power in decibel measured in the main band.
    adjacent_channel_power : tensor size = (..., len(acf))
        The total power in decibel measured in the each adjacent bands.
    """
    if acf is None or acb is None:
        acf, acb = [], []

    elif not isinstance(acf, (list, tuple)):
        raise TypeError("Adjacent Channel Frequency offests must be a "
                        "list or a tuple.")

    if isinstance(acb, (int, float)):
        acb = type(acf)([acb] * len(acf))

    elif not isinstance(acb, (list, tuple)):
        raise TypeError("Adjacent Channel Bandwidth must be a list or "
                        "a tuple.")

    # form the bands
    bands = [(-0.5 * mcb + mcf, +0.5 * mcb + mcf)]
    for f, b in zip(acf, acb):
        bands.append((-0.5 * b + f, +0.5 * b + f))

    # compute the power in each band
    ff, px, channel = bandwidth_power(signal, sample_rate, bands,
                                      dim=dim, nperseg=nperseg,
                                      n_overlap=0, scaling="spectrum")

    return channel[..., :1], channel[..., 1:]
