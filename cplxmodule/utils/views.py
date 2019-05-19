import torch
import warnings


def fix_dim(dim, n_dim):
    r"""For the given dimensionality make sure the `axis` is nonnegative."""
    axis = (n_dim + dim) if dim < 0 else dim
    if not 0 <= axis < n_dim:
        raise ValueError(f"""Dimension {dim} is out of range for {n_dim}.""")

    return axis


def complex_view(x, dim=-1, squeeze=True):
    r"""Returns a real and imaginary views into the complex tensor, assumed
    to be in interleaved layout (double-real, i.e. re-im). The returned tensor
    is half the size along the specified dimension and is not, in general,
    contiguous in memory.

    Arguments
    ---------
    x : torch.tensor
        Time series of measurement values
    dim : int
        Axis along which the periodogram is computed, i.e. ``dim=-1``.

    Returns
    -------
    real : torch.tensor
        The view into a real part of the tensor.
    imag : torch.tensor
        The view into a real part of the tensor.
    """
    dim = fix_dim(dim, x.dim())
    shape, strides = list(x.size()), list(x.stride())
    offset = x.storage_offset()

    # compute new shape and strides
    strided_size, rem = divmod(shape[dim], 2)
    if rem != 0:
        warnings.warn(f"Odd dimension size for the complex data unpacking: "
                      f"taking the least size that fits.", RuntimeWarning)

    # new shape and stride structure
    if shape[dim] == 2 and squeeze:
        # if the complex dimension is exactly two, then just drop it
        shape_view = shape[:dim] + shape[dim+1:]
        strides_view = strides[:dim] + strides[dim+1:]

    else:
        # otherwise, half the size and double the stride
        size, rem = divmod(shape[dim], 2)
        shape_view = shape[:dim] + [size] + shape[dim+1:]
        strides_view = strides[:dim] + [2 * strides[dim]] + strides[dim+1:]

    # differentiable strided view into real and imaginary parts
    real = torch.as_strided(x, shape_view, strides_view, offset)
    imag = torch.as_strided(x, shape_view, strides_view, offset + strides[dim])

    return real, imag


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
