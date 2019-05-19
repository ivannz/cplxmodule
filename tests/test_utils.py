import torch
import pytest
import numpy as np

from numpy.testing import assert_allclose


@pytest.fixture
def random_state():
    return np.random.RandomState(None)  # (1249563438)


def test_window_view(random_state):
    from cplxmodule.utils import window_view

    np_x = random_state.randn(2, 3, 1024, 2, 2)
    tr_x = torch.tensor(np_x)

    dim, size, stride = -3, 5, 2
    dim = (tr_x.dim() + dim) if dim < 0 else dim

    tr_x_view = window_view(tr_x, dim, size, stride)
    for i in range(tr_x_view.shape[dim]):
        slice_ = np.r_[i * stride:i * stride + size]
        a = tr_x_view.index_select(dim, torch.tensor(i)).squeeze(dim)
        b = tr_x.index_select(dim, torch.tensor(slice_))
        assert_allclose(a, b)

    assert_allclose(window_view(tr_x, dim, size, stride, at=-1),
                    tr_x.unfold(dim, size, stride))


def test_complex_view(random_state):
    from cplxmodule.utils import complex_view

    for shape in [3, 4, 5, 6, 7, 8]:
        tr_x = torch.tensor(random_state.randn(16, 10, shape))

        if shape % 2:
            with pytest.warns(RuntimeWarning, match="Odd dimension"):
                real, imag = complex_view(tr_x, -1, squeeze=False)

            # odd
            assert_allclose(real, tr_x[..., 0:-1:2].clone())
            assert_allclose(imag, tr_x[..., 1:-1:2].clone())

        else:
            real, imag = complex_view(tr_x, -1, squeeze=False)

            # slice
            assert_allclose(real, tr_x[..., 0::2].clone())
            assert_allclose(imag, tr_x[..., 1::2].clone())

            # reshape
            input = tr_x.reshape(*tr_x.shape[:-1], -1, 2)
            assert_allclose(real, input[..., 0])
            assert_allclose(imag, input[..., 1])
        # end if
    # end for
