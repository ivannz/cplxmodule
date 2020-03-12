import pytest
import torch

from cplxmodule.nn.masked import LinearMasked


def test_emptymask():
    lin = LinearMasked(11, 13)

    with pytest.raises(RuntimeError, match=r"has no sparsity mask"):
        lin.weight_masked

    with pytest.raises(RuntimeError, match=r"has no sparsity mask"):
        lin(torch.ones(1, 11))


def test_setattr_interface():
    shape = 101, 97
    lin = LinearMasked(*shape)

    lin.mask = mask = torch.tensor(1.0)
    assert torch.allclose(*torch.broadcast_tensors(lin.mask, mask))
    assert torch.allclose(lin.weight_masked, mask * lin.weight)

    lin.mask = None
    assert lin.mask is None

    with pytest.raises(RuntimeError, match=r"has no sparsity mask"):
        lin.weight_masked

    # enable/disable
    lin.mask = mask = torch.randint(2, size=(1, 1)).float()
    assert torch.allclose(*torch.broadcast_tensors(lin.mask, mask))
    assert torch.allclose(lin.weight_masked, mask * lin.weight)

    # output masking
    lin.mask = mask = torch.randint(2, size=(shape[1], 1)).float()
    assert torch.allclose(*torch.broadcast_tensors(lin.mask, mask))
    assert torch.allclose(lin.weight_masked, mask * lin.weight)

    # input masking
    lin.mask = mask = torch.randint(2, size=(1, shape[0],)).float()
    assert torch.allclose(*torch.broadcast_tensors(lin.mask, mask))
    assert torch.allclose(lin.weight_masked, mask * lin.weight)

    # unstructured masking
    lin.mask = mask = torch.randint(2, size=(shape[1], shape[0],)).float()
    assert torch.allclose(*torch.broadcast_tensors(lin.mask, mask))
    assert torch.allclose(lin.weight_masked, mask * lin.weight)

    # mask is overwritten, to `.copy_` raises
    with pytest.raises(RuntimeError, match=r"must match the size"):
        lin.mask = torch.ones(shape[1], 2)

    # mask is set anew, to `torch.expand` raises
    lin.mask = None
    with pytest.raises(RuntimeError, match=r"must match the existing size"):
        lin.mask = torch.ones(0, shape[0])


def test_method_interface():
    shape = 11, 13
    lin = LinearMasked(*shape)

    mask = torch.tensor(1.0)
    lin.mask_(mask)
    assert torch.allclose(*torch.broadcast_tensors(lin.mask, mask))
    assert torch.allclose(lin.weight_masked, mask * lin.weight)

    lin.mask_(None)
    assert lin.mask is None

    with pytest.raises(RuntimeError, match=r"has no sparsity mask"):
        lin.weight_masked

    # enable/disable
    mask = torch.randint(2, size=(1, 1)).float()
    lin.mask_(mask)
    assert torch.allclose(*torch.broadcast_tensors(lin.mask, mask))
    assert torch.allclose(lin.weight_masked, mask * lin.weight)

    # output masking
    mask = torch.randint(2, size=(shape[1], 1)).float()
    lin.mask_(mask)
    assert torch.allclose(*torch.broadcast_tensors(lin.mask, mask))
    assert torch.allclose(lin.weight_masked, mask * lin.weight)

    # input masking
    mask = torch.randint(2, size=(1, shape[0],)).float()
    lin.mask_(mask)
    assert torch.allclose(*torch.broadcast_tensors(lin.mask, mask))
    assert torch.allclose(lin.weight_masked, mask * lin.weight)

    # unstructured masking
    mask = torch.randint(2, size=(shape[1], shape[0],)).float()
    lin.mask_(mask)
    assert torch.allclose(*torch.broadcast_tensors(lin.mask, mask))
    assert torch.allclose(lin.weight_masked, mask * lin.weight)

    # mask is overwritten, to `.copy_` raises
    with pytest.raises(RuntimeError, match=r"must match the size"):
        lin.mask_(torch.ones(shape[1], 2))

    # mask is set anew, to `torch.expand` raises
    lin.mask_(None)
    with pytest.raises(RuntimeError, match=r"must match the existing size"):
        lin.mask_(torch.ones(0, shape[0]))


def test_state_dict_loading():
    shape = 101, 97

    # unmasked --> masked
    origin = torch.nn.Linear(*shape)
    masked = LinearMasked(*shape)

    masked.load_state_dict(origin.state_dict(), strict=False)
    assert torch.allclose(masked.weight, origin.weight)

    with pytest.raises(RuntimeError, match=r"Missing key\(s\)"):
        masked.load_state_dict(origin.state_dict(), strict=True)

    # masked --> unmasked
    origin = torch.nn.Linear(*shape)
    masked = LinearMasked(*shape)

    origin.load_state_dict(masked.state_dict(), strict=False)
    assert torch.allclose(origin.weight, masked.weight)

    masked.mask = None
    origin.load_state_dict(masked.state_dict(), strict=True)

    masked.mask = torch.tensor(1.)
    with pytest.raises(RuntimeError, match=r"Unexpected key\(s\)"):
        origin.load_state_dict(masked.state_dict(), strict=True)

    # loading masks with load_state_dict
    origin = torch.nn.Linear(*shape)
    masked = LinearMasked(*shape)

    # loading `None` into an unset mask is ok
    masked.load_state_dict({**origin.state_dict(), "mask": None})

    # unsetting a mask with state_dict having `None` should work, too
    masked.mask = mask = torch.randint(2, size=(shape[1], shape[0],)).float()
    masked.load_state_dict({**origin.state_dict(), "mask": None})
    assert masked.mask is None

    # see if the existing mask is unaffected when mask is missing
    masked.mask = mask = torch.randint(2, size=(shape[1], shape[0],)).float()
    masked.load_state_dict(origin.state_dict(), strict=False)
    assert torch.allclose(masked.mask, mask)

    with pytest.raises(RuntimeError, match=r"Missing key\(s\)"):
        masked.load_state_dict(origin.state_dict(), strict=True)

    # setting mask
    for mshape in [(1,), (shape[1], 1), (1, shape[0]), (shape[1], shape[0])]:
        origin = torch.nn.Linear(*shape)
        masked = LinearMasked(*shape).mask_(None)

        mask = torch.randint(2, size=mshape).float()
        masked.load_state_dict({**origin.state_dict(), "mask": mask})

        assert torch.allclose(*torch.broadcast_tensors(masked.mask, mask))
        assert torch.allclose(masked.weight_masked, mask * origin.weight)
