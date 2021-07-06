import pytest
import copy

import torch

from cplxmodule import nn
from cplxmodule.nn import CplxToCplx
from cplxmodule import cplx, Cplx

from functools import lru_cache, wraps


def cplx_allclose(input, other):
    return torch.allclose(input.real, other.real) and \
           torch.allclose(input.imag, other.imag)


def _cplx_emulate_module(Module):
    class template(CplxToCplx):
        @wraps(Module)
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.real = Module(*args, **kwargs)
            self.imag = Module(*args, **kwargs)

        def forward(self, input):
            re = self.real(input.real) - self.imag(input.imag)
            im = self.imag(input.real) + self.real(input.imag)

            # undo the double application of bias
            if getattr(self.real, 'bias', None) is not None:
                broadcast = []
                if isinstance(self.real, torch.nn.modules.conv._ConvNd):
                    broadcast = len(self.real.kernel_size) * [1]
                re += self.imag.bias.reshape(-1, *broadcast)
                im -= self.real.bias.reshape(-1, *broadcast)

            return Cplx(re, im)

        def state_dict(self):
            state_dict = super().state_dict()
            # weight
            out = {
                'weight.real': state_dict.pop('real.weight'),
                'weight.imag': state_dict.pop('imag.weight'),
            }
            if getattr(self.real, 'bias', None) is not None:
                out.update({
                    'bias.real': state_dict.pop('real.bias'),
                    'bias.imag': state_dict.pop('imag.bias'),
                })

            return {**out, **state_dict}

    # set up meta info
    template.__name__ = f"EmulatedCplxModule{Module.__name__}"
    template.__qualname__ = f"<runtime type `{template.__name__}`>"
    template.__doc__ = f"Cplx emulation based on `{Module.__name__}`"
    return template


class _EmulatedCplxModuleMeta(type):
    """Meta class for promoting torch.nn Modules to complex ones."""
    @lru_cache(maxsize=None)
    def __getitem__(self, Base):
        if isinstance(Base, type) and issubclass(Base, torch.nn.Module):
            # make sure that base is not an instance, and that no
            #  nested wrapping takes place.
            if issubclass(Base, (EmulatedCplxModule, CplxToCplx)):
                return Base

            if Base is torch.nn.Module:
                return CplxToCplx

            return _cplx_emulate_module(Base)

        raise TypeError(f"Expecting a torch.nn.Module subclass. Got `{type(Base)}`.")


class EmulatedCplxModule(metaclass=_EmulatedCplxModuleMeta):
    pass


def test_linear():
    emulated = EmulatedCplxModule[torch.nn.Linear](23, 17, bias=False).double()
    implemented = nn.CplxLinear(23, 17, bias=False).double()
    implemented.load_state_dict(emulated.state_dict())

    z = cplx.randn(32, 31, 23, dtype=torch.double)
    with torch.no_grad():
        assert cplx_allclose(implemented(z), emulated(z))

    emulated = EmulatedCplxModule[torch.nn.Linear](127, 63, bias=True).double()
    implemented = nn.CplxLinear(127, 63, bias=True).double()
    implemented.load_state_dict(emulated.state_dict())

    z = cplx.randn(32, 31, 127, dtype=torch.double)
    with torch.no_grad():
        assert cplx_allclose(implemented(z), emulated(z))

    duplicated = copy.deepcopy(implemented)

    dup = duplicated.state_dict()
    ref = implemented.state_dict()
    assert all([torch.allclose(dup[p], ref[p]) for p in ref.keys()])

    with torch.no_grad():
        assert cplx_allclose(duplicated(z), implemented(z))


def do_test_conv(Layer, CplxLayer, in_channels, out_channels, kernel_size,
                 **kwargs):
    emulated = EmulatedCplxModule[Layer](
        in_channels, out_channels, kernel_size, **kwargs).double()
    implemented = CplxLayer(
        in_channels, out_channels, kernel_size, **kwargs).double()

    # ensure identical parameters
    implemented.load_state_dict(emulated.state_dict())

    # use float64 for testing
    shape = 4, in_channels, *[16]*len(emulated.real.kernel_size)
    with torch.no_grad():
        z = cplx.randn(*shape, dtype=torch.double)
        assert cplx_allclose(implemented(z), emulated(z))

    duplicated = copy.deepcopy(implemented)

    dup = duplicated.state_dict()
    ref = implemented.state_dict()
    assert all([torch.allclose(dup[p], ref[p]) for p in ref.keys()])

    with torch.no_grad():
        assert cplx_allclose(duplicated(z), implemented(z))


@pytest.mark.parametrize('case', [
    (torch.nn.Conv1d, nn.CplxConv1d, (3,)),
    (torch.nn.Conv2d, nn.CplxConv2d, (3, 4)),
    (torch.nn.Conv3d, nn.CplxConv3d, (3, 4, 5)),
])
def test_conv(case):
    Layer, CplxLayer, kernel_size = case

    common = Layer, CplxLayer, 12, 16

    # test different kernel sizes
    do_test_conv(*common, kernel_size, bias=True)

    # test no bias
    do_test_conv(*common, kernel_size, bias=False)

    # test same kernel size
    do_test_conv(*common, kernel_size[0], bias=True)

    # test stride=2
    do_test_conv(*common, kernel_size, bias=True, stride=3)

    # test dilation=3
    do_test_conv(*common, kernel_size, bias=True, dilation=3)

    # test padding=2
    do_test_conv(*common, kernel_size, bias=True, padding=2)

    # test padding=2 w. circular
    with pytest.xfail(reason="always xfail"):
        do_test_conv(*common, kernel_size, bias=True, padding=2,
                     padding_mode="circular")

    # test groups=2
    do_test_conv(*common, kernel_size, bias=True, groups=2)


@pytest.mark.parametrize('case', [
    (torch.nn.ConvTranspose1d, nn.CplxConvTranspose1d, (3,)),
    (torch.nn.ConvTranspose2d, nn.CplxConvTranspose2d, (3, 4)),
    (torch.nn.ConvTranspose3d, nn.CplxConvTranspose3d, (3, 4, 5)),
])
def test_conv_transpose(case):
    Layer, CplxLayer, kernel_size = case

    common = Layer, CplxLayer, 12, 16

    # test different kernel sizes
    do_test_conv(*common, kernel_size, bias=True)

    # test no bias
    do_test_conv(*common, kernel_size, bias=False)

    # test same kernel size
    do_test_conv(*common, kernel_size[0], bias=True)

    # test stride=2
    do_test_conv(*common, kernel_size, bias=True, stride=3)

    # test dilation=3
    do_test_conv(*common, kernel_size, bias=True, dilation=3)

    # test padding=2
    do_test_conv(*common, kernel_size, bias=True, padding=2)

    # test groups=2
    do_test_conv(*common, kernel_size, bias=True, groups=2)

    # test output_padding=3
    do_test_conv(*common, kernel_size, bias=True, stride=2, output_padding=1)


@pytest.mark.parametrize('pair', [
    (nn.modules.casting.InterleavedRealToCplx,
     nn.modules.casting.CplxToInterleavedReal),
    (nn.modules.casting.ConcatenatedRealToCplx,
     nn.modules.casting.CplxToConcatenatedReal),
    (nn.modules.casting.TensorToCplx,
     nn.modules.casting.CplxToTensor),
    (nn.modules.casting.AsTypeCplx,
     nn.modules.linear.CplxReal),
])
def test_casting(pair):
    to_cplx, to_real = pair
    tensor = torch.randn(32, 31, 1024, 2)

    assert torch.allclose(to_real()(to_cplx()(tensor)), tensor)
