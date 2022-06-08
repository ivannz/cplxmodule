import pytest
import torch

from cplxmodule import cplx
from cplxmodule.nn import CplxParameter


def test_loading():
    shape = 11, 13, 7

    # load complex parameter in full
    state_dict = {"real": torch.randn(*shape), "imag": torch.randn(*shape)}
    par = CplxParameter(cplx.Cplx(real=torch.ones(*shape), imag=torch.zeros(*shape)))
    par.load_state_dict(state_dict, strict=True)

    assert torch.allclose(par.real, state_dict["real"])
    assert torch.allclose(par.imag, state_dict["imag"])

    # no effect if parameter ot components are missing entirely
    par.load_state_dict({}, strict=False)
    assert torch.allclose(par.real, state_dict["real"])
    assert torch.allclose(par.imag, state_dict["imag"])

    with pytest.raises(RuntimeError, match=r"Missing key\(s\)"):
        par.load_state_dict({}, strict=True)

    # promote real tensor
    state_dict = {"": torch.randn(*shape)}
    par = CplxParameter(cplx.Cplx(real=torch.ones(*shape), imag=torch.zeros(*shape)))
    par.load_state_dict(state_dict, strict=True)

    assert torch.allclose(par.real, state_dict[""])
    assert torch.allclose(par.imag, torch.zeros_like(state_dict[""]))


def make_module(*shape):
    module = torch.nn.Module()
    module.mod = torch.nn.Module()
    module.mod.par = CplxParameter(
        cplx.Cplx(real=torch.ones(*shape), imag=torch.zeros(*shape))
    )
    return module


def test_nested_loading():
    shape = 11, 13, 7

    # load complex parameter in full
    base = CplxParameter(cplx.randn(*shape))

    state_dict = {f"mod.par.{k}": v for k, v in base.state_dict().items()}
    module = make_module(*shape)
    module.load_state_dict(state_dict, strict=True)

    assert torch.allclose(module.mod.par.real, base.real)
    assert torch.allclose(module.mod.par.imag, base.imag)

    # no effect if parameter ot components are missing entirely
    module.load_state_dict({}, strict=False)
    assert torch.allclose(module.mod.par.real, base.real)
    assert torch.allclose(module.mod.par.imag, base.imag)

    with pytest.raises(RuntimeError, match=r"Missing key\(s\)"):
        module.load_state_dict({}, strict=True)

    # promote real tensor
    base = CplxParameter(cplx.Cplx(torch.randn(*shape)))

    module = make_module(*shape)
    module.load_state_dict({"mod.par": base.real}, strict=True)

    assert torch.allclose(module.mod.par.real, base.real)
    assert torch.allclose(module.mod.par.imag, base.imag)


def test_malformed():
    shape = 11, 13, 7

    # parital complex parameter
    par = CplxParameter(cplx.randn(*shape))
    with pytest.raises(RuntimeError, match=r"Complex parameter requires both"):
        par.load_state_dict({"real": torch.randn(*shape)})

    with pytest.raises(RuntimeError, match=r"Complex parameter requires both"):
        par.load_state_dict({"imag": torch.randn(*shape)})

    with pytest.raises(RuntimeError, match=r"disallows redundant"):
        par.load_state_dict(
            {
                "real": torch.randn(*shape),
                "imag": torch.randn(*shape),
                "bar": torch.randn(*shape),
                "foo": torch.randn(*shape),
            },
            strict=True,
        )

    with pytest.raises(RuntimeError, match=r"size mismatch for"):
        par.load_state_dict(
            {
                "real": torch.randn(1, 1),
                "imag": torch.randn(1, 1),
            },
            strict=True,
        )

    with pytest.raises(RuntimeError, match=r"size mismatch for"):
        par.load_state_dict(
            {
                "": torch.randn(1, 1),
            },
            strict=True,
        )


def test_nested_malformed():
    shape = 11, 13, 7

    # parital complex parameter
    module = make_module(*shape)
    with pytest.raises(RuntimeError, match=r"Complex parameter requires both"):
        module.load_state_dict({"mod.par.real": torch.randn(*shape)})

    with pytest.raises(RuntimeError, match=r"Complex parameter requires both"):
        module.load_state_dict({"mod.par.imag": torch.randn(*shape)})

    # redundant keys pertaining to the complex parameter are forbidden
    with pytest.raises(RuntimeError, match=r"disallows redundant"):
        module.load_state_dict(
            {
                "mod.par.real": torch.randn(*shape),
                "mod.par.imag": torch.randn(*shape),
                "mod.par.bar": torch.randn(*shape),
                "mod.par.foo": torch.randn(*shape),
            },
            strict=True,
        )

    # redundant keys unrelated to the complex parameter are ignored
    module.load_state_dict(
        {
            "mod.par.real": torch.zeros(*shape),
            "mod.par.imag": torch.ones(*shape),
            "bar": torch.randn(*shape),
            "foo": torch.randn(*shape),
        },
        strict=False,
    )

    with pytest.raises(RuntimeError, match=r"size mismatch for"):
        module.load_state_dict(
            {
                "mod.par.real": torch.randn(1, 1),
                "mod.par.imag": torch.randn(1, 1),
            },
            strict=True,
        )

    with pytest.raises(RuntimeError, match=r"size mismatch for"):
        module.load_state_dict(
            {
                "mod.par": torch.randn(1, 1),
            },
            strict=True,
        )

    assert torch.allclose(module.mod.par.real, torch.zeros(*shape))
    assert torch.allclose(module.mod.par.imag, torch.ones(*shape))
