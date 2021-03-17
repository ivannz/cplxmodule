import pytest

import torch
import tempfile

try:
    import onnx
    from onnxruntime import InferenceSession

except ImportError as e:
    pytestmark = pytest.mark.skip(reason=str(e))

from cplxmodule import nn
from cplxmodule.nn import relevance
from cplxmodule.nn import masked

from cplxmodule.nn.modules import casting


def onnx_export_to(filename, module, input, *,
                   training=False, opset_version=11):
    # jit compile and export into ONNX
    torch.onnx.export(
        module, (input,), filename, verbose=False,
        export_params=True, training=training,
        opset_version=opset_version, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        # all axes are static
        dynamic_axes={'input': [], 'output': []})


def do_onnx_export_test(module, input, *, training=False):
    with tempfile.NamedTemporaryFile() as file:
        onnx_export_to(file.name, module, input, training=training)

        # Load the ONNX model and check that the IR is well formed
        onnx.checker.check_model(onnx.load(file.name), full_check=True)


def do_onnx_inference_test(module, input, *, training=False):
    with tempfile.NamedTemporaryFile() as file:
        onnx_export_to(file.name, module, input, training=training)
        onnx.checker.check_model(onnx.load(file.name), full_check=True)

        # run inference through the exported model
        input = torch.randn_like(input)
        output, = InferenceSession(file.name).run(
            ['output'], {'input': input.numpy()}
        )

        assert torch.allclose(
            torch.from_numpy(output),
            module(input), rtol=1e-4, atol=1e-5
        )


def wrap_cplxtoreal(*modules):
    return torch.nn.Sequential(casting.TensorToCplx(), *modules)


def wrap_cplxtocplx(*modules):
    return torch.nn.Sequential(casting.TensorToCplx(),
                               *modules,
                               casting.CplxToTensor())


@pytest.mark.parametrize('training', [
    True, False, None
])
def test_linear_onnx_export(training):
    module, input = torch.nn.Linear(16, 32), torch.randn(2, 3, 5, 16)
    do_onnx_export_test(module, input, training=training)
    do_onnx_inference_test(module, input, training=training)

    module = relevance.LinearVD(16, 32)
    do_onnx_export_test(module, input, training=training)

    module = relevance.LinearARD(16, 32)
    do_onnx_export_test(module, input, training=training)

    module = masked.LinearMasked(16, 32)
    module.mask = torch.randint(0, 2, size=module.weight.shape)
    do_onnx_export_test(module, input, training=training)


@pytest.mark.parametrize('training', [
    True, False, None
])
def test_conv1d_onnx_export(training):
    module, input = torch.nn.Conv1d(16, 32, 5), torch.randn(2, 16, 25)
    do_onnx_export_test(module, input, training=training)
    do_onnx_inference_test(module, input, training=training)

    module = relevance.Conv1dVD(16, 32, 5)
    do_onnx_export_test(module, input, training=training)

    module = relevance.Conv1dARD(16, 32, 5)
    do_onnx_export_test(module, input, training=training)

    module = masked.Conv1dMasked(16, 32, 5)
    module.mask = torch.randint(0, 2, size=module.weight.shape)
    do_onnx_export_test(module, input, training=training)


@pytest.mark.parametrize('training', [
    True, False, None
])
def test_conv2d_onnx_export(training):
    module, input = torch.nn.Conv2d(16, 32, 5), torch.randn(2, 16, 25, 25)
    do_onnx_export_test(module, input, training=training)
    do_onnx_inference_test(module, input, training=training)

    module = relevance.Conv2dVD(16, 32, 5)
    do_onnx_export_test(module, input, training=training)

    module = relevance.Conv2dARD(16, 32, 5)
    do_onnx_export_test(module, input, training=training)

    module = masked.Conv2dMasked(16, 32, 5)
    module.mask = torch.randint(0, 2, size=module.weight.shape)
    do_onnx_export_test(module, input, training=training)


@pytest.mark.parametrize('training', [
    True, False, None
])
def test_conv3d_onnx_export(training):
    module, input = torch.nn.Conv3d(16, 32, 5), torch.randn(2, 16, 25, 25, 25)
    do_onnx_export_test(module, input, training=training)
    do_onnx_inference_test(module, input, training=training)

    module = relevance.Conv3dVD(16, 32, 5)
    do_onnx_export_test(module, input, training=training)

    module = relevance.Conv3dARD(16, 32, 5)
    do_onnx_export_test(module, input, training=training)

    module = masked.Conv3dMasked(16, 32, 5)
    module.mask = torch.randint(0, 2, size=module.weight.shape)
    do_onnx_export_test(module, input, training=training)


@pytest.mark.parametrize('training', [
    True, False, None
])
def test_batchnorm_onnx_export(training):
    input = torch.randn(2, 32, 256)
    # with pytest.skip():
    #     module = torch.nn.BatchNorm1d(32, track_running_stats=False)
    #     do_onnx_export_test(module, input, training=training)

    module = torch.nn.BatchNorm1d(32, track_running_stats=True)
    do_onnx_export_test(module, input, training=training)


@pytest.mark.parametrize('training', [
    True, False, None
])
def test_cplx_linear_onnx_export(training):
    module, input = nn.CplxLinear(16, 32), torch.randn(2, 3, 5, 16, 2)
    do_onnx_export_test(wrap_cplxtocplx(module), input, training=training)
    do_onnx_inference_test(wrap_cplxtocplx(module), input, training=training)

    module = relevance.CplxLinearVD(16, 32)
    do_onnx_export_test(wrap_cplxtocplx(module), input, training=training)

    module = relevance.CplxLinearARD(16, 32)
    do_onnx_export_test(wrap_cplxtocplx(module), input, training=training)

    module = masked.CplxLinearMasked(16, 32)
    module.mask = torch.randint(0, 2, size=module.weight.shape)
    do_onnx_export_test(wrap_cplxtocplx(module), input, training=training)


@pytest.mark.parametrize('training', [
    True, False, None
])
def test_cplx_conv1d_onnx_export(training):
    module, input = nn.CplxConv1d(16, 32, 5), torch.randn(3, 16, 25, 2)
    do_onnx_export_test(wrap_cplxtocplx(module), input, training=training)

    do_onnx_inference_test(wrap_cplxtocplx(module), input,
                           training=training)

    module = relevance.CplxConv1dVD(16, 32, 5)
    do_onnx_export_test(wrap_cplxtocplx(module), input, training=training)

    module = relevance.CplxConv1dARD(16, 32, 5)
    do_onnx_export_test(wrap_cplxtocplx(module), input, training=training)

    module = masked.CplxConv1dMasked(16, 32, 5)
    module.mask = torch.randint(0, 2, size=module.weight.shape)
    do_onnx_export_test(wrap_cplxtocplx(module), input, training=training)


@pytest.mark.parametrize('training', [
    True, False, None
])
def test_cplx_conv2d_onnx_export(training):
    module, input = nn.CplxConv2d(16, 32, 5), torch.randn(3, 16, 25, 25, 2)
    do_onnx_export_test(wrap_cplxtocplx(module), input, training=training)

    do_onnx_inference_test(wrap_cplxtocplx(module), input,
                           training=training)

    module = relevance.CplxConv2dVD(16, 32, 5)
    do_onnx_export_test(wrap_cplxtocplx(module), input, training=training)

    module = relevance.CplxConv2dARD(16, 32, 5)
    do_onnx_export_test(wrap_cplxtocplx(module), input, training=training)

    module = masked.CplxConv2dMasked(16, 32, 5)
    module.mask = torch.randint(0, 2, size=module.weight.shape)
    do_onnx_export_test(wrap_cplxtocplx(module), input, training=training)


@pytest.mark.parametrize('training', [
    True, False, None
])
def test_cplx_conv3d_onnx_export(training):
    module, input = nn.CplxConv3d(16, 32, 5), torch.randn(3, 16, 25, 25, 25, 2)
    do_onnx_export_test(wrap_cplxtocplx(module), input, training=training)

    do_onnx_inference_test(wrap_cplxtocplx(module), input,
                           training=training)

    module = relevance.CplxConv3dVD(16, 32, 5)
    do_onnx_export_test(wrap_cplxtocplx(module), input, training=training)

    module = relevance.CplxConv3dARD(16, 32, 5)
    do_onnx_export_test(wrap_cplxtocplx(module), input, training=training)

    module = masked.CplxConv3dMasked(16, 32, 5)
    module.mask = torch.randint(0, 2, size=module.weight.shape)
    do_onnx_export_test(wrap_cplxtocplx(module), input, training=training)


@pytest.mark.parametrize('training', [
    True, False, None
])
def test_cplx_batchnorm_onnx_export(training):
    module = nn.CplxBatchNorm1d(16, track_running_stats=False)
    do_onnx_export_test(wrap_cplxtocplx(module).float(),
                        torch.randn(2, 16, 256, 2).float(), training=training)

    module = nn.CplxBatchNorm1d(32, track_running_stats=True)
    do_onnx_export_test(wrap_cplxtocplx(module).float(),
                        torch.randn(2, 32, 256, 2).float(), training=training)


@pytest.mark.parametrize('training', [
    True, False, None
])
def test_cplx_activations_onnx_export(training):
    module = nn.CplxModReLU()
    do_onnx_export_test(wrap_cplxtocplx(module).float(),
                        torch.randn(2, 3, 5, 16, 2).float(), training=training)

    module = nn.CplxAdaptiveModReLU()
    do_onnx_export_test(wrap_cplxtocplx(module).float(),
                        torch.randn(2, 3, 5, 16, 2).float(), training=training)

    module = nn.CplxModulus()
    do_onnx_export_test(wrap_cplxtoreal(module).float(),
                        torch.randn(2, 3, 5, 16, 2).float(), training=training)

    with pytest.raises(RuntimeError, match="the operator atan2 to ONNX"):
        module = nn.CplxAngle()
        do_onnx_export_test(wrap_cplxtoreal(module).float(),
                            torch.randn(2, 3, 5, 16, 2).float(),
                            training=training)


@pytest.mark.parametrize('training', [
    True, False, None
])
def test_cplx_concatenated_casting_float_onnx_export(training):
    module = torch.nn.Sequential(casting.ConcatenatedRealToCplx(),
                                 nn.CplxIdentity(),
                                 casting.CplxToConcatenatedReal())
    input = torch.randn(2, 16, 256)

    do_onnx_export_test(module.float(), input.float(), training=training)

    do_onnx_inference_test(module.float(), input.float(), training=training)

    with pytest.xfail(reason="double is not implemented in ONNX"):
        do_onnx_export_test(module.double(), input.double(),
                            training=training)

    with pytest.xfail(reason="double is not implemented in ONNX"):
        do_onnx_inference_test(module.double(), input.double(),
                               training=training)


@pytest.mark.xfail(reason="torch.as_strided is not supported by ONNX")
@pytest.mark.parametrize('training', [
    True, False, None
])
def test_cplx_interleaved_casting_onnx_export(training):
    module = torch.nn.Sequential(casting.InterleavedRealToCplx(),
                                 nn.CplxIdentity(),
                                 casting.CplxToInterleavedReal())
    input = torch.randn(2, 16, 256)

    do_onnx_export_test(module.float(), input.float(), training=training)
    do_onnx_inference_test(module.float(), input.float(), training=training)

    with pytest.xfail(reason="double is not implemented in ONNX"):
        do_onnx_export_test(module.double(), input.double(),
                            training=training)

    with pytest.xfail(reason="double is not implemented in ONNX"):
        do_onnx_inference_test(module.double(), input.double(),
                               training=training)


@pytest.mark.parametrize('training', [
    True, False, None
])
def test_cplx_astype_casting_float_onnx_export(training):
    module = torch.nn.Sequential(casting.AsTypeCplx(),
                                 nn.CplxIdentity(),
                                 nn.modules.linear.CplxReal())
    input = torch.randn(2, 16, 256)

    do_onnx_export_test(module.float(), input.float(), training=training)

    with pytest.raises(Exception, match="Invalid Feed"):
        do_onnx_inference_test(module.float(), input.float(), training=training)

    with pytest.xfail(reason="double is not implemented in ONNX"):
        do_onnx_export_test(module.double(), input.double(),
                            training=training)

    with pytest.xfail(reason="double is not implemented in ONNX"):
        do_onnx_inference_test(module.double(), input.double(),
                               training=training)
