import tqdm
import pytest
import numpy as np

import torch
import torch.nn.functional as F

from itertools import starmap
from cplxmodule import cplx
from cplxmodule.nn import CplxBatchNorm1d, RealToCplx, CplxToReal
from cplxmodule.nn.modules.batchnorm import cplx_batch_norm
from cplxmodule.nn.modules.batchnorm import whiten2x2, whitendxd


@pytest.fixture
def random_state():
    return np.random.RandomState(None)  # (1249563438)


def test_whitening(random_state, nugget=1e-8):
    randn = random_state.randn

    # generate re-im data paired over the last axis
    z = (randn(1000, 10, 17) + 1j * randn(1000, 10, 17)) * 1e-1
    z += randn(1, 10, 1) * (1 + 1j)
    x = np.stack([z.real, z.imag], axis=0)

    # center x and get variance
    axes = 1, *range(3, x.ndim)
    n_samples = np.prod([x.shape[a] for a in axes])
    c = x - x.mean(axis=axes, keepdims=True)

    M = np.einsum("ubfs, vbfs->uvf", c, c) / n_samples

    # compute the inverse root
    v_rr, v_ri = M[0, 0] + nugget, M[0, 1]
    v_ir, v_ii = M[1, 0], M[1, 1] + nugget

    sqrdet = np.sqrt(v_rr * v_ii - v_ri * v_ir)
    denom = np.sqrt(v_rr + 2 * sqrdet + v_ii) * sqrdet
    R = np.array([[v_ii + sqrdet, -v_ri], [-v_ir, v_rr + sqrdet]]) / denom

    # verify inverse root
    RMR = np.einsum("uif, ijf, jvf->fuv", R, M, R)
    assert np.allclose(RMR, np.eye(2)[np.newaxis], atol=1e-3)

    # verify stats
    res = np.einsum("ubfs, uvf->vbfs", c, R)

    m = res.mean(axis=axes, keepdims=True)
    assert np.allclose(m, 0.)

    v = np.einsum("ubfs, vbfs->fuv", res - m, res - m) / n_samples
    assert np.allclose(v, np.eye(2)[np.newaxis], atol=1e-3)

    # verify torch
    res2x2 = whiten2x2(torch.from_numpy(x), nugget=nugget)
    assert np.allclose(res2x2.numpy(), res)

    # resdxd = whitendxd(torch.from_numpy(x), nugget=nugget)
    # assert np.allclose(resdxd.numpy(), res)


def fit(X, y, model, n_steps=2000):
    model.train()
    feed = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=32, shuffle=True)

    optim = torch.optim.Adam(model.parameters())
    with tqdm.tqdm(range(n_steps)) as bar:
        for i in bar:
            for bx, by in feed:
                optim.zero_grad()

                F.mse_loss(model(bx), by).backward()

                optim.step()

    return model.eval()


def predict(X, model):
    model.eval()
    feed = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X),
        batch_size=256, shuffle=False)
    with torch.no_grad():
        return torch.cat([*starmap(model, feed)], dim=0)


def test_real_batchnorm(random_state):
    x = random_state.randn(1000, 10) * 10. + 5.
    x = torch.from_numpy(x).to(torch.float)

    model = torch.nn.BatchNorm1d(10, affine=True)

    fit(x, x, model, n_steps=5)

    out = predict(x, model)
    out -= model.bias
    out /= model.weight

    assert np.isclose(float(out.mean()), 0.0, atol=1e-1)
    assert np.isclose(float(out.std()), 1.0, atol=1e-1)


def test_cplx_batchnorm(random_state):
    randn = random_state.randn
    noise = (randn(1000, 10, 17) + 1j * randn(1000, 10, 17)) * 1e-1
    z = cplx.Cplx.from_numpy(noise + randn(1, 10, 1) * (1 + 1j))

    z = z.to(torch.float)
    out = cplx_batch_norm(z, None, None).numpy()

    re, im = out.real, out.imag
    assert np.isclose(float(re.mean()), 0., atol=1e-1)
    assert np.isclose(float(im.mean()), 0., atol=1e-1)
    assert np.isclose(float((re*re).mean()), 1., atol=1e-1)
    assert np.isclose(float((im*im).mean()), 1., atol=1e-1)
    assert np.isclose(float((re*im).mean()), 0., atol=1e-1)


def test_cplx_batchnorm_layer_noaffine(random_state):
    randn = random_state.randn

    noise = (randn(1000, 10, 17) + 1j * randn(1000, 10, 17)) * 1e-1
    z = randn(1, 10, 1) * (1 + 1j) + noise
    z = np.stack([z.real, z.imag], axis=-1).reshape(*z.shape[:-1], -1)
    x = torch.from_numpy(z).to(torch.float)

    model = torch.nn.Sequential(
        RealToCplx(),
        CplxBatchNorm1d(10, affine=False),
        CplxToReal(),
    )

    model.train()
    feed = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x),
        batch_size=32, shuffle=True)

    # with torch.autograd.detect_anomaly():
    for i in range(5):
        for bx, in feed:
            F.mse_loss(model(bx), bx)

    model.eval()

    out = predict(x, model)
    re, im = out[..., 0::2], out[..., 1::2]
    assert np.isclose(float(re.mean()), 0., atol=1e-1)
    assert np.isclose(float(im.mean()), 0., atol=1e-1)
    assert np.isclose(float((re*re).mean()), 1., atol=1e-1)
    assert np.isclose(float((im*im).mean()), 1., atol=1e-1)
    assert np.isclose(float((re*im).mean()), 0., atol=1e-1)


def test_cplx_batchnorm_layer_affine(random_state):
    randn = random_state.randn

    noise = (randn(1000, 10, 17) + 1j * randn(1000, 10, 17)) * 1e-1
    z = randn(1, 10, 1) * (1 + 1j) + noise
    z = np.stack([z.real, z.imag], axis=-1).reshape(*z.shape[:-1], -1)
    x = torch.from_numpy(z).to(torch.float)

    model = torch.nn.Sequential(
        RealToCplx(),
        CplxBatchNorm1d(10, affine=True),
        CplxToReal(),
    )

    # with torch.autograd.detect_anomaly():
    fit(x, x, model, n_steps=5)

    out = predict(x, model).reshape(*x.shape[:-1], -1, 2)
    with torch.no_grad():
        out = out - model[1].bias.reshape(1, 10, 1, 2)
        res, _ = torch.solve(out.transpose(-1, -2), model[1].weight.permute(2, 0, 1))

    re, im = res[..., 0, :], res[..., 1, :]
    assert np.isclose(float(re.mean()), 0., atol=1e-1)
    assert np.isclose(float(im.mean()), 0., atol=1e-1)
    assert np.isclose(float((re*re).mean()), 1., atol=1e-1)
    assert np.isclose(float((im*im).mean()), 1., atol=1e-1)
    assert np.isclose(float((re*im).mean()), 0., atol=1e-1)
