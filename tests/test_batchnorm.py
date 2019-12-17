import tqdm
import pytest
import numpy as np

import torch
import torch.nn.functional as F

from itertools import starmap
from cplxmodule import cplx
from cplxmodule.batchnorm import CplxBatchNorm1d, cplx_batch_norm
from cplxmodule.layers import RealToCplx, CplxToReal


@pytest.fixture
def random_state():
    return np.random.RandomState(None)  # (1249563438)


def test_invsqr2x2(random_state, eps=1e-8):
    from cplxmodule.batchnorm import invsqrt_2x2

    """Computes inverse root of the 2x2 covariance matrix."""
    randn = random_state.randn

    # generate re-im data paired over the last axis
    z = (randn(1000, 10, 17) + 1j * randn(1000, 10, 17)) * 1e-1
    z += randn(1, 10, 1) * (1 + 1j)
    x = np.stack([z.real, z.imag], axis=-1)

    # center x and get variance
    axes = 0, *range(2, x.ndim - 1)  # note `-1`
    c = x - x.mean(axis=axes, keepdims=True)
    M = np.einsum("bfsu, bfsv->uvf", c, c) / len(c)

    # compute the inverse root
    (v_rr, v_ri), (v_ir, v_ii) = M

    # sqrdet = np.sqrt(np.maximum(v_rr * v_ii - v_ri * v_ir, eps))
    sqrdet = np.sqrt((v_rr + eps) * (v_ii + eps) - v_ri * v_ir)
    denom = np.sqrt(v_rr + v_ii + 2 * sqrdet + 2 * eps) * sqrdet
    R = np.array([[v_ii + sqrdet, -v_ri], [-v_ir, v_rr + sqrdet]]) / denom

    # verify inverse root
    RMR = np.einsum("uif, ijf, jvf->fuv", R, M, R)
    assert np.allclose(RMR, np.eye(2)[np.newaxis], atol=1e-3)

    # verify stats
    res = np.einsum("bfsu, uvf->bfsv", c, R)

    m = res.mean(axis=axes, keepdims=True)
    assert np.allclose(m, 0.)

    v = np.einsum("bfsu, bfsv->fuv", res - m, res - m) / len(res)
    assert np.allclose(v, np.eye(2)[np.newaxis], atol=1e-3)

    # verify torch
    abcd = [*map(torch.from_numpy, (v_rr, v_ri, v_ir, v_ii))]
    pqrs = torch.stack(invsqrt_2x2(*abcd, eps=eps), dim=0)
    assert np.allclose(pqrs.numpy().reshape(2, 2, -1), R)


def test_batchnorm_layer():
    pass


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
    x = torch.from_numpy(np.random.normal(
        loc=5.0, scale=10.0, size=(1000, 10)
    )).to(torch.float)

    model = torch.nn.BatchNorm1d(10, affine=True)

    fit(x, x, model, n_steps=5)

    out = predict(x, model)
    out -= model.bias
    out /= model.weight

    assert np.isclose(float(out.mean()), 0.0, atol=1e-1)
    assert np.isclose(float(out.std()), 1.0, atol=1e-1)


def test_cplx_batchnorm(random_state):
    z = 1e-2 * (np.random.randn(1000, 10) + 1j * np.random.randn(1000, 10))
    z += np.random.randn(1, 10) * (1 + 1j)
    z = cplx.Cplx.from_numpy(z)

    out = cplx_batch_norm(z, None, None).numpy()
    re, im = out.real, out.imag

    assert np.isclose(re.mean(), 0.) and np.isclose(im.mean(), 0.)
    assert np.isclose((re*re).mean(), 1.) and np.isclose((im*im).mean(), 1.)


    x = torch.stack([
        torch.from_numpy(z.real),
        torch.from_numpy(z.imag),
    ], dim=-1).flatten(-2).to(torch.float)

    model = torch.nn.Sequential(
        RealToCplx(),
        CplxBatchNorm1d(10, affine=False),
        CplxToReal(),
    )

    model.train()
    feed = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x),
        batch_size=32, shuffle=True)
    for i in range(5):
        for bx, in feed:
            F.mse_loss(model(bx), bx)

    model.eval()


    out = predict(x, model)
    out -= model[1].bias
    out /= model[1].weight

    assert np.isclose(float(out.mean()), 0.0, atol=1e-1)
    assert np.isclose(float(out.std()), 1.0, atol=1e-1)


