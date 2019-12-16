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

    # by hand
    z = 1e-2 * (np.random.randn(1000, 10) + 1j * np.random.randn(1000, 10))
    z += np.random.randn(1000, 10) * (1 + 1j)

    # center and get variance
    x = np.stack([z.real, z.imag], axis=-1)
    c = x - x.mean(axis=0, keepdims=True)
    var = np.einsum("biu, biv->iuv", c, c) / len(c)

    # compute the inverse root
    eps = 1e-8
    a, b, d = var[:, 0, 0], var[:, 0, 1], var[:, 1, 1]
    s = np.sqrt(np.maximum(a * d - b * b, eps))
    t = np.sqrt(a + d + 2 * s) * s
    M = np.array([[d + s, -b], [-b, a + s]]) / t

    res = np.einsum("biu, uvi->biv", c, M)

    m = res.mean(axis=0, keepdims=True)
    v = np.einsum("biu, biv->iuv", res - m, res - m) / len(res)
    assert np.allclose(m, 0.)
    assert np.allclose(v, np.eye(2)[np.newaxis], atol=1e-1)

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


