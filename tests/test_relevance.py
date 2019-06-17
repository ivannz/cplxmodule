import torch
import pytest
import numpy as np

from numpy.testing import assert_allclose

import torch.nn
import torch.sparse

import torch.nn.functional as F

from cplxmodule import Cplx

from torch.nn import Linear
from cplxmodule.layers import CplxLinear

from cplxmodule.relevance import LinearARD
from cplxmodule.relevance import CplxLinearARD

from cplxmodule.masked import LinearMasked
from cplxmodule.masked import CplxLinearMasked

from cplxmodule.relevance import penalties, sparsity
from cplxmodule.masked import deploy_masks, compute_ard_masks


@pytest.fixture
def random_state():
    return np.random.RandomState(None)  # (1249563438)


def test_torch_expi(random_state):
    from scipy.special import expi
    from cplxmodule.relevance.utils import torch_expi

    npy = random_state.randn(200)
    trx = torch.tensor(npy)

    assert_allclose(torch_expi(trx), expi(npy))

    assert torch.autograd.gradcheck(torch_expi, trx.requires_grad_(True))


def example(cplx=False):
    r"""An example, illustrating pre-training."""

    def train_model(X, y, model, n_steps=20000, threshold=1.0, klw=1e-3):
        import tqdm

        model.train()
        optim = torch.optim.Adam(model.parameters())

        losses = []
        with tqdm.tqdm(range(n_steps)) as bar:
            for i in bar:
                optim.zero_grad()

                y_pred = model(X)

                mse = F.mse_loss(y_pred, y)
                kl_d = sum(penalties(model))

                loss = mse + klw * kl_d
                loss.backward()

                optim.step()

                losses.append(float(loss))
                f_sparsity = sparsity(model, threshold)
                bar.set_postfix_str(f"{f_sparsity:.1%} {float(mse):.3e} {float(kl_d):.3e}")
            # end for
        # end with
        return model.eval(), losses

    def test_model(X, y, model, threshold=1.0):
        model.eval()
        with torch.no_grad():
            mse = F.mse_loss(model(X), y)
            kl_d = sum(penalties(model))

        print(f"{sparsity(model, threshold):.1%} "
              f"{mse.item():.3e} {float(kl_d):.3e}")
        return model

    def construct_real(linear):
        from collections import OrderedDict

        return torch.nn.Sequential(OrderedDict([
            ("body", torch.nn.Sequential(OrderedDict([
                # ("linear", linear(n_features, n_features, bias=True)),
                # ("relu", torch.nn.LeakyReLU()),
            ]))),
            ("final", linear(n_features, n_output, bias=False)),
        ]))

    def construct_cplx(linear):
        from collections import OrderedDict
        from cplxmodule.layers import RealToCplx, CplxToReal
        from cplxmodule.activation import CplxAdaptiveModReLU

        return torch.nn.Sequential(OrderedDict([
            ("cplx", RealToCplx()),
            ("body", torch.nn.Sequential(OrderedDict([
                # ("linear", linear(n_features // 2, n_features // 2, bias=True)),
                # ("relu", CplxAdaptiveModReLU(n_features // 2)),
            ]))),
            ("final", linear(n_features // 2, n_output // 2, bias=False)),
            ("real", CplxToReal()),
        ]))

    device_ = torch.device("cpu")
    if cplx:
        layers = [CplxLinear, CplxLinearARD, CplxLinearMasked]
        construct = construct_cplx
        phases = {
            "CplxLinear": 1000,
            "CplxLinearARD": 4000,
            "CplxLinearMasked": 500
        }
    else:
        layers = [Linear, LinearARD, LinearMasked]
        construct = construct_real
        phases = {
            "Linear": 1000,
            "LinearARD": 4000,
            "LinearMasked": 500
        }

    tau = 0.73105  # p = a / 1 + a, a = p / (1 - p)
    threshold = np.log(tau) - np.log(1 - tau)
    print(f"{tau:.1%} - {threshold:.3g}")

    n_features = 500 if cplx else 250
    n_output = 20 if cplx else 10

    # a simple dataset
    X = torch.randn(100, n_features)
    y = X[:, :n_output].clone()

    X, y = X.to(device_), y.to(device_)

    # construct models
    models = {"none": None}
    models.update({
        l.__name__: construct(l) for l in layers
    })

    # train a sequence of models
    names, losses = list(models.keys()), {}
    for src, dst in zip(names[:-1], names[1:]):

        # load the current model with the last one's weights
        model = models[dst]
        if models[src] is not None:
            model.load_state_dict(models[src].state_dict(), strict=False)

        model.to(device_)

        # compute the dropout masks and
        masks = compute_ard_masks(models[src], threshold=threshold)

        # conditionally deploy the computed dropout masks
        model = deploy_masks(model, state_dict=masks)

        model, losses[dst] = train_model(X, y, model, n_steps=phases[dst],
                                         threshold=threshold, klw=1e-1)
    # end for

    # get scores on test
    X = torch.randn(10000, n_features)
    y = X[:, :n_output].clone()

    X, y = X.to(device_), y.to(device_)

    for key, model in models.items():
        if model is None:
            continue

        test_model(X, y, model, threshold=threshold)
        # print([*model.named_parameters()])
        if cplx:
            weight = model.final.weight
            print(Cplx(weight.real, weight.imag))
        else:
            print(model.final.weight)


if __name__ == '__main__':
    example(False)
    example(True)
