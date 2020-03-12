import torch
import pytest
import numpy as np

from numpy.testing import assert_allclose

import torch.nn
import torch.sparse

import torch.nn.functional as F

from cplxmodule import Cplx

from torch.nn import Linear
from cplxmodule.nn import CplxLinear

from cplxmodule.nn.relevance import LinearARD, LinearL0, LinearLASSO
from cplxmodule.nn.relevance import CplxLinearARD

from cplxmodule.nn.masked import LinearMasked
from cplxmodule.nn.masked import CplxLinearMasked

from torch.nn import Bilinear
from cplxmodule.nn.relevance import BilinearARD
from cplxmodule.nn.masked import BilinearMasked

from cplxmodule.nn import CplxBilinear
from cplxmodule.nn.relevance import CplxBilinearARD
from cplxmodule.nn.masked import CplxBilinearMasked

from cplxmodule.nn.relevance import penalties, compute_ard_masks
from cplxmodule.nn.masked import deploy_masks, named_masks
from cplxmodule.nn.masked import binarize_masks
from cplxmodule.nn.utils.sparsity import sparsity, named_sparsity


@pytest.fixture
def random_state():
    return np.random.RandomState(None)  # (1249563438)


def test_torch_expi(random_state):
    from scipy.special import expi
    from cplxmodule.nn.relevance.complex import torch_expi

    npy = random_state.randn(200)
    trx = torch.tensor(npy)

    assert_allclose(torch_expi(trx), expi(npy))

    assert torch.autograd.gradcheck(torch_expi, trx.requires_grad_(True))


def model_train(X, y, model, n_steps=20000, threshold=1.0,
                reduction="mean", klw=1e-3, verbose=True):
    import tqdm

    model.train()
    optim = torch.optim.Adam(model.parameters())

    losses = []
    with tqdm.tqdm(range(n_steps)) as bar:
        for i in bar:
            optim.zero_grad()

            y_pred = model(X)

            mse = F.mse_loss(y_pred, y)
            kl_d = sum(penalties(model, reduction=reduction))

            loss = mse + klw * kl_d
            loss.backward()

            optim.step()

            losses.append(float(loss))
            if verbose:
                f_sparsity = sparsity(model, hard=True,
                                      threshold=threshold)
            else:
                f_sparsity = float("nan")

            bar.set_postfix_str(f"{f_sparsity:.1%} {float(mse):.3e} {float(kl_d):.3e}")
        # end for
    # end with
    return model.eval(), losses


def model_test(X, y, model, threshold=1.0):
    model.eval()
    with torch.no_grad():
        mse = F.mse_loss(model(X), y)
        kl_d = sum(penalties(model))

    f_sparsity = sparsity(model, hard=True, threshold=threshold)
    print(f"{f_sparsity:.1%} {mse.item():.3e} {float(kl_d):.3e}")
    return model


def example(kind="cplx"):
    r"""An example, illustrating pre-training."""

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
        from cplxmodule.nn import RealToCplx, CplxToReal
        from cplxmodule.nn import CplxAdaptiveModReLU

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
    if kind == "cplx":
        layers = [CplxLinear, CplxLinearARD, CplxLinearMasked]
        construct = construct_cplx
        reduction = "mean"
        phases = {
            "CplxLinear": (1000, 0.0),
            "CplxLinearARD": (4000, 1e-1),
            "CplxLinearMasked": (500, 0.0)
        }

    elif kind == "real-ard":
        layers = [Linear, LinearARD, LinearMasked]
        construct = construct_real
        reduction = "mean"
        phases = {
            "Linear": (1000, 0.0),
            "LinearARD": (4000, 1e-1),
            "LinearMasked": (500, 0.0)
        }

    elif kind == "real-l0":
        layers = [Linear, LinearL0, LinearMasked]
        construct = construct_real
        reduction = "sum"
        phases = {
            "Linear": (1000, 0.0),
            "LinearL0": (4000, 2e-2),
            "LinearMasked": (500, 0.0)
        }

    elif kind == "real-lasso":
        layers = [Linear, LinearLASSO, LinearMasked]
        construct = construct_real
        reduction = "mean"
        phases = {
            "Linear": (1000, 0.0),
            "LinearLASSO": (4000, 1e-1),
            "LinearMasked": (500, 0.0)
        }

    if kind == "real-lasso":
        tau = 0.25
    else:
        tau = 0.73105  # p = a / 1 + a, a = p / (1 - p)
    threshold = np.log(tau) - np.log(1 - tau)
    print(f"\n{80*'='}\n{tau:.1%} - {threshold:.3g}")

    n_features = 500 if "cplx" in kind else 250
    n_output = 20 if "cplx" in kind else 10

    # a simple dataset
    X = torch.randn(10100, n_features)
    y = - X[:, :n_output].clone()
    X, y = X.to(device_), y.to(device_)

    train_X, train_y = X[:100], y[:100]
    test_X, test_y = X[100:], y[100:]

    # construct models
    models = {"none": None}
    models.update({
        l.__name__: construct(l) for l in layers
    })

    # train a sequence of models
    names, losses = list(models.keys()), {}
    for src, dst in zip(names[:-1], names[1:]):
        print(f">>>>>> {dst}")
        n_steps, klw = phases[dst]

        # load the current model with the last one's weights
        model = models[dst]
        if models[src] is not None:
            # compute the dropout masks and normalize them
            state_dict = models[src].state_dict()
            masks = compute_ard_masks(models[src], hard=False,
                                      threshold=threshold)

            state_dict, masks = binarize_masks(state_dict, masks)

            # deploy old weights onto the new model
            model.load_state_dict(state_dict, strict=False)

            # conditionally deploy the computed dropout masks
            model = deploy_masks(model, state_dict=masks)

        model.to(device_)

        model, losses[dst] = model_train(train_X, train_y, model,
                                         n_steps=n_steps, threshold=threshold,
                                         klw=klw, reduction=reduction)
    # end for

    # get scores on test
    for key, model in models.items():
        if model is None:
            continue

        print(f"\n>>>>>> {key}")
        model_test(test_X, test_y, model, threshold=threshold)
        print(model.final.weight)
        print([*named_masks(model)])
        print([*named_sparsity(model, hard=True, threshold=threshold)])


def example_bilinear(kind="real"):
    r"""An example, illustrating pre-training."""
    from cplxmodule.nn import RealToCplx, CplxToReal
    from cplxmodule.cplx import from_real, to_real

    class BilinearTest(torch.nn.Module):
        def __init__(self, bilinear):
            super().__init__()
            self.final = bilinear(n_features, n_features, 1, bias=False)

        def forward(self, input):
            return self.final(input, input)

    class CplxBilinearTest(torch.nn.Module):
        def __init__(self, bilinear):
            super().__init__()
            self.cplx = RealToCplx()
            self.final = bilinear(n_features // 2, n_features // 2, 1, bias=False)
            self.real = CplxToReal()

        def forward(self, input):
            z = self.cplx(input)
            return self.real(self.final(z, z))

    device_ = torch.device("cpu")
    reduction = "mean"
    if kind == "cplx":
        layers = [CplxBilinear, CplxBilinearARD, CplxBilinearMasked]
        construct = CplxBilinearTest
        reduction = "mean"
        phases = {
            "CplxBilinear": (1000, 0.0),
            "CplxBilinearARD": (10000, 1e-1),
            "CplxBilinearMasked": (500, 0.0)
        }

    elif kind == "real":
        layers = [Bilinear, BilinearARD, BilinearMasked]
        phases = {
            "Bilinear": (1000, 0.0),
            "BilinearARD": (10000, 1e-1),
            "BilinearMasked": (500, 0.0)
        }
        construct = BilinearTest

    tau = 0.73105  # p = a / 1 + a, a = p / (1 - p)
    threshold = np.log(tau) - np.log(1 - tau)
    print(f"\n{80*'='}\n{tau:.1%} - {threshold:.3g}")

    n_features, n_output = 50, 10

    # a simple dataset : larger than in linear ARD!
    X = torch.randn(10500, n_features)
    out = X[:, :n_output]
    if "cplx" in kind:
        z = from_real(out, copy=False)
        y = - to_real(z.conj * z, flatten=False).mean(dim=-2)

    else:
        y = - (out * out).mean(dim=-1, keepdim=True)

    X, y = X.to(device_), y.to(device_)

    train_X, train_y = X[:500], y[:500]
    test_X, test_y = X[500:], y[500:]

    # construct models
    models = {"none": None}
    models.update({
        l.__name__: construct(l) for l in layers
    })

    # train a sequence of models
    names, losses = list(models.keys()), {}
    for src, dst in zip(names[:-1], names[1:]):
        print(f">>>>>> {dst}")
        n_steps, klw = phases[dst]

        # load the current model with the last one's weights
        model = models[dst]
        if models[src] is not None:
            # compute the dropout masks and normalize them
            state_dict = models[src].state_dict()
            masks = compute_ard_masks(models[src], hard=False,
                                      threshold=threshold)

            state_dict, masks = binarize_masks(state_dict, masks)

            # deploy old weights onto the new model
            print(model.load_state_dict(state_dict, strict=False))

            # conditionally deploy the computed dropout masks
            model = deploy_masks(model, state_dict=masks)

        model.to(device_)

        model, losses[dst] = model_train(train_X, train_y, model,
                                         n_steps=n_steps, threshold=threshold,
                                         klw=klw, reduction=reduction)
    # end for

    # get scores on test
    for key, model in models.items():
        if model is None:
            continue

        print(f"\n>>>>>> {key}")
        model_test(test_X, test_y, model, threshold=threshold)
        print(model.final.weight)
        print([*named_masks(model)])
        print([*named_sparsity(model, hard=True, threshold=threshold)])


if __name__ == '__main__':
    example("real-ard")
    example("real-l0")
    example("real-lasso")
    example("cplx")
    example_bilinear("real")
    example_bilinear("cplx")
