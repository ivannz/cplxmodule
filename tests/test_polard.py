import torch
import numpy as np

import torch.nn
import torch.sparse

import torch.nn.functional as F

from cplxmodule import Cplx
from cplxmodule import cplx

from cplxmodule.nn import CplxLinear
from cplxmodule.nn.relevance import CplxLinearVD, CplxLinearARD
from cplxmodule.nn.relevance.extensions.complex.polard import CplxLinearPolARD
from cplxmodule.nn.masked import CplxLinearMasked

from cplxmodule.nn.relevance import penalties, compute_ard_masks
from cplxmodule.nn.masked import deploy_masks, named_masks
from cplxmodule.nn.masked import binarize_masks
from cplxmodule.nn.utils.sparsity import sparsity, named_sparsity


def model_train(X, y, model, n_steps=20000, threshold=1.0,
                reduction="sum", klw=1e-3, verbose=True):
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

    def construct_cplx(linear):
        from collections import OrderedDict
        from cplxmodule.nn import RealToCplx, CplxToReal
        from cplxmodule.nn import CplxAdaptiveModReLU

        return torch.nn.Sequential(OrderedDict([
            ("cplx", RealToCplx()),
            # ("body", torch.nn.Sequential(OrderedDict([
            #     ("linear", linear(n_features // 2, n_features // 2, bias=True)),
            #     ("relu", CplxAdaptiveModReLU(n_features // 2)),
            # ]))),
            ("final", linear(n_features // 2, n_output // 2, bias=False)),
            ("real", CplxToReal()),
        ]))

    device_ = torch.device("cpu")
    if kind == "cplx-vd":
        layers = [CplxLinear, CplxLinearVD, CplxLinearMasked]
        construct = construct_cplx
        reduction = "mean"
        phases = {
            "CplxLinear": (1000, 0.0),
            "CplxLinearVD": (14000, 1e0),
            "CplxLinearMasked": (500, 0.0)
        }

    elif kind == "cplx-ard":
        layers = [CplxLinear, CplxLinearARD, CplxLinearMasked]
        construct = construct_cplx
        reduction = "mean"
        phases = {
            "CplxLinear": (1000, 0.0),
            "CplxLinearARD": (14000, 1e0),
            "CplxLinearMasked": (500, 0.0)
        }

    elif kind == "cplx-polar":
        layers = [CplxLinear, CplxLinearPolARD, CplxLinearMasked]
        construct = construct_cplx
        reduction = "mean"
        phases = {
            "CplxLinear": (1000, 0.0),
            "CplxLinearPolARD": (14000, 1e0),
            "CplxLinearMasked": (500, 0.0)
        }

    tau = 0.73105  # p = a / 1 + a, a = p / (1 - p)
    # tau = 0.5  # p = a / 1 + a, a = p / (1 - p)
    threshold = np.log(tau) - np.log(1 - tau)
    print(f"\n{80*'='}\n{tau:.1%} - {threshold:.3g}")

    n_features = 500 if "cplx" in kind else 250
    n_output = 20 if "cplx" in kind else 10

    # a simple dataset
    X = torch.randn(10100, n_features)
    y = - X[:, :n_output].clone()
    X, y = X.to(device_), cplx.from_interleaved_real(y.to(device_))
    # y = cplx.to_interleaved_real(y * y)
    y = cplx.to_interleaved_real(y * (1j))

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
    import matplotlib.pyplot as plt
    for key, model in models.items():
        if model is None:
            continue

        print(f"\n>>>>>> {key}")
        model_test(test_X, test_y, model, threshold=threshold)
        print(model.final.weight)
        print([*named_masks(model)])
        print([*named_sparsity(model, hard=True, threshold=threshold)])

        with torch.no_grad():
            if isinstance(model.final, (CplxLinearVD, CplxLinearARD,
                                        CplxLinearPolARD)):

                plt.hist(model.final.log_alpha.cpu().numpy().ravel(),
                         bins=21)
                plt.show()

                plt.hist(model.final.log_sigma2.cpu().data.numpy().ravel(),
                         bins=21)
                plt.show()

            if isinstance(model.final, CplxLinearPolARD):
                # print(model.final.log_eta)
                # print(model.final.log_phi)
                # print(model.final.log_var)

                eta = torch.tanh(model.final.log_eta / 2)
                phi = torch.atan(model.final.log_phi)

                fig = plt.figure(figsize=(7, 7))
                ax = fig.add_subplot(111, projection="polar")
                # ax.scatter(np.angle(roots), np.abs(roots), c=colors, s=50)
                eta = eta.cpu().numpy().ravel()
                phi = phi.cpu().numpy().ravel()
                ax.scatter(np.where(eta < 0, phi + np.pi, phi), abs(eta), s=50)
                # ax.set_rlim(0, 1.1)

                plt.show()
                # mult = cplx.Cplx(vareta * torch.cos(phi),
                #                  vareta * torch.sin(phi))


if __name__ == '__main__':
    example("cplx-vd")
    example("cplx-ard")
    example("cplx-polar")
