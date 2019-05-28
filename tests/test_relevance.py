import torch
import pytest
import numpy as np

from numpy.testing import assert_allclose

import torch.nn
import torch.sparse

import torch.nn.functional as F

from cplxmodule.layers import CplxLinear

from cplxmodule.relevance import penalties, sparsity, make_sparse
from cplxmodule.relevance import LinearARD

from cplxmodule.relevance import CplxLinearARD
from cplxmodule.relevance.complex import CplxLinearARDApprox
from cplxmodule.relevance.complex import CplxLinearARDBogus


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
    def build_cplx_model(dropout=True):
        from cplxmodule.layers import RealToCplx, CplxToReal
        from cplxmodule.activation import CplxModReLU

        # linear = CplxLinear if not dropout else CplxLinearARDBogus
        linear = CplxLinear if not dropout else CplxLinearARD
        return torch.nn.Sequential(
            RealToCplx(),
            linear(250, 10, bias=False),

            # linear(250, 500, bias=True),
            # CplxModReLU(threshold=0.05),
            # linear(500, 10, bias=False),

            CplxToReal()
        )

    def build_real_model(dropout=True):
        linear = torch.nn.Linear if not dropout else LinearARD
        return torch.nn.Sequential(
            linear(250, 10, bias=False),

            # linear(250, 20, bias=True),
            # torch.nn.LeakyReLU(negative_slope=0.05),
            # linear(20, 10, bias=False),

            # linear(250, 100, bias=True),
            # torch.nn.LeakyReLU(negative_slope=0.05),
            # linear(100, 50, bias=True),
            # torch.nn.LeakyReLU(negative_slope=0.05),
            # linear(50, 10, bias=False),

            # linear(250, 500, bias=True),
            # torch.nn.LeakyReLU(negative_slope=0.05),
            # linear(500, 10, bias=False),

            # linear(250, 100, bias=True),
            # torch.nn.LeakyReLU(negative_slope=0.05),
            # linear(100, 50, bias=True),
            # torch.nn.LeakyReLU(negative_slope=0.05),
            # linear(50, 20, bias=True),
            # torch.nn.LeakyReLU(negative_slope=0.05),
            # linear(20, 10, bias=False),
        )

    def train_model(X, y, model, n_steps=20000, klw=1e-3, verbose=True):
        import tqdm

        model.train()
        optim = torch.optim.Adam(model.parameters())
        with tqdm.tqdm(range(n_steps)) as bar:
            for i in bar:
                optim.zero_grad()

                y_pred = model(X)

                mse = F.mse_loss(y_pred, y)
                kl_d = sum(penalties(model))

                loss = mse + klw * kl_d
                loss.backward()

                optim.step()

                sprsty = sparsity(model) if verbose else float("nan")
                bar.set_postfix_str(
                    f"""{sprsty:.0%} {mse.item():.3e} """
                    f"""{float(kl_d):.3e}"""
                )
            # end for
        # end with
        return model.eval()

    def test_model(X, y, model):
        model.eval()
        with torch.no_grad():
            mse = F.mse_loss(model(X), y)
            kl_d = sum(penalties(model))

        print(f"""{sparsity(model):.1%} {mse.item():.3e} {float(kl_d):.3e}""")
        return model

    build_model = build_cplx_model if cplx else build_real_model
    device_ = torch.device("cpu")

    n_features = 500 if cplx else 250
    n_output = 20 if cplx else 10

    # a simple dataset
    X = torch.randn(100, n_features)
    y = X[:, :n_output].clone()

    X, y = X.to(device_), y.to(device_)

    # train simple dense net
    model = build_model(False)
    model.to(device_)

    model = train_model(X, y, model, 1000)

    # use pre-trained for automatic relevance detection
    model_sparse = build_model(True)
    # if cplx:
    #     model_sparse[1].exact = False
    model_sparse.to(device_)
    model_sparse.load_state_dict(model.state_dict(), strict=False)

    model_sparse = train_model(X, y, model_sparse, n_steps=9000, klw=1e-1)

    # sparsify the model
    model_sparse.eval()
    print(model_sparse)

    print(make_sparse(model_sparse, 1.0, mode="sparse"))
    if not cplx:
        print(model_sparse[0].sparse_weight_)
    else:
        print(model_sparse[1].sparse_re_weight_)
        print(model_sparse[1].sparse_im_weight_)

    # get scores on test
    X = torch.randn(1250, n_features)
    y = X[:, :n_output].clone()

    X, y = X.to(device_), y.to(device_)

    test_model(X, y, model)
    test_model(X, y, model_sparse)


if __name__ == '__main__':
    example(False)
    example(True)
