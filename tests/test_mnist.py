import tqdm

import torch
import torch.nn.functional as F

from torchvision import datasets, transforms

from cplxmodule.nn.relevance import penalties
from cplxmodule.nn.utils.sparsity import sparsity

from torch.nn import Linear, Conv2d
from cplxmodule.nn.relevance import LinearARD, Conv2dARD
from cplxmodule.nn.masked import LinearMasked, Conv2dMasked

from cplxmodule.nn.relevance import compute_ard_masks
from cplxmodule.nn.masked import binarize_masks, deploy_masks
from cplxmodule.nn.masked import named_masks
from cplxmodule.nn.utils.sparsity import named_sparsity


def model_fit(model, feed, optim, n_steps=100, threshold=1.0,
              klw=1e-3, reduction="mean", verbose=True):
    losses = []
    with tqdm.tqdm(range(n_steps)) as bar:
        model.train()
        for i in bar:
            for data, target in feed:
                optim.zero_grad()

                n_ll = F.nll_loss(model(data), target, reduction=reduction)
                kl_d = sum(penalties(model, reduction=reduction))

                loss = n_ll + klw * kl_d
                loss.backward()

                optim.step()

                losses.append(float(loss))
                if verbose:
                    f_sparsity = sparsity(model, hard=True,
                                          threshold=threshold)
                else:
                    f_sparsity = float("nan")

                bar.set_postfix_str(
                    f"{f_sparsity:.1%} {float(n_ll):.3e} {float(kl_d):.3e}"
                )
            # end for
        # end for
    # end with
    return model.eval(), losses


def model_predict(model, feed):
    """Compute the model prediction on data from the feed."""
    pred, fact = [], []
    with tqdm.tqdm(feed) as bar, torch.no_grad():
        model.eval()
        for data, *rest in bar:
            pred.append(model(data))
            if rest:
                fact.append(rest[0])

    fact = torch.cat(fact, dim=0).cpu() if fact else None
    return torch.cat(pred, dim=0).cpu(), fact


def model_score(model, feed, threshold=1.0):
    from sklearn.metrics import confusion_matrix
    import re

    model.eval()
    pred, fact = model_predict(model, feed)

    n_ll = F.nll_loss(pred, fact, reduction="mean")
    kl_d = sum(penalties(model))

    f_sparsity = sparsity(model, hard=True, threshold=threshold)

    # C_{ij} = \hat{P}(y = i & \hat{y} = j)
    cm = confusion_matrix(fact.numpy(), pred.numpy().argmax(axis=-1))

    tp = cm.diagonal()
    fp, fn = cm.sum(axis=1) - tp, cm.sum(axis=0) - tp

    # format the arrays and remove clutter
    p_str = re.sub("[',]", "", str([f"{p:4.0%}" for p in tp / (tp + fp)]))
    r_str = re.sub("[',]", "", str([f"{p:4.0%}" for p in tp / (tp + fn)]))
    print(
        f"(S) {f_sparsity:.1%} ({float(kl_d):.2e}) "
        f"(A) {tp.sum() / cm.sum():.1%} ({n_ll.item():.2e})"
        f"\n(P) {p_str}"  # \approx (y = i \mid \hat{y} = i)
        f"\n(R) {r_str}"  # \approx (\hat{y} = i \mid y = i)
    )
    print(re.sub(r"(?<=\D)0", ".", str(cm)))

    return model


class FeedWrapper(object):
    """A wrapper for a dataLoader that puts batches on device on the fly.

    Parameters
    ----------
    feed : torch.utils.data.DataLoader
        The data loader instance to be wrapped.

    **kwargs : keyword arguments
        The keyword arguments to be passed to `torch.Tensor.to()`.
    """
    def __init__(self, feed, **kwargs):
        assert isinstance(feed, torch.utils.data.DataLoader)
        self.feed, self.kwargs = feed, kwargs

    def __len__(self):
        return len(self.feed)

    def __iter__(self):
        if not self.kwargs:
            yield from iter(self.feed)

        else:
            for batch in iter(self.feed):
                yield tuple(b.to(**self.kwargs)
                            for b in batch)


class Model(torch.nn.Module):
    """A convolutional net."""
    def __init__(self, conv2d=Conv2d, linear=Linear):
        super().__init__()

        self.conv1 = conv2d(1, 20, 5, 1)
        self.conv2 = conv2d(20, 50, 5, 1)
        self.fc1 = linear(4 * 4 * 50, 500)
        self.fc2 = linear(500, 10)

    def forward(self, x):
        x = F.avg_pool2d(F.relu(self.conv1(x)), 2, 2)
        x = F.avg_pool2d(F.relu(self.conv2(x)), 2, 2)
        x = F.relu(self.fc1(x.reshape(-1, 4 * 4 * 50)))
        return F.log_softmax(self.fc2(x), dim=1)


if __name__ == '__main__':
    threshold = 3.0
    device_ = torch.device("cuda:3")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # reverse the roles of the MNIST train-test split
    mnist_test = datasets.MNIST('./data', transform=transform,
                                train=True, download=True)
    mnist_train = datasets.MNIST('./data', transform=transform,
                                 train=False)

    # define wrapped data feeds
    feeds = {
        "train": FeedWrapper(
            torch.utils.data.DataLoader(
                mnist_train, batch_size=32, shuffle=True),
            device=device_),
        "test": FeedWrapper(
            torch.utils.data.DataLoader(
                mnist_test, batch_size=256, shuffle=False),
            device=device_),
    }

    # Models and training setttings
    models = {
        "none": None,
        "dense": Model(Conv2d, Linear),
        "ard": Model(Conv2dARD, LinearARD),
        "masked": Model(Conv2dMasked, LinearMasked),
    }

    phases = {
        "dense": (40, 0.0),
        "ard": (40, 1e-2),
        "masked": (20, 0.0),
    }

    # the main loop: transfer weights and masks and then train
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
        optim = torch.optim.Adam(model.parameters())
        model, losses[dst] = model_fit(
            model, feeds["train"], optim, n_steps=n_steps,
            threshold=threshold, klw=klw, reduction="mean")

    # run tests
    for key, model in models.items():
        if model is None:
            continue

        print(f"\n>>>>>> {key}")
        model_score(model, feeds["test"], threshold=threshold)
        # print([*named_masks(model)])
        print([*named_sparsity(model, hard=True, threshold=threshold)])
