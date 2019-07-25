import tqdm

import torch
import torch.nn.functional as F

from torchvision import datasets, transforms

from cplxmodule.relevance import penalties
from cplxmodule.utils.stats import sparsity

from cplxmodule.relevance.real import LinearARD, Conv2dARD

from cplxmodule.relevance import compute_ard_masks
from cplxmodule.masked import binarize_masks, deploy_masks
from cplxmodule.masked import named_masks
from cplxmodule.utils.stats import named_sparsity


class SimpleNet(torch.nn.Module):
    def __init__(self, conv2d=torch.nn.Conv2d, linear=torch.nn.Linear):
        super().__init__()

        self.conv1 = conv2d(1, 20, 5, 1)
        self.conv2 = conv2d(20, 50, 5, 1)
        self.fc1 = linear(4 * 4 * 50, 500)
        self.fc2 = linear(500, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2, 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2, 2)
        x = self.fc2(F.relu(self.fc1(x.view(-1, 4 * 4 * 50))))
        return F.log_softmax(x, dim=1)


def train_model(model, feed, optim, n_steps=100, threshold=1.0,
                reduction="mean", klw=1e-3, verbose=True):
    model.train()
    losses = []
    with tqdm.tqdm(range(n_steps)) as bar:
        for i in bar:
            for data, target in feed:
                optim.zero_grad()

                n_ll = F.nll_loss(model(data), target)
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

                bar.set_postfix_str(f"{f_sparsity:.1%} {float(n_ll):.3e} {float(kl_d):.3e}")
            # end for
        # end for
    # end with
    return model.eval(), losses


def test_model(model, feed, threshold=1.0):
    model.eval()
    with tqdm.tqdm(feed) as bar, torch.no_grad():
        n_ll = torch.cat([
            F.nll_loss(model(data), target, reduction=None)
            for data, target in bar
        ], dim=0).mean()

        kl_d = sum(penalties(model))

    f_sparsity = sparsity(model, hard=True, threshold=threshold)
    print(f"{f_sparsity:.1%} {n_ll.item():.3e} {float(kl_d):.3e}")
    return model


class FeedWrapper():
    def __init__(self, feed, **kwargs):
        self.feed, self.kwargs = feed, kwargs

    def __iter__(self):
        self.iter_ = iter(self.feed)
        return self

    def __next__(self):
        return tuple(
            b.to(**self.kwargs)
            for b in next(self.iter_)
        )

    def __len__(self):
        return len(self.feed)


threshold = 3.0
device_ = torch.device("cpu")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# reverse the roles of the MNIST train-test split
mnist_test = datasets.MNIST('./data', transform=transform, train=True, download=True)
mnist_train = datasets.MNIST('./data', transform=transform, train=False)

feeds = {
    "train": FeedWrapper(
        torch.utils.data.DataLoader(
            mnist_train, batch_size=256, shuffle=True),
        device=device_),
    "test": FeedWrapper(
        torch.utils.data.DataLoader(
            mnist_test, batch_size=256, shuffle=False),
        device=device_),
}


models = {"none": None}
models.update({
    "dense": SimpleNet(torch.nn.Conv2d, torch.nn.Linear),
    "ard": SimpleNet(Conv2dARD, LinearARD),
})

phases = {
    "dense": (40, 0.0),
    "ard": (80, 1e-1),
}


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
    model, losses[dst] = train_model(
        model,
        feeds["train"],
        optim, n_steps=n_steps, threshold=threshold, klw=klw, reduction="mean")


for key, model in models.items():
    if model is None:
        continue

    print(f"\n>>>>>> {key}")
    test_model(model, feeds["test"], threshold=threshold)
    print([*named_masks(model)])
    print([*named_sparsity(model, hard=True, threshold=threshold)])
