import tqdm

import torch
import torch.nn.functional as F

from torchvision import datasets, transforms

from cplxmodule.relevance import penalties
from cplxmodule.utils.stats import sparsity

from cplxmodule.relevance.real import LinearARD
from cplxmodule.relevance.real import Conv2dARD


class Net(torch.nn.Module):
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


def train_model(model, feed, optim, threshold=1.0,
                reduction="mean", klw=1e-3, verbose=True):
    model.train()
    losses = []
    with tqdm.tqdm(feed) as bar:
        for data, target in bar:
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
    # end with
    return model.eval(), losses


def test_model(model, feed, threshold=1.0):
    model.eval()
    losses = []
    with tqdm.tqdm(feed) as bar, torch.no_grad():
        for data, target in bar:
            n_ll = F.nll_loss(model(data), target)

        kl_d = sum(penalties(model))

    f_sparsity = sparsity(model, hard=True, threshold=threshold)
    print(f"{f_sparsity:.1%} {n_ll.item():.3e} {float(kl_d):.3e}")
    return model


transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
])

train_feed = torch.utils.data.DataLoader(
        datasets.MNIST('./data', transform=transform, train=True, download=True),
        batch_size=64, shuffle=True)

test_feed = torch.utils.data.DataLoader(
        datasets.MNIST('./data', transform=transform, train=False),
        batch_size=256, shuffle=False)


device_ = torch.device("cpu")
model = Net(Conv2dARD, LinearARD).to(device_)
optim = torch.optim.Adam(model.parameters())

train_model(model, train_feed, optim)
test_model(model, test_feed)
