import scipy
import scipy.special

import torch
import torch.sparse

import torch.nn.functional as F


class ExpiFunction(torch.autograd.Function):
    r"""Pythonic differentiable port of scipy's Exponential Integral Ei.

    $$
        Ei
            \colon \mathbb{R} \to \mathbb{R} \cup \{\pm \infty\}
            \colon x \mapsto \int_{-\infty}^x \tfrac{e^t}{t} dt
        \,. $$

    Notes
    -----
    This may potentially introduce a memory transfer and compute bottleneck
    during the forward pass due to CPU-GPU device switch. Backward pass does
    not suffer from this issue and is compute on-device.
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)

        x_cpu = x.data.cpu().numpy()
        output = scipy.special.expi(x_cpu, dtype=x_cpu.dtype)
        return torch.from_numpy(output).to(x.device)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[-1]
        return grad_output * torch.exp(x) / x


torch_expi = ExpiFunction.apply


def kldiv_approx(log_alpha, coef, reduction):
    r"""Sofplus-sigmoid approximation.
    $$
        \alpha \mapsto
            k_1 \sigma(k_2 + k_3 \log \alpha) + C
                - k_4 \log (1 + e^{-\log \alpha})
        \,, $$
    for $C = - k_1$.
    """
    if reduction is not None and reduction not in ("mean", "sum"):
        raise ValueError("""`reduction` must be either `None`, "sum" """
                         """or "mean".""")

    k1, k2, k3, k4 = coef
    C = -k1

    # $x \mapsto \log(1 + e^x)$ is softplus and needs different
    #  compute paths depending on the sign of $x$:
    #  $$ x\mapsto \log(1+e^{-\lvert x\rvert}) + \max{\{x, 0\}} \,. $$
    sigmoid = torch.sigmoid(k2 + k3 * log_alpha)
    softplus = - k4 * F.softplus(- log_alpha)
    kl_div = k1 * sigmoid + softplus + C

    if reduction == "mean":
        return kl_div.mean()

    elif reduction == "sum":
        return kl_div.sum()

    return kl_div
