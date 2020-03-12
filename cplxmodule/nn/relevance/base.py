import torch


class BaseARD(torch.nn.Module):
    r"""\alpha-based variational dropout.

    Attributes
    ----------
    penalty : computed torch.Tensor, read-only
        The Kullback-Leibler divergence between the mean field approximate
        variational posterior of the weights and the scale-free log-uniform
        prior:
        $$
            KL(\mathcal{N}(w\mid \theta, \alpha \theta^2) \|
                    \tfrac1{\lvert w \rvert})
                = \mathbb{E}_{\xi \sim \mathcal{N}(1, \alpha)}
                    \log{\lvert \xi \rvert}
                - \tfrac12 \log \alpha + C
            \,. $$

    log_alpha : computed torch.Tensor, read-only
        Log-variance of the multiplicative scaling noise. Computed as a log
        of the ratio of the variance of the weight to the squared absolute
        value of the weight. The higher the log-alpha the less relevant the
        parameter is.

    Get the dropout mask based on the confidence level $\tau \in (0, 1)$:
        $$
            \Pr(\lvert w_i \rvert > 0)
                \leq \Pr(z_i \neq 0)
                = 1 - \sigma\bigl(
                    \log\alpha + \beta \log \tfrac{-\gamma}{\zeta}
                \bigr)
                \leq \tau
            \,. $$
        For $\tau=0.25$ and $\beta=0.66$ we have `threshold=2.96`.

    """
    @property
    def penalty(self):
        """Get the penalty induced by the variational approximation.

        Returns
        -------
        mask : torch.Tensor, differentiable, read-only
            The penalty term to be computed, collected, and added to the
            negative log-likelihood. In variational dropout and automatic
            relevance determination methods this is the kl-divergence term
            of the ELBO, which depends on the variational approximation,
            and not the input data (like in VAE). The requires the use of
            forward hooks with specific trait modules that compute the
            differentiable penalty on forward pass. Which is currently out
            of the scope of this package.

        Details
        -------
        Making penalty into a property emphasizes its read-only-ness, however
        the same could've been achieved with a method.
        """

        # if the property in a derived class raises then it gets
        # silently undefined due to inheritance from torch.nn.Module.
        raise NotImplementedError("Derived classes must compute "
                                  "their own penalty.")

    def relevance(self, **kwargs):
        r"""Get the dropout mask based on the provided parameters.

        Returns
        -------
        mask : torch.Tensor
            A nonnegative tensor of the same shape as the `.weight` parameter
            with explicit zeros indicating a dropped out parameter, a value to
            be set to and fixed at zero. A nonzero value indicates a relevant
            parameter, which is to be kept. A binary mask is `hard`, whereas a
            `soft` mask may use arbitrary positive values (not necessarily in
            [0, 1]) to represent retained parameters. Soft masks might occur in
            sparsification methods, where the sparsity mask is learnt and
            likely co-adapts to the weights, e.g. in \ell_0 probabilistic
            regularization. Soft masks would require cleaning up to eliminate
            the result of such co-adaptation (see `nn.masked.binarize_masks`).
        """
        raise NotImplementedError("Derived classes must implement a float "
                                  "mask of relevant coefficients.")


def named_penalties(module, reduction="sum", prefix=""):
    """Returns an iterator over all penalties in the network, yielding
    both the name of a variational submodule as well as the value of its
    penalty.

    Parameters
    ----------
    module : torch.nn.Module
        The network, which is scanned for variational modules.

    reduction : string, default='sum'
        The reduction method for the penalty tensors collected from variational
        modules. Allowed settings are 'mean', 'sum', and None, with the latter
        disabling reduction and returning penalty tensors as-is.

    prefix : string, default empty
        The prefix for the yielded names.

    Yields
    ------
    (string, torch.Tensor):
        Name and the differentiable value of the penalty for the variational
        approximation used in the submodule.

    Note
    ----
    Penalties from duplicate (shared or recurrent) modules are computed and
    returned only once.

    Details
    -------
    Summing the KL-divergences is the recommended setting, as it is grounded
    in variational inference theory. 'mean' option is provided for future
    extensions only, since averaging KL-divergence penalties adversely drives
    `smaller` layers (convolutional versus dense) to higher sparsity, which
    is the opposite of the desired outcome from variational dropout.
    """

    if reduction is not None and reduction not in ("mean", "sum"):
        raise ValueError(f"`reduction` must be either `None`,"
                         f" `sum` or `mean`. Got {reduction}.")

    # yields own penalty and penalties of all descendants
    for name, mod in module.named_modules(prefix=prefix):
        if isinstance(mod, BaseARD):
            penalty = mod.penalty
            if reduction == "sum":
                penalty = penalty.sum()

            elif reduction == "mean":
                penalty = penalty.mean()

            yield name, penalty


def penalties(module, reduction="sum"):
    """Returns an simplified iterator over all penalties in the network.
    See notes and details in `named_penalties`.

    Parameters
    ----------
    module : torch.nn.Module
        The network, which is scanned for variational modules.

    reduction : string, default='sum'
        The reduction method for the penalty tensors collected from variational
        modules. Allowed settings are 'mean', 'sum', and None, with the latter
        disabling reduction and returning penalty tensors as-is.

    Yields
    ------
    torch.Tensor
        A differentiable value of the penalty.
    """
    for name, penalty in named_penalties(module, reduction=reduction):
        yield penalty


def named_relevance(module, prefix="", **kwargs):
    """A generator of relevance masks and submodules owning them.

    Parameters
    ----------
    module : torch.nn.Module
        The network, which is scanned for variational modules.

    prefix : string, default empty
        The prefix for the yielded names.

    **kwargs : variable keyworded arguments
        The parameters of the relevance masks to pass to upstream
        implementations.

    Yields
    ------
    torch.Tensor
        A non-differentiable tensor of the computed relevance mask.
    """
    for name, mod in module.named_modules(prefix=prefix):
        if isinstance(mod, BaseARD):
            yield name, mod.relevance(**kwargs).detach()


def compute_ard_masks(module, *, prefix="", **kwargs):
    """Create a dict of properly named masks, compatible with `nn.masked`.

    Parameters
    ----------
    module : torch.nn.Module
        The network, which is scanned for variational modules.

    prefix : string, default empty
        The prefix for the yielded mask names.

    **kwargs : variable keyworded arguments
        The parameters of the relevance masks to pass to upstream
        implementations.

    Yields
    ------
    torch.Tensor
        A non-differentiable tensor of the computed relevance mask.
    """
    if not isinstance(module, torch.nn.Module):
        return {}

    relevance = named_relevance(module, prefix=prefix, **kwargs)
    return {
        name + ("." if name else "") + "mask": mask
        for name, mask in relevance
    }
