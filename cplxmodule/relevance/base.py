import torch


class BaseARD(torch.nn.Module):
    r"""\alpha-based variational dropout."""
    __ard_ignore__ = ()

    @property
    def penalty(self):
        # if the property in a derived class raises then it gets
        # silently undefined due to inheritance from torch.nn.Module.
        raise NotImplementedError("Derived classes must compute "
                                  "their own penalty.")

    def relevance(self, threshold, hard=False):
        raise NotImplementedError("Derived classes must implement a float "
                                  "mask of relevant coefficients.")

    def _sparsity(self, threshold, hard=True):
        raise NotImplementedError("Derived classes must implement "
                                  "a method to estimate sparsity.")


def named_penalties(module, reduction="mean", prefix=""):
    """Generator of named penalty terms with specified reduction."""
    if reduction is not None and reduction not in ("mean", "sum"):
        raise ValueError(f"""`reduction` must be either `None`, "sum" """
                         f"""or "mean". Got {reduction}.""")

    # yields own penalty and penalties of all descendants
    for name, mod in module.named_modules(prefix=prefix):
        if isinstance(mod, BaseARD):
            kl_div = mod.penalty
            if reduction == "mean":
                kl_div = kl_div.mean()

            elif reduction == "sum":
                kl_div = kl_div.sum()

            yield name, kl_div


def penalties(module, reduction="mean"):
    for name, penalty in named_penalties(module, reduction=reduction):
        yield penalty


def named_sparsity(module, threshold, hard=True, prefix=""):
    """A generator of parameter names and their sparsity statistics.

    Details
    -------
    Relies on the fact that BaseARD subclasses identify parameters, they wish
    to report sparsity for, by their id(). This is done to prevent incorrect
    estimation in case of parameter sharing.
    """

    # gather the dropout statistics and service parameters to ignore
    n_dropout, p_service = {}, set()

    # query the modules about their immediate parameters
    for name, mod in module.named_modules(prefix=prefix):
        if isinstance(mod, BaseARD):
            # collect parameters to ignore
            name = name + ("." if name else "")
            p_service.update(name + k for k in mod.__ard_ignore__)

            # collect parameter id-keyed number of hard zeros
            n_dropout.update(mod._sparsity(threshold=threshold, hard=hard))

    for name, par in module.named_parameters(prefix=prefix):
        if name not in p_service:
            yield name, (n_dropout.get(id(par), 0.), par.numel())


def sparsity(module, threshold=1.0, hard=True):
    pairs = (s for n, s in named_sparsity(module, hard=hard,
                                          threshold=threshold))
    n_zer, n_par = map(sum, zip(*pairs))
    return n_zer / max(n_par, 1)


def named_relevance(module, threshold=1.0, hard=False, prefix=""):
    """A generator of relevance masks and submodules owning them."""

    for name, mod in module.named_modules(prefix=prefix):
        if isinstance(mod, BaseARD):
            yield name, mod.relevance(threshold, hard=hard).detach()
