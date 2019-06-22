import torch
from ..relevance.base import named_relevance


class SparsityStats(object):
    __sparsity_ignore__ = ()

    def sparsity(self, **kwargs):
        raise NotImplementedError("Derived classes must implement "
                                  "a method to estimate sparsity.")


def compute_ard_masks(module, *, prefix="", **kwargs):
    if not isinstance(module, torch.nn.Module):
        return {}

    relevance = named_relevance(module, prefix=prefix, **kwargs)

    return {
        name + ("." if name else "") + "mask": mask
        for name, mask in relevance
    }


def named_sparsity(module, prefix="", **kwargs):
    """A generator of parameter names and their sparsity statistics.

    Details
    -------
    Relies on the fact that SparsityStats subclasses identify parameters,
    they wish to report sparsity for, by their id(). This is done to prevent
    incorrect estimation in case of parameter sharing.
    """

    # gather the dropout statistics and service parameters to ignore
    n_dropout, p_service = {}, set()

    # query the modules about their immediate parameters
    for name, mod in module.named_modules(prefix=prefix):
        if isinstance(mod, SparsityStats):
            # collect parameters to ignore
            name = name + ("." if name else "")
            p_service.update(name + k for k in mod.__sparsity_ignore__)

            # collect parameter id-keyed number of hard zeros
            n_dropout.update(mod._sparsity(**kwargs))

    for name, par in module.named_parameters(prefix=prefix):
        if name not in p_service:
            yield name, (n_dropout.get(id(par), 0.), par.numel())


def sparsity(module, **kwargs):
    pairs = (s for n, s in named_sparsity(module, **kwargs))
    n_zer, n_par = map(sum, zip(*pairs))
    return n_zer / max(n_par, 1)
