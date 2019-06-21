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


def named_penalties(module, prefix=""):
    # yields own penalty and penalties of all descendants
    for name, mod in module.named_modules(prefix=prefix):
        if isinstance(mod, BaseARD):
            yield name, mod.penalty


def penalties(module):
    for name, penalty in named_penalties(module):
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
