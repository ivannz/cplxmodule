import torch
import warnings


class SparsityStats(object):
    __sparsity_ignore__ = ()

    def sparsity(self, **kwargs):
        raise NotImplementedError("Derived classes must implement "
                                  "a method to estimate sparsity.")


def named_sparsity(module, prefix="", **kwargs):
    """A generator of parameter names and their sparsity statistics.

    Details
    -------
    Relies on the fact that SparsityStats subclasses identify parameters,
    they wish to report sparsity for, by their id(). This is done to prevent
    incorrect estimation in case of parameter sharing.
    """

    warnings.warn("Since v2020.06 module's buffers are also accounted "
                  "by `named_sparsity`.", FutureWarning)

    # gather the dropout statistics and service parameters to ignore
    n_dropout, p_service = {}, set()

    # query the modules about their immediate parameters
    for name, mod in module.named_modules(prefix=prefix):
        if isinstance(mod, SparsityStats):
            # collect parameters to ignore
            name = name + ("." if name else "")
            p_service.update(name + k for k in mod.__sparsity_ignore__)

            # collect parameter id-keyed number of hard zeros
            n_dropout.update(mod.sparsity(**kwargs))

    for name, par in module.named_parameters(prefix=prefix):
        if name not in p_service:
            yield name, (n_dropout.get(id(par), 0.), par.numel())

    for name, buf in module.named_buffers(prefix=prefix):
        if name not in p_service:
            yield name, (n_dropout.get(id(buf), 0.), buf.numel())


def sparsity(module, **kwargs):
    pairs = (s for n, s in named_sparsity(module, **kwargs))
    n_zer, n_par = map(sum, zip(*pairs))
    return n_zer / max(n_par, 1)
