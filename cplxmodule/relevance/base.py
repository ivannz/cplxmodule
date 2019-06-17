import torch


class BaseARD(torch.nn.Module):
    r"""\alpha-based variational dropout."""
    @property
    def penalty(self):
        # if the porperty in a derived class raises then it gets
        # silently undefined due to inheritance from troch.nn.Module.
        raise NotImplementedError("Derived classes must compute "
                                  "their own penalty.")

    def get_sparsity_mask(self, threshold):
        raise NotImplementedError("Derived classes must implement a method to "
                                  "get a binary mask of dropped out values.")

    def num_zeros(self, threshold):
        raise NotImplementedError("Derived classes must implement a "
                                  "zero-counting method.")


def named_penalties(module, prefix=""):
    # yields own penalty and penalties of all descendants
    for name, mod in module.named_modules(prefix=prefix):
        if isinstance(mod, BaseARD):
            yield name, mod.penalty


def penalties(module):
    for name, penalty in named_penalties(module):
        yield penalty


def sparsity(module, threshold=1.0):
    # traverse all parameters
    n_par = sum(par.numel()
                for name, par in module.named_parameters()
                if "log_sigma2" not in name
                   and "log_alpha" not in name)

    # query the modules about their immediate parameters
    n_zer = sum(mod.num_zeros(threshold)
                for name, mod in module.named_modules()
                if isinstance(mod, BaseARD))

    return n_zer / max(n_par, 1)
