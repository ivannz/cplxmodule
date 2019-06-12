import torch


class BaseARD(torch.nn.Module):
    r"""\alpha-based variational dropout."""
    @property
    def penalty(self):
        # if the porperty in a derived class raises then it gets
        # silently undefined due to inheritance from troch.nn.Module.
        raise NotImplementedError("""Derived classes must compute their own """
                                  """variational penalty.""")


def named_penalties(module, prefix=""):
    # yields own penalty and penalties of all descendants
    for name, mod in module.named_modules():
        if isinstance(mod, BaseARD):
            yield name, mod.penalty


def penalties(module):
    for name, penalty in named_penalties(module):
        yield mod.penalty


class BaseLinearARD(BaseARD):
    @property
    def log_alpha(self):
        raise NotImplementedError("""Derived classes must implement """
                                  """log-relevance as a read-only property.""")

    def get_sparsity_mask(self, threshold):
        r"""Get the dropout mask based on the log-relevance."""
        with torch.no_grad():
            return torch.ge(self.log_alpha, threshold)

    @property
    def is_sparse(self):
        mode = getattr(self, "sparsity_mode_", None)
        return mode is not None

    def sparsify(self, threshold=1.0, mode="dense"):
        raise NotImplementedError("""Derived classes must implement a """
                                  """sparsification method.""")

    def num_zeros(self, threshold=1.0):
        raise NotImplementedError("""Derived classes must implement a """
                                  """zero-counting method.""")

    # def extra_repr(self):
    #     return "" if not self.is_sparse else f"mode={self.sparsity_mode_}"


def sparsity(module, threshold=1.0):
    # traverse all parameters
    n_par = sum(par.numel()
                for name, par in module.named_parameters()
                if "log_sigma2" not in name)

    # query the modules about their immediate parameters
    n_zer = sum(mod.num_zeros(threshold)
                for name, mod in module.named_modules()
                if isinstance(mod, BaseLinearARD))

    return n_zer / max(n_par, 1)


def make_sparse(module, threshold=1.0, mode="dense"):
    def do_sparsify(mod):
        if isinstance(mod, BaseLinearARD):
            mod.sparsify(threshold, mode=mode)

    return module.apply(do_sparsify)
