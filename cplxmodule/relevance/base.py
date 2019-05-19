import torch


def apply(module, fn, memo=None):
    r"""Traverse the modules and iterate over generator `fn`."""
    if memo is None:
        # The set of visited modules to forbid reentry
        memo = set()

    if module not in memo:
        memo.add(module)
        yield from fn(module)

        # `.children()` traverses all directly descendant modules.
        for submod in module.children():
            yield from apply(submod, fn, memo)
    # end if


class BaseARD(torch.nn.Module):
    r"""\alpha-based variational dropout."""
    @property
    def penalty(self):
        # if the porperty in a derived class raises then it gets
        # silently undefined due to inheritance from troch.nn.Module.
        raise NotImplementedError("""Derived classes must compute their own """
                                  """variational penalty.""")


def penalties(module):
    # yields own penalty and penalties of all descendants
    def get_penalty(mod):
        if isinstance(mod, BaseARD):
            yield mod.penalty

    yield from apply(module, get_penalty)


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
    n_par = sum(par.numel()
                for name, par in module.named_parameters()
                if "log_sigma2" not in name)

    def get_sparsity(mod):
        if isinstance(mod, BaseLinearARD):
            yield mod.num_zeros(threshold)

    n_zer = sum(apply(module, get_sparsity))
    return n_zer / max(n_par, 1)


def make_sparse(module, threshold=1.0, mode="dense"):
    def do_sparsify(mod):
        if isinstance(mod, BaseLinearARD):
            mod.sparsify(threshold, mode=mode)

    return module.apply(do_sparsify)
