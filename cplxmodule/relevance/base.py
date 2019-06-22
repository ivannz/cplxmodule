import torch


class BaseARD(torch.nn.Module):
    r"""\alpha-based variational dropout."""
    @property
    def penalty(self):
        # if the property in a derived class raises then it gets
        # silently undefined due to inheritance from torch.nn.Module.
        raise NotImplementedError("Derived classes must compute "
                                  "their own penalty.")

    def relevance(self, **kwargs):
        raise NotImplementedError("Derived classes must implement a float "
                                  "mask of relevant coefficients.")


def named_penalties(module, reduction="mean", prefix=""):
    """Generator of named penalty terms with specified reduction."""
    if reduction is not None and reduction not in ("mean", "sum"):
        raise ValueError(f"""`reduction` must be either `None`, "sum" """
                         f"""or "mean". Got {reduction}.""")

    # yields own penalty and penalties of all descendants
    for name, mod in module.named_modules(prefix=prefix):
        if isinstance(mod, BaseARD):
            penalty = mod.penalty
            if reduction == "mean":
                penalty = penalty.mean()

            elif reduction == "sum":
                penalty = penalty.sum()

            yield name, penalty


def penalties(module, reduction="mean"):
    for name, penalty in named_penalties(module, reduction=reduction):
        yield penalty


def named_relevance(module, prefix="", **kwargs):
    """A generator of relevance masks and submodules owning them."""
    for name, mod in module.named_modules(prefix=prefix):
        if isinstance(mod, BaseARD):
            yield name, mod.relevance(**kwargs).detach()


def compute_ard_masks(module, *, prefix="", **kwargs):
    if not isinstance(module, torch.nn.Module):
        return {}

    relevance = named_relevance(module, prefix=prefix, **kwargs, hard=True)

    return {
        name + ("." if name else "") + "mask": mask
        for name, mask in relevance
    }
