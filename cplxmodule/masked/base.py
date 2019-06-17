import torch
from ..relevance.base import BaseARD


class BaseMasked(torch.nn.Module):
    @property
    def is_sparse(self):
        """Check if the layer is in sparse mode."""
        raise NotImplementedError("Derived classes must "
                                  "implement sparsity flag.")

    def mask_(self, mask):
        raise NotImplementedError("Derived classes must implement "
                                  "`sparsify` action.")

    def __setattr__(self, name, value):
        """Special routing syntax like `.require_grad = ...`."""
        if name != "mask":
            return super().__setattr__(name, value)

        self.mask_(value)


def is_sparse(module):
    if isinstance(module, BaseMasked):
        return module.is_sparse

    return False


def named_masks(module, prefix=""):
    # yields own mask and masks of every descendant
    for name, mod in module.named_modules(prefix=prefix):
        if isinstance(mod, BaseMasked):
            yield name, getattr(mod, "mask", None)


def deploy_masks(module, *, state_dict=None, prefix=""):
    if not isinstance(state_dict, dict) \
       or not isinstance(module, torch.nn.Module):
        return module

    for name, mod in module.named_modules(prefix=prefix):
        if isinstance(mod, BaseMasked):
            name = name + ("." if name else "") + "mask"
            mod.mask = state_dict.get(name, None)

    return module


def compute_ard_masks(module, *, threshold=None, prefix=""):
    if not isinstance(threshold, (float, int)) \
       or not isinstance(module, torch.nn.Module):
        return {}

    masks = {}
    for name, mod in module.named_modules(prefix=prefix):
        if isinstance(mod, BaseARD):
            mask = mod.get_sparsity_mask(threshold).detach().clone()
            name = name + ("." if name else "") + "mask"
            masks[name] = ~mask

    return masks
