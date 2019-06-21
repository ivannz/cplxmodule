import torch
from ..relevance.base import BaseARD


class BaseMasked(torch.nn.Module):
    @property
    def is_sparse(self):
        """Check if the layer is in sparse mode."""
        return isinstance(getattr(self, "mask", None), torch.Tensor)

    def mask_(self, value):
        if not self.is_sparse and value is not None:
            # None -> sparse : register mask, turning on sparsity
            device, dtype = self.weight.device, self.weight.dtype
            self.register_buffer("mask", value.detach().to(device, dtype))

        elif self.is_sparse and value is not None:
            # sparse -> sparse : mask update
            device, dtype = self.weight.device, self.weight.dtype
            self.mask.data = value.detach().to(device, dtype)

        elif self.is_sparse and value is None:
            # sparse -> None : remove the mask
            del self.mask

        elif not self.is_sparse and value is None:
            # None -> None : nothing
            pass

        return self

    def __setattr__(self, name, value):
        """Special routing syntax like `.require_grad = ...`."""
        if name != "mask":
            return super().__setattr__(name, value)

        self.mask_(value)


class SparseWeightMixin(BaseMasked):
    @property
    def weight_masked(self):
        """Return a sparsified weight of the parent *Linear."""
        # __rmul__ by the mask in case of weird weight types
        return (self.weight * self.mask) if self.is_sparse else self.weight


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
    if not isinstance(module, torch.nn.Module):
        return {}

    masks = {}
    for name, mod in module.named_modules(prefix=prefix):
        if isinstance(mod, BaseARD):
            name = name + ("." if name else "") + "mask"
            masks[name] = mod.relevance(threshold).detach()

    return masks
