"""Magnitude pruning with `nn.masked` layers."""
import torch
from math import ceil

from ..masked.base import BaseMasked


def named_masked_modules(module, prefix=""):
    # yields every descendant maskable module
    for name, mod in module.named_modules(prefix=prefix):
        if isinstance(mod, BaseMasked):
            yield name, mod


def compute_sparsity_mask(weight, sparsity):
    """Get a mask with the specified sparsity.

    Details
    -------
    Following Zhu and Gupta (2017) this procedure sorts the absolute magnitudes
    of weights and masks the smallest ones so that the desired sparsity level
    is reached. Since the modules derived from `BaseMasked` apply the mask
    multiplicatively, the pruned parameters' gradients are explicitly zeroed.
    """
    with torch.no_grad():
        tensor = abs(weight)

        rank = ceil(float(sparsity) * (tensor.numel() - 1))
        val, idx = torch.kthvalue(tensor.flatten(), rank)

        return tensor.gt(val).to(val)


def _propagate(modules, values, prefix=""):
    yield prefix, values.get("")

    children = {}
    for name in modules:
        parent, dot, child = name.partition(".")
        if parent:
            children.setdefault(parent, set()).add(child)

    for parent, submodules in children.items():
        # by default children inherit value from parent
        value = values.get(parent, values.get(""))
        subvalues = dict.fromkeys(submodules, value)

        # add child-specific values
        for name, value in values.items():
            if name.startswith(parent + "."):
                subvalues[name[len(parent) + 1:]] = value

        subprefix = prefix + ("." if prefix else "") + parent
        yield from _propagate(submodules, subvalues, subprefix)


def propagate_sparsity_targets(modules, sparsity):
    return {k: v for k, v in _propagate(modules, sparsity)
            if k in modules and v is not None}


def magprune(module, **sparsity):
    modules = dict(named_masked_modules(module))
    if not modules:
        raise TypeError(f"No maskable modules found.")

    sparsity = propagate_sparsity_targets(modules, sparsity)

    # if module has a layer, but sparsity does not, then leave mask as is
    for name, level in sparsity.items():
        mod = modules[name]
        mod.mask = compute_sparsity_mask(mod.weight, level)
