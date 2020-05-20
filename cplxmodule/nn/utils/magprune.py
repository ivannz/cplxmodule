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


def propagate(lookup, G, prefix="", value=None):
    """Assign values propagating if necessary.

    Details
    -------
    Yields all prefixes of `G` with valued taken from `lookup` or
    propagated from parent.
    """
    # '' is a the parent of all nodes (except itself)
    if "" in G:
        yield from propagate(lookup, set(n for n in G if n), prefix, value)
        return

    # lookup (or inherit) the parent's value
    value = lookup.get(prefix, value)  # lookup.get(prefix, 1.) * value
    yield prefix, value

    # collect children of the current prefix (aka `parent`)
    children, prefix = {}, prefix + ("." if prefix else "")
    for node in G:
        name, dot, child = node.partition(".")
        children.setdefault(prefix + name, set())
        if child:
            children[prefix + name].add(child)

    # propagate this parent's value to its children
    for prefix, G in children.items():
        yield from propagate(lookup, G, prefix, value)


def magprune(module, **sparsity):
    modules = dict(named_masked_modules(module))
    if not modules:
        raise TypeError(f"No maskable modules found.")

    # if module has a layer, but sparsity does not, then leave mask as is
    sparsity = dict(propagate(sparsity, modules))
    for name, mod in modules.items():
        if sparsity[name] is not None:
            mod.mask = compute_sparsity_mask(mod.weight, sparsity[name])
