import torch


class BaseMasked(torch.nn.Module):
    """The base class for linear layers that should have fixed sparsity
    pattern.

    Attributes
    ----------
    is_sparse : bool, read-only
        Indicates if the instance has a valid usable mask.

    mask : torch.Tensor
        The current mask used for the weights. Always guaranteed to have
        the same shape, dtype (float, double) and be on the same device
        as `.weight`.

    Details
    -------
    As of pytorch 1.1.0 there is no mechanism to preallocate runtime buffers
    before loading state_dict. Thus we use a custom `__init__` and 'piggy-back'
    on the documented, but private method `Module._load_from_state_dict` to
    conditionally allocate or free mask buffers via `.mask_` method. This API
    places a restriction on the order of bases classes when subclassing. In
    order for `super().__init__` in subclasses to do the necessary mask setting
    up and initialize the `torch.nn.Module` itself, the `BaseMasked` should be
    placed as far to the right in base class list, but before `torch.nn.Module`.

    The masks could be set either manually through setting `.mask`, or loaded in
    bulk with `deploy_masks`, or via `model.load_state_dict(..., strict=False)`.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("mask", None)

    @property
    def is_sparse(self):
        """Check if the layer is in sparse mode."""
        return isinstance(self.mask, torch.Tensor)

    def mask_(self, mask):
        """Update or reset the mask to a new one, broadcasting it if necessary.

        Arguments
        ---------
        mask : torch.Tensor or None
            The mask to be used. Device migration, dtype conversion and
            broadcasting are done automatically to conform to `.weight`.

        Details
        -------
        Effectively switches on / off masking of the weights.
        """
        if mask is not None and not isinstance(mask, torch.Tensor):
            raise TypeError(f"`mask` must be either a Tensor or "
                            f"`None`. Got {type(mask).__name__}.")

        if mask is not None:
            # None -> sparse : register mask, turning on sparsity
            # sparse -> sparse : mask update

            # Detach (storage no-copy), move to device (likely no-copy) ...
            mask = mask.detach().to(self.weight.device, self.weight.dtype)
            # uses "clumsy" way of `.to()`, because `weight` might be Cplx.

            #  ... expand (no-copy) and make contiguous (copy).
            mask = mask.expand(self.weight.shape).contiguous()
            self.register_buffer("mask", mask)

        elif self.is_sparse and mask is None:
            # sparse -> None : remove the mask and re-register the buffer
            del self.mask

            self.register_buffer("mask", None)

        elif not self.is_sparse and mask is None:
            # None -> None : nothing
            pass

        return self

    def __setattr__(self, name, value):
        """Special routing syntax like `.require_grad = ...`."""
        if name != "mask":
            return super().__setattr__(name, value)

        self.mask_(value)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        """Surgically load the state with the runtime masks from a dict."""
        # this next call loads everything, expect mask into this module only!
        mask = prefix + "mask"
        super()._load_from_state_dict(
            {k: v for k, v in state_dict.items() if k != mask}, prefix,
            local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        # here: mask in missing <=> buffer exists and not None <=> is_sparse
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/module.py#L758
        mask_in_missing = mask in missing_keys

        # mask was explicitly given: set own mask, mask cannot be in unexpected
        if mask in state_dict:
            if mask_in_missing:
                missing_keys.remove(mask)

            self.mask_(state_dict[mask])

        # must report absent mask, regardless if self is sparse or not
        elif strict:
            if not mask_in_missing:
                missing_keys.append(mask)

        # ignore missing mask if not strict
        elif mask_in_missing:  # mask not in state_dict and not strict
            missing_keys.remove(mask)


class MaskedWeightMixin():
    """A mixin for accessing read-only masked weight,"""
    @property
    def weight_masked(self):
        """Return a sparsified weight of the parent *Linear."""
        if not self.is_sparse:
            msg = f"`{type(self).__name__}` has no sparsity mask. Please, " \
                  f"either set a mask attribute, or call `deploy_masks()`."
            raise RuntimeError(msg)

        # __rmul__ by the mask in case of weird weight types (like Cplx)
        return self.weight * self.mask


def is_sparse(module):
    """Check if a module is sparse and has a set mask."""
    if isinstance(module, BaseMasked):
        return module.is_sparse

    return False


def named_masks(module, prefix=""):
    """Returns an iterator over all masks in the network, yielding both
    the name of a maskable submodule as well as its current mask.

    Parameters
    ----------
    module : torch.nn.Module
        The network, which is scanned for variational modules.

    prefix : string, default empty
        The prefix for the yielded names.

    Yields
    ------
    (string, torch.Tensor):
        Name and the mask used to activate/deactivate the parameters.

    Note
    ----
    Masks from duplicate (shared or recurrent) modules are returned only once.
    """

    # yields own mask and masks of every descendant
    for name, mod in module.named_modules(prefix=prefix):
        if isinstance(mod, BaseMasked):
            yield name, mod.mask


def deploy_masks(module, *, state_dict=None, prefix="", reset=False):
    """Apply mask to the Masked submodules.

    Arguments
    ---------
    module : torch.nn.Module
        The module, children of which will have their masks updated or reset.

    state_dict : dict, keyword-only
        The dictionary of masks. Keys indicate which layer's mask to update
        or reset and the value represents the masking tensor. Explicitly
        providing `None` under some key resets disables the sparsity of that
        layer.

    prefix : str, default
        The prefix to append to submodules' names during module tree walk.

    reset : bool, default False
        Whether to forcefully reset the sparsity of the layers, masks for
        which were NOT provided in `state_dict`.
    """
    if not isinstance(state_dict, dict) \
       or not isinstance(module, torch.nn.Module):
        return module

    for name, mod in module.named_modules(prefix=prefix):
        if isinstance(mod, BaseMasked):
            name = name + ("." if name else "") + "mask"
            if name in state_dict:
                # mask was given :: update
                mod.mask = state_dict[name]

            elif reset:
                # mask is missing and reset :: set to None
                mod.mask = None

            else:
                # mask is missing and not reset :: nothing
                pass

    return module


def binarize_masks(state_dict, masks):
    """Normalize the weight and masks, so that masks are binary.

    Arguments
    ---------
    state_dict : dict of tensors
        A dictionary of tensors, usually from `.state_dict()` call.

    masks : dict of tensors
        A dictionary of masks for the weights in `state_dict`.

    Returns
    -------
    state_dict : dict of tensors
        A dictionary of tensors, multiplied by their corresponding mask.

    masks : dict of tensors
        A dictionary of binarized masks.
    """
    with torch.no_grad():
        new_state_dict = {}
        for name, par in state_dict.items():
            if "weight" in name:
                mask = name.rsplit("weight", 1)[0] + "mask"
                if mask in masks:
                    par = par * masks[mask].to(par)

                    # clean up negative hard zeros (sign bit is set)
                    par[abs(par) == 0] = 0

            new_state_dict[name] = par

        new_masks = {name: torch.ne(mask, 0).to(mask)
                     for name, mask in masks.items()}

    return new_state_dict, new_masks
