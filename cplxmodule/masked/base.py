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
    As of pytorch 1.1.0 there is no mechanism to preallocate runtime
    buffers before loading state_dict. Therefore as such masks should
    be loaded with `deploy_masks` and the whole model -- with
    `model.load_state_dict(..., strict=False)`.
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

        if not self.is_sparse and mask is not None:
            # None -> sparse : register mask, turning on sparsity

            # Detach (storage no-copy), move to device (likely no-copy) ...
            mask = mask.detach().to(self.weight.device, self.weight.dtype)
            # uses "clumsy" way of `.to()`, because `weight` might be Cplx.

            #  ... expand (no-copy) and make contiguous (copy).
            mask = mask.expand(self.weight.shape).contiguous()
            self.register_buffer("mask", mask)

        elif self.is_sparse and mask is not None:
            # sparse -> sparse : mask update
            self.mask.copy_(mask.detach())

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
        """Surgically load the state with the runtime masks from a dict.

        Details
        -------
        As of pytorch 1.1.0 there is no mechanism to preallocate runtime
        buffers before loading state_dict. Therefore we overload documented
        but hidden `_load_from_state_dict` to load the masks conditionally.
        """
        # this next call loads into this module only!
        missing, unexpected = [], []
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing, unexpected, error_msgs)

        mask = prefix + "mask"
        # mask was explicitly given: set own mask
        if mask in state_dict:
            self.mask_(state_dict[mask])

            # mask not in self => mask might be in unexpected (not in missing)
            if mask in unexpected:
                unexpected.remove(mask)

        # mask was not given => mask might be in missing (not in unexpected)
        elif mask not in missing and strict:
            # report missing mask if self has mask
            missing.append(mask)

        # mask was not given, and is either in missing, or not strict
        else:
            pass

        unexpected_keys.extend(unexpected)
        missing_keys.extend(missing)


class MaskedWeightMixin(BaseMasked):
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
    """Walk over the submodule tree yielding names and masks."""
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
