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
                                      True, missing, unexpected, error_msgs)

        mask = prefix + "mask"
        self.mask_(state_dict.get(mask, None))

        # clean up the missing/unexpected lists
        if mask in unexpected:
            # state comes from an actively masked layer
            unexpected.remove(mask)
        unexpected_keys.extend(unexpected)

        if mask in missing:
            # state comes from an unmasked layer
            missing.remove(mask)
        missing_keys.extend(missing)


class MaskedWeightMixin(BaseMasked):
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
            yield name, mod.mask


def deploy_masks(module, *, state_dict=None, prefix=""):
    if not isinstance(state_dict, dict) \
       or not isinstance(module, torch.nn.Module):
        return module

    for name, mod in module.named_modules(prefix=prefix):
        if isinstance(mod, BaseMasked):
            name = name + ("." if name else "") + "mask"
            mod.mask = state_dict.get(name, None)

    return module
