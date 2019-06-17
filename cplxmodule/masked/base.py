import torch


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


def apply_masks(masks, *, source, target=None):
    assert False, """BAD DESIGN: Use a state-dict transformer interface."""
#     def do_sparsify(mod):
#         if isinstance(mod, BaseMasked):
#             mod.sparsify(~mod.get_sparsity_mask(threshold), mode=mode)

#     return module.apply(do_sparsify)
