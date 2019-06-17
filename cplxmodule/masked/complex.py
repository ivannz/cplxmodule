import torch

from ..layers import CplxLinear
from .base import BaseMasked

from ..cplx import Cplx


class CplxLinearMasked(CplxLinear, BaseMasked):
    @property
    def weight(self):
        """Return a sparsified weight.

        This works both for the default `forward()` inherited from Linear,
        and for what the user expects to see when they request weight from
        the layer (masked zero values).
        """
        weight = super().__getattr__("weight")
        if not self.is_sparse:
            return weight

        return Cplx(weight.real, weight.imag) * self.mask

    @property
    def is_sparse(self):
        """Check if the layer is in sparse mode."""
        return isinstance(getattr(self, "mask", None), torch.Tensor)

    def mask_(self, value):
        if not self.is_sparse and value is not None:
            # None -> sparse : register mask, turning on sparsity
            self.register_buffer("mask", value.detach().to(self.weight.imag))

        elif self.is_sparse and value is not None:
            # sparse -> sparse : mask update
            self.mask.copy_(value.detach())

        elif self.is_sparse and value is None:
            # sparse -> None : remove the mask
            del self.mask

        elif not self.is_sparse and value is None:
            # None -> None : nothing
            pass

        return self
