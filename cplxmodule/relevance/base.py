import warnings

import torch
import torch.nn.functional as F

from .utils import torch_sparse_linear, torch_sparse_tensor
from .utils import parameter_to_buffer, buffer_to_parameter


class BaseARD(torch.nn.Module):
    r"""\alpha-based variational dropout."""
    @property
    def penalty(self):
        # if the porperty in a derived class raises then it gets
        # silently undefined due to inheritance from troch.nn.Module.
        raise NotImplementedError("Derived classes must compute "
                                  "their own penalty.")

    def get_sparsity_mask(self, threshold):
        raise NotImplementedError("Derived classes must implement a method to "
                                  "get a binary mask of dropped out values.")

    def num_zeros(self, threshold):
        raise NotImplementedError("Derived classes must implement a "
                                  "zero-counting method.")


def named_penalties(module, prefix=""):
    # yields own penalty and penalties of all descendants
    for name, mod in module.named_modules(prefix=prefix):
        if isinstance(mod, BaseARD):
            yield name, mod.penalty


def penalties(module):
    for name, penalty in named_penalties(module):
        yield penalty


def sparsity(module, threshold=1.0):
    # traverse all parameters
    n_par = sum(par.numel()
                for name, par in module.named_parameters()
                if "log_sigma2" not in name
                   and "log_alpha" not in name)

    # query the modules about their immediate parameters
    n_zer = sum(mod.num_zeros(threshold)
                for name, mod in module.named_modules()
                if isinstance(mod, BaseARD))

    return n_zer / max(n_par, 1)


class SparseModeMixin(object):
    def forward_sparse(self, input):
        if self.sparsity_mode_ == "dense":
            # Profiling -- Elementwise multiplication faster than `.where()`
            return F.linear(input, self.weight_ * self.nonzero_, self.bias)

        elif self.sparsity_mode_ == "sparse":
            weight_ = torch_sparse_tensor(
                self.nonzero_, self.weight_, self.weight.shape)
            return torch_sparse_linear(input, weight_, self.bias)

        raise RuntimeError(f"Unrecognized sparsity mode. "
                           f"Got `{self.sparsity_mode_}`")

    @property
    def is_sparse(self):
        mode = getattr(self, "sparsity_mode_", None)
        return mode is not None

    def sparsify(self, mask=None, mode="dense"):
        if not hasattr(self, "sparsity_mode_"):
            self.sparsity_mode_ = None

        if mode is not None and mode not in ("dense", "sparse"):
            raise ValueError(f"`mode` must be either 'dense', 'sparse' "
                             f"or `None`. Got '{mode}'.")

        if mode == "sparse":
            # creating a sparse tensor time and time again is expensive,
            #  but is required for backward, so cannot be cached. And the
            #  current hurdles of spasre matrix multiplcation just aren't
            #  worth it. Besides (cu)BLAS is blazingly fast anyway.
            warnings.warn("mode 'sparse' will likely be discontinued "
                          "and later deprecated.", DeprecationWarning)

        if mask is not None and (mask.dtype not in (torch.bool, torch.uint8)
           or mask.shape != self.weight.shape):
            raise RuntimeError(f"`mask` must be None or a binary matrix "
                               f"{self.weight.shape}. Got '{mask.shape}'.")

        if mask is None:
            mode = None

        # make weight into a buffer (load_state dict doesn't care about
        #  param/buffer distinction!)

        # None -> sparse/dense : mutate par-to-buf
        if not self.is_sparse and mode is not None:
            nonzero_ = mask.to(self.weight)
            if mode == "sparse":
                # truly sparse mode: using torch sparse tensor
                weight_ = torch.masked_select(self.weight.detach(),
                                              mask.to(self.weight.device))
                nonzero_ = nonzero_.nonzero().t()

            elif mode == "dense":
                # simulated sparse mode: using dense matrices with hard zeros
                weight_ = self.weight.detach() * nonzero_

            self.register_parameter("weight_", torch.nn.Parameter(weight_))
            self.register_buffer("nonzero_", nonzero_)

            # lastly, mutate the original parameter into a no-grad buffer
            parameter_to_buffer(self, "weight")

        # sparse/dense -> None : mutate buf-to-par
        elif self.is_sparse and mode is None:
            # some copying on new learnt weights could take place here.
            pass

            del self.nonzero_, self.weight_
            buffer_to_parameter(self, "weight")

        # sparse / dense -> dense / sparse : re-mutatation
        elif self.is_sparse and mode is not None:
            # sparse -> sparse or dense -> dense : check mask
            if self.sparsity_mode_ == mode:
                # binary masks or nonzero indices are exactly equal : nothing
                if mode == "sparse":
                    nonzero_ = mask.nonzero().t()
                    if torch.equal(nonzero_.to(self.nonzero_), self.nonzero_):
                        return self

                elif mode == "dense":
                    if torch.equal(mask.to(self.nonzero_), self.nonzero_):
                        return self

            # sparse -> dense or dense -> sparse : re-mutate
            else:
                pass

            # perform "sparse/dense -> None -> sparse/dense" : discards data
            self.sparsify(mask, mode=None)
            return self.sparsify(mask, mode=mode)

        # None -> None : nothing
        elif not self.is_sparse and mode is None:
            pass

        self.sparsity_mode_ = mode
        return self


def make_sparse(module, threshold, mode="dense"):
    def do_sparsify(mod):
        if isinstance(mod, SparseModeMixin) and isinstance(mod, BaseARD):
            mod.sparsify(~mod.get_sparsity_mask(threshold), mode=mode)

    return module.apply(do_sparsify)
