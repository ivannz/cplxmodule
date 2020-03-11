import torch

from .base import CplxToCplx
from .container import CplxSequential
from ... import cplx


class CplxDropout(torch.nn.Dropout2d, CplxToCplx):
    r"""
    Complex 1d dropout layer: simultaneous dropout on both real and
    imaginary parts.

    See torch.nn.Dropout1d for reference on the input dimensions and arguments.
    """
    def forward(self, input):
        *head, n_last = input.shape

        # shape -> [*shape, 2] : re-im are feature maps!
        tensor = torch.stack([input.real, input.imag], dim=-1)
        output = super().forward(tensor.reshape(-1, 1, 2))

        # [-1, 1, 2] -> [*head, n_last * 2]
        output = output.reshape(*head, -1)

        # [*head, n_last * 2] -> [*head, n_last]
        return cplx.from_interleaved_real(output, False, -1)


class CplxAvgPool1d(torch.nn.AvgPool1d, CplxToCplx):
    r"""
    Complex 1d average pooling layer: simultaneously pools both real
    and imaginary parts.

    See torch.nn.AvgPool1d for reference on the input dimensions and arguments.
    """
    def forward(self, input):
        # apply parent.forward to re and im parts
        return input.apply(super().forward)


# i am lazy to rewrite code, so here is a factory
def torch_module(fn, base, name=None):
    # a class template
    class template_(*base):
        def forward(self, input):
            return fn(input)

    # update the name and qualname
    name_ = name if name is not None else fn.__name__.title()
    setattr(template_, "__name__", name_)
    setattr(template_, "__qualname__", name_)

    # update defaults
    for attr in ('__module__', '__doc__'):
        try:
            value = getattr(fn, attr)
        except AttributeError:
            pass
        else:
            setattr(template_, attr, value)
    # end for

    return template_


CplxExp = torch_module(cplx.exp, (CplxToCplx,), "CplxExp")

CplxLog = torch_module(cplx.log, (CplxToCplx,), "CplxLog")
