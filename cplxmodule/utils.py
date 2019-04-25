import torch
import torch.nn

from .base import CplxToCplx
from .base import CplxToReal
from .base import RealToCplx


def is_from_cplx(module):
    if isinstance(module, (CplxToCplx, CplxToReal)):
        return True

    if isinstance(module, torch.nn.Sequential):
        return is_from_cplx(module[0])

    return False


def is_to_cplx(module):
    if isinstance(module, (CplxToCplx, RealToCplx)):
        return True

    if isinstance(module, torch.nn.Sequential):
        return is_to_cplx(module[-1])

    return False


def is_cplx_to_cplx(module):
    return is_from_cplx(module) and is_to_cplx(module)


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
