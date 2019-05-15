import torch
import torch.nn

from collections import OrderedDict

from .base import CplxToCplx
from .utils import is_cplx_to_cplx
from .cplx import cplx_add


class CplxSequential(torch.nn.Sequential, CplxToCplx):
    r"""
    Sequence of complex-to-complex modules:
    $$
        z_l = F_l(z_{l-1})
        \,, $$
    for $l=1..L$ and the complex input $z_0$.
    """
    def __init__(self, *args):
        # make a simple typecheck on the passed arguments
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            modules = args[0].items()
        else:
            modules = enumerate(args)

        bad_modules = [str(n) for n, m in modules if not is_cplx_to_cplx(m)]
        if bad_modules:
            raise TypeError(f"""Only complex-to-complex modules can be used """
                            f"""in {self.__class__.__name__}. The following """
                            f"""modules failed: {bad_modules}.""")

        super().__init__(*args)


class CplxResidualBottleneck(CplxToCplx):
    r"""A tag for Complex Linear layer, that alters the behaviour
    of the residual container.
    """
    pass


class CplxResidualSequential(CplxSequential):
    r"""
    Sequence of complex-to-complex residual modules:
    $$
        z_l = F_l(z_{l-1}) + z_{l-1}
        \,, $$
    for $l=1..L$ and the complex input $z_0$. In the case when $F_l$ is
    of type `CplxResidualBottleneck`, the update becomes:
    $$
        z_l = F_l(z_{l-1})
        \,. $$
    """
    def forward(self, input):
        for module in self:
            if isinstance(module, CplxResidualBottleneck):
                input = module(input)

            else:
                input = cplx_add(input, module(input))

        return input


class CplxBusResidualSequential(CplxSequential):
    r"""
    Sequence of complex-to-complex residual modules with a common ground
    bus `z_{bus}`:
    $$
        z_l = F_l(z_{l-1}) + z_{bus}
        \,, $$
    for $l=1..L$ and the complex input $z_0 = z_{bus}$.
    """
    def forward(self, input):
        # loop over the layers in sequence and add them to the bus
        bus = input
        for module in self:
            input = cplx_add(bus, module(input))
        return input
