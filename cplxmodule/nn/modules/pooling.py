from .base import CplxToCplx
from ... import cplx


class CplxMaxPoolNd(CplxToCplx):
    r"""An almost verbatim copy of `_MaxPoolNd` from torch/nn/modules/pooling.py"""

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def extra_repr(self) -> str:
        return (
            "kernel_size={kernel_size}, stride={stride}, padding={padding}"
            ", dilation={dilation}, ceil_mode={ceil_mode}".format(**self.__dict__)
        )


class CplxMaxPool1d(CplxMaxPoolNd):
    r"""Applies a 1D max pooling over a complex input signal composed of
    several input planes.

    See Also
    --------
    See docs for the parameters of `torch.nn.MaxPool1d`.
    """

    def forward(self, input):
        return cplx.max_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
        )


class CplxMaxPool2d(CplxMaxPoolNd):
    r"""Applies a 2D max pooling over a complex input signal composed of
    several input planes.

    See Also
    --------
    See docs for the parameters of `torch.nn.MaxPool2d`.
    """

    def forward(self, input):
        return cplx.max_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
        )


class CplxMaxPool3d(CplxMaxPoolNd):
    r"""Applies a 3D max pooling over a complex input signal composed of
    several input planes.

    See Also
    --------
    See docs for the parameters of `torch.nn.MaxPool3d`.
    """

    def forward(self, input):
        return cplx.max_pool3d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
        )
