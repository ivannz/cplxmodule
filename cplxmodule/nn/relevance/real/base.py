import torch
import torch.nn

import torch.nn.functional as F


class GaussianMixin:
    r"""Trait class with log-alpha property for variational dropout.

    Attributes
    ----------
    log_alpha : computed torch.Tensor, read-only
        Log-variance of the multiplicative scaling noise. Computed as a log
        of the ratio of the variance of the weight to the squared absolute
        value of the weight. The higher the log-alpha the less relevant the
        parameter is.
    """
    def reset_variational_parameters(self):
        # initially everything is relevant
        self.log_sigma2.data.uniform_(-10, -10)

    @property
    def log_alpha(self):
        r"""Get $\log \alpha$ from $(\theta, \sigma^2)$ parameterization."""
        return self.log_sigma2 - 2 * torch.log(abs(self.weight) + 1e-12)


class LinearGaussian(GaussianMixin, torch.nn.Linear):
    """Linear layer with variational dropout.

    Details
    -------
    See `torch.nn.Linear` for reference on the dimensions and parameters.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)

        self.log_sigma2 = torch.nn.Parameter(torch.Tensor(*self.weight.shape))
        self.reset_variational_parameters()

    def forward(self, input):
        mu = super().forward(input)
        if not self.training:
            return mu

        s2 = F.linear(input * input, torch.exp(self.log_sigma2), None)
        return mu + torch.randn_like(s2) * torch.sqrt(torch.clamp(s2, 1e-8))


class BilinearGaussian(GaussianMixin, torch.nn.Bilinear):
    """Bilinear layer with variational dropout.

    Details
    -------
    See `torch.nn.Bilinear` for reference on the dimensions and parameters.
    """

    def __init__(self, in1_features, in2_features, out_features, bias=True):
        super().__init__(in1_features, in2_features, out_features, bias=bias)

        self.log_sigma2 = torch.nn.Parameter(torch.Tensor(*self.weight.shape))
        self.reset_variational_parameters()

    def forward(self, input1, input2):
        """Forward pass of the SGVB method for a bilinear layer.

        Straightforward generalization of the local reparameterization trick.
        """
        mu = super().forward(input1, input2)
        if not self.training:
            return mu

        s2 = F.bilinear(input1 * input1, input2 * input2,
                        torch.exp(self.log_sigma2), None)

        # .normal reports a grad-fn, but weirdly does not pass grads!
        return mu + torch.randn_like(s2) * torch.sqrt(torch.clamp(s2, 1e-8))


class ConvNdGaussianMixin(GaussianMixin):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups,
                         bias=bias, padding_mode=padding_mode)

        if self.padding_mode != "zeros":
            raise ValueError(f"Only `zeros` padding mode is supported. "
                             f"Got `{self.padding_mode}`.")

        self.log_sigma2 = torch.nn.Parameter(torch.Tensor(*self.weight.shape))
        self.reset_variational_parameters()

    def _forward_impl(self, input, conv):
        r"""Forward pass of the SGVB method for a Nd convolutional layer.

        Details
        -------
        A convolution can be represented as matrix-vector product of the doubly
        block-circulant embedding (Toeplitz) of the kernel and the unravelled
        input. As such, it is an implicit linear layer with block structured
        weight matrix, but unlike it, the local reparameterization trick has
        a little caveat. If the kernel itself is assumed to have the specified
        variational distribution, then the outputs will be spatially correlated
        due to the same weight block being reused at each location:
        $$
            cov(y_{f\beta}, y_{k\omega})
                = \delta_{f=k} \sum_{c \alpha}
                    \sigma^2_{fc \alpha}
                    x_{c i_\beta(\alpha)}
                    x_{c i_\omega(\alpha)}
            \,, $$
        where $i_\beta(\alpha)$ is the location in $x$ for the output location
        $\beta$ and kernel offset $\alpha$ (depends on stride and dilation).
        In contrast, if instead the Toeplitz embedding blocks are assumed iid
        draws from the variational distribution, then covariance becomes
        $$
            cov(y_{f\beta}, y_{k\omega})
                = \delta_{f\beta = k\omega} \sum_{c \alpha}
                    \sigma^2_{fc \alpha}
                    \lvert x_{c i_\omega(\alpha)} \rvert^2
            \,. $$
        Molchanov et al. (2017) implicitly assume that kernels is are iid draws
        from the variational distribution for different spatial locations. This
        effectively zeroes the spatial cross-correlation in the output, reduces
        the variance of the gradient in SGVB method.
        """
        mu = super().forward(input)
        if not self.training:
            return mu

        s2 = conv(input * input, torch.exp(self.log_sigma2), None,
                  self.stride, self.padding, self.dilation, self.groups)
        return mu + torch.randn_like(s2) * torch.sqrt(torch.clamp(s2, 1e-8))


class Conv1dGaussian(ConvNdGaussianMixin, torch.nn.Conv1d):
    """1D convolution layer with variational dropout.

    Details
    -------
    See `torch.nn.Conv1d` for reference on the dimensions and parameters. See
    `cplxmodule.nn.relevance.ConvNdGaussianMixin` for details about the
    implementation of the reparameterization trick.
    """

    def forward(self, input):
        return self._forward_impl(input, F.conv1d)


class Conv2dGaussian(ConvNdGaussianMixin, torch.nn.Conv2d):
    """2D convolution layer with variational dropout.

    Details
    -------
    See `torch.nn.Conv2d` for reference on the dimensions and parameters. See
    `cplxmodule.nn.relevance.ConvNdGaussianMixin` for details about the
    implementation of the reparameterization trick.
    """

    def forward(self, input):
        return self._forward_impl(input, F.conv2d)


class Conv3dGaussian(ConvNdGaussianMixin, torch.nn.Conv3d):
    """3D convolution layer with variational dropout.

    Details
    -------
    See `torch.nn.Conv3d` for reference on the dimensions and parameters. See
    `cplxmodule.nn.relevance.ConvNdGaussianMixin` for details about the
    implementation of the reparameterization trick.
    """

    def forward(self, input):
        return self._forward_impl(input, F.conv3d)
