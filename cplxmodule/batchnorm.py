import torch

from torch.nn import init

from .layers import CplxToCplx, CplxParameter
from .cplx import Cplx


def whiten2x2(tensor, training=True, running_mean=None, running_cov=None,
              momentum=0.1, nugget=1e-5):
    r"""Solve R M R = I for R and a given 2x2 matrix M = [[a, b], [c, d]].

    Details
    -------
    Using (tril) L L^T = V seems to 'favour' the first dimension (re), so
    Trabelsi et al. (2017) used explicit 2x2 root of M: such R that M = RR.

    For M = [[a, b], [c, d]] we have the following facts:
        (1) inv M = 1/(ad - bc) [[d, -b], [-c, a]]
        (2) \sqrt{M} =  [[a + s, b], [c, d + s]] 1/t
            for s = \sqrt{ad - bc}, t = \sqrt{a + d + 2 \sqrt{s}}
            det \sqrt{M} = t^{-2} (ad + s(d + a) + s^2 - bc) = s

    Therefore `inv \sqrt{M} = [[p, q], [r, s]]`, where
        [[p, q], [r, s]] = 1/(t s) [[d + s, -b], [-c, a + s]]
    """

    # compute reduction axes and broadcast shape (tail) ? x 1 x f x ...
    axes = 0, *range(2, tensor.dim() - 1)
    shape = 1, tensor.shape[1], *([1] * (tensor.dim() - 3))

    # get feature mean and covariance
    # 1. compute batch mean [F x 2] and center the batch
    if training:
        mean = tensor.mean(dim=axes)
        if running_mean is not None:
            running_mean += momentum * (mean.data - running_mean)
    else:
        mean = running_mean
    tensor = tensor - mean.reshape(*shape, 2)

    # 2. per feature real-imag 2x2 covariance matrix
    #  using naïve means (biased estimator)
    if training:
        # P x B x F x ... -> F x P x [B x ...]
        perm = tensor.permute(1, -1, *axes).flatten(2, -1)
        cov = torch.matmul(perm, perm.transpose(-1, -2)) / perm.shape[-1]
        if running_cov is not None:
            running_cov += momentum * (cov.data - running_cov)

    else:
        cov = running_cov

    # 3. get R = [[p, q], [r, s]], with E R c c^T R^T = R M R = I
    cov = cov.reshape(*shape, 2, 2)
    a, b = cov[..., 0, 0] + nugget, cov[..., 0, 1]
    c, d = cov[..., 1, 0], cov[..., 1, 1] + nugget

    # (unsure if that was intentional) the inv-root in Trabelsi (2017) uses
    #  numpy `np.sqrt` instead of `K.sqrt` so grads are not passed through
    #  properly, i.e. constants, [complex_standardization](bn.py#L56-57).
    sqrdet = torch.sqrt(a * d - b * c)
    denom = sqrdet * torch.sqrt(a + 2 * sqrdet + d)
    p, q = (d + sqrdet) / denom, -b / denom
    r, s = -c / denom, (a + sqrdet) / denom

    # 4. apply Q to x (manually)
    out = torch.stack([
        tensor[..., 0] * p + tensor[..., 1] * r,
        tensor[..., 0] * q + tensor[..., 1] * s,
    ], dim=-1)
    return out  # , torch.cat([p, q, r, s], dim=0).reshape(2, 2, -1)


def whitendxd(tensor, training=True, running_mean=None, running_cov=None,
              momentum=0.1, nugget=1e-5):
    """Jointly whiten features in tensors [B x F x ... x D]: take D vectors and
    whiten individually for each F over [B x ...].

    Details
    -------
    Comutes the mean along all axes but F and D, then gets F biased estimates
    of the covariance between D. The covariances are regularized by a `nugget`
    and then their batched cholesky decomposition is used in triangular solve
    to do the whitening.
    """

    # compute reduction axes and broadcast shape (tail) ? x 1 x f x ...
    axes = 0, *range(2, tensor.dim() - 1)
    shape = 1, tensor.shape[1], *([1] * (tensor.dim() - 3)), tensor.shape[-1]

    # get feature mean and covariance
    if training:
        mean = tensor.mean(dim=axes)
        if running_mean is not None:
            running_mean += momentum * (mean.data - running_mean)
    else:
        mean = running_mean
    tensor = tensor - mean.reshape(shape)

    if training:
        # P x B x F x ... -> F x P x [B x ...]
        perm = tensor.permute(1, -1, *axes).flatten(2, -1)
        cov = torch.matmul(perm, perm.transpose(-1, -2)) / perm.shape[-1]
        if running_cov is not None:
            running_cov += momentum * (cov.data - running_cov)

    else:
        cov = running_cov

    # invert cholesky decomposition.
    eye = nugget * torch.eye(tensor.shape[-1]).unsqueeze(0)
    ell = torch.cholesky(cov + eye, upper=True)
    soln = torch.triangular_solve(
        tensor.unsqueeze(-1), ell.reshape(*shape, tensor.shape[-1]))

    return soln.solution.squeeze(-1)


def cplx_batch_norm(
    input,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    training=True,
    momentum=0.1,
    eps=1e-05,
):
    # check arguments
    assert ((running_mean is None and running_var is None)
            or (running_mean is not None and running_var is not None))
    assert ((weight is None and bias is None)
            or (weight is not None and bias is not None))

    # stack along the last axis ... -> ... x 2
    x = torch.stack([input.real, input.imag], dim=-1)
    shape = 1, x.shape[1], *([1] * (x.dim() - 3))

    # whiten and apply affine transformation
    z = whiten2x2(x, training=training, running_mean=running_mean,
                  running_cov=running_var, momentum=momentum, nugget=eps)

    if weight is not None and bias is not None:
        weight, bias = weight.reshape(*shape, 2, 2), bias.reshape(*shape, 2)
        z = torch.stack([
            z[..., 0] * weight[..., 0, 0] + z[..., 1] * weight[..., 0, 1],
            z[..., 0] * weight[..., 1, 0] + z[..., 1] * weight[..., 1, 1],
        ], dim=-1) + bias

    return Cplx(z[..., 0], z[..., 1])


class _CplxBatchNorm(CplxToCplx):
    def __init__(
        self,
        num_features,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = torch.nn.Parameter(torch.empty(num_features, 2, 2))
            self.bias = torch.nn.Parameter(torch.empty(num_features, 2))

        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.empty(num_features, 2))
            self.register_buffer('running_var', torch.empty(num_features, 2, 2))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)

        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.num_batches_tracked.zero_()

            self.running_mean.zero_()
            self.running_var[:, 0, 0].fill_(1)
            self.running_var[:, 1, 0].zero_()
            self.running_var[:, 0, 1].zero_()
            self.running_var[:, 1, 1].fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight[:, 0, 0])
            init.zeros_(self.weight[:, 1, 0])
            init.zeros_(self.weight[:, 0, 1])
            init.ones_(self.weight[:, 1, 1])
            init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input):
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        return cplx_batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**vars(self))


class CplxBatchNorm1d(_CplxBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


class CplxBatchNorm2d(_CplxBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class CplxBatchNorm3d(_CplxBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
