import torch

from torch.nn import init

from .base import CplxToCplx
from ... import cplx


def whiten2x2(tensor, training=True, running_mean=None, running_cov=None,
              momentum=0.1, nugget=1e-5):
    r"""Solve R M R = I for R and a given 2x2 matrix M = [[a, b], [c, d]].

    Arguments
    ---------
    tensor : torch.tensor
        The input data expected to be at least 3d, with shape [2, B, F, ...],
        where `B` is the batch dimension, `F` -- the channels/features,
        `...` -- the spatial dimensions (if present). The leading dimension
        `2` represents real and imaginary components (stacked).

    training : bool, default=True
        Determines whether to update running feature statistics, if they are
        provided, or use them instead of batch computed statistics. If `False`
        then `running_mean` and `running_cov` MUST be provided.

    running_mean : torch.tensor, or None
        The tensor with running mean statistics having shape [2, F]. Ignored
        if explicitly `None`.

    running_cov : torch.tensor, or None
        The tensor with running real-imaginary covariance statistics having
        shape [2, 2, F]. Ignored if explicitly `None`.

    momentum : float, default=0.1
        The weight in the exponential moving average used to keep track of the
        running feature statistics.

    nugget : float, default=1e-05
        The ridge coefficient to stabilise the estimate of the real-imaginary
        covariance.

    Details
    -------
    Using (tril) L L^T = V seems to 'favour' the first dimension (re), so
    Trabelsi et al. (2018) used explicit 2x2 root of M: such R that M = RR.

    For M = [[a, b], [c, d]] we have the following facts:
        (1) inv M = \frac1{ad - bc} [[d, -b], [-c, a]]
        (2) \sqrt{M} = \frac1{t} [[a + s, b], [c, d + s]]
            for s = \sqrt{ad - bc}, t = \sqrt{a + d + 2 s}
            det \sqrt{M} = t^{-2} (ad + s(d + a) + s^2 - bc) = s

    Therefore `inv \sqrt{M} = [[p, q], [r, s]]`, where
        [[p, q], [r, s]] = \frac1{t s} [[d + s, -b], [-c, a + s]]
    """
    # assume tensor is 2 x B x F x ...

    # tail shape for broadcasting ? x 1 x F x [*1]
    tail = 1, tensor.shape[2], *([1] * (tensor.dim() - 3))
    axes = 1, *range(3, tensor.dim())

    # 1. compute batch mean [2 x F] and center the batch
    if training:
        mean = tensor.mean(dim=axes)
        if running_mean is not None:
            running_mean += momentum * (mean.data - running_mean)

    else:
        mean = running_mean

    tensor = tensor - mean.reshape(2, *tail)

    # 2. per feature real-imaginary 2x2 covariance matrix
    if training:
        # faster than doing mul and then mean. Stabilize by a small ridge.
        var = tensor.var(dim=axes, unbiased=False) + nugget
        cov_uu, cov_vv = var[0], var[1]

        # has to mul-mean here anyway (naïve) : reduction axes shifted left.
        cov_vu = cov_uv = (tensor[0] * tensor[1]).mean([a - 1 for a in axes])
        if running_cov is not None:
            cov = torch.stack([
                cov_uu.data, cov_uv.data,
                cov_vu.data, cov_vv.data,
            ], dim=0).reshape(2, 2, -1)
            running_cov += momentum * (cov - running_cov)

    else:
        cov_uu, cov_uv = running_cov[0, 0], running_cov[0, 1]
        cov_vu, cov_vv = running_cov[1, 0], running_cov[1, 1]

    # 3. get R = [[p, q], [r, s]], with E R c c^T R^T = R M R = I
    # (unsure if intentional, but the inv-root in Trabelsi et al. (2018) uses
    # numpy `np.sqrt` instead of `K.sqrt` so grads are not passed through
    # properly, i.e. constants, [complex_standardization](bn.py#L56-57).
    sqrdet = torch.sqrt(cov_uu * cov_vv - cov_uv * cov_vu)
    # torch.det uses svd, so may yield -ve machine zero

    denom = sqrdet * torch.sqrt(cov_uu + 2 * sqrdet + cov_vv)
    p, q = (cov_vv + sqrdet) / denom, -cov_uv / denom
    r, s = -cov_vu / denom, (cov_uu + sqrdet) / denom

    # 4. apply Q to x (manually)
    out = torch.stack([
        tensor[0] * p.reshape(tail) + tensor[1] * r.reshape(tail),
        tensor[0] * q.reshape(tail) + tensor[1] * s.reshape(tail),
    ], dim=0)
    return out  # , torch.cat([p, q, r, s], dim=0).reshape(2, 2, -1)


def whitendxd(tensor, training=True, running_mean=None, running_cov=None,
              momentum=0.1, nugget=1e-5):
    """Jointly whiten features in tensors [B x F x ... x D]: take D vectors and
    whiten individually for each F over [B x ...].

    Details
    -------
    Computes the mean along all axes but F and D, then gets F biased estimates
    of the covariance between D. The covariances are regularized by a `nugget`
    and then their batched Cholesky decomposition is used in triangular solve
    to do the whitening.

    Warning
    -------
    `torch.triangular_solve` uses MAGMA which seems to have issues with batch
    sizes altogether exceeding 100k vectors.

    Please refer to this thread:
    https://github.com/pytorch/pytorch/issues/24403#issuecomment-521655390
    """

    # compute reduction axes and broadcast shape (tail) P x 1 x F x ...
    axes, d = (1, *range(3, tensor.dim())), tensor.shape[0]
    shape = 1, tensor.shape[2], *([1] * (tensor.dim() - 3))

    # get feature mean and covariance
    if training:
        mean = tensor.mean(dim=axes)
        if running_mean is not None:
            running_mean += momentum * (mean.data - running_mean)
    else:
        mean = running_mean
    tensor = tensor - mean.reshape(d, *shape)

    if training:
        # P x B x F x ... -> F x P x [B x ...]
        perm = tensor.permute(2, 0, *axes).flatten(2, -1)
        cov = torch.matmul(perm, perm.transpose(-1, -2)) / perm.shape[-1]
        if running_cov is not None:
            running_cov += momentum * (cov.data.permute(1, 2, 0) - running_cov)

    else:
        cov = running_cov.permute(2, 0, 1)

    # invert Cholesky decomposition.
    eye = nugget * torch.eye(d, device=cov.device, dtype=cov.dtype).unsqueeze(0)
    ell = torch.cholesky(cov + eye, upper=True)
    soln = torch.triangular_solve(
        tensor.unsqueeze(-1).permute(*range(1, tensor.dim()), 0, -1),
        ell.reshape(*shape, d, d))

    soln = soln.solution.squeeze(-1)
    return torch.stack(torch.unbind(soln, dim=-1), dim=0)


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
    """Applies complex-valued Batch Normalization as described in
    (Trabelsi et al., 2018) for each channel across a batch of data.

    Arguments
    ---------
    input : complex-valued tensor
        The input complex-valued data is expected to be at least 2d, with
        shape [B, F, ...], where `B` is the batch dimension, `F` -- the
        channels/features, `...` -- the spatial dimensions (if present).

    running_mean : torch.tensor, or None
        The tensor with running mean statistics having shape [2, F]. Ignored
        if explicitly `None`.

    running_var : torch.tensor, or None
        The tensor with running real-imaginary covariance statistics having
        shape [2, 2, F]. Ignored if explicitly `None`.

    weight : torch.tensor, default=None
        The 2x2 weight matrix of the affine transformation of real and
        imaginary parts post normalization. Has shape [2, 2, F] . Ignored
        together with `bias` if explicitly `None`.

    bias : torch.tensor, or None
        The offest (bias) of the affine transformation of real and imaginary
        parts post normalization. Has shape [2, F] . Ignored together with
        `weight` if explicitly `None`.

    training : bool, default=True
        Determines whether to update running feature statistics, if they are
        provided, or use them instead of batch computed statistics. If `False`
        then `running_mean` and `running_var` MUST be provided.

    momentum : float, default=0.1
        The weight in the exponential moving average used to keep track of the
        running feature statistics.

    eps : float, default=1e-05
        The ridge coefficient to stabilise the estimate of the real-imaginary
        covariance.

    Details
    -------
    Has non standard interface for running stats and weight and bias of the
    affine transformation for purposes of improved memory locality (noticeable
    speedup both on host and device computations).
    """
    # check arguments
    assert ((running_mean is None and running_var is None)
            or (running_mean is not None and running_var is not None))
    assert ((weight is None and bias is None)
            or (weight is not None and bias is not None))

    # stack along the first axis
    x = torch.stack([input.real, input.imag], dim=0)

    # whiten and apply affine transformation
    z = whiten2x2(x, training=training, running_mean=running_mean,
                  running_cov=running_var, momentum=momentum, nugget=eps)

    if weight is not None and bias is not None:
        shape = 1, x.shape[2], *([1] * (x.dim() - 3))
        weight = weight.reshape(2, 2, *shape)
        z = torch.stack([
            z[0] * weight[0, 0] + z[1] * weight[0, 1],
            z[0] * weight[1, 0] + z[1] * weight[1, 1],
        ], dim=0) + bias.reshape(2, *shape)

    return cplx.Cplx(z[0], z[1])


class _CplxBatchNorm(CplxToCplx):
    """The base clas for Complex-valeud batch normalization layer.

    Taken from `torch.nn.modules.batchnorm` verbatim.
    """
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
            self.weight = torch.nn.Parameter(torch.empty(2, 2, num_features))
            self.bias = torch.nn.Parameter(torch.empty(2, num_features))

        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.empty(2, num_features))
            self.register_buffer('running_var', torch.empty(2, 2, num_features))
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
            self.running_var[0, 0].fill_(1)
            self.running_var[1, 0].zero_()
            self.running_var[0, 1].zero_()
            self.running_var[1, 1].fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight[0, 0])
            init.zeros_(self.weight[1, 0])
            init.zeros_(self.weight[0, 1])
            init.ones_(self.weight[1, 1])
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
    """Complex-valued batch normalization for 2D or 3D data.
    See torch.nn.BatchNorm1d for details.
    """
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


class CplxBatchNorm2d(_CplxBatchNorm):
    """Complex-valued batch normalization for 4D data.
    See torch.nn.BatchNorm2d for details.
    """
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class CplxBatchNorm3d(_CplxBatchNorm):
    """Complex-valued batch normalization for 5D data.
    See torch.nn.BatchNorm3d for details.
    """
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
