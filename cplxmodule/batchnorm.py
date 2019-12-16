import torch

from torch.nn import init

from .layers import CplxToCplx, CplxParameter
from .cplx import Cplx


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

    # (input) [N, C, ...]
    axes = 0, *range(2, input.dim())
    size = 1, input.shape[1], *(len(axes)-1) * [1]
    r, i = input.real, input.imag

    # 1. compute batch mean [C] and center the batch
    if training:
        m_r = r.mean(dim=axes)
        m_i = i.mean(dim=axes)
        if running_mean is not None:
            mean = torch.cat([
                m_r.data, m_i.data,
            ], dim=0).reshape(2, -1)
            running_mean += momentum * (mean - running_mean)

    else:
        m_r, m_i = running_mean  # torch.unbind(running_mean, dim=0)
    r, i = r - m_r.reshape(size), i - m_i.reshape(size)

    # 2. per feature real-imag 2x2 covariance matrix
    #  using naïve means (biased estimator)
    if training:
        v_rr = (r * r).mean(dim=axes)
        v_ri = (r * i).mean(dim=axes)
        v_ii = (i * i).mean(dim=axes)
        if running_var is not None:
            var = torch.cat([  # 25% waste, needs only the upper triangle
                v_rr.data, v_ri.data,
                v_ri.data, v_ii.data,
            ], dim=0).reshape(2, 2, -1)
            running_var += momentum * (var - running_var)

    else:
        # v_rr, v_ri, _, v_ii = torch.unbind(running_var.reshape(4, -1), dim=0)
        (v_rr, v_ri), (_, v_ii) = running_var

    # 3. using tril LL^T = V "favours" the first dimension.
    #  Trabelsi et al. (2017) used explicit 2x2 root of V:
    #   V = Q^2, Q^T = Q, V = [[a, b], [b, d]].
    # For M = [[a, b], [c, d]]
    #  (1) inv M = 1/(ad - bc) [[d, -b], [-c, a]]
    #  (2) √M = [[a + s, b], [c, d + s]] 1/t
    #    for s = √(ad - bc), t = √(a + d + 2√s)
    #    det √M = t^{-2}(ad + s(d + a) + s^2 - bc) = s
    #  (3) inv √M = 1/(t s) [[d + s, -b], [-c, a + s]]
    root_det = torch.sqrt(torch.clamp(v_rr * v_ii - v_ri * v_ri, 0) + eps)
    denom = root_det * torch.sqrt(v_rr + 2 * root_det + v_ii)

    # Trabelsi (2017): the inv-root uses numpy (np.) instead of K.,
    #  so no autodiff (unsure if that was intentional).

    # 3.5 compose Q = [[rr, ri], [ir, ii]], then E Qxx^TQ^T = QMQ = I
    q_rr = (v_ii + root_det) / denom
    q_ri = -v_ri / denom
    q_ii = (v_rr + root_det) / denom

    # 4. apply Q to x (manually), and apply affine transformation
    re = r * q_rr.reshape(size) + i * q_ri.reshape(size)
    im = r * q_ri.reshape(size) + i * q_ii.reshape(size)
    if weight is not None and bias is not None:
        w_rr, w_ri, w_ii = weight
        u = re * w_rr.reshape(size) + im * w_ri.reshape(size)
        v = re * w_ri.reshape(size) + im * w_ii.reshape(size)

        b_r, b_i = bias
        re, im = u + b_r.reshape(size), v + b_i.reshape(size)
    return Cplx(re, im)


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
            self.weight = torch.nn.Parameter(torch.empty(3, num_features))
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
        self.reset_running_stats()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.zero_()
            self.running_var[0, 0].fill_(1)
            self.running_var[1, 1].fill_(1)

            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.zeros_(self.bias)
            init.ones_(self.weight)
            init.zeros_(self.weight[1])

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
