import torch

from torch.nn import init

from .layers import CplxToCplx, CplxParameter
from .cplx import Cplx


def torch_invsqrt_2x2(a, b, c, d, eps=1e-8):
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
    # (unsure if that was intentional) the inv-root in Trabelsi (2017) uses
    #  numpy `np.sqrt` instead of `K.sqrt` so grads are not passed through
    #  properly, i.e. constants, [complex_standardization](bn.py#L56-57).
    sqrdet = torch.sqrt((a + eps) * (d + eps) - b * c)
    denom = sqrdet * torch.sqrt(a + 2 * sqrdet + d + 2 * eps)

    p, q = (d + sqrdet) / denom, -c / denom
    r, s = -b / denom, (a + sqrdet) / denom
    return p, q, r, s


def torch_matmul_2x2(u, v, a, b, c, d):
    return u * a + v * b, u * c + v * d


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

    x = torch.stack([input.real, input.imag], dim=0)
    axes = 1, *range(3, x.dim())
    size = 1, input.shape[1], *[1]*(input.dim()-2)

    # 1. compute batch mean [C] and center the batch
    m = x.mean(dim=axes)
    if training:
        pass

    x = x - m.reshape(2, *size)

    # 2. per feature real-imag 2x2 covariance matrix using naïve means
    # (biased) [2, B, F, ...] -> [F, 2, ...] -> [F, 2, 2] -> [4, F]
    p = x.permute(2, 0, *axes).flatten(2, -1)
    var = torch.matmul(p, p.transpose(-1, -2)) / p.shape[2]
    var = var.flatten(-2, -1).t()
    if training:
        pass

    # 3. get R = [[rr, ri], [ir, ii]], with E R c c^T R^T = RMR = I
    Q = torch.cat(torch_invsqrt_2x2(*var, eps=1e-8), dim=0).reshape(2, 2, -1)

    # 4. apply Q to x (manually), and apply affine transformation
    # [2, B, F, ...] * [2, 2, F] 'ubf..., uvf -> vbf...'
    x = torch.einsum('ubf..., uvf -> vbf...', x, Q)  # batch mm maybe?

    # 5. Affine tranformation
    if weight is not None and bias is not None:
        pass
"""
A = torch.randn(5, 2, 2).requires_grad_(True).triu()



a = torch.randn(10, 3, 3).requires_grad_(True)

V = torch.matmul(a, a.transpose(-1, -2)) + 1e-05 * torch.eye(3).unsqueeze(0)

L = torch.cholesky(V)

torch.triangular_solve(a, torch.cholesky(V, upper=True)).solution



x = torch.randn(2, 1000, 10, 17)

axes = 1, *range(3, x.dim())
size = 1, x.shape[2], *[1]*(x.dim()-3)

p = x.permute(2, 0, *axes)
shape = p.shape

p = p.flatten(2, -1)
V = torch.matmul(p, p.transpose(-1, -2)) + 1e-05 * torch.eye(2).unsqueeze(0)
ell = torch.cholesky(V, upper=True)
p = torch.triangular_solve(p, ell).solution

res = p.reshape(shape).permute(1, 2, 0, 3)

"""

    # (input) [N, C, ...]
    axes = 0, *range(2, input.dim())
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

    # 3. get R = [[rr, ri], [ir, ii]], with E R c c^T R^T = RMR = I
    q_rr, q_ri, q_ir, q_ii = torch_invsqrt_2x2(v_rr, v_ri, v_ri, v_ii)

    # 4. apply Q to x (manually), and apply affine transformation
    q_rr, q_ri = q_rr.reshape(size), q_ri.reshape(size)
    q_ir, q_ii = q_ir.reshape(size), q_ii.reshape(size)
    re, im = torch_matmul_2x2(r, i, q_rr, q_ri, q_ir, q_ii)
    if weight is not None and bias is not None:
        w_rr, w_ri, w_ii = weight.reshape(3, *size)
        re, im = torch_matmul_2x2(re, im, w_rr, w_ri, w_ri, w_ii)

        b_r, b_i = bias.reshape(3, *size)
        re, im = re + b_r, im + b_i

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
