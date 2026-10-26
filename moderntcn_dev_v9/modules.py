"""Independent frozen cubic B-spline readout; reference papers supply no code.

The original linear head remains in the native model. This module returns only
the mean spline increment in normalized prediction units.
"""
import torch
from torch import nn
from torch.nn import functional as F
from moderntcn_dev_v3.modules import ConditionalTemporalFilter
from moderntcn_dev_v1.modules import stats


def fixed_knots():
    return -1 + (torch.arange(12, dtype=torch.float32) - 3) * (2.0 / 5.0)


class SplineTargetReadout(nn.Module):
    def __init__(self, input_width, pred_len):
        super().__init__()
        if not isinstance(input_width, int) or not isinstance(pred_len, int) or min(input_width, pred_len) <= 0:
            raise ValueError('positive integer spline I/P required')
        self.I, self.P = input_width, pred_len
        self.coef = nn.Parameter(torch.zeros(pred_len, input_width, 8, dtype=torch.float32))
        self.register_buffer('knots', fixed_knots())

    def basis(self, a):
        if a.ndim != 2 or a.shape[1] != self.I or a.dtype != torch.float32:
            raise ValueError('spline input must be float32 [B,I]')
        if self.knots.shape != (12,) or self.knots.dtype != torch.float32 or self.coef.dtype != torch.float32:
            raise ValueError('frozen float32 spline parameter/buffer format')
        if not bool(torch.isfinite(a).all() & torch.isfinite(self.knots).all() & torch.isfinite(self.coef).all()):
            raise ValueError('nonfinite spline input/knots/coef')
        if not bool((self.knots[1:] > self.knots[:-1]).all()):
            raise ValueError('strictly increasing knots required')
        u = torch.tanh(a).unsqueeze(-1)
        k = self.knots
        basis = ((u >= k[:-1]) & (u < k[1:])).to(a.dtype)
        for degree in range(1, 4):
            count = 11 - degree
            left = (u - k[:count]) / (k[degree:degree+count] - k[:count])
            right = (k[degree+1:degree+1+count] - u) / (k[degree+1:degree+1+count] - k[1:1+count])
            basis = left * basis[..., :count] + right * basis[..., 1:count+1]
        if not bool(torch.isfinite(basis).all()):
            raise ValueError('nonfinite spline basis')
        return basis

    def forward(self, a, diagnostics=None):
        basis = self.basis(a)
        delta = F.linear(basis.reshape(a.shape[0], self.I*8), self.coef.reshape(self.P, self.I*8)) / self.I
        if not bool(torch.isfinite(delta).all()):
            raise ValueError('nonfinite spline increment')
        if diagnostics is not None:
            diagnostics['U_spline_readout'] = dict(input=stats(a), coordinate=stats(torch.tanh(a)),
                basis=stats(basis), delta=stats(delta), scale='1/I', location='post_block_skip_target_before_denorm')
        return delta
