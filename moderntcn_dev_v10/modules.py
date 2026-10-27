"""Fixed endpoint horizon interpolation of v9 spline control outputs.
Independent formula implementation; no N-HiTS source code copied.
"""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from moderntcn_dev_v9.modules import SplineTargetReadout, fixed_knots
from moderntcn_dev_v3.modules import ConditionalTemporalFilter
from moderntcn_dev_v1.modules import stats


def control_count(pred_len):
    if type(pred_len) is not int or pred_len < 1:
        raise ValueError('positive integer prediction length required')
    return max(1, (pred_len + 3)//4)


def interpolation_matrix(pred_len):
    K = control_count(pred_len)
    if K == 1:
        return torch.ones(pred_len, 1, dtype=torch.float32)
    t = torch.arange(pred_len, dtype=torch.float32) * (K-1) / (pred_len-1)
    left = torch.minimum(torch.floor(t).to(torch.int64), torch.tensor(K-2))
    weight = t - left
    M = torch.zeros(pred_len, K, dtype=torch.float32)
    rows = torch.arange(pred_len)
    M[rows, left] = 1-weight
    M[rows, left+1] = weight
    return M


def interpolation_identity(pred_len, input_width):
    K = control_count(pred_len)
    M = interpolation_matrix(pred_len)
    return dict(P=pred_len, K=K, I=input_width, coef_shape=[K,input_width,8],
                M_shape=list(M.shape), M_sha256=hashlib.sha256(M.numpy().tobytes()).hexdigest(),
                M_registered=pred_len!=1, P1_source='moderntcn_dev_v9.modules.SplineTargetReadout')


class HorizonCoupledSplineReadout(SplineTargetReadout):
    def __init__(self, input_width, pred_len):
        if pred_len == 1:
            raise ValueError('P1 must use original v9 class directly')
        K = control_count(pred_len)
        super().__init__(input_width, K)
        self.P, self.K = pred_len, K
        self.register_buffer('M', interpolation_matrix(pred_len))

    def forward(self, a, diagnostics=None):
        basis = self.basis(a)
        if self.coef.shape != (self.K,self.I,8) or self.M.shape != (self.P,self.K) or self.M.dtype != torch.float32:
            raise ValueError('horizon parameter/buffer format')
        if not bool(torch.isfinite(self.M).all()):
            raise ValueError('nonfinite fixed interpolation')
        q = F.linear(basis.reshape(a.shape[0],self.I*8), self.coef.reshape(self.K,self.I*8)) / self.I
        delta = F.linear(q, self.M)
        if not bool(torch.isfinite(q).all() & torch.isfinite(delta).all()):
            raise ValueError('nonfinite spline control/output')
        if diagnostics is not None:
            diagnostics['U_spline_readout'] = dict(input=stats(a), coordinate=stats(torch.tanh(a)),
                basis=stats(basis), controls=stats(q), delta=stats(delta), scale='1/I',
                P=self.P,K=self.K,location='post_block_skip_target_before_denorm')
        return delta


def validate_fixed_state(model, state):
    """Reject corrupted fixed grids before writing any live tensor."""
    if not getattr(model, 'use_u', False):
        return
    I, P = model.spec.value['D']*model.spec.value['N'], model.spec.value['pred_len']
    if state['u.coef'].shape != (control_count(P), I, 8):
        raise ValueError('checkpoint coefficient shape')
    if not torch.equal(state['u.knots'].cpu(), fixed_knots()):
        raise ValueError('checkpoint frozen spline knots mismatch')
    if P != 1 and not torch.equal(state['u.M'].cpu(), interpolation_matrix(P)):
        raise ValueError('checkpoint fixed interpolation mismatch')
