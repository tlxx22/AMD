"""Conditional C-axis Cayley rotation shared across D/N; independent OFT-inspired formula.

Native weights remain trainable. No COFT radius, amplitude, static merge or fallback.
"""
import math
import torch
from torch import nn
from moderntcn_dev_v1.modules import stats
from moderntcn_dev_v3.modules import ConditionalTemporalFilter


def cayley(skew, identity):
    """Prescribed solve order, no cast/epsilon/alternate algorithm.

    Production arguments are float32; explicit local double fixtures test derivatives.
    """
    if not torch.isfinite(skew).all():
        raise ValueError('nonfinite Cayley input')
    rotation = torch.linalg.solve(identity - skew, identity + skew)
    if not torch.isfinite(rotation).all():
        raise ValueError('nonfinite Cayley output')
    return rotation


class ConditionalVariableRotation(nn.Module):
    def __init__(self, L, C, D, target, aux):
        super().__init__()
        if not 0 <= target < C or len(set(aux)) != len(aux) or any(j == target or not 0 <= j < C for j in aux):
            raise ValueError('target/ordered auxiliary schema')
        self.L, self.C, self.D, self.target, self.aux = L, C, D, target, tuple(aux)
        self.A = nn.Parameter(torch.empty(C, 8, 2))
        self.e = nn.Parameter(torch.zeros(C, 8))
        self.b = nn.Parameter(torch.zeros(len(aux)))
        self.Theta = nn.Parameter(torch.zeros(8, C * (C - 1) // 2))
        self.register_buffer('identity', torch.eye(C, dtype=torch.float32))
        self.register_buffer('upper_indices', torch.triu_indices(C, C, offset=1))
        with torch.no_grad():
            for a in self.A:
                nn.init.xavier_uniform_(a)

    def controller(self, X):
        if X.ndim != 3 or tuple(X.shape[1:]) != (self.L, self.C):
            raise ValueError('U history schema')
        X = X.float()
        if not torch.isfinite(X).all():
            raise ValueError('nonfinite U input')
        mu = X.mean(1)
        var = (X - mu[:, None]).square().mean(1)
        s = torch.stack((mu, torch.log1p(var)), -1)
        if not torch.isfinite(s).all():
            raise ValueError('nonfinite U statistics')
        h = torch.tanh(torch.einsum('crs,bcs->bcr', self.A, s) + self.e)
        if self.aux:
            hj = h[:, self.aux, :]
            g = torch.sigmoid((h[:, self.target, None, :] * hj).sum(-1) / math.sqrt(8) + self.b)
            q = (g[:, :, None] * hj).sum(1) / (1 + g.sum(1, keepdim=True))
        else:
            g = h.new_empty((X.shape[0], 0))
            q = h.new_zeros((X.shape[0], 8))
        if not torch.isfinite(q).all() or not torch.isfinite(g).all():
            raise ValueError('nonfinite U controller output')
        return q, g

    def rotation(self, q):
        theta = q @ self.Theta
        upper = theta.new_zeros((q.shape[0], self.C, self.C))
        upper[:, self.upper_indices[0], self.upper_indices[1]] = theta
        skew = upper - upper.transpose(-1, -2)
        return cayley(skew, self.identity), skew

    def forward(self, z, X, diagnostics=None):
        if z.ndim != 4 or tuple(z.shape[1:3]) != (self.C, self.D) or z.shape[0] != X.shape[0]:
            raise ValueError('U C-axis carrier schema')
        if z.dtype != torch.float32 or not torch.isfinite(z).all():
            raise ValueError('U requires finite float32 carrier')
        q, g = self.controller(X)
        rotation, skew = self.rotation(q)
        rotated = torch.bmm(rotation, z.reshape(z.shape[0], self.C, -1)).reshape_as(z)
        if not torch.isfinite(rotated).all():
            raise ValueError('nonfinite U rotation output')
        if diagnostics is not None:
            diagnostics['U_shared_rotation'] = {
                'q': stats(q), 'gate_mean': float(g.detach().mean()) if g.numel() else None,
                'skew': stats(skew), 'rotation_minus_identity': stats(rotation - self.identity),
                'actual_carrier_change': stats(rotated - z),
                'axes': 'one B,C,C rotation shared over all D,N', 'basis_count': 8,
            }
        return rotated
