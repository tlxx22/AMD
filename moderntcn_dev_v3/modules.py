"""v3: only the V router gains explicit pre-RevIN target-window state."""
import torch
from torch import nn
from torch.nn import functional as F
from moderntcn_dev_v1.modules import (
    WindowStateModulation, ConditionalTemporalFilter as _V2Filter, stats,
)


class ConditionalTemporalFilter(_V2Filter):
    def __init__(self, L, D, N, large, C, target_idx):
        super().__init__(L, D, N, large)
        if not 0 <= target_idx < C:
            raise ValueError('V target/schema index')
        self.L, self.C, self.target_idx = L, C, target_idx
        # No random draw: all existing V initialization and caller RNG preserved.
        self.P_s = nn.Parameter(torch.zeros(16, 2))

    def target_state(self, X):
        if X.ndim != 3 or tuple(X.shape[1:]) != (self.L, self.C):
            raise ValueError('V target history shape/schema')
        if not torch.isfinite(X).all():
            raise ValueError('nonfinite V pre-RevIN input')
        y = X[:, :, self.target_idx].float()
        mu = y.mean(dim=1)
        var = (y - mu[:, None]).square().mean(dim=1)
        state = torch.stack((mu, torch.log1p(var)), dim=-1)
        if not torch.isfinite(state).all():
            raise ValueError('nonfinite V target statistics')
        return state

    def _route(self, Z, X):
        if Z.ndim != 3 or tuple(Z.shape[1:]) != (self.C, self.L) or Z.shape[0] != X.shape[0]:
            raise ValueError('V normalized history shape/schema')
        if not torch.isfinite(Z).all():
            raise ValueError('nonfinite V normalized input')
        state = self.target_state(X)
        # State statistics stay float32. The cast only supports explicit synthetic
        # double-precision algebra fixtures; registered training weights are float32.
        context = F.linear(state.to(self.P_s.dtype), self.P_s, bias=None)
        a = torch.tanh(self.W2(F.gelu(self.W1(Z) + context[:, None, :], approximate='none')))
        if not torch.isfinite(context).all() or not torch.isfinite(a).all():
            raise ValueError('nonfinite V target-conditioned route')
        return a, state, context

    def route(self, Z, X):
        return self._route(Z, X)[0]

    def forward(self, H, Z, X, diagnostics=None):
        B, C, D, N = H.shape
        if (C, D, N) != (self.C, self.D, self.N) or X.shape[0] != B:
            raise ValueError('fixed V profile shape')
        a, state, context = self._route(Z, X)
        bases = F.conv1d(H.reshape(B*C*D, 1, N), self.T[:, None, :],
                         padding=self.Kv//2).reshape(B, C, D, 4, N)
        delta = (bases*a[:, :, None, :, None]*self.A[None, None, :, :, None]).sum(3)
        if diagnostics is not None:
            kernel = torch.einsum('bcr,dr,rk->bcdk', a, self.A, self.T)
            diagnostics['V'] = {
                'target_state': stats(state), 'projected_target_state': stats(context),
                'target_idx': self.target_idx, 'state_axes': 'B,[mean,log1p_population_variance]',
                'a': stats(a), 'delta_kernel': stats(kernel), 'actual_delta': stats(delta),
            }
        return delta
