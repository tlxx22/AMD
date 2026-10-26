"""Independent noncircular lag scoring and aligned auxiliary content transport."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from moderntcn_dev_v1.modules import stats
from moderntcn_dev_v3.modules import ConditionalTemporalFilter


class LaggedAuxiliaryTransport(nn.Module):
    def __init__(self, C, D, N, target, aux):
        super().__init__()
        if D != 64 or N < 1 or not 0 <= target < C or tuple(aux) != tuple(j for j in range(C) if j != target):
            raise ValueError('target/ordered auxiliary/D/N schema')
        self.C, self.D, self.N = C, D, N
        self.target, self.aux = target, tuple(aux)
        self.J, self.K = len(aux), (N - 1) // 2 + 1
        self.W_Q = nn.Parameter(torch.empty(8, D))
        self.W_K = nn.Parameter(torch.empty(self.J, 8, D))
        self.W_V = nn.Parameter(torch.empty(self.J, 8, D))
        self.W_O = nn.Parameter(torch.zeros(D, 8))
        nn.init.xavier_uniform_(self.W_Q)
        for j in range(self.J):
            nn.init.xavier_uniform_(self.W_K[j])
        for j in range(self.J):
            nn.init.xavier_uniform_(self.W_V[j])
        positions = torch.arange(N)[:, None]
        lags = torch.arange(self.K)[None, :]
        self.register_buffer('valid', lags <= positions)
        self.register_buffer('source_index', (positions - lags).clamp_min(0))
        self.register_buffer('overlap', (N - torch.arange(self.K)).to(torch.float32))

    @staticmethod
    def finite(*values):
        if any(not torch.isfinite(v).all() for v in values):
            raise ValueError('nonfinite lagged auxiliary input or response')

    def project(self, z):
        if z.ndim != 4 or tuple(z.shape[1:]) != (self.C, self.D, self.N) or z.dtype != torch.float32:
            raise ValueError('finite float32 B,C,D,N required')
        self.finite(z)
        zn = F.layer_norm(z.permute(0, 1, 3, 2), (self.D,), weight=None, bias=None, eps=1e-5)
        q = F.linear(zn[:, self.target], self.W_Q)
        k = torch.einsum('bjnd,jrd->bjnr', zn[:, self.aux], self.W_K)
        value = torch.einsum('bjnd,jrd->bjnr', z[:, self.aux].permute(0, 1, 3, 2), self.W_V)
        self.finite(q, k, value)
        return q, k, value

    def transport(self, q, k, value):
        B = q.shape[0]
        if q.shape != (B, self.N, 8) or k.shape != (B, self.J, self.N, 8) or value.shape != k.shape:
            raise ValueError('projected lag schema')
        self.finite(q, k, value)
        if not self.J:
            return q.new_zeros(B, self.N, 8), q.new_zeros(B, 0, self.K), q.new_zeros(B, self.N, 0, self.K)
        # Expand only projected r=8 content. Invalid safe-index fills never score.
        shifted_k = k[:, :, self.source_index, :]
        shifted_k = shifted_k.masked_fill(~self.valid[None, None, :, :, None], 0)
        score = torch.einsum('bnr,bjnkr->bjk', q, shifted_k) / (self.overlap * math.sqrt(8))[None, None, :]
        self.finite(score)
        masked = score[:, None, :, :].expand(-1, self.N, -1, -1).masked_fill(~self.valid[None, :, None, :], -torch.inf)
        alpha = torch.softmax(masked.reshape(B, self.N, self.J * self.K), dim=-1).reshape(B, self.N, self.J, self.K)
        shifted_value = value[:, :, self.source_index, :]
        message = torch.einsum('bnjk,bjnkr->bnr', alpha, shifted_value)
        self.finite(alpha, message)
        return message, score, alpha

    def forward(self, z, diagnostics=None):
        q, k, value = self.project(z)
        message, score, alpha = self.transport(q, k, value)
        delta = F.linear(message, self.W_O).permute(0, 2, 1)
        self.finite(delta)
        result = z.clone()
        result[:, self.target] = z[:, self.target] + delta
        self.finite(result)
        if diagnostics is not None:
            diagnostics['U_lagged_auxiliary'] = dict(score=stats(score) if self.J else None, alpha=stats(alpha) if self.J else None, message=stats(message), delta_target=stats(delta), lag_count=self.K, noncircular=True, weights_are_not_causal_importance=True)
        return result
