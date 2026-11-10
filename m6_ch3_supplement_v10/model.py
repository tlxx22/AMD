"""Analysis-only G/R parameterization; frozen production files are untouched."""
import torch
import hashlib
from torch import nn
from torch.nn import functional as F
from moderntcn_dev_v10 import adapter
from moderntcn_dev_v10.model import ModernTCNUV
from moderntcn_dev_v10.modules import interpolation_matrix, control_count
from moderntcn_dev_v3.modules import ConditionalTemporalFilter


class AnalysisSpline(nn.Module):
    def __init__(self, I, P, G):
        super().__init__()
        if G not in (3, 5, 7):
            raise ValueError('unapproved grid')
        self.I, self.P, self.G = I, P, G
        self.K = control_count(P)
        self.coef = nn.Parameter(torch.zeros(self.K, I, G + 3))
        self.register_buffer('knots', -1 + (torch.arange(G + 7, dtype=torch.float32) - 3) * (2.0 / G))
        if P != 1:
            self.register_buffer('M', interpolation_matrix(P))

    def basis(self, a):
        if a.dtype != torch.float32 or a.ndim != 2 or a.shape[1] != self.I:
            raise ValueError('float32 [B,I] required')
        if not bool(torch.isfinite(a).all() & torch.isfinite(self.coef).all() & torch.isfinite(self.knots).all()):
            raise ValueError('nonfinite spline')
        if not bool((self.knots[1:] > self.knots[:-1]).all()):
            raise ValueError('invalid knots')
        u, k = torch.tanh(a).unsqueeze(-1), self.knots
        b = ((u >= k[:-1]) & (u < k[1:])).to(a.dtype)
        for d in range(1, 4):
            n = self.G + 6 - d
            left = (u - k[:n]) / (k[d:d+n] - k[:n])
            right = (k[d+1:d+1+n] - u) / (k[d+1:d+1+n] - k[1:1+n])
            b = left * b[..., :n] + right * b[..., 1:n+1]
        if not bool(torch.isfinite(b).all()):
            raise ValueError('nonfinite basis')
        return b

    def forward(self, a, diagnostics=None):
        b = self.basis(a)
        q = F.linear(b.reshape(a.shape[0], -1), self.coef.reshape(self.K, -1)) / self.I
        delta = q if self.P == 1 else F.linear(q, self.M)
        if not bool(torch.isfinite(delta).all()):
            raise ValueError('nonfinite increment')
        return delta


class AnalysisFilter(ConditionalTemporalFilter):
    def __init__(self, L, D, N, large, C, target_idx, R):
        nn.Module.__init__(self)
        if R not in (2, 4, 6) or not 0 <= target_idx < C:
            raise ValueError('unapproved rank/schema')
        self.L, self.D, self.N, self.C = L, D, N, C
        self.R, self.target_idx, self.Kv = R, target_idx, min(large, 2*N-1)
        if R > self.Kv:
            raise ValueError('row-orthogonal basis required')
        # Same construction and explicit initialization order as original V.
        self.W1 = nn.Linear(L, 16, bias=False)
        self.W2 = nn.Linear(16, R, bias=False)
        self.A = nn.Parameter(torch.zeros(D, R))
        self.T = nn.Parameter(torch.empty(R, self.Kv))
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.orthogonal_(self.T)
        self.P_s = nn.Parameter(torch.zeros(16, 2))

    def forward(self, H, Z, X, diagnostics=None):
        B, C, D, N = H.shape
        if (C, D, N) != (self.C, self.D, self.N) or X.shape[0] != B:
            raise ValueError('fixed filter profile shape')
        route, _, _ = self._route(Z, X)
        bases = F.conv1d(H.reshape(B*C*D, 1, N), self.T[:, None, :], padding=self.Kv//2).reshape(B, C, D, self.R, N)
        return (bases * route[:, :, None, :, None] * self.A[None, None, :, :, None]).sum(3)


def build(profile, G=5, R=4, *, task='MS', fold=6):
    if (G, R) not in ((5, 4), (3, 4), (7, 4), (5, 2), (5, 6)):
        raise ValueError('outside frozen sensitivity grid')
    model = adapter.build(profile, use_u=True, use_v=True, task=task, fold=fold)
    if (G, R) == (5, 4):
        return model
    v = model.spec.value
    # Native B has already been completely initialized. Replacement modules
    # use their own original substreams and cannot consume caller dropout RNG.
    with torch.random.fork_rng(devices=[]):
        if G != 5:
            torch.random.default_generator.manual_seed(v['module_init_seed'] + 17011)
            model.u = AnalysisSpline(v['D']*v['N'], v['pred_len'], G)
        if R != 4:
            torch.random.default_generator.manual_seed(v['module_init_seed'] + 17021)
            model.v = AnalysisFilter(v['L'], v['D'], v['N'], v['native_options']['large_size'][0], v['C'], v['target_idx'], R)
    model.analysis_identity = dict(base_candidate_id='moderntcn-uv-dev-v10',
        analysis_variant_id=f'ch3-sensitivity-G{G}-R{R}-v1', G=G, R=R,
        basis_count=G+3, purpose='validation_only_sensitivity',
        coef_shape=list(model.u.coef.shape), A_shape=list(model.v.A.shape),
        T_shape=list(model.v.T.shape), W2_shape=list(model.v.W2.weight.shape),
        knots_sha256=hashlib.sha256(model.u.knots.numpy().tobytes()).hexdigest(),
        P=v['pred_len'], I=v['D']*v['N'], scale='1/I', degree=3,
        M_sha256=hashlib.sha256(model.u.M.numpy().tobytes()).hexdigest() if hasattr(model.u, 'M') else None)
    return model


def validate_state(model, state, identity):
    if identity != getattr(model, 'analysis_identity', None):
        raise ValueError('analysis identity mismatch before live writes')
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError('state keys')
    for k, value in state.items():
        if value.shape != expected[k].shape or value.dtype != expected[k].dtype or not bool(torch.isfinite(value).all()):
            raise ValueError('state tensor format: ' + k)
        if k in ('u.knots', 'u.M') and not torch.equal(value.cpu(), expected[k].cpu()):
            raise ValueError('fixed buffer mismatch: ' + k)
    return state
