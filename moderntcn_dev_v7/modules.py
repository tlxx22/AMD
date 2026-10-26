"""Independent tied quadratic basis; not a full N-BEATS/NBEATSx stack."""
import torch
from torch import nn
from torch.nn import functional as F
from moderntcn_dev_v1.modules import stats
from moderntcn_dev_v3.modules import ConditionalTemporalFilter


class TiedHistoryComponent(nn.Module):
    def __init__(self, L, C, P, target, aux, horizon, urban):
        super().__init__()
        if L <= 1 or not 0 <= target < C or tuple(aux) != tuple(j for j in range(C) if j != target):
            raise ValueError('history/target/ordered auxiliary schema')
        if (urban and (L != 12 or P != 1 or horizon not in (3, 6, 9, 12))) or (not urban and (L, P, C, horizon) != (96, 24, 3, 24)):
            raise ValueError('frozen history and true forecast offsets')
        self.L, self.C, self.P, self.target, self.aux = L, C, P, target, tuple(aux)
        self.W_enc = nn.Parameter(torch.empty(32, C * (L + 2)))
        self.b_enc = nn.Parameter(torch.zeros(32))
        self.W_coef = nn.Parameter(torch.zeros(3, 32))
        nn.init.xavier_uniform_(self.W_enc)
        tb = (torch.arange(L, dtype=torch.float32) - (L - 1)) / (L - 1)
        tf = (torch.tensor([horizon], dtype=torch.float32) if urban else torch.arange(1, P + 1, dtype=torch.float32)) / (L - 1)
        self.register_buffer('Phi_b', torch.stack((torch.ones_like(tb), tb, tb * tb), -1))
        self.register_buffer('Phi_f', torch.stack((torch.ones_like(tf), tf, tf * tf), -1))

    def encode(self, X, Z0):
        if X.ndim != 3 or tuple(X.shape[1:]) != (self.L, self.C) or tuple(Z0.shape) != (X.shape[0], self.C, self.L):
            raise ValueError('current history / original RevIN schema')
        Xf = X.float()
        if Z0.dtype != torch.float32 or not torch.isfinite(Xf).all() or not torch.isfinite(Z0).all():
            raise ValueError('finite float32 history required')
        mu = Xf.mean(dim=1)
        var = ((Xf - mu[:, None, :]) ** 2).mean(dim=1)
        s = torch.stack((mu, torch.log1p(var)), dim=-1)
        if not torch.isfinite(s).all():
            raise ValueError('nonfinite population statistics')
        r = torch.cat((Z0.reshape(X.shape[0], self.C * self.L), s.reshape(X.shape[0], 2 * self.C)), dim=-1)
        eta = F.relu(F.linear(r, self.W_enc, self.b_enc))
        theta = F.linear(eta, self.W_coef, bias=None)
        if not torch.isfinite(theta).all():
            raise ValueError('nonfinite tied coefficients')
        return theta

    def forward(self, X, Z0, diagnostics=None):
        theta = self.encode(X, Z0)
        backcast = theta @ self.Phi_b.T
        forecast = theta @ self.Phi_f.T
        if not torch.isfinite(backcast).all() or not torch.isfinite(forecast).all():
            raise ValueError('nonfinite basis component')
        if diagnostics is not None:
            diagnostics['U_tied_component'] = dict(theta=stats(theta), backcast=stats(backcast), forecast=stats(forecast), unit='original native RevIN normalized target', shared_coefficients=True)
        return backcast, forecast
