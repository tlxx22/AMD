"""Independent, approved SOFTS/APT-inspired output corrections; no author ports."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def receiver_indices(channels, task, target_idx, aux_idx):
    if type(channels) is not int or channels < 1 or task not in ('MS', 'M'):
        raise ValueError('invalid channels/task')
    if type(target_idx) is not int or not 0 <= target_idx < channels:
        raise ValueError('invalid target_idx')
    if not isinstance(aux_idx, (list, tuple)) or any(type(j) is not int for j in aux_idx):
        raise ValueError('explicit ordered integer aux_idx required')
    if len(set(aux_idx)) != len(aux_idx) or any(j < 0 or j >= channels or j == target_idx for j in aux_idx):
        raise ValueError('invalid ordered aux_idx')
    return (target_idx,) if task == 'MS' else tuple(range(channels))


def init_linear_ln(module):
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.LayerNorm):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)


def window_statistics(x):
    """Current preprocessed window only; no state, affine, detach or scaler fit."""
    if x.ndim != 3 or x.shape[1] < 1:
        raise ValueError('history must have shape [b,L,C] with L>=1')
    mu = x.mean(dim=1)
    centered = x - mu[:, None, :]
    sigma = torch.sqrt(centered.square().mean(dim=1) + 1e-5)
    return mu, sigma, centered / sigma[:, None, :]


class SelectiveOutputResidual(nn.Module):
    """U: receiver-conditioned auxiliary core, bounded directly in output space."""
    kappa = 1.0

    def __init__(self, length, pred_len, channels, task, target_idx, aux_idx):
        super().__init__()
        self.length, self.pred_len, self.channels = length, pred_len, channels
        self.receivers = receiver_indices(channels, task, target_idx, aux_idx)
        self.sources = tuple(aux_idx) if task == 'MS' else tuple(range(channels))
        self.task = task
        self.linear_z = nn.Linear(length, 32)
        self.norm = nn.LayerNorm(32, eps=1e-5)
        self.linear_a = nn.Linear(2, 32)
        self.v1, self.v2 = nn.Linear(32, 32), nn.Linear(32, 16)
        self.aq, self.ak = nn.Linear(32, 32, bias=False), nn.Linear(32, 32, bias=False)
        self.b_cond = nn.Parameter(torch.zeros(32))
        self.ao = nn.Linear(32, 16)
        self.gate = nn.Linear(48, pred_len)
        self.out = nn.Linear(16, pred_len, bias=False)
        init_linear_ln(self)
        nn.init.zeros_(self.out.weight)

    def forward(self, x, *, return_diagnostics=False):
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.length, self.channels):
            raise ValueError('U expects [b,L,C] registered history')
        if not self.sources or (self.task == 'M' and self.channels == 1):
            delta = x.new_zeros(x.shape[0], len(self.receivers), self.pred_len)
            return (delta, None) if return_diagnostics else delta
        mu, sigma, z = window_statistics(x)
        history = self.norm(self.linear_z(z.transpose(1, 2)))
        level = self.linear_a(torch.stack((mu, torch.log1p(sigma)), dim=-1))
        e = history + level
        values = self.v2(F.gelu(self.v1(e)))
        er = e[:, self.receivers, :]
        logits = self.ao(F.gelu(self.aq(er).unsqueeze(2) + self.ak(e).unsqueeze(1) + self.b_cond))
        allowed = torch.zeros(len(self.receivers), self.channels, device=x.device, dtype=torch.bool)
        if self.task == 'MS':
            allowed[:, self.sources] = True
        else:
            allowed[:] = True
            allowed.fill_diagonal_(False)
        logits = logits.masked_fill(~allowed[None, :, :, None], -torch.inf)
        null = logits.new_zeros(x.shape[0], len(self.receivers), 1, 16)
        alpha = torch.cat((logits, null), dim=2).softmax(dim=2)
        core = (alpha[:, :, :-1, :] * values[:, None, :, :]).sum(dim=2)
        gate = self.gate(torch.cat((er, core), dim=-1)).sigmoid()
        d_u = gate * self.out(core)
        delta = self.kappa * torch.tanh(d_u / self.kappa)
        if return_diagnostics:
            return delta, dict(alpha=alpha, core=core, gate=gate, mu=mu, sigma=sigma,
                               dU=d_u, Delta_U=delta, allowed=allowed,
                               history_embedding=history, statistics_embedding=level)
        return delta


class PrototypeOutputCalibration(nn.Module):
    """V: history-only timestamp/state prototypes calibrate original Y0 alone."""
    kappa = 1.0
    prototype_count, prototype_dim = 8, 16

    def __init__(self, length, pred_len, channels, marks_dim, task, target_idx, aux_idx):
        super().__init__()
        if type(marks_dim) is not int or marks_dim < 1:
            raise ValueError('registered native mark dimension required')
        self.length, self.pred_len, self.channels, self.marks_dim = length, pred_len, channels, marks_dim
        self.receivers = receiver_indices(channels, task, target_idx, aux_idx)
        self.time = nn.Linear(2 * marks_dim, 16)
        self.state = nn.Linear(4, 16)
        self.norm = nn.LayerNorm(16, eps=1e-5)
        self.prototypes = nn.Parameter(torch.empty(8, 16))
        self.hidden = nn.Linear(16, 16)
        self.final = nn.Linear(16, 2 * pred_len)
        init_linear_ln(self)
        nn.init.xavier_uniform_(self.prototypes)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, x, marks, y0, *, return_diagnostics=False):
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.length, self.channels):
            raise ValueError('V history shape mismatch')
        if marks is None or tuple(marks.shape) != (x.shape[0], self.length, self.marks_dim):
            raise ValueError('V registered historical native marks required')
        if tuple(y0.shape) != (x.shape[0], self.pred_len, self.channels):
            raise ValueError('V original native Y0 shape mismatch')
        mu, sigma, _ = window_statistics(x)
        change = (x[:, 1:, :] - x[:, :-1, :]).abs().mean(dim=1) if self.length > 1 else torch.zeros_like(mu)
        state = torch.stack((mu, torch.log1p(sigma), x[:, -1, :], change), dim=-1)[:, self.receivers, :]
        m = torch.cat((marks[:, 0, :], marks[:, -1, :]), dim=-1)
        query = self.norm(F.gelu(self.time(m)[:, None, :] + self.state(state)))
        pi = (query @ self.prototypes.T / math.sqrt(16)).softmax(dim=-1)
        context = pi @ self.prototypes
        a_v, b_v = self.final(F.gelu(self.hidden(context))).split(self.pred_len, dim=-1)
        original = y0[:, :, self.receivers].transpose(1, 2)
        d_v = a_v * (original - mu[:, self.receivers, None]) + b_v
        delta = self.kappa * torch.tanh(d_v / self.kappa)
        if return_diagnostics:
            return delta, dict(query=query, pi=pi, context=context, aV=a_v, bV=b_v,
                               dV=d_v, Delta_V=delta, mu=mu, sigma=sigma, state=state, marks=m)
        return delta
