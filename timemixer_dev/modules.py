"""Independent implementation of the approved U/V formulas (not source ports)."""
import torch
from torch import nn
from torch.nn import functional as F


def init_linear_ln(module):
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.LayerNorm):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)


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


class SelectiveVariableInteraction(nn.Module):
    """U: receiver-conditioned core, fixed null source, gated history residual."""
    def __init__(self, length, channels, task, target_idx, aux_idx):
        super().__init__()
        self.length, self.channels = length, channels
        self.receivers = receiver_indices(channels, task, target_idx, aux_idx)
        self.sources = tuple(aux_idx) if task == 'MS' else tuple(range(channels))
        self.task = task
        self.linear_z = nn.Linear(length, 32)
        self.norm = nn.LayerNorm(32, eps=1e-5)
        self.linear_a = nn.Linear(2, 32)
        self.v1, self.v2 = nn.Linear(32, 32), nn.Linear(32, 16)
        self.aq, self.ak = nn.Linear(32, 32, bias=False), nn.Linear(32, 32, bias=False)
        self.ba = nn.Parameter(torch.zeros(32))
        self.ao = nn.Linear(32, 16)
        self.gate = nn.Linear(48, length)
        self.out = nn.Linear(16, length, bias=False)
        self.gamma = nn.Parameter(torch.tensor(0.01))
        init_linear_ln(self)

    def forward(self, z, x, *, return_diagnostics=False):
        if z.ndim != 3 or tuple(z.shape[1:]) != (self.length, self.channels) or x.shape != z.shape:
            raise ValueError('U expects matching [b,L,C] histories')
        # No allowed source: exact bypass, including M/C=1.
        if not self.sources or (self.task == 'M' and self.channels == 1):
            return (z, None) if return_diagnostics else z
        mu = x.mean(dim=1)
        sigma = torch.sqrt(x.var(dim=1, unbiased=False) + 1e-5)
        stats = torch.stack((mu, torch.log1p(sigma)), dim=-1)
        e = self.norm(self.linear_z(z.transpose(1, 2))) + self.linear_a(stats)
        values = self.v2(F.gelu(self.v1(e)))
        er = e[:, self.receivers, :]
        pair = self.aq(er).unsqueeze(2) + self.ak(e).unsqueeze(1) + self.ba
        logits = self.ao(F.gelu(pair))
        allowed = torch.zeros(len(self.receivers), self.channels, dtype=torch.bool, device=z.device)
        if self.task == 'MS':
            allowed[:, self.sources] = True
        else:
            allowed[:] = True
            allowed[torch.arange(self.channels, device=z.device), torch.arange(self.channels, device=z.device)] = False
        logits = logits.masked_fill(~allowed[None, :, :, None], -torch.inf)
        null = logits.new_zeros(z.shape[0], len(self.receivers), 1, 16)
        alpha = torch.cat((logits, null), dim=2).softmax(dim=2)
        core = (alpha[:, :, :-1, :] * values[:, None, :, :]).sum(dim=2)
        gate = torch.sigmoid(self.gate(torch.cat((er, core), dim=-1)))
        delta = gate * self.out(core)
        result = z.clone()
        result[:, :, self.receivers] = z[:, :, self.receivers] + self.gamma * delta.transpose(1, 2)
        if return_diagnostics:
            return result, dict(alpha=alpha, core=core, delta=delta, embeddings=e, allowed=allowed)
        return result


class HistoryConditionedScaleAdjustment(nn.Module):
    """V: dense per-position weights over existing PDM/FMM scales, rho=0.5."""
    rho = 0.5

    def __init__(self, d_model, scales, pred_len, channels, task, target_idx, aux_idx, channel_independence):
        super().__init__()
        self.receivers = receiver_indices(channels, task, target_idx, aux_idx)
        if channel_independence != (1 if task == 'M' else 0):
            raise ValueError('candidate requires MS/CI=0 or M/CI=1')
        self.d_model, self.scales, self.pred_len = d_model, scales, pred_len
        self.channels, self.ci = channels, channel_independence
        self.h = nn.Linear(3 * d_model, 16)
        self.norm = nn.LayerNorm(16, eps=1e-5)
        self.eta = nn.Parameter(torch.zeros(scales, 16))
        self.f = nn.Linear(32, 16)
        self.p = nn.Linear(16, pred_len, bias=False)
        init_linear_ln(self)
        nn.init.normal_(self.p.weight, mean=0.0, std=1e-3)

    def descriptors(self, states, batch):
        if len(states) != self.scales:
            raise ValueError('PDM scale count mismatch')
        descriptions = []
        for state in states:
            expected = batch * self.channels if self.ci else batch
            if state.ndim != 3 or state.shape[0] != expected or state.shape[2] != self.d_model or state.shape[1] < 1:
                raise ValueError('PDM [b*C,Ls,D] / [b,Ls,D] mismatch')
            g = state.reshape(batch, self.channels if self.ci else 1, state.shape[1], self.d_model)
            mean, last = g.mean(dim=2), g[:, :, -1, :]
            change = (g[:, :, 1:, :] - g[:, :, :-1, :]).abs().mean(dim=2) if g.shape[2] > 1 else torch.zeros_like(mean)
            descriptions.append(torch.cat((mean, last, change), dim=-1))
        return torch.stack(descriptions, dim=2)  # [b,R,S,3D]

    def probabilities(self, states, batch):
        desc = self.descriptors(states, batch)
        h = self.norm(F.gelu(self.h(desc))) + self.eta[None, None, :, :]
        hbar = h.mean(dim=2, keepdim=True).expand_as(h)
        logits = self.p(F.gelu(self.f(torch.cat((h, hbar), dim=-1))))
        return logits.softmax(dim=2)

    def forward(self, states, predictions, *, return_diagnostics=False):
        if len(predictions) != self.scales:
            raise ValueError('FMM scale count mismatch')
        batch = predictions[0].shape[0]
        if any(tuple(y.shape) != (batch, self.pred_len, self.channels) for y in predictions):
            raise ValueError('FMM must retain native C outputs')
        pi = self.probabilities(states, batch)
        stacked = torch.stack(predictions, dim=-1)  # [b,P,C,S]
        base = stacked.sum(-1)  # original sum, never mean
        yr = stacked[:, :, self.receivers, :].permute(0, 2, 3, 1)
        correction = self.rho * ((self.scales * pi - 1) * yr).sum(dim=2)
        result = base.clone()
        result[:, :, self.receivers] = base[:, :, self.receivers] + correction.transpose(1, 2)
        if return_diagnostics:
            return result, dict(pi=pi, weights=1 - self.rho + self.rho * self.scales * pi, correction=correction)
        return result
