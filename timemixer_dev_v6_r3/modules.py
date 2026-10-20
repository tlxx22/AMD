"""User-frozen v6 r3 internal submodule adaptations (MIT sources, notices included).

U: XLinear Gating_Block two-layer gate -> target-conditioned auxiliary bases.
V: TimeBridge Q/K relation vs raw Value separation -> real-L temporal operator.
No full author U/V model, extra predictor, normalization block, or free aux Value.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def receiver_indices(channels, task, target_idx, aux_idx):
    if type(channels) is not int or channels < 1 or task not in ('MS', 'M'):
        raise ValueError('explicit positive C and MS/M required')
    if type(target_idx) is not int or not 0 <= target_idx < channels:
        raise ValueError('explicit valid target index required')
    if len(set(aux_idx)) != len(aux_idx) or any(type(j) is not int or not 0 <= j < channels or j == target_idx for j in aux_idx):
        raise ValueError('ordered unique auxiliaries excluding target required')
    return (target_idx,) if task == 'MS' else tuple(range(channels))


class TargetConditionedAuxBasisGate(nn.Module):
    """Gate [B,R,D], shared over P; receiver conditioning modulates A_mix itself."""
    basis_count = 8
    def __init__(self, length, d_model, channels, task, target_idx, aux_idx):
        super().__init__()
        if any(type(n) is not int or n < 1 for n in (length, d_model, channels)):
            raise ValueError('positive L/D/C required')
        self.length, self.d_model, self.channels, self.task = length, d_model, channels, task
        self.receivers = receiver_indices(channels, task, target_idx, aux_idx)
        slots = tuple(aux_idx) if task == 'MS' else tuple(range(channels))
        self.has_aux = bool(slots) if task == 'MS' else channels > 1
        self.register_buffer('source_slot_indices', torch.tensor(slots, dtype=torch.long))
        self.register_buffer('receiver_order', torch.tensor(self.receivers, dtype=torch.long))
        self.history = nn.Linear(length, d_model, bias=True)
        self.mix_weight = nn.Parameter(torch.empty(self.basis_count, len(slots)))
        if slots:
            nn.init.kaiming_uniform_(self.mix_weight, a=math.sqrt(5))  # exact nn.Linear default
        self.kappa = nn.Linear(1, self.basis_count, bias=True)
        self.target_term = nn.Linear(1, self.basis_count, bias=True)
        self.output = nn.Linear(self.basis_count, 1, bias=True)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def auxiliary_bases(self, embedded):
        """Exclude a receiver before its aggregation; never total-minus-self."""
        if self.task == 'MS':
            source = embedded.index_select(1, self.source_slot_indices)
            mixed = torch.einsum('bjd,kj->bdk', source, self.mix_weight)
            return mixed[:, None, :, :] / len(self.source_slot_indices)
        # [B,C,D,K], no receiver-source C x C tensor. Each prefix/suffix excludes r.
        weighted = embedded.unsqueeze(-1) * self.mix_weight.transpose(0, 1)[None, :, None, :]
        zero = weighted.new_zeros((weighted.shape[0], 1, weighted.shape[2], weighted.shape[3]))
        prefix = torch.cat((zero, weighted[:, :-1].cumsum(dim=1)), dim=1)
        suffix = torch.cat((weighted[:, 1:].flip(1).cumsum(dim=1).flip(1), zero), dim=1)
        return (prefix + suffix) / (self.channels - 1)

    def forward(self, x, *, return_diagnostics=False):
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.length, self.channels):
            raise ValueError('U requires original loader history [B,L,C]')
        if not self.has_aux:
            gate = x.new_ones((x.shape[0], len(self.receivers), self.d_model))
            return (gate, dict(no_aux=True, gate=gate)) if return_diagnostics else gate
        embedded = self.history(x.transpose(1, 2))
        query = embedded.index_select(1, self.receiver_order)
        bases = self.auxiliary_bases(embedded)
        kappa = 2 * torch.sigmoid(self.kappa(query.unsqueeze(-1)))
        hidden = F.relu(kappa * bases + self.target_term(query.unsqueeze(-1)))
        logits = self.output(hidden).squeeze(-1)
        gate = 2 * torch.sigmoid(logits)
        if return_diagnostics:
            return gate, dict(embedding=embedded, target=query, bases=bases, kappa=kappa,
                              hidden=hidden, logits=logits, gate=gate, no_aux=False)
        return gate


def gated_native_projection(carrier, native, gate, receivers, *, batch_size, channels, channel_independence):
    """Exact gate=1 native identity; algebraic native W*(m*z)+b otherwise.

    native(z) + W*((m-1)*z) retains the original projection call and its unscaled
    bias, while avoiding [B,R,P,D] products and all [B,R,P,D,K] materialization.
    This is the same frozen formula, not an additional learned prediction head.
    """
    base = native(carrier)
    if gate.ndim != 3 or gate.shape != (batch_size, len(receivers), carrier.shape[-1]):
        raise ValueError('explicit receiver/latent gate shape required')
    if channel_independence == 0:
        if carrier.shape[0] != batch_size or native.out_features != channels:
            raise ValueError('CI0 native carrier/output axes')
        index = torch.tensor(receivers, dtype=torch.long, device=carrier.device)
        effective = native.weight.index_select(0, index)[None, :, :] * (gate - 1)
        correction = torch.bmm(carrier, effective.transpose(1, 2))
        value = base.clone()
        value.index_copy_(-1, index, base.index_select(-1, index) + correction)
        return value
    if channel_independence != 1 or native.out_features != 1 or carrier.shape[0] != batch_size * channels:
        raise ValueError('CI1 b-major variable-minor native carrier axes')
    full_gate = gate.new_ones((batch_size, channels, gate.shape[-1]))
    index = torch.tensor(receivers, dtype=torch.long, device=carrier.device)
    full_gate.index_copy_(1, index, gate)
    effective = (full_gate.reshape(batch_size * channels, -1) - 1) * native.weight[0]
    correction = torch.bmm(carrier, effective.unsqueeze(-1))
    return base + correction


class TemporalQKRawValueOperator(nn.Module):
    """Only relation attention with unprojected latent Values; no parent block."""
    heads, relation_dim, relation_head_dim = 4, 32, 8
    def __init__(self, d_model):
        super().__init__()
        if type(d_model) is not int or d_model < 1 or d_model % self.heads:
            raise ValueError('V requires positive D divisible by four')
        self.d_model = d_model
        self.query = nn.Linear(d_model, self.relation_dim, bias=False)
        self.key = nn.Linear(d_model, self.relation_dim, bias=False)
        self.a = nn.Parameter(torch.zeros(()))

    @staticmethod
    def relation_input(g):
        average = F.avg_pool1d(F.pad(g.transpose(1, 2), (2, 0), mode='replicate'),
                              kernel_size=3, stride=1).transpose(1, 2)
        return (g - average).detach()

    def forward(self, g, *, return_diagnostics=False):
        if g.ndim != 3 or g.shape[1] < 1 or g.shape[2] != self.d_model:
            raise ValueError('V requires current finest [B_eff,L,D]')
        b, length, dim = g.shape
        relation = self.relation_input(g)
        q = self.query(relation).reshape(b, length, 4, 8).transpose(1, 2)
        k = self.key(relation).reshape(b, length, 4, 8).transpose(1, 2)
        attention = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(8), dim=-1)
        raw_values = g.reshape(b, length, 4, dim // 4).transpose(1, 2)
        y = torch.matmul(attention, raw_values).transpose(1, 2).reshape(b, length, dim)
        alpha = 0.5 * torch.tanh(self.a)
        value = g + alpha * (y - g)
        if return_diagnostics:
            return value, dict(relation=relation, query=q, key=k, attention=attention,
                               raw_values=raw_values, redistributed=y, alpha=alpha)
        return value
