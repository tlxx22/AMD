"""User-frozen v6 r4 internal submodule adaptations (MIT sources, notices included).

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

    def forward(self, x, *, return_diagnostics=False, return_logits=False):
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.length, self.channels):
            raise ValueError('U requires original loader history [B,L,C]')
        if not self.has_aux:
            gate = x.new_ones((x.shape[0], len(self.receivers), self.d_model))
            result = torch.zeros_like(gate) if return_logits else gate
            return (result, dict(no_aux=True, gate=gate)) if return_diagnostics else result
        embedded = self.history(x.transpose(1, 2))
        query = embedded.index_select(1, self.receiver_order)
        bases = self.auxiliary_bases(embedded)
        kappa = 2 * torch.sigmoid(self.kappa(query.unsqueeze(-1)))
        hidden = F.relu(kappa * bases + self.target_term(query.unsqueeze(-1)))
        logits = self.output(hidden).squeeze(-1)
        gate = 2 * torch.sigmoid(logits)
        result = logits if return_logits else gate
        if return_diagnostics:
            return result, dict(embedding=embedded[:8], target=query[:8], bases=bases[:8], kappa=kappa[:8],
                              hidden=hidden[:8], logits=logits[:8], gate=gate[:8], no_aux=False)
        return result


def norm_preserving_carrier(carrier, logits):
    """Along D only: ||z|| * (m*z)/||m*z||, m=2*sigmoid(logits).

    Local float64 arithmetic leaves model/parameters float32. log(2) and the
    common max(logsigmoid) cancel. Scaled Euclidean norms avoid squared large
    carriers and squared subnormal masses. All scales/norms retain autograd.
    Zero z uses safe positive norm inputs *before* where. No eps, clipping,
    gate floor, gain, detach, or gate==1 bypass is used.
    """
    if carrier.shape[-1] != logits.shape[-1]:
        raise ValueError('U norm/relative logits require matching D axis')
    z = carrier.to(torch.float64)
    l = logits.to(torch.float64)
    if not torch.isfinite(z).all() or not torch.isfinite(l).all():
        raise FloatingPointError('U requires finite native carrier and controller logits')
    log_gate = F.logsigmoid(l)  # log(2) cancels; this is NOT the raw gate
    relative_gate = torch.exp(log_gate - log_gate.amax(dim=-1, keepdim=True))
    radius_scale = z.abs().amax(dim=-1, keepdim=True)
    nonzero = radius_scale > 0
    safe_radius = torch.where(nonzero, radius_scale, torch.ones_like(radius_scale))
    unit_z = z / safe_radius
    v = unit_z * relative_gate
    v_scale = v.abs().amax(dim=-1, keepdim=True)
    if torch.any(nonzero & (v_scale == 0)):
        raise FloatingPointError('U nonzero carrier has numerical zero gated norm')
    safe_v_scale = torch.where(nonzero, v_scale, torch.ones_like(v_scale))
    unit_v = v / safe_v_scale
    z_energy = torch.where(nonzero, unit_z.square().sum(-1, keepdim=True), torch.ones_like(radius_scale))
    v_energy = torch.where(nonzero, unit_v.square().sum(-1, keepdim=True), torch.ones_like(radius_scale))
    # Initial constant logits give relative_gate=1, ratio exactly one; no
    # conditional bypass may remove the initial gate gradient.
    factor = torch.sqrt(z_energy) / torch.sqrt(v_energy)
    changed = z + safe_radius * (unit_v * factor - unit_z)
    result = torch.where(nonzero, changed, z).to(carrier.dtype)
    if not torch.isfinite(result).all():
        raise FloatingPointError('U norm-preserving carrier not representable in native dtype')
    return result


def gated_native_projection(carrier, native, logits, receivers, *, batch_size, channels,
                            channel_independence, diagnostics=None):
    """Original native bias plus original W applied to norm-preserved carrier.

    Compute original native(carrier)+W*(z_star-carrier) to preserve the exact
    initialization path. Receiver loop bounds temporary memory to [B,P,D];
    no [B,R,P,D,K] tensor exists. The gate controller remains shared over P.
    """
    base = native(carrier)
    if logits.shape != (batch_size, len(receivers), carrier.shape[-1]):
        raise ValueError('explicit receiver/latent logits shape required')
    if channel_independence not in (0, 1):
        raise ValueError('native CI contract')
    if channel_independence == 0 and (carrier.shape[0] != batch_size or native.out_features != channels):
        raise ValueError('CI0 native carrier/output axes')
    if channel_independence == 1 and (carrier.shape[0] != batch_size*channels or native.out_features != 1):
        raise ValueError('CI1 b-major variable-minor native carrier axes')
    value = base.clone()
    for i, r in enumerate(receivers):
        z = carrier if channel_independence == 0 else carrier.reshape(batch_size, channels, *carrier.shape[1:])[:, r]
        changed = norm_preserving_carrier(z, logits[:, i, None, :])
        weight = native.weight[r:r+1] if channel_independence == 0 else native.weight
        correction = F.linear(changed-z, weight, None)
        if channel_independence == 0:
            value[:, :, r:r+1] = base[:, :, r:r+1] + correction
        else:
            value.reshape(batch_size,channels,*value.shape[1:])[:,r] = base.reshape(batch_size,channels,*base.shape[1:])[:,r] + correction
        if diagnostics is not None:
            # Explicitly requested, <=8 samples per receiver; no persistent state.
            with torch.no_grad():
                zz=z[:8].double(); ss=changed[:8].double()
                nz=zz.norm(dim=-1); ns=ss.norm(dim=-1)
                diagnostics.append(dict(receiver=r, carrier_norm=nz, changed_norm=ns,
                    norm_difference=ns-nz, inner_product=(zz*ss).sum(-1),
                    zero_carrier=nz==0, raw_gate=2*torch.sigmoid(logits[:8,i]),
                    raw_logits=logits[:8,i], native_correction=correction[:8],
                    relative_gate_is_not_raw_gate=True))
    return value


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
        previous = F.pad(g.transpose(1,2), (2,0), mode='replicate').transpose(1,2)
        return ((g-previous[:,1:-1]) + (g-previous[:,:-2])).div(3).detach()

    def forward(self, g, *, return_diagnostics=False):
        if g.ndim != 3 or g.shape[1] < 1 or g.shape[2] != self.d_model:
            raise ValueError('V requires current finest [B_eff,L,D]')
        b, length, dim = g.shape
        relation = self.relation_input(g)
        q = self.query(relation).reshape(b, length, 4, 8).transpose(1, 2)
        k = self.key(relation).reshape(b, length, 4, 8).transpose(1, 2)
        # Native float32 training; preserve double only for independent equation
        # checks. RMS statistics use local float64 to avoid square overflow;
        # normalized vectors/dot/softmax use native float32 (double for references).
        calc_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        qq, kk = q.to(torch.float64), k.to(torch.float64)
        qms, kms = qq.square().mean(-1,keepdim=True), kk.square().mean(-1,keepdim=True)
        qstar = (qq / torch.sqrt(qms + 1e-6)).to(calc_dtype)
        kstar = (kk / torch.sqrt(kms + 1e-6)).to(calc_dtype)
        scores = torch.matmul(qstar, kstar.transpose(-2,-1)) / math.sqrt(8)
        attention = torch.softmax(scores,dim=-1)
        raw_values = g.reshape(b,length,4,dim//4).transpose(1,2)
        y = torch.matmul(attention,raw_values.to(calc_dtype)).transpose(1,2).reshape(b,length,dim).to(g.dtype)
        alpha = 0.5*torch.tanh(self.a)
        value = g + alpha*(y-g)
        if return_diagnostics:
            return value, dict(relation=relation[:8],query=q[:8],key=k[:8],
                normalized_query=qstar[:8],normalized_key=kstar[:8],attention=attention[:8],
                q_mean_square=qms[:8], k_mean_square=kms[:8],
                q_eps_dominant=qms[:8]<1e-6,k_eps_dominant=kms[:8]<1e-6,
                raw_values=raw_values[:8],redistributed=y[:8],alpha=alpha,
                hidden_response=value[:8]-g[:8],diagnostic_sample_limit=8)
        return value
