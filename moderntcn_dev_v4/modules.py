# Copyright 2026 The Google Research Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Modified PyTorch TFT historical VSN core; not a full TFT reproduction.

Source: google-research/google-research, commit
8bedbbd982e7dcf5c5280e6ae2f0769295c3369e, tft/libs/tft_model.py.
Project changes: auxiliary native patches; current target latent context;
8-dimensional selector GRN then score; zero output; target-only update; no dropout.
The unchanged V3 implementation is imported explicitly, not reimplemented.
"""
import torch
from torch import nn
from torch.nn import functional as F
from moderntcn_dev_v3.modules import ConditionalTemporalFilter
from moderntcn_dev_v1.modules import stats


def linear(inputs, outputs, bias=True):
    layer = nn.Linear(inputs, outputs, bias=bias)
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class ZeroOutput(nn.Module):
    """Bias-free linear with directly zero-created weights, no random draw."""
    def __init__(self, inputs, outputs):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(outputs, inputs))

    def forward(self, x):
        return F.linear(x, self.weight, bias=None)


class GRN(nn.Module):
    """TFT ELU -> linear -> GLU -> skip+LayerNorm, fixed r=8, dropout=0."""
    def __init__(self, inputs, *, context_dim=None, projected_skip=False):
        super().__init__()
        self.L_in = linear(inputs, 8)
        if context_dim is not None:
            self.L_ctx = linear(context_dim, 8, bias=False)
        self.L_hidden = linear(8, 8)
        self.L_gate = linear(8, 8)
        self.L_value = linear(8, 8)
        self.skip = linear(inputs, 8) if projected_skip else nn.Identity()
        if not projected_skip and inputs != 8:
            raise ValueError('identity GRN skip requires eight input coordinates')
        self.norm = nn.LayerNorm(8, eps=1e-3, elementwise_affine=True)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(self, a, context=None):
        if a.dtype != torch.float32:
            raise ValueError('registered GRN computation is float32')
        q = self.L_in(a)
        if hasattr(self, 'L_ctx'):
            if context is None or context.dtype != torch.float32 or context.shape[:-1] != a.shape[:-1]:
                raise ValueError('selector context axes/dtype')
            q = q + self.L_ctx(context)
        elif context is not None:
            raise ValueError('variable content GRN has no context parameters')
        q2 = self.L_hidden(F.elu(q, alpha=1.0))
        g = torch.sigmoid(self.L_gate(q2)) * self.L_value(q2)
        return self.norm(self.skip(a) + g)


class AuxiliaryPatchSelection(nn.Module):
    """Historical auxiliary content, target-latent-conditioned variable selection."""
    def __init__(self, L, C, D, N, patch_len, stride, target_idx, aux_idx):
        super().__init__()
        if D != 64 or not 0 <= target_idx < C or list(aux_idx) != [j for j in range(C) if j != target_idx]:
            raise ValueError('fixed D/target/ordered auxiliary schema')
        if len(aux_idx) < 2 or (L, patch_len, stride, N) not in ((12, 1, 1, 12), (96, 8, 4, 24)):
            raise ValueError('registered history/patch profile required')
        self.L, self.C, self.D, self.N = L, C, D, N
        self.patch_len, self.stride = patch_len, stride
        self.target, self.aux = target_idx, tuple(aux_idx)
        self.embeddings = nn.ModuleList([linear(patch_len, 8) for _ in self.aux])
        self.content_grns = nn.ModuleList([GRN(8) for _ in self.aux])
        self.selector = GRN(len(self.aux)*8, context_dim=D, projected_skip=True)
        self.score = linear(8, len(self.aux))
        self.W_out = ZeroOutput(8, D)

    def patches(self, X):
        if X.ndim != 3 or tuple(X.shape[1:]) != (self.L, self.C) or X.dtype != torch.float32:
            raise ValueError('pre-RevIN history schema/float32 required')
        if not torch.isfinite(X).all():
            raise ValueError('nonfinite auxiliary input')
        # Ordered aux only: target never enters embeddings or content GRNs.
        x = X[:, :, self.aux].transpose(1, 2)
        if self.patch_len != self.stride:
            x = F.pad(x, (0, self.patch_len-self.stride), mode='replicate')
        p = x.unfold(-1, self.patch_len, self.stride).permute(0, 2, 1, 3)
        if tuple(p.shape[1:]) != (self.N, len(self.aux), self.patch_len):
            raise ValueError('native stem patch alignment')
        return p

    def content(self, X):
        patches = self.patches(X)
        e = torch.stack([layer(patches[:, :, j]) for j, layer in enumerate(self.embeddings)], dim=2)
        z = torch.stack([layer(e[:, :, j]) for j, layer in enumerate(self.content_grns)], dim=2)
        return e, z

    def select(self, e, z, context):
        # B,N,J,8 flattened aux-major / embedding-minor. LN remains eight-wide.
        t = self.selector(e.flatten(-2), context)
        alpha = torch.softmax(self.score(t), dim=-1)
        m = (alpha.unsqueeze(-1) * z).sum(dim=2)
        return alpha, m

    def forward(self, H, X, diagnostics=None):
        if H.ndim != 4 or tuple(H.shape[1:]) != (self.C, self.D, self.N) or H.shape[0] != X.shape[0] or H.dtype != torch.float32:
            raise ValueError('U carrier axes/dtype')
        if not torch.isfinite(H).all():
            raise ValueError('nonfinite U carrier')
        e, z = self.content(X)
        context = H[:, self.target].transpose(1, 2)
        alpha, m = self.select(e, z, context)
        delta = self.W_out(m).transpose(1, 2)
        # No in-place write: other native variable slots stay byte-identical.
        updated = (H[:, self.target] + delta).unsqueeze(1)
        out = torch.cat((H[:, :self.target], updated, H[:, self.target+1:]), dim=1)
        if diagnostics is not None:
            diagnostics['U_auxiliary_content'] = {
                'axes': 'alpha B,N,J; content B,N,J,8; context B,N,D; delta B,D,N',
                'target_idx': self.target, 'ordered_aux_idx': list(self.aux),
                'alpha': stats(alpha), 'content': stats(z), 'context': stats(context),
                'weighted_content': stats(m), 'actual_target_delta': stats(delta),
                'interpretation': 'selection weights are not causal importance',
            }
        return out
