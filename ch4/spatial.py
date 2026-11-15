"""Approved PDFormer-inspired supports and receiver-conditioned low-rank response.

Offsets align hidden patches, not physical propagation delays. No dense N*N logits.
"""
import math
import torch
from torch import nn


class RelationBranch(nn.Module):
    def __init__(self, offsets):
        super().__init__()
        self.offsets = tuple(offsets)
        self.q = nn.Linear(66, 64, bias=False)
        self.k = nn.Linear(66, 64, bias=False)
        self.v = nn.Linear(64, 64, bias=False)
        self.w0 = nn.Linear(64, 64, bias=False)
        self.left = nn.Linear(64, 8, bias=False)
        self.right = nn.Linear(8, 64, bias=False)
        self.condition = nn.Sequential(nn.Linear(82, 32), nn.GELU(), nn.Linear(32, 8), nn.Tanh())
        if 1 in offsets:
            self.offset_bias = nn.Parameter(torch.zeros(4))
        else:
            self.register_parameter('offset_bias', None)

    def projections(self, z, state):
        b, k, n, _ = z.shape
        zs = torch.cat((z, state[:, None].expand(-1, k, -1, -1)), -1)
        return (self.q(zs).reshape(b, k, n, 4, 16),
                self.k(zs).reshape(b, k, n, 4, 16),
                self.v(z).reshape(b, k, n, 4, 16))

    def forward(self, z, state, embedding, neighbors, available):
        b, k, n, _ = z.shape
        if neighbors.ndim != 2 or neighbors.shape[0] != n:
            raise ValueError('neighbors must be [N,S], padded with -1')
        if torch.any(neighbors < -1) or torch.any(neighbors >= n):
            raise ValueError('neighbor outside node order')
        q, key, value = self.projections(z, state)
        safe = neighbors.clamp_min(0)
        keys, values, masks, biases = [], [], [], []
        for offset in self.offsets:
            times = torch.arange(k, device=z.device) - offset
            keys.append(key[:, times.clamp_min(0)][:, :, safe])
            values.append(value[:, times.clamp_min(0)][:, :, safe])
            masks.append(available[:, times.clamp_min(0)][:, :, safe]
                         & (times >= 0)[None, :, None, None]
                         & (neighbors >= 0)[None, None]
                         & available[:, :, :, None])
            biases.append(self.offset_bias if offset else z.new_zeros(4))
        key = torch.stack(keys, dim=4)       # B,K,N,S,O,heads,16
        value = torch.stack(values, dim=4)
        legal = torch.stack(masks, dim=-1)   # B,K,N,S,O
        logits = torch.einsum('bknhf,bknsohf->bknhso', q, key) / math.sqrt(16)
        logits = logits + torch.stack(biases, -1)[None, None, None, :, None, :]
        legal = legal.unsqueeze(3).expand_as(logits)
        logits = logits.flatten(-2)
        legal = legal.flatten(-2)
        valid = legal.any(-1, keepdim=True)
        # Never evaluate softmax on an all -inf row.
        safe_logits = torch.where(valid, logits.masked_fill(~legal, -torch.inf), torch.zeros_like(logits))
        weights = torch.softmax(safe_logits, -1).masked_fill(~legal, 0)
        weights = weights.reshape(b, k, n, 4, neighbors.shape[1], len(self.offsets))
        message = torch.einsum('bknhso,bknsohf->bknhf', weights, value).reshape(b, k, n, 64)
        context = torch.cat((z, embedding[None, None].expand(b, k, -1, -1),
                             state[:, None].expand(-1, k, -1, -1)), -1)
        gamma = self.condition(context)
        response = self.w0(message) + self.right(self.left(message) * gamma)
        return response, valid[..., 0].any(-1)


class SpatialResponse(nn.Module):
    def __init__(self, nodes, seed=2024):
        super().__init__()
        # CPU-only local RNG: constructing spatial parameters leaves temporal RNG unchanged.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 37001)
            self.project = nn.Linear(64, 64, bias=False)
            self.embedding = nn.Embedding(nodes, 16)
            self.geo = RelationBranch((0, 1))
            self.sem = RelationBranch((0,))
            self.output = nn.Linear(128, 64, bias=False)
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            nn.init.normal_(self.embedding.weight, 0, .02)
            nn.init.eye_(self.geo.w0.weight)
            nn.init.eye_(self.sem.w0.weight)
            nn.init.zeros_(self.output.weight)

    def forward(self, h, state, geo, sem, available, node_indices=None):
        if not torch.isfinite(h).all() or not torch.isfinite(state).all():
            raise ValueError('filled history/hidden state must be finite; missingness uses independent masks')
        n = h.shape[2]
        if node_indices is None:
            node_indices = torch.arange(n, device=h.device)
        embedding = self.embedding(node_indices)
        z = self.project(h)
        tg, vg = self.geo(z, state, embedding, geo, available)
        ts, vs = self.sem(z, state, embedding, sem, available)
        valid = vg | vs
        delta = self.output(torch.cat((tg, ts), -1))
        return delta.masked_fill(~valid[..., None], 0), valid
