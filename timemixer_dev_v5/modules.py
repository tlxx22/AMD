"""Independent project adaptations: Crossformer DSW/variable axis and ModernTCN DW/PW roles.

Reference-only author code: Crossformer Apache-2.0, ModernTCN MIT. This is not
an implementation of complete TSA/router or full ModernTCN/ConvFFN2/BN.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def receiver_indices(channels, task, target_idx, aux_idx):
    if type(channels) is not int or channels < 1 or task not in ('MS','M'):
        raise ValueError('explicit positive C and MS/M required')
    if type(target_idx) is not int or not 0 <= target_idx < channels:
        raise ValueError('explicit valid target index required')
    if len(set(aux_idx)) != len(aux_idx) or any(type(j) is not int or not 0 <= j < channels or j == target_idx for j in aux_idx):
        raise ValueError('ordered unique auxiliary indices excluding target required')
    return (target_idx,) if task == 'MS' else tuple(range(channels))


def initialize(module):
    for layer in module.modules():
        if isinstance(layer, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None: nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.LayerNorm):
            nn.init.ones_(layer.weight); nn.init.zeros_(layer.bias)


class SegmentedVariableInteraction(nn.Module):
    """Same-patch receiver/source attention; restore L positions into latent D."""
    patch_len = stride = 3
    embedding_dim, heads, head_dim = 32, 4, 8
    eps = 1e-5

    def __init__(self, length, d_model, channels, task, target_idx, aux_idx):
        super().__init__()
        if type(length) is not int or length < 1 or type(d_model) is not int or d_model < 1:
            raise ValueError('positive L and D required')
        self.length, self.d_model, self.channels = length, d_model, channels
        self.task = task
        self.receivers = receiver_indices(channels, task, target_idx, aux_idx)
        self.sources = tuple(aux_idx) if task == 'MS' else tuple(range(channels))
        self.patch_embedding = nn.Linear(3,32)
        self.norm = nn.LayerNorm(32,eps=self.eps)
        self.statistics = nn.Linear(2,32)
        self.query, self.key, self.value = (nn.Linear(32,32) for _ in range(3))
        self.bridge = nn.Linear(32,3*d_model,bias=False)
        initialize(self); nn.init.zeros_(self.bridge.weight)

    def forward(self, x, *, return_diagnostics=False):
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.length,self.channels):
            raise ValueError('U raw history L/C shape mismatch')
        b,L,C = x.shape; pad = (-L)%3; N = (L+pad)//3; R = len(self.receivers)
        padded = torch.cat([x[:,:1,:].expand(-1,pad,-1),x],dim=1) if pad else x
        raw = padded.reshape(b,N,3,C).permute(0,1,3,2)
        mu = raw.mean(-1); centered = raw-mu[...,None]
        sigma = torch.sqrt(centered.square().mean(-1)+self.eps)
        z = centered/sigma[...,None]
        e = self.norm(self.patch_embedding(z)) + self.statistics(torch.stack([mu,torch.log1p(sigma)],-1))
        if not self.sources or (self.task == 'M' and C == 1):
            update = x.new_zeros((b,R,L,self.d_model))
            return (update,dict(raw_patch=raw,mu=mu,sigma=sigma,z=z,embedding=e,source_indices=self.sources,no_sources=True)) if return_diagnostics else update
        J = len(self.sources)
        q = self.query(e[:,:,self.receivers,:]).reshape(b,N,R,4,8).permute(0,1,3,2,4)
        k = self.key(e[:,:,self.sources,:]).reshape(b,N,J,4,8).permute(0,1,3,2,4)
        val = self.value(e[:,:,self.sources,:]).reshape(b,N,J,4,8).permute(0,1,3,2,4)
        allowed = x.new_ones((R,J),dtype=torch.bool)
        if self.task == 'M':allowed = ~torch.eye(C,dtype=torch.bool,device=x.device)
        score = torch.matmul(q,k.transpose(-1,-2))/math.sqrt(8)
        score = score.masked_fill(~allowed[None,None,None,:,:],-torch.inf)
        alpha = torch.softmax(torch.cat([score,score.new_zeros((*score.shape[:-1],1))],-1),dim=-1)
        context = torch.matmul(alpha[...,:-1],val).permute(0,1,3,2,4).contiguous().reshape(b,N,R,32)
        restored = self.bridge(context).reshape(b,N,R,3,self.d_model).permute(0,2,1,3,4).contiguous().reshape(b,R,3*N,self.d_model)
        update = restored[:,:,pad:,:]
        if return_diagnostics:
            return update,dict(raw_patch=raw,mu=mu,sigma=sigma,z=z,embedding=e,query=q,key=k,value=val,alpha=alpha,context=context,source_indices=self.sources,allowed=allowed,pad=pad,update=update)
        return update


class LatentTemporalRefinement(nn.Module):
    """D is the latent feature axis, never the business-variable C axis."""
    def __init__(self, d_model):
        super().__init__()
        if type(d_model) is not int or d_model < 1:raise ValueError('positive latent D required')
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model,eps=1e-5)
        self.dw7 = nn.Conv1d(d_model,d_model,7,groups=d_model,bias=False)
        self.dw3 = nn.Conv1d(d_model,d_model,3,groups=d_model,bias=False)
        self.expand = nn.Linear(d_model,2*d_model)
        self.project = nn.Linear(2*d_model,d_model)
        initialize(self);nn.init.zeros_(self.project.weight);nn.init.zeros_(self.project.bias)

    def forward(self, g, *, return_diagnostics=False):
        if g.ndim != 3 or g.shape[-1] != self.d_model or g.shape[1] < 1:
            raise ValueError('V effective-batch/L/D shape mismatch')
        a = self.norm(g).transpose(1,2)
        q = self.dw7(F.pad(a,(3,3),mode='replicate')) + self.dw3(F.pad(a,(1,1),mode='replicate'))
        delta = self.project(F.gelu(self.expand(q.transpose(1,2))))
        result = g+delta
        return (result,dict(normalized=a,temporal=q,update=delta)) if return_diagnostics else result
