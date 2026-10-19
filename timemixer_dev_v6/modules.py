"""v6 r2: leaf router core and serial nonlinear depthwise temporal operator.

Source extraction scope is narrowed by the user; no complete TSA/ModernTCN Block.
"""
import torch
from torch import nn
from .author_components import DSW_embedding, AttentionLayer, conv_bn


def receiver_indices(channels, task, target_idx, aux_idx):
    if type(channels)is not int or channels<1 or task not in ('MS','M'):
        raise ValueError('explicit positive C and MS/M required')
    if type(target_idx)is not int or not 0<=target_idx<channels:
        raise ValueError('explicit valid target index required')
    if len(set(aux_idx))!=len(aux_idx)or any(type(j)is not int or not 0<=j<channels or j==target_idx for j in aux_idx):
        raise ValueError('ordered unique auxiliary indices excluding target required')
    return (target_idx,)if task=='MS'else tuple(range(channels))


class SegmentedVariableInteraction(nn.Module):
    """DSW adapted to target-conditioned router messages; no parent TSA block."""
    patch_len=stride=3
    embedding_dim,heads,head_dim,router_count=32,4,8,2
    def __init__(self,length,d_model,channels,task,target_idx,aux_idx):
        super().__init__()
        if type(length)is not int or length<1 or type(d_model)is not int or d_model<1:raise ValueError('positive L and D required')
        self.length,self.d_model,self.channels,self.task=length,d_model,channels,task
        self.receivers=receiver_indices(channels,task,target_idx,aux_idx)
        self.sources=tuple(tuple(aux_idx)if task=='MS'else tuple(j for j in range(channels)if j!=r)for r in self.receivers)
        self.segment_count=(length+2)//3
        self.enc_value_embedding=DSW_embedding(3,32)
        self.enc_pos_embedding=nn.Parameter(torch.randn(1,channels,self.segment_count,32))
        self.pre_norm=nn.LayerNorm(32,eps=1e-5)
        self.router=nn.Parameter(torch.randn(self.segment_count,2,32))
        self.dim_sender=AttentionLayer(32,4,dropout=0.)
        self.dim_receiver=AttentionLayer(32,4,dropout=0.)
        self.bridge=nn.Linear(32,3*d_model,bias=False);nn.init.zeros_(self.bridge.weight)
    def forward(self,x,*,return_diagnostics=False):
        if x.ndim!=3 or tuple(x.shape[1:])!=(self.length,self.channels):raise ValueError('U raw history L/C shape mismatch')
        b,L,C=x.shape;pad=(-L)%3;N=self.segment_count
        padded=torch.cat([x[:,:1,:].expand(-1,pad,-1),x],dim=1)if pad else x
        embedded=self.pre_norm(self.enc_value_embedding(padded)+self.enc_pos_embedding)
        E=embedded.permute(0,2,1,3)
        updates=[];messages=[];buffers=[];queries=[]
        for r,A in zip(self.receivers,self.sources):
            if not A:
                # No attention on an empty set and no output bias for a missing proposal.
                update=x.new_zeros((b,L,self.d_model))
                message=x.new_zeros((b,N,32));buffer=x.new_zeros((b,N,2,32));conditioned=self.router[None]+E[:,:,r,None,:]
            else:
                target=E[:,:,r,:]
                conditioned=self.router[None]+target[:,:,None,:]
                legal=E[:,:,A,:].reshape(b*N,len(A),32)
                buffer=self.dim_sender(conditioned.reshape(b*N,2,32),legal,legal).reshape(b,N,2,32)
                message=self.dim_receiver(target.reshape(b*N,1,32),buffer.reshape(b*N,2,32),buffer.reshape(b*N,2,32)).reshape(b,N,32)
                update=self.bridge(message).reshape(b,N,3,self.d_model).reshape(b,3*N,self.d_model)[:,pad:,:]
            updates.append(update)
            if return_diagnostics:messages.append(message);buffers.append(buffer);queries.append(conditioned)
        update=torch.stack(updates,dim=1)
        if return_diagnostics:return update,dict(embedding=embedded,router_state=E,message=torch.stack(messages,dim=1),buffer=torch.stack(buffers,dim=2),conditioned_router=torch.stack(queries,dim=2),pad=pad,sources=self.sources,update=update)
        return update


class SerialTemporalOperator(nn.Module):
    """Nonlinear local-to-dilated operator; cannot use linear reparam merging."""
    def __init__(self,D):
        super().__init__()
        self.local=conv_bn(D,D,3,1,1,D,dilation=1,bias=False)
        self.activation=nn.GELU()
        self.expanded=conv_bn(D,D,3,1,3,D,dilation=3,bias=False)
    def forward(self,x):return self.expanded(self.activation(self.local(x)))


class LatentTemporalRefinement(nn.Module):
    """Only a serial depthwise temporal operator and skip; no parent Block."""
    def __init__(self,d_model):
        super().__init__()
        if type(d_model)is not int or d_model<1:raise ValueError('positive latent D required')
        self.d_model=d_model;self.operator=SerialTemporalOperator(d_model)
        nn.init.zeros_(self.operator.expanded.bn.weight)
        nn.init.zeros_(self.operator.expanded.bn.bias)
    def forward(self,g,*,return_diagnostics=False):
        if g.ndim!=3 or g.shape[-1]!=self.d_model or g.shape[1]<1:raise ValueError('V effective-batch/L/D shape mismatch')
        update=self.operator(g.transpose(1,2)).transpose(1,2)
        value=g+update
        return (value,dict(update=update))if return_diagnostics else value
