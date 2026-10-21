"""Frozen U/V formulas. No normalization, dropout or learnable outer gates added."""
import math
import torch
from torch import nn
from torch.nn import functional as F

def stats(x):
    t=x.detach();return {'rms':float(t.float().square().mean().sqrt()),'max_abs':float(t.abs().max())}

class WindowStateModulation(nn.Module):
    def __init__(self,C,D,target,aux):
        super().__init__();self.C=C;self.D=D;self.target=target;self.aux=tuple(aux);r=8
        self.A=nn.Parameter(torch.empty(C,r,2));self.e=nn.Parameter(torch.zeros(C,r));self.b=nn.Parameter(torch.zeros(len(aux)))
        self.Wh=nn.Linear(D,r,bias=False);self.Wo=nn.Linear(r,D,bias=False);self.Wb=nn.Linear(r,D,bias=False)
        for a in self.A:nn.init.xavier_uniform_(a)
        nn.init.xavier_uniform_(self.Wh.weight);nn.init.zeros_(self.Wo.weight);nn.init.zeros_(self.Wb.weight)
    def controller(self,X):
        X=X.float()
        if not torch.isfinite(X).all():raise ValueError('nonfinite U input')
        mu=X.mean(1);var=(X-mu[:,None]).square().mean(1);s=torch.stack((mu,torch.log1p(var)),-1)
        if not torch.isfinite(s).all():raise ValueError('nonfinite U statistics')
        h=torch.tanh(torch.einsum('crs,bcs->bcr',self.A,s)+self.e)
        if self.aux:
            hj=h[:,self.aux,:];g=torch.sigmoid((h[:,self.target,None,:]*hj).sum(-1)/math.sqrt(8)+self.b)
            q=(g[:,:,None]*hj).sum(1)/(1+g.sum(1,keepdim=True))
        else:g=h.new_empty((X.shape[0],0));q=h.new_zeros((X.shape[0],8))
        return q,g
    def forward(self,H,X,diagnostics=None):
        q,g=self.controller(X);hy=H[:,self.target].transpose(1,2)
        multiplicative=self.Wo(q[:,None]*self.Wh(hy));state=self.Wb(q)[:,None];delta=multiplicative+state
        out=H.clone();out[:,self.target]=H[:,self.target]+delta.transpose(1,2)
        if diagnostics is not None:
            diagnostics['U']={'q':stats(q),'gate_mean':float(g.detach().mean()) if g.numel() else None,'multiplicative':stats(multiplicative),'state':stats(state),'actual_delta':stats(delta)}
        return out

class ConditionalTemporalFilter(nn.Module):
    def __init__(self,L,D,N,large):
        super().__init__();self.D=D;self.N=N;self.Kv=min(large,2*N-1);self.R=4
        self.W1=nn.Linear(L,16,bias=False);self.W2=nn.Linear(16,4,bias=False)
        self.A=nn.Parameter(torch.zeros(D,4));self.T=nn.Parameter(torch.empty(4,self.Kv))
        nn.init.xavier_uniform_(self.W1.weight);nn.init.xavier_uniform_(self.W2.weight);nn.init.orthogonal_(self.T)
    def route(self,Z):return torch.tanh(self.W2(F.gelu(self.W1(Z),approximate='none')))
    def forward(self,H,Z,diagnostics=None):
        B,C,D,N=H.shape
        if (D,N)!=(self.D,self.N):raise ValueError('fixed V profile shape')
        a=self.route(Z)
        bases=F.conv1d(H.reshape(B*C*D,1,N),self.T[:,None,:],padding=self.Kv//2).reshape(B,C,D,4,N)
        delta=(bases*a[:,:,None,:,None]*self.A[None,None,:,:,None]).sum(3)
        if diagnostics is not None:
            kernel=torch.einsum('bcr,dr,rk->bcdk',a,self.A,self.T)
            diagnostics['V']={'a':stats(a),'delta_kernel':stats(kernel),'actual_delta':stats(delta)}
        return delta
