"""User-frozen conditional native variable operator; independent implementation.
Design references: CondConv 1904.04971v3 / LoRA 2106.09685v2.
Not author-code port or frozen pretrained LoRA; no TFT execution dependency.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from moderntcn_dev_v1.modules import stats
from moderntcn_dev_v3.modules import ConditionalTemporalFilter

class ConditionalVariableOperator(nn.Module):
    def __init__(self,L,C,D,target,aux):
        super().__init__()
        if not 0<=target<C or len(set(aux))!=len(aux) or any(j==target or not 0<=j<C for j in aux):
            raise ValueError('target/ordered auxiliary schema')
        self.L,self.C,self.D,self.target,self.aux=L,C,D,target,tuple(aux)
        self.A=nn.Parameter(torch.empty(C,8,2))
        self.e=nn.Parameter(torch.zeros(C,8))
        self.b=nn.Parameter(torch.zeros(len(aux)))
        self.P_weight=nn.Parameter(torch.empty(D*8,C,1))
        self.Q_weight=nn.Parameter(torch.zeros(D*C,8,1))
        # No random constructors or legacy Wh/Wo/Wb: exact prescribed RNG order.
        with torch.no_grad():
            for a in self.A:nn.init.xavier_uniform_(a)
            for p in self.P_weight.view(D,8,C,1):nn.init.xavier_uniform_(p[:,:,0])

    def controller(self,X):
        if X.ndim!=3 or tuple(X.shape[1:])!=(self.L,self.C):raise ValueError('U history schema')
        X=X.float()
        if not torch.isfinite(X).all():raise ValueError('nonfinite U input')
        mu=X.mean(1);var=(X-mu[:,None]).square().mean(1);s=torch.stack((mu,torch.log1p(var)),-1)
        if not torch.isfinite(s).all():raise ValueError('nonfinite U statistics')
        h=torch.tanh(torch.einsum('crs,bcs->bcr',self.A,s)+self.e)
        if self.aux:
            hj=h[:,self.aux,:];g=torch.sigmoid((h[:,self.target,None,:]*hj).sum(-1)/math.sqrt(8)+self.b)
            q=(g[:,:,None]*hj).sum(1)/(1+g.sum(1,keepdim=True))
        else:g=h.new_empty((X.shape[0],0));q=h.new_zeros((X.shape[0],8))
        if not torch.isfinite(q).all() or not torch.isfinite(g).all():raise ValueError('nonfinite U controller output')
        return q,g

    def forward(self,x_flat,X,diagnostics=None):
        if x_flat.ndim!=3 or x_flat.shape[1]!=self.D*self.C or x_flat.shape[0]!=X.shape[0]:raise ValueError('U grouped carrier schema')
        if not torch.isfinite(x_flat).all():raise ValueError('nonfinite U carrier')
        B,_,N=x_flat.shape;q,g=self.controller(X)
        t=F.conv1d(x_flat,self.P_weight,groups=self.D).reshape(B,self.D,8,N)
        tq=t*q[:,None,:,None]
        delta=F.conv1d(tq.reshape(B,self.D*8,N),self.Q_weight,groups=self.D)
        if not torch.isfinite(delta).all():raise ValueError('nonfinite U operator output')
        if diagnostics is not None:
            diagnostics['U_variable_operator']={'q':stats(q),'gate_mean':float(g.detach().mean()) if g.numel() else None,'factor_activation':stats(t),'actual_pre_activation_delta':stats(delta),'axes':'carrier B,D*C,N; q B,8; per-D operator C->8->C','rank_bound':min(8,self.C)}
        return delta
