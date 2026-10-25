"""Reviewed numerical-preserving guard integrated in v5 only; original E4 prototype retained."""
import torch
class GuardError(FloatingPointError):
 def __init__(self,boundary,name):super().__init__(f'{boundary}: {name}');self.boundary=boundary;self.name=name

def loss_guard(loss):
 if not bool(torch.isfinite(loss).all()):raise GuardError('before_backward','loss')

def gradients_original(named):
 active=False
 for name,p in named:
  if p.grad is not None:
   if not bool(torch.isfinite(p.grad).all()):raise GuardError('before_Adam',name)
   active |= bool(torch.count_nonzero(p.grad))
 if not active:raise GuardError('before_Adam','no active loss gradient')

def gradients_collected(named):
 entries=[(n,p.grad) for n,p in named if p.grad is not None]
 # Same independent tensor predicates, same order. No norm/flatten/concatenated parameters.
 flags=[torch.stack((torch.isfinite(g).all(),torch.count_nonzero(g)!=0)) for n,g in entries]
 host=torch.stack(flags).detach().cpu().tolist() if flags else []
 active=False
 for (name,g),(finite,nonzero) in zip(entries,host):
  if not finite:raise GuardError('before_Adam',name)
  active |= nonzero
 if not active:raise GuardError('before_Adam','no active loss gradient')

def parameters_original(named):
 for name,p in named:
  if not bool(torch.isfinite(p).all()):raise GuardError('after_Adam_before_scheduler',name)

def parameters_collected(named):
 entries=list(named);flags=[torch.isfinite(p).all() for n,p in entries]
 host=torch.stack(flags).detach().cpu().tolist() if flags else []
 for (name,p),finite in zip(entries,host):
  if not finite:raise GuardError('after_Adam_before_scheduler',name)
