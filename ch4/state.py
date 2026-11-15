"""Exact epoch/optimizer/RNG recovery. No automatic technical retry."""
import math
import os
import random
from pathlib import Path
import numpy as np
import torch


def update_best(state,mse,epoch,patience):
    if not math.isfinite(mse):raise ValueError('nonfinite full-validation MSE')
    improved=state.get('best_mse') is None or mse<state['best_mse']
    if improved:state.update(best_mse=mse,best_epoch=epoch,stale=0)
    else:state['stale']+=1
    return improved,state['stale']>=patience


def scheduler_state(scheduler):
    value=scheduler.state_dict()
    # torch 2.0 stores a bound Python method. Bind its algorithm identity,
    # never serialize another scheduler/optimizer instance through that method.
    method=value.pop('anneal_func',None)
    if method is not None:value['_anneal_func_name']=method.__name__
    return value


def capture(model,optimizer,scheduler,generator,state,binding,device):
    return dict(binding=binding,model=model.state_dict(),optimizer=optimizer.state_dict(),
                scheduler=scheduler_state(scheduler),progress=dict(state),python_rng=random.getstate(),
                numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                data_rng=generator.get_state(),cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)


def restore(checkpoint,model,optimizer,scheduler,generator,binding,device):
    if checkpoint['binding']!=binding:raise ValueError('checkpoint task/config/data/source binding changed')
    model.load_state_dict(checkpoint['model'],strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    schedule=dict(checkpoint['scheduler']);name=schedule.pop('_anneal_func_name',None)
    if name is not None and scheduler.anneal_func.__name__!=name:raise ValueError('annealing algorithm mismatch')
    scheduler.load_state_dict(schedule)
    random.setstate(checkpoint['python_rng']);np.random.set_state(checkpoint['numpy_rng'])
    torch.set_rng_state(checkpoint['torch_rng']);generator.set_state(checkpoint['data_rng'])
    if device.type=='cuda':torch.cuda.set_rng_state(checkpoint['cuda_rng'],device)
    state=dict(checkpoint['progress'])
    if scheduler.last_epoch!=state['updates']:raise ValueError('scheduler/completed update mismatch')
    return state


def save_checkpoint(path,payload):
    path=Path(path);tmp=path.with_name(path.name+'.staging')
    if tmp.exists():raise FileExistsError('preserve incomplete checkpoint staging')
    with tmp.open('xb') as stream:
        torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)
