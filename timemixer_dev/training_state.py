"""Epoch-boundary checkpoints with complete validation before live state writes."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from utils.ch3_contract import BestState
from utils.ch3_onecycle import Schedule

SCHEMA='timemixer-uv-training-state-v2'

def parameter_names(model,opt):
    names={id(p):name for name,p in model.named_parameters()}
    return [[names[id(p)] for p in group['params']] for group in opt.param_groups]

def rng_devices(model,cuda_states,identity):
    device=str(next(model.parameters()).device)
    execution=identity['execution_device']
    if execution['device']!=device:raise ValueError('checkpoint training device binding')
    devices=[]
    if device.startswith('cuda:'):
        if not torch.cuda.is_initialized():raise ValueError('CUDA training RNG not initialized')
        if device!='cuda:0' or len(cuda_states)!=1 or os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or not isinstance(execution['gpu_uuid'],str) or not execution['gpu_uuid'].startswith('GPU-'):raise ValueError('single visible approved GPU UUID binding required')
        for index,value in enumerate(cuda_states):
            props=torch.cuda.get_device_properties(index)
            devices.append(dict(index=index,name=props.name,uuid=execution['gpu_uuid'],total_memory=props.total_memory,
                                capability=[props.major,props.minor],state_bytes=value.numel()))
    elif execution['gpu_uuid'] is not None:raise ValueError('CPU training cannot carry a GPU UUID')
    return dict(training_device=device,gpu_uuid=execution['gpu_uuid'],cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),cuda_devices=devices)

def validate_cuda_rng(metadata,states,expected):
    """Pure metadata/tensor validation; CPU tests do not initialize CUDA."""
    if metadata!=expected or not isinstance(states,list):raise ValueError('CUDA RNG device metadata identity')
    devices=expected['cuda_devices'];device=expected['training_device']
    if device.startswith('cuda:'):
        index=int(device.split(':')[1])
        if not devices or index>=len(devices) or [d['index'] for d in devices]!=list(range(len(devices))) or not isinstance(expected.get('gpu_uuid'),str) or not expected['gpu_uuid'].startswith('GPU-') or devices[index]['uuid']!=expected['gpu_uuid']:raise ValueError('CUDA RNG training device missing')
    elif device!='cpu' or devices:raise ValueError('unsupported RNG training device')
    if len(states)!=len(devices):raise ValueError('CUDA RNG device state missing')
    for value,description in zip(states,devices):
        length=description['state_bytes']
        if type(length)!=int or length<=0 or not isinstance(value,torch.Tensor) or value.device.type!='cpu' or value.dtype!=torch.uint8 or value.ndim!=1 or value.numel()!=length or not value.is_contiguous():raise ValueError('CUDA RNG tensor format/shape/length')

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def atomic_json(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with tmp.open('rb') as f:os.fsync(f.fileno())
    os.replace(tmp,path)

def snapshot(model,opt,scheduler,best,epoch,steps,generator,identity,*,journals):
    from ch3_runner import cpu_tree
    if scheduler.updates!=steps:raise ValueError('scheduler/update accounting')
    saved_opt=cpu_tree(opt.state_dict());names=parameter_names(model,opt)
    id_names={key:name for group,group_names in zip(saved_opt['param_groups'],names) for key,name in zip(group['params'],group_names)}
    activity={id_names[key]:int(value['step'].item()) for key,value in saved_opt['state'].items()}
    cuda_states=torch.cuda.get_rng_state_all() if next(model.parameters()).device.type=='cuda' else []
    devices=rng_devices(model,cuda_states,identity);validate_cuda_rng(devices,cuda_states,devices)
    return dict(schema=SCHEMA,identity=copy.deepcopy(identity),boundary='after_epoch_validation_checkpoint',
        model=cpu_tree(model.state_dict()),optimizer=saved_opt,scheduler=scheduler.state_dict(),parameter_names=names,adam_activity=activity,rng_devices=devices,
        best=copy.deepcopy(vars(best)),epoch=epoch,steps=steps,journals=copy.deepcopy(journals),
        rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                 cuda=cuda_states,generator=generator.get_state()))

def save(path,state):
    path=Path(path);tmp=path.with_name(path.name+'.tmp');torch.save(state,tmp)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    os.replace(tmp,path);atomic_json(str(path)+'.sha256',{'sha256':file_sha(path),'schema':SCHEMA})

def read(path):
    path=Path(path);expected=json.loads(Path(str(path)+'.sha256').read_text())
    if expected!={'sha256':file_sha(path),'schema':SCHEMA}:raise ValueError('checkpoint checksum/schema mismatch')
    return torch.load(path,map_location='cpu')

def static_groups(groups):
    result=copy.deepcopy(groups)
    for g in result:
        g['lr']='OneCycle';g['betas']=['OneCycle',g['betas'][1]]
    return result

def validate(state,model,opt,scheduler,identity):
    if set(state)!={'schema','identity','boundary','model','optimizer','scheduler','best','epoch','steps','rng','parameter_names','journals','adam_activity','rng_devices'} or state['parameter_names']!=parameter_names(model,opt):raise ValueError('checkpoint top-level/parameter-name schema')
    if set(state['journals'])!={'history_sha256','updates_sha256'} or any(not isinstance(v,str) or len(v)!=64 for v in state['journals'].values()):raise ValueError('checkpoint journal schema')
    if state.get('schema')!=SCHEMA or state.get('identity')!=identity or state.get('boundary')!='after_epoch_validation_checkpoint':
        raise ValueError('checkpoint task/arm/F/H/protocol/data/source/schema identity mismatch')
    expected=model.state_dict();saved=state['model']
    if set(saved)!=set(expected):raise ValueError('model keys mismatch')
    for k,v in expected.items():
        s=saved[k]
        if not isinstance(s,torch.Tensor) or s.shape!=v.shape or s.dtype!=v.dtype or not torch.isfinite(s).all():raise ValueError('model tensor schema/finite: '+k)
    epoch,steps=state['epoch'],state['steps'];tr=identity['profile']['training']
    if type(epoch)!=int or not 0<=epoch<=tr['epochs'] or type(steps)!=int or steps!=epoch*tr['scheduler']['steps_per_epoch']:raise ValueError('unsupported non-epoch boundary')
    best=state['best']
    if set(best)!={'best','epoch','bad','patience'} or best['patience']!=tr['patience'] or type(best['bad'])!=int or best['bad']<0:raise ValueError('BestState schema')
    if epoch<1 or type(best['epoch'])!=int or not 1<=best['epoch']<=epoch or not np.isfinite(best['best']):raise ValueError('best progress')
    current=opt.state_dict();saved_opt=state['optimizer']
    if set(saved_opt)!={'state','param_groups'} or static_groups(saved_opt['param_groups'])!=static_groups(current['param_groups']):raise ValueError('optimizer group identity/order')
    ids=[i for g in current['param_groups'] for i in g['params']]
    parameters=[p for g in opt.param_groups for p in g['params']]
    if not set(saved_opt['state'])<=set(ids):raise ValueError('foreign optimizer state')
    id_names=dict(zip(ids,[name for group in state['parameter_names'] for name in group]));activity=state['adam_activity']
    if not isinstance(activity,dict) or not activity or set(activity)!={id_names[key] for key in saved_opt['state']}:raise ValueError('missing active Adam state/identity')
    for key,value in saved_opt['state'].items():
        p=parameters[ids.index(key)]
        if set(value)!={'step','exp_avg','exp_avg_sq'}:raise ValueError('Adam state schema')
        for name,v in value.items():
            if not isinstance(v,torch.Tensor) or not torch.isfinite(v).all():raise ValueError('Adam finite tensor')
            if name=='step':
                recorded=activity[id_names[key]]
                if type(recorded)!=int or v.shape!=torch.Size([]) or v.dtype!=torch.float32 or not 1<=v.item()<=steps or v.item()!=int(v.item()) or v.item()!=recorded:raise ValueError('Adam active step accounting')
            elif v.shape!=p.shape or v.dtype!=p.dtype or (name=='exp_avg_sq' and bool((v<0).any())):raise ValueError('Adam tensor shape/dtype/variance')
    # Validate scheduler and optimizer on private dummy parameters, not live state.
    groups=[{'params':[torch.nn.Parameter(torch.empty_like(p,device='cpu')) for p in g['params']]} for g in opt.param_groups]
    shadow=torch.optim.Adam(groups,lr=tr['lr'],betas=tuple(tr['betas']),eps=tr['eps'],weight_decay=tr['weight_decay'])
    shadow_scheduler=Schedule(shadow,tr['scheduler']);shadow.load_state_dict(copy.deepcopy(saved_opt));shadow_scheduler.load_state_dict(copy.deepcopy(state['scheduler']),steps)
    rng=state['rng']
    if set(rng)!={'python','numpy','torch','cuda','generator'}:raise ValueError('RNG schema')
    random.Random().setstate(rng['python']);np.random.RandomState().set_state(rng['numpy'])
    for name in ('torch','generator'):
        v=rng[name]
        if not isinstance(v,torch.Tensor) or v.device.type!='cpu' or v.dtype!=torch.uint8 or v.ndim!=1 or not v.is_contiguous() or v.numel()!=torch.Generator().get_state().numel():raise ValueError('RNG tensor dtype/shape/length')
        torch.Generator().set_state(v)
    current_cuda=torch.cuda.get_rng_state_all() if next(model.parameters()).device.type=='cuda' else []
    validate_cuda_rng(state['rng_devices'],rng['cuda'],rng_devices(model,current_cuda,identity))
    return state

def restore(path,model,opt,scheduler,generator,identity):
    state=validate(read(path),model,opt,scheduler,identity)
    model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'],state['steps'])
    best=BestState();best.__dict__.update(state['best'])
    random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy']);torch.set_rng_state(state['rng']['torch'])
    if state['rng']['cuda']:torch.cuda.set_rng_state_all(state['rng']['cuda'])
    generator.set_state(state['rng']['generator'])
    return best,state['epoch'],state['steps']
