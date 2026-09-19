"""Shared formal runner, fixed-wave scheduler and six-step resource worker.

No business import on module import. Formal start requires external approval,
clean reviewed closure, data bindings and a reviewed probe decision.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from utils.ch3_contract import (ROOT, PROFILE_FILE, digest, read_profiles, validate_manifest,
                               profile, task_by_id, BestState, summarize)


def dump(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()


def code_binding():
    files=['ch3_runner.py','utils/ch3_contract.py','utils/ch3_data.py','models/ch3_adapter.py',
           'models/tsAMD.py','models/tsAMD_enhanced.py','models/common.py','models/tsmoe.py',
           'utils/dataloader_urbanev.py','utils/feature_schema.py']
    files += ['tests/test_ch3_formal.py','scripts/ch3/start_model.sh','scripts/ch3/start_probe.sh',
              'tools/restricted_regression/m5_formal_entry.py','tools/restricted_regression/restricted_io_guard.py',
              'tools/restricted_regression/sitecustomize.py','tools/restricted_regression/bundle.sha256']
    files += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'models/modules').glob('*.py'))]
    return {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}


def environment_binding():
    from importlib.metadata import version
    return dict(python=sys.version, executable=sys.executable,
                packages={k:version(k) for k in ('torch','numpy','pandas','scipy','scikit-learn','einops')})


def hardware_binding():
    return dict(gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,driver_version','--format=csv,noheader,nounits'],text=True).strip(),
                cpu_affinity=sorted(os.sched_getaffinity(0)),threads=4)


def validate_probe_report(c, report):
    if report.get('protocol_sha')!=digest(c) or report.get('Q')!=sum(g['q'] for g in c['groups']):
        raise ValueError('probe configuration/worker coverage mismatch')
    if set(report.get('decisions',{}))!={g['id'] for g in c['groups']}:
        raise ValueError('probe group coverage mismatch')
    for group in c['groups']:
        decision=report['decisions'][group['id']]
        if decision.get('status')=='Passed':
            verified_waves(group,decision)
            from utils.ch3_contract import numeric_probe_policy
            rule=numeric_probe_policy(c,task_by_id(c,group['representatives'][0]))
            if rule and decision['concurrency']>1:
                checked=decision.get(str(decision['concurrency']),{}).get('numerical_comparisons',[])
                if len(checked)!=group['q']:raise ValueError('numeric admission coverage missing')
                import math
                for row in checked:
                    if (row.get('passed') is not True or row.get('mode')!='bounded_numeric'
                            or row.get('policy_sha')!=digest(rule) or row.get('atol')!=rule['atol']
                            or row.get('rtol')!=0 or row.get('exact_residual_state') is not True):
                        raise ValueError('numeric admission policy mismatch')
                    maxima=list(row.get('tensor_max_abs',{}).values())+[row.get('scalar_normalized_max_abs')]
                    if len(maxima)!=5 or any(v is None or not math.isfinite(v) or v<0 or v>rule['atol'] for v in maxima):
                        raise ValueError('numeric admission exceeds bound')
    return report


def validate_model_completion(c, model, report):
    expected=[t['id'] for t in c['tasks'] if t['model']==model]
    if (not expected or report.get('protocol_sha')!=digest(c) or report.get('model')!=model
            or report.get('task_ids')!=expected):raise ValueError('foreign/incomplete model completion')
    groups={g['id']:g for g in c['groups'] if g['model']==model}
    if set(report.get('results',{}))!=set(groups):raise ValueError('model summary coverage mismatch')
    for key,value in report['results'].items():
        if value.get('input_variant')!=groups[key]['input_variant']:raise ValueError('model input identity mismatch')
    return report


def preflight(c,model,approval=None,probe=False):
    validate_manifest(c)
    if model is not None and model not in {t['model'] for t in c['tasks']}:raise ValueError('model group not registered')
    reasons=[]
    from utils.ch3_contract import training_blockers
    for task in c['tasks']:
        if model is None or task['model']==model:
            reasons.extend(training_blockers(c,task))
    reasons=list(dict.fromkeys(reasons))
    from utils.ch3_data import verify_source_state
    for name,d in c['datasets'].items():
        try:verify_source_state(d)
        except (OSError,ValueError) as exc:reasons.append(name+': '+str(exc))
        if probe:
            reasons.extend(name+': '+reason for reason in d['mandatory_blockers'])
            if name!='UrbanEV' and d['endpoints'] is None:reasons.append(name+': probe tail endpoints missing')
    if probe:
        reasons.extend(c['execution']['probe'].get('mandatory_blockers',[]))
        from utils.ch3_contract import followup_limits
        try:followup_limits(c,approval)
        except (ValueError,KeyError,TypeError) as exc:reasons.append('follow-up: '+str(exc))
    if git('status','--porcelain','--untracked-files=all'):reasons.append('reviewed clean closure required')
    if approval is None:reasons.append('explicit review/closure authorization missing')
    else:
        if approval.get('commit')!=git('rev-parse','HEAD'):reasons.append('approval commit mismatch')
        if approval.get('protocol_sha')!=digest(c):reasons.append('approval configuration mismatch')
        if approval.get('code')!=code_binding():reasons.append('approval source mismatch')
        if approval.get('environment')!=environment_binding():reasons.append('approval environment mismatch')
        if approval.get('hardware')!=hardware_binding():reasons.append('approval hardware/CPU allocation mismatch')
        if probe:
            if approval.get('purpose')!='ch3_resource_probe' or approval.get('reviewed') is not True:reasons.append('probe review not approved')
        else:
            if approval.get('purpose')!='ch3_formal' or approval.get('structure_frozen') is not True or approval.get('m6_authorized') is not True:
                reasons.append('structure freeze and M6 authorization required')
            for domain in {t['dataset'] for t in c['tasks'] if t['model']==model}:
                unresolved=c['datasets'][domain]['mandatory_blockers']
                if unresolved:reasons.append(domain+': '+','.join(unresolved))
                if domain not in approval.get('data_bindings',{}):reasons.append(domain+': reviewed data prefix binding missing')
            if not approval.get('probe_report_sha'):reasons.append('reviewed resource decisions missing')
    for src in c['sources'].values():
        if subprocess.check_output(['git','-C',src['repository'],'rev-parse','HEAD'],text=True).strip()!=src['commit']:
            reasons.append('author commit changed: '+src['repository'])
        for p,h in src['files'].items():
            if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:reasons.append('author bytes changed: '+p)
    return reasons


class GPULock:
    def __init__(self,c):self.path=Path(c['execution']['evidence'])/'project-gpu0.lock';self.handle=None
    def __enter__(self):
        self.handle=self.path.open('a+')
        try:fcntl.flock(self.handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:self.handle.close();raise RuntimeError('another project model/probe owns GPU0')
        return self
    def __exit__(self,*args):fcntl.flock(self.handle,fcntl.LOCK_UN);self.handle.close()


def tensor_digest(value):
    import torch
    h=hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):
            a=v.detach().cpu().contiguous();h.update(str((a.dtype,tuple(a.shape))).encode())
            if a.numel():h.update(memoryview(a.numpy()).cast('B'))
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(str(k).encode());add(v[k])
        elif isinstance(v,(list,tuple)):
            for x in v:add(x)
        else:h.update(repr(v).encode())
    add(value);return h.hexdigest()


def cpu_tree(value):
    import torch
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def save_state(path,model,opt,identity,best,epoch,steps,generator):
    import torch,random,numpy as np
    value=dict(schema='ch3-state-v1',identity=identity,model=cpu_tree(model.state_dict()),
               optimizer=cpu_tree(opt.state_dict()),best=vars(best),epoch=epoch,steps=steps,
               rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
                        generator=generator.get_state()))
    tmp=Path(str(path)+'.tmp');torch.save(value,tmp);os.replace(tmp,path)


def restore_state(path,model,opt,identity,generator):
    import torch,random,numpy as np
    state=torch.load(path,map_location='cpu')
    if state.get('schema')!='ch3-state-v1' or state.get('identity')!=identity:raise ValueError('resume identity mismatch; no cross-protocol recovery')
    model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer'])
    random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy']);torch.set_rng_state(state['rng']['torch'])
    if state['rng']['cuda']:torch.cuda.set_rng_state_all(state['rng']['cuda'])
    generator.set_state(state['rng']['generator'])
    best=BestState();best.__dict__.update(state['best']);epoch,steps=state['epoch'],state['steps']
    del state
    return best,epoch,steps


def formal_identity(c, task, metadata, approval):
    p=profile(c,task)
    return dict(purpose='ch3_formal',run_id=task['id'],input_variant=task['input_variant'],
                protocol_sha=digest(c),profile_sha=digest(p),data_sha=digest(metadata),
                source=code_binding(),commit=approval['commit'])


def audit_resume(out,approval,run):
    """Metadata and approved fingerprints before any torch deserialization."""
    out=Path(out);record=approval.get('resume_audits',{}).get(run)
    if record is None:raise PermissionError('explicit audit of retained run required')
    mode=record.get('mode')
    names={'complete':['result.json','manifest.json'],
           'resume':['manifest.json','last.pt','best.pt','history.jsonl'],
           'not_started':[]}.get(mode)
    if names is None:raise ValueError('unsupported recovery mode')
    if mode=='not_started':
        if out.exists():raise FileExistsError('cannot fresh-restart retained output')
        return mode
    # Reject foreign protocol/input identity from metadata before any checkpoint read.
    c=read_profiles();task=task_by_id(c,run)
    manifest=json.loads((out/'manifest.json').read_text())
    identity=manifest['identity']
    if (identity.get('protocol_sha')!=digest(c) or identity.get('profile_sha')!=digest(profile(c,task))
            or identity.get('input_variant')!=task['input_variant']
            or manifest.get('task')!=task or manifest.get('profile')!=profile(c,task)):
        raise ValueError('foreign protocol/input identity before checkpoint access')
    for name in names:
        if hashlib.sha256((out/name).read_bytes()).hexdigest()!=record.get('sha256',{}).get(name):raise ValueError('recovery audit bytes mismatch: '+name)
    if manifest['identity']['run_id']!=run or manifest['identity']['purpose']!='ch3_formal':raise ValueError('foreign recovery identity')
    if (out/'result.json').exists() and mode!='complete':raise ValueError('completed run cannot resume')
    return mode


def finite(t):
    import torch
    if not bool(torch.isfinite(t).all()):raise FloatingPointError('nonfinite tensor')


def update(model,opt,x,y,p,device):
    import torch
    from models.ch3_adapter import target_prediction
    model.train();opt.zero_grad(set_to_none=True)
    pred,aux=target_prediction(model,x.to(device),p);target=y.to(device)
    finite(pred);loss=torch.nn.functional.mse_loss(pred,target)
    if aux is not None:finite(aux);loss=loss+aux
    finite(loss);loss.backward()
    active=[]
    for parameter in model.parameters():
        if parameter.grad is not None:
            finite(parameter.grad)
            if bool(torch.count_nonzero(parameter.grad)):active.append(parameter)
    if not active:raise FloatingPointError('no target/loss gradient')
    opt.step()
    for parameter in model.parameters():finite(parameter)
    return float(loss.detach())


def evaluate(model,batches,p,device):
    import torch
    from models.ch3_adapter import target_prediction
    model.eval();sse=sae=0.;count=0
    with torch.no_grad():
        for x,y in batches:
            pred,_=target_prediction(model,x.to(device),p);err=pred-y.to(device);finite(err)
            sse+=float(err.double().square().sum());sae+=float(err.double().abs().sum());count+=err.numel()
    if count==0:raise ValueError('empty evaluation')
    return dict(mse=sse/count,mae=sae/count,sse=sse,sae=sae,elements=count)


def init_training(c,task,device):
    import torch,random,numpy as np
    from models.ch3_adapter import build
    p=profile(c,task);torch.set_num_threads(4);random.seed(2024);np.random.seed(2024);torch.manual_seed(2024)
    model=build(c,task).to(device);t=p['training']
    opt=torch.optim.Adam(model.parameters(),lr=t['lr'],betas=tuple(t['betas']),eps=t['eps'],weight_decay=t['weight_decay'])
    return p,model,opt,torch.Generator().manual_seed(2024)


def numeric_probe_snapshot(model, opt, rule, step):
    """Small CPU values for one named tensor; all residual state stays exact."""
    import torch
    name=rule['parameter'];params=dict(model.named_parameters())
    if name not in params:raise ValueError('approved numeric parameter missing')
    parameter=params[name];raw=opt.state_dict();ids=[]
    for objects,stored in zip(opt.param_groups,raw['param_groups']):
        if len(objects['params'])!=len(stored['params']):raise ValueError('optimizer mapping mismatch')
        ids += [sid for obj,sid in zip(objects['params'],stored['params']) if obj is parameter]
    if len(ids)!=1:raise ValueError('numeric parameter optimizer identity is not unique')
    sid=ids[0];state=raw['state'].get(sid,{})
    if set(state)!={'step','exp_avg','exp_avg_sq'}:raise ValueError('unapproved optimizer state schema')
    def pack(v):
        if not torch.is_tensor(v) or str(v.dtype)!=rule['dtype'] or list(v.shape)!=rule['shape']:
            raise ValueError('numeric tensor dtype/shape changed')
        finite(v)
        return dict(dtype=str(v.dtype),shape=list(v.shape),values=v.detach().cpu().reshape(-1).tolist())
    values=dict(parameter=pack(parameter),gradient=pack(parameter.grad),
                exp_avg=pack(state['exp_avg']),exp_avg_sq=pack(state['exp_avg_sq']))
    residual=dict(raw);residual['state']=dict(raw['state']);residual['state'][sid]={k:v for k,v in state.items() if k not in ('exp_avg','exp_avg_sq')}
    return dict(step=step,parameter_name=name,optimizer_parameter_id=sid,
                tensors=values,
                exact_model=tensor_digest({k:v for k,v in model.state_dict().items() if k!=name}),
                exact_optimizer=tensor_digest(residual),
                exact_gradients=tensor_digest({k:v.grad for k,v in params.items() if k!=name and v.grad is not None}))


def compare_probe_trajectories(c, task, reference, actual):
    """Fail closed on identity; return auditable exact or bounded equivalence."""
    import math
    from utils.ch3_contract import numeric_probe_policy
    rule=numeric_probe_policy(c,task)
    identity=('id','profile_sha','initial','initial_rng','batch_ids','validation_tail','steps','final_rng')
    for key in identity:
        if key not in reference or key not in actual or reference[key]!=actual[key]:
            raise ValueError('exact identity/RNG/batch mismatch: '+key)
    if reference['id']!=task['id'] or reference['profile_sha']!=digest(profile(c,task)):
        raise ValueError('foreign task profile')
    if reference.get('finite') is not True or actual.get('finite') is not True:
        raise ValueError('finite flag missing')
    if reference['steps']!=6 or len(reference['batch_ids'])!=6:raise ValueError('incomplete short trajectory')
    exact_fields=('trajectory','validation','final')
    if any(k not in x for x in (reference,actual) for k in exact_fields):raise ValueError('required trajectory evidence missing')
    if any(len(x['trajectory'])!=6 for x in (reference,actual)):raise ValueError('six-step coverage missing')
    raw_equal=all(reference[k]==actual[k] for k in exact_fields)
    if rule is None:
        return dict(passed=raw_equal,mode='exact',bitwise_equal=raw_equal,
                    reason=None if raw_equal else 'exact numerical mismatch')
    if reference.get('numeric_policy_sha')!=digest(rule) or actual.get('numeric_policy_sha')!=digest(rule):
        raise ValueError('missing or foreign numeric policy evidence')
    traces=[reference.get('numeric_trace'),actual.get('numeric_trace')]
    if any(not isinstance(x,list) or len(x)!=6 for x in traces):raise ValueError('numeric trace missing')
    if any(len(x.get('trajectory',[]))!=6 for x in (reference,actual)):raise ValueError('six steps required')
    maxima={k:0.0 for k in rule['tensor_fields']};scalar_max=0.;failures=[];count=0
    def number(v):
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):
            raise ValueError('nonfinite or nonnumeric comparison value')
        return float(v)
    def compare_number(a,b,label):
        nonlocal scalar_max
        delta=abs(number(a)-number(b));scalar_max=max(scalar_max,delta)
        if delta>rule['atol']:failures.append(label)
    for step,(a,b) in enumerate(zip(*traces),1):
        required={'step','parameter_name','optimizer_parameter_id','tensors','exact_model','exact_optimizer','exact_gradients'}
        if set(a)!=required or set(b)!=required:raise ValueError('numeric evidence schema changed')
        if a['step']!=step or b['step']!=step or a['parameter_name']!=rule['parameter'] or b['parameter_name']!=rule['parameter']:
            raise ValueError('numeric step/parameter mismatch')
        for key in ('optimizer_parameter_id','exact_model','exact_optimizer','exact_gradients'):
            if a[key]!=b[key]:failures.append('step%d exact %s'%(step,key))
        if set(a['tensors'])!=set(rule['tensor_fields']) or set(b['tensors'])!=set(rule['tensor_fields']):
            raise ValueError('numeric tensor coverage mismatch')
        for key in rule['tensor_fields']:
            aa,bb=a['tensors'][key],b['tensors'][key]
            if set(aa)!={'dtype','shape','values'} or set(bb)!={'dtype','shape','values'}:raise ValueError('packed tensor schema')
            if any(x['dtype']!=rule['dtype'] or x['shape']!=rule['shape'] for x in (aa,bb)):
                raise ValueError('numeric shape/dtype mismatch')
            size=math.prod(rule['shape'])
            if len(aa['values'])!=size or len(bb['values'])!=size:raise ValueError('numeric element count mismatch')
            delta=max((abs(number(x)-number(y)) for x,y in zip(aa['values'],bb['values'])),default=0.)
            count+=size;maxima[key]=max(maxima[key],delta)
            if delta>rule['atol']:failures.append('step%d %s exceeds absolute tolerance'%(step,key))
        ta,tb=reference['trajectory'][step-1],actual['trajectory'][step-1]
        if set(ta)!=set(tb) or set(ta)!={'step','loss','state','optimizer'} or ta['step']!=step or tb['step']!=step:
            raise ValueError('trajectory schema changed')
        compare_number(ta['loss'],tb['loss'],'step%d loss'%step)
    a,b=reference['validation'],actual['validation']
    if set(a)!={'mse','mae','sse','sae','elements'} or set(a)!=set(b):raise ValueError('validation schema changed')
    if type(a['elements']) is not int or type(b['elements']) is not int or a['elements']<=0 or a['elements']!=b['elements']:
        raise ValueError('validation element count mismatch')
    for x in (a,b):
        for key,total in (('mse','sse'),('mae','sae')):
            if number(x[key])!=number(x[total])/x['elements']:raise ValueError('validation aggregation inconsistent')
    for key in ('mse','mae'):compare_number(a[key],b[key],'validation '+key)
    for key in ('sse','sae'):compare_number(a[key]/a['elements'],b[key]/b['elements'],'normalized validation '+key)
    return dict(passed=not failures,mode='bounded_numeric',policy_id=rule['id'],policy_sha=digest(rule),
                atol=rule['atol'],rtol=rule['rtol'],equal_nan=False,
                bitwise_equal=raw_equal and all(v==0 for v in maxima.values()),
                tensor_max_abs=maxima,scalar_normalized_max_abs=scalar_max,compared_elements=count,
                exact_residual_state=not any('exact' in f for f in failures),failures=failures)


def memory_growth_review(memory):
    """Keep strict four-point rule; CPU pre-hash still includes prior history."""
    if len(memory)<4:return {'blocked':False,'triggers':[],'scope':'insufficient observations'}
    recent=memory[-4:]
    triggers=[key for key in ('allocated','rss_before_hash')
              if all(b[key]>a[key] for a,b in zip(recent,recent[1:]))]
    after_grows=all(b['rss_after_hash']>a['rss_after_hash'] for a,b in zip(recent,recent[1:]))
    return dict(blocked=bool(triggers),triggers=triggers,after_hash_grows=after_grows,
                scope='six-step check; pre-hash RSS may retain earlier hash allocator effects; not proof of GPU leak')


def probe_worker(c,task,out,device='cuda:0',capture_states=False):
    import torch
    out=Path(out);p,model,opt,generator=init_training(c,task,device)
    from utils.ch3_contract import numeric_probe_policy
    numeric_rule=numeric_probe_policy(c,task);numeric_trace=[None]*6 if numeric_rule else None
    import random,numpy as np
    def rng():return tensor_digest((random.getstate(),np.random.get_state(),torch.get_rng_state(),
                                   torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],generator.get_state()))
    def rss():
        return next(int(line.split()[1])*1024 for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))
    def capture(label):
        if capture_states:
            torch.save(dict(model=cpu_tree(model.state_dict()),optimizer=cpu_tree(opt.state_dict()),
                            gradients={k:cpu_tree(v.grad) for k,v in model.named_parameters() if v.grad is not None}),out/(label+'.pt'))
    initial=tensor_digest(model.state_dict());initial_rng=rng();capture('initial')
    b=p['training']['batch'];v=p['training']['eval_batch']
    d=c['datasets'][task['dataset']]
    if task['dataset']=='UrbanEV':
        a,z,_=c['urban_folds'][task['fold']-1];validation_samples=(z-a-p['T']-task['h']+1)*275
    else:
        if d['endpoints'] is None:raise ValueError('version endpoints required for actual validation tail')
        a,z,_=d['endpoints'];validation_samples=z-a-p['pred_len']+1
    tail=validation_samples%v
    def batch(size):
        x=torch.randn(size,p['T'],p['C'],generator=generator)
        y=torch.randn(size,p['pred_len'],1,generator=generator)
        return x,y
    trajectory=[None]*6;batch_ids=[None]*6;memory=[None]*6
    for step in range(6):
        x,y=batch(b);batch_ids[step]=tensor_digest((x,y))
        loss=update(model,opt,x,y,p,device)
        # Measurement is before CPU state hashing; previous steps' hashing can
        # still affect RSS. Both phases remain visible and no tolerance changes.
        if str(device).startswith('cuda'):torch.cuda.synchronize()
        before=rss();allocated=torch.cuda.memory_allocated() if str(device).startswith('cuda') else 0
        trajectory[step]=dict(step=step+1,loss=loss,state=tensor_digest(model.state_dict()),optimizer=tensor_digest(opt.state_dict()))
        if numeric_rule:numeric_trace[step]=numeric_probe_snapshot(model,opt,numeric_rule,step+1)
        after=rss()
        if step==1:validation=evaluate(model,[batch(v),batch(tail or v)],p,device)
        del x,y
        if str(device).startswith('cuda'):torch.cuda.synchronize()
        memory[step]=dict(step=step+1,allocated=allocated,
             reserved=torch.cuda.memory_reserved() if str(device).startswith('cuda') else 0,
             rss_before_hash=before,rss_after_hash=after,rss=rss())
        capture('step-'+str(step+1))
    result=dict(id=task['id'],profile_sha=digest(p),initial=initial,initial_rng=initial_rng,
                batch_ids=batch_ids,trajectory=trajectory,validation=validation,validation_tail=tail,steps=6,
                final=tensor_digest(model.state_dict()),final_rng=rng(),memory=memory,
                allocated=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0,
                reserved=torch.cuda.max_memory_reserved() if str(device).startswith('cuda') else 0,
                affinity=sorted(os.sched_getaffinity(0)),threads=torch.get_num_threads(),finite=True,
                diagnostic_state_capture=capture_states,memory_review=memory_growth_review(memory))
    if numeric_rule:result.update(numeric_policy_sha=digest(numeric_rule),numeric_trace=numeric_trace)
    dump(out/'trajectory.json',result)
    if result['memory_review']['blocked']:
        raise RuntimeError('persistent memory growth in measured updates; group blocked pending diagnosis')
    return result



def formal_worker(c,task,out,approval,resume=False):
    import torch
    from utils.ch3_data import load,batches
    out=Path(out);p=profile(c,task)
    datasets,metadata=load(c,task)
    binding=approval['data_bindings'][task['dataset']]
    if binding.get(task['id'])!=digest(metadata):raise ValueError('data prefix/scaler/task binding mismatch')
    identity=formal_identity(c,task,metadata,approval)
    if (out/'result.json').exists():raise FileExistsError('completed run cannot restart')
    p,model,opt,generator=init_training(c,task,'cuda:0');best=BestState(p['training']['patience']);epoch=steps=0
    if resume:
        audit_resume(out,approval,task['id'])
        if json.loads((out/'manifest.json').read_text())['identity']!=identity:raise ValueError('manifest identity mismatch before checkpoint load')
        best,epoch,steps=restore_state(out/'last.pt',model,opt,identity,generator)
    elif (out/'last.pt').exists():raise FileExistsError('staging retained; audit then explicit resume required')
    if not resume:dump(out/'manifest.json',dict(identity=identity,task=task,profile=p,metadata=metadata))
    train=batches(datasets['train'],p,'train',generator);val=batches(datasets['validation'],p,'validation')
    for epoch_index in range(epoch+1,p['training']['epochs']+1):
        if best.stopped:break
        for x,y in train:update(model,opt,x,y,p,'cuda:0');steps+=1
        metrics=evaluate(model,val,p,'cuda:0')
        if best.update(metrics['mse'],epoch_index):save_state(out/'best.pt',model,opt,identity,best,epoch_index,steps,generator)
        save_state(out/'last.pt',model,opt,identity,best,epoch_index,steps,generator)
        with (out/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(epoch=epoch_index,steps=steps,validation=metrics,best_epoch=best.epoch))+'\n')
    del train,val,datasets,opt
    saved=torch.load(out/'best.pt',map_location='cpu')
    if saved['identity']!=identity:raise ValueError('best identity mismatch')
    model.load_state_dict(saved['model'],strict=True);del saved
    # Only the separately approved formal worker can enter this final test path.
    datasets,test_metadata=load(c,task,test_capability={'purpose':'ch3_formal_test','protocol_sha':digest(c),'run_id':task['id']})
    result=evaluate(model,batches(datasets['test'],p,'test'),p,'cuda:0')
    result.update(id=task['id'],input_variant=task['input_variant'],purpose='ch3_formal',protocol_sha=digest(c),best_epoch=best.epoch,
                  seed=2024,std=None,metric_space='train-standardized',stability='Not evaluated')
    dump(out/'result.json',result)


def verified_waves(group,decision):
    q=decision.get('concurrency')
    if q not in (1,2,4) or q>group['q']:raise ValueError('unverified concurrency')
    if decision.get('representatives')!=group['representatives'] or decision.get('status')!='Passed':raise ValueError('group evidence mismatch')
    return [wave[i:i+q] for wave in group['waves'] for i in range(0,len(wave),q)]


def dispatch(waves,command,out,*,placeholder=False):
    """Fixed waves, no refill; own subprocesses only; retain failed output."""
    out=Path(out);active=[]
    def stop(sig,frame):
        for p in active:
            if p.poll() is None:p.terminate()
        raise InterruptedError('safe-stop requested; staging retained')
    old=signal.signal(signal.SIGTERM,stop)
    try:
        for number,wave in enumerate(waves):
            if (out/'STOP').exists():raise InterruptedError('STOP requested between waves')
            active=[];handles=[]
            for run in wave:
                log=(out/(run+'.log')).open('x',encoding='utf-8');handles.append(log)
                active.append(subprocess.Popen(command(run),stdout=log,stderr=subprocess.STDOUT))
            codes=[p.wait() for p in active]
            for f in handles:f.close()
            dump(out/f'wave-{number}.json',dict(tasks=wave,returncodes=codes))
            if any(codes):raise RuntimeError('wave failed; no automatic fresh rerun')
    finally:
        for p in active:
            if p.poll() is None:p.terminate();p.wait()
        signal.signal(signal.SIGTERM,old)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model');parser.add_argument('action',choices=['preflight','dry-run','start','status','logs','complete','safe-stop','worker'],nargs='?',default='preflight')
    parser.add_argument('--approval');parser.add_argument('--probe-report');parser.add_argument('--output');parser.add_argument('--run-id');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();c=validate_manifest(read_profiles());out=Path(args.output or (Path(c['execution']['evidence'])/('formal-'+str(args.model))))
    if args.action in ['status','logs','complete','safe-stop']:
        if args.action=='safe-stop':
            state=json.loads((out/'controller.json').read_text());pid=state['pid']
            current=Path(f'/proc/{pid}/stat').read_text().split()[21]
            if current!=state['start_ticks']:raise RuntimeError('PID reused; refusing signal')
            os.kill(pid,signal.SIGTERM)
        else:
            complete=(out/'complete.json').exists()
            if complete:validate_model_completion(c,args.model,json.loads((out/'complete.json').read_text()))
            print(json.dumps({'output':str(out),'complete':complete,'logs':[str(p) for p in out.glob('*.log')]}))
        return
    approval=json.loads(Path(args.approval).read_text()) if args.approval else None
    reasons=preflight(c,args.model,approval)
    if args.action in ['preflight','dry-run']:
        print(json.dumps(dict(model=args.model,blocked=reasons,tasks=[t['id'] for t in c['tasks'] if t['model']==args.model]),indent=2))
        if args.action=='preflight' and reasons:raise SystemExit(2)
        return
    if reasons:raise RuntimeError('; '.join(reasons))
    if args.action=='worker':
        from restricted_io_guard import require_installed
        state=require_installed()
        if state.get('purpose')!='ch3_formal':raise PermissionError('formal guarded worker required')
        return formal_worker(c,task_by_id(c,args.run_id),out,approval,args.resume)
    report=json.loads(Path(args.probe_report).read_text())
    validate_probe_report(c,report)
    if hashlib.sha256(Path(args.probe_report).read_bytes()).hexdigest()!=approval['probe_report_sha']:raise ValueError('probe report SHA mismatch')
    if report['protocol_sha']!=digest(c) or report['code']!=code_binding():raise ValueError('probe source/config mismatch')
    if report['hardware']!=hardware_binding() or report['environment']!=environment_binding():raise ValueError('probe hardware/environment mismatch')
    # Reject before creating artifacts; start one model group only.
    waves=[]
    for group in c['groups']:
        if group['model']==args.model:waves.extend(verified_waves(group,report['decisions'][group['id']]))
    recovery={}
    if args.resume:
        if (out/'complete.json').exists():raise FileExistsError('completed group cannot resume')
        for t in c['tasks']:
            if t['model']==args.model:recovery[t['id']]=audit_resume(out/t['id'],approval,t['id'])
        waves=[[run for run in wave if recovery[run]!='complete'] for wave in waves]
        waves=[wave for wave in waves if wave]
    else:out.mkdir(parents=True,exist_ok=False)
    controller_out=out/('recovery-'+str(time.time_ns())) if args.resume else out
    controller_out.mkdir(exist_ok=True)
    with GPULock(c):
        dump(out/'controller.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],model=args.model))
        def command(run):
            cmd=[c['execution']['python'],str(ROOT/'tools/restricted_regression/m5_formal_entry.py'),
                    'formal-worker','--run-id',run,'--output',str(out/run),'--approval',args.approval]
            if recovery.get(run)=='resume':cmd.append('--resume')
            return cmd
        dispatch(waves,command,controller_out)
        rows=[json.loads((out/t['id']/'result.json').read_text()) for t in c['tasks'] if t['model']==args.model]
        dump(out/'complete.json',dict(model=args.model,protocol_sha=digest(c),
             task_ids=[t['id'] for t in c['tasks'] if t['model']==args.model],results=summarize(c,rows)))


if __name__=='__main__':main()
