"""Actual candidate train/validation/test worker; no legacy model routing."""
import contextlib
import copy
import fcntl
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import numpy as np
import torch
from . import adapter
from .spec import StructuralSpec
from .training_config import ROOT,validate,digest
from .training_state import atomic_json,file_sha,snapshot,save,read,restore,validate as validate_state
from .artifacts import publish,audit
from .gate_policy import validate_config_policy
from utils.ch3_contract import BestState
from utils.ch3_onecycle import Schedule
from utils.ch3_data import load,batches
from utils.ch3_time_marks import batch_parts
from ch3_runner import finite

RUN_NAMESPACE=ROOT.parent/'amd-execution-evidence/m6/timemixer-uv-v4-development-runs'
SIGNAL_STOP_REQUESTED=False

def request_stop(signum=None,frame=None):
    global SIGNAL_STOP_REQUESTED
    SIGNAL_STOP_REQUESTED=True

def synthetic_root_allowed(folder):
    folder=Path(folder).resolve()
    return folder.parent==ROOT and folder.name.startswith('.uv-v4-runner-')

def successful_identity(identity):
    # Scientific training identity is independent of report/config wrapper hashes.
    # StructuralSpec binds model implementation; task/profile/data bind training.
    value=copy.deepcopy({k:identity[k] for k in ('candidate_id','task','profile','structural_spec',
                  'data_sha','purpose','reference_original','training_protocol')})
    implementation=value['structural_spec']['implementation']
    core=('__init__.py','adapter.py','model.py','modules.py','source.py','spec.py','source_binding.json')
    value['structural_spec']['implementation']={k:implementation[k] for k in core}
    return digest(value)

def source_binding():
    paths=list((ROOT/'timemixer_dev_v4').glob('*.py'))+list((ROOT/'timemixer_dev_v4').glob('*.json'))
    paths += [ROOT/f for f in ('timemixer_train_v4_entry.py','scripts/ch3/start_timemixer_uv_v4.sh','utils/__init__.py','utils/ch3_data.py','utils/ch3_onecycle.py','utils/ch3_contract.py','utils/ch3_time_marks.py','utils/feature_schema.py','utils/dataloader_urbanev.py','ch3_runner.py','models/__init__.py','models/ch3_adapter.py','timemixer_dev/__init__.py','timemixer_dev/training_state.py')]
    return {str(p.relative_to(ROOT)):file_sha(p) for p in sorted(paths) if p.is_file()}

def environment():
    import pandas,sklearn,platform
    return dict(python=sys.executable,python_version=sys.version,torch=torch.__version__,numpy=np.__version__,
                pandas=pandas.__version__,sklearn=sklearn.__version__,platform=platform.platform(),cuda_build=torch.version.cuda,
                threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),bytecode_disabled=bool(sys.dont_write_bytecode),
                thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})

@contextlib.contextmanager
def identity_lock(root,key):
    folder=Path(root)/'identity-locks';folder.mkdir(parents=True,exist_ok=True)
    with (folder/(key+'.lock')).open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield

def append(path,value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n');f.flush()

def evaluate(model,dataset,p,device,predict,route,phase):
    model.eval();C=p['C'] if p.get('task')=='M' else 1
    sse=torch.zeros(C,dtype=torch.float64);sae=sse.clone();n=0
    with torch.no_grad():
        for batch in batches(dataset,p,phase):
            x,y,mark=batch_parts(batch);pred=predict(model,x.to(device),mark.to(device) if mark is not None else None)
            if pred.shape!=y.shape:raise ValueError('evaluation shape/broadcast mismatch')
            error=pred-y.to(device);finite(error)
            sse+=error.double().square().sum((0,1)).cpu();sae+=error.double().abs().sum((0,1)).cpu();n+=error.shape[0]*error.shape[1]
            route[phase]+=1
    if n!=len(dataset)*p['pred_len']:raise ValueError('evaluation tail dropped')
    return dict(mse=float(sse.sum()/(n*C)),mae=float(sae.sum()/(n*C)),sse=float(sse.sum()),sae=float(sae.sum()),elements=n*C,
                channels=[dict(index=i,mse=float(sse[i]/n),mae=float(sae[i]/n)) for i in range(C)])

def run_worker(config,task_id,root,attempt,*,device='cpu',resume=False,path_override=None,
               pause_after_epoch=None,reference_original=False,fault=None,launch_context=None):
    validate(config);task=next(t for t in config['tasks'] if t['id']==task_id);synthetic=config['purpose']=='synthetic'
    launch_provenance=None
    root=Path(root).resolve()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,90}',attempt):raise ValueError('attempt syntax')
    if synthetic:
        if device!='cpu' or not synthetic_root_allowed(config['synthetic_root']) or path_override is None or not root.is_relative_to(Path(config['synthetic_root']).resolve()) or not Path(path_override).resolve().is_relative_to(Path(config['synthetic_root']).resolve()):raise ValueError('synthetic isolation')
        if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    else:
        if device!='cuda:0' or path_override is not None or pause_after_epoch is not None or reference_original or fault:raise ValueError('real worker requires reviewed GPU launch, no fixtures')
        if not root.is_relative_to(RUN_NAMESPACE):raise ValueError('independent candidate result namespace required')
        from .launcher import validate_worker_context
        runtime_ref=copy.deepcopy(launch_context)
        launch_context=validate_worker_context(config,task_id,launch_context)
        launch_provenance=dict(runtime_ref=runtime_ref,authorization_ref=launch_context['authorization_ref'])
    p=copy.deepcopy(task['structural_source_profile']);tr=p['training'];data=task['data_config']
    if synthetic and (tr['threads']!=1 or tr['epochs']>4 or task['arithmetic']['max_optimizer_steps']>12):raise ValueError('bounded single-thread synthetic contract required')
    torch.set_num_threads(tr['threads'])
    def check_data_stat():
        if not synthetic:
            for name,expected in task['data_stat'].items():
                s=Path(name).stat()
                if {k:getattr(s,k) for k in expected}!=expected:raise ValueError('bound data stat changed')
    check_data_stat()
    datasets,metadata=load(data,task,resolved_profile=p,path_override=path_override)
    if not synthetic and metadata.get('prefixes',{})!=task['baseline_prefix_metadata']:raise ValueError('second-round normalized record prefix mismatch')
    if not synthetic and {k:metadata.get(k) for k in task['baseline_record_prefix']}!=task['baseline_record_prefix']:raise ValueError('second-round record prefix mismatch')
    check_data_stat()
    if metadata['window_counts']!={k:v for k,v in task['arithmetic']['windows'].items() if k!='test'}:raise ValueError('real loader/window arithmetic mismatch')
    if reference_original and (not synthetic or task['arm']!='B'):raise ValueError('reference path only for synthetic off equivalence')
    # Match the second-round worker: load deterministic data, then seed init.
    random.seed(2024);np.random.seed(2024);torch.manual_seed(2024)
    spec=StructuralSpec.from_profile(p,use_u=task['use_u'],use_v=task['use_v'],task=task['task'],fold=task['fold']).value
    model=(adapter.build_original(p) if reference_original else adapter.build(p,use_u=task['use_u'],use_v=task['use_v'],task=task['task'],fold=task['fold'])).to(device)
    if reference_original:
        from models.ch3_adapter import target_prediction
        predict=lambda m,x,marks:target_prediction(m,x,p,marks)[0]
    else:predict=adapter.prediction
    opt=torch.optim.Adam(model.parameters(),lr=tr['lr'],betas=tuple(tr['betas']),eps=tr['eps'],weight_decay=tr['weight_decay'])
    scheduler=Schedule(opt,tr['scheduler']);generator=torch.Generator().manual_seed(2024)
    env=environment()
    identity=dict(schema='timemixer-uv-run-identity-v2',candidate_id=task['candidate_id'],task=task,profile=p,structural_spec=spec,
                  config_sha=digest(config),data_sha=digest(metadata),environment=env,source=source_binding(),purpose=config['purpose'],
                  reference_original=reference_original,training_protocol=config['training_protocol'],gate_policy=copy.deepcopy(validate_config_policy(config)),
                  execution_device=dict(device=device,gpu_uuid=None if synthetic else launch_context['gpu_identity']['uuid']))
    import subprocess
    identity['producer_commit']=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    key=digest(identity);name=task['dataset']+'-'+task['input_variant']+'-f'+str(task['fold'])+'-h'+str(task['h'])+'-'+task['arm']+'-'+key[:12]
    guard_key=successful_identity(identity)
    folder=root/attempt;staging=folder/'staging'/name;final=folder/'runs'/name;registry=(root if synthetic else RUN_NAMESPACE)/'completed'/guard_key
    route=dict(train=0,validation=0,test=0);steps=0;epoch=0;best=BestState(tr['patience'])
    with identity_lock(root if synthetic else RUN_NAMESPACE,guard_key):
        if final.exists():raise FileExistsError('successful identity already exists')
        if registry.exists():
            registered=json.loads(registry.read_text())
            if not (resume and registered.get('status')=='publishing' and registered.get('staging')==str(staging) and staging.is_dir()):raise FileExistsError('successful/reserved identity already exists')
        if resume:
            if not staging.is_dir():raise FileNotFoundError('owned staging required')
            previous=json.loads((staging/'run-state.json').read_text())
            if previous['identity_sha']!=key:raise ValueError('resume staging identity')
            if (staging/'test-claim.json').exists() and not (staging/'test-result.json').exists():raise ValueError('test may have been accessed; cannot repeat automatically')
            updates=[json.loads(l) for l in (staging/'updates.jsonl').read_text().splitlines()]
            history=[json.loads(l) for l in (staging/'history.jsonl').read_text().splitlines()]
            if not (staging/'last.pt').is_file():raise ValueError('unsupported interruption before first epoch checkpoint')
            checkpoint=read(staging/'last.pt')
            if checkpoint['journals']!={'history_sha256':file_sha(staging/'history.jsonl'),'updates_sha256':file_sha(staging/'updates.jsonl')}:raise ValueError('resume journal checksum mismatch')
            if len(updates)!=checkpoint['steps'] or len(history)!=checkpoint['epoch'] or [u['step'] for u in updates]!=list(range(1,len(updates)+1)):raise ValueError('unsupported interrupted batch/validation boundary; no rewind')
            if history[-1]['best_epoch']!=checkpoint['best']['epoch']:raise ValueError('history/checkpoint best mismatch')
            best,epoch,steps=restore(staging/'last.pt',model,opt,scheduler,generator,identity)
            for k in route:route[k]=previous['routes'][k]
        else:
            staging.mkdir(parents=True,exist_ok=False)
            for file,value in [('config.json',config),('spec.json',spec),('environment.json',env),('command.json',dict(argv=sys.argv,cwd=os.getcwd())),('identity.json',identity),('data-metadata.json',metadata),('launch.json',launch_provenance)]:atomic_json(staging/file,value)
            for file in ('updates.jsonl','history.jsonl','worker.log'):(staging/file).touch()
            append(staging/'worker.log',dict(event='start',task=task_id,attempt=attempt,model_class=type(model).__name__,synthetic=synthetic,initial_lr=opt.param_groups[0]['lr'],initial_beta1=opt.param_groups[0]['betas'][0]))
        append(staging/'commands.jsonl',dict(argv=sys.argv,cwd=os.getcwd(),resume=resume,epoch=epoch,steps=steps))
        append(staging/'launches.jsonl',dict(provenance=launch_provenance,resume=resume,epoch=epoch,steps=steps))
        def state(phase):
            atomic_json(staging/'run-state.json',dict(phase=phase,identity_sha=key,epochs=epoch,steps=steps,routes=route))
        def stopped():
            if SIGNAL_STOP_REQUESTED or (folder/'STOP').exists():raise InterruptedError('owned STOP; no further work')
        try:
            state('epoch_boundary');stopped()
            while epoch<tr['epochs'] and not best.stopped and not (staging/'test-result.json').exists():
                state('training');model.train();start_steps=steps
                for batch_index,batch in enumerate(batches(datasets['train'],p,'train',generator)):
                    stopped();x,y,mark=batch_parts(batch);opt.zero_grad(set_to_none=True)
                    pred=predict(model,x.to(device),mark.to(device) if mark is not None else None);target=y.to(device)
                    loss=adapter.loss(pred,target);finite(loss);loss.backward()
                    active=False
                    for parameter in model.parameters():
                        if parameter.grad is not None:finite(parameter.grad);active|=bool(torch.count_nonzero(parameter.grad))
                    if not active:raise FloatingPointError('no active loss gradient')
                    opt.step();steps+=1;route['train']+=1
                    for parameter in model.parameters():finite(parameter)
                    scheduler.after_successful_update()
                    item=dict(step=steps,epoch=epoch+1,batch=batch_index,loss=float(loss.detach()),lr=opt.param_groups[0]['lr'],beta1=opt.param_groups[0]['betas'][0])
                    if synthetic:item['batch_signature']=hashlib_of_tensor(x)
                    append(staging/'updates.jsonl',item)
                    if fault=='after_update':raise RuntimeError('synthetic injected worker failure')
                if steps-start_steps!=task['arithmetic']['train_batches']:raise ValueError('train batch count')
                stopped();before=scheduler.updates
                metrics=evaluate(model,datasets['validation'],p,device,predict,route,'validation')
                if scheduler.updates!=before:raise ValueError('validation changed scheduler')
                epoch+=1;improved=best.update(metrics['mse'],epoch)
                append(staging/'history.jsonl',dict(epoch=epoch,steps=steps,validation=metrics,best_epoch=best.epoch,bad=best.bad,improved=improved,lr=opt.param_groups[0]['lr'],beta1=opt.param_groups[0]['betas'][0]))
                saved=snapshot(model,opt,scheduler,best,epoch,steps,generator,identity,journals=dict(history_sha256=file_sha(staging/'history.jsonl'),updates_sha256=file_sha(staging/'updates.jsonl')))
                if improved:save(staging/'best.pt',saved)
                save(staging/'last.pt',saved);state('epoch_boundary')
                append(staging/'worker.log',dict(event='epoch_boundary',epoch=epoch,steps=steps,best=best.epoch,bad=best.bad))
                if pause_after_epoch==epoch:return dict(status='paused',staging=str(staging),steps=steps,identity_sha=key)
            stopped()
            if not (staging/'test-result.json').exists():
                if (staging/'test-claim.json').exists():raise ValueError('uncertain test access; manual review required')
                best_state=validate_state(read(staging/'best.pt'),model,opt,scheduler,identity)
                model.load_state_dict(best_state['model'],strict=True)
                with (staging/'test-claim.json').open('x') as f:json.dump(dict(identity_sha=key,calls=1,best_epoch=best.epoch),f)
                state('test_claimed')
                if fault=='after_test_claim':raise RuntimeError('synthetic test uncertainty')
                check_data_stat()
                testing,test_meta=load(data,task,resolved_profile=p,path_override=path_override,test_capability=dict(purpose='ch3_formal_test',protocol_sha=digest(data),run_id=task_id));check_data_stat()
                if test_meta['scaler']!=metadata['scaler']:raise ValueError('test changed train-only scaler')
                before=scheduler.updates;result=evaluate(model,testing['test'],p,device,predict,route,'test')
                if scheduler.updates!=before:raise ValueError('test changed scheduler')
                atomic_json(staging/'test-data-metadata.json',test_meta)
                result.update(identity_sha=key,test_calls=1,best_epoch=best.epoch,epochs=epoch,steps=steps,metric_space='train-standardized',metric_scope='all_channels' if task['task']=='M' else 'target_only',task_id=task_id,arm=task['arm'],synthetic=synthetic,test_data_sha=digest(test_meta))
                atomic_json(staging/'test-result.json',result);atomic_json(staging/'test-result.sha256',dict(sha256=file_sha(staging/'test-result.json')));state('test_saved')
            else:
                if json.loads((staging/'test-result.sha256').read_text())!={'sha256':file_sha(staging/'test-result.json')}:raise ValueError('saved test checksum')
                result=json.loads((staging/'test-result.json').read_text())
                if result['identity_sha']!=key or result['test_calls']!=1:raise ValueError('saved test identity')
            if fault=='before_publish':raise RuntimeError('synthetic sealing failure after saved test')
            state('completed');append(staging/'worker.log',dict(event='complete',epochs=epoch,steps=steps,test_calls=1))
            manifest=dict(schema='timemixer-uv-run-manifest-v1',status='completed',identity_sha=key,successful_identity_key=guard_key,identity=identity,task_id=task_id,epochs=epoch,steps=steps,max_steps=task['arithmetic']['max_optimizer_steps'],best=vars(best),routes=route,test_calls=1,synthetic=synthetic,launch_provenance=launch_provenance,gate='Not evaluated')
            registry.parent.mkdir(parents=True,exist_ok=True)
            if not registry.exists():
                with registry.open('x') as f:json.dump(dict(status='publishing',staging=str(staging),final=str(final)),f,sort_keys=True)
            receipt=publish(staging,final,manifest,result)
            atomic_json(registry,dict(status='completed',receipt=receipt))
            return receipt
        except BaseException as error:
            if staging.is_dir():
                append(staging/'failures.jsonl',dict(type=type(error).__name__,message=str(error),epochs=epoch,steps=steps,time=time.time()))
                append(staging/'worker.log',dict(event='failure',error=str(error),epochs=epoch,steps=steps))
            raise

def hashlib_of_tensor(value):
    import hashlib
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
