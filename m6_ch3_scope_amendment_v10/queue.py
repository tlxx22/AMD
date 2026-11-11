"""One fixed 29-wave queue. A technical failure prevents all later dispatch."""
import os
import sys
import time
import json
import math
import fcntl
import subprocess
import traceback
from pathlib import Path
from .contract import ROOT,SCOPE,permission,read,sha,load_ref,write_new,test_key,verify_source_refs
from moderntcn_input_ablation_v10.launcher import proc_start,gpu_startup


def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True,timeout=30).strip()


def preflight(config_path,auth_path):
    a=read(auth_path);c=permission(config_path,a)
    from m6_ch3_supplement_v10.environment import verify
    verify(c);verify_source_refs(load_ref(c['sources_ref']))
    h=git('rev-parse','HEAD')
    if h!=a['commit'] or git('rev-parse','@{u}')!=h or git('status','--porcelain'):raise PermissionError('bound clean Git required')
    if git('ls-remote','origin','refs/heads/'+git('branch','--show-current')).split()[0]!=h:raise PermissionError('actual remote mismatch')
    if (Path(c['result_root'])/c['attempt']).exists():raise FileExistsError('attempt already used; no automatic retry')
    return c


def context(config_path,runtime_path):
    r=read(runtime_path);a=load_ref(r['auth_ref']);c=permission(config_path,a);folder=Path(runtime_path).parent
    if r['scope']!=SCOPE or r['config_ref']!=dict(path=str(config_path),sha256=sha(config_path)) or r['commit']!=a['commit'] or r['waves']!=c['waves']:raise PermissionError('runtime binding')
    if folder.resolve()!=(Path(c['result_root'])/c['attempt']).resolve():raise PermissionError('owned attempt path')
    o=read(folder/'owner.json')
    if o['runtime_sha256']!=sha(runtime_path) or proc_start(o['pid'])!=o['start'] or os.getppid()!=o['pid']:raise PermissionError('owned child PID/start/parent')
    if git('rev-parse','HEAD')!=r['commit'] or git('status','--porcelain'):raise PermissionError('local committed source changed')
    from m6_ch3_supplement_v10.environment import verify
    verify(c);return c,folder


def validate_result(c,j,r):
    if r['job']!=j:raise ValueError('result job')
    if j['phase']=='T':
        t=load_ref(c['sources_ref'])['prototypes'][j['prototype']]['task'];n=r['epochs']
        if not 1<=n<=20 or r['updates']!=n*t['arithmetic']['train_batches'] or r['test_calls']!=0 or r['status']!='validation_only_completed':raise ValueError('train budget/test boundary')
        if r['routes']!=dict(train=r['updates'],validation=n*t['arithmetic']['eval_batches']['validation']) or not 1<=r['best_epoch']<=n:raise ValueError('validation/best coverage')
        if not all(math.isfinite(r['validation'][m]) for m in ('mse','mae')):raise ValueError('finite validation')
    elif j['phase'] in ('S','P','C'):
        if (r['calls'],r['presentations'],r['elements'])!=tuple(j['expected'][k] for k in ('batches','samples','elements')):raise ValueError('test coverage')
        if r['split']!='test' or r['new_test_access'] is not True or any(r[k]!=0 for k in ('train','backward','optimizer')):raise ValueError('frozen test only')
    elif r.get('sensitivity_cells')!=25 or r.get('position_cells')!=15 or r.get('case_models')!=5 or r.get('case_plots')!=4:raise ValueError('offline coverage')


def checked(c,folder,j):
    if j not in c['jobs']:raise PermissionError('unknown receipt job')
    folder=Path(folder);r=read(folder/'receipts'/(j['id']+'.json'));out=folder/('plots' if j['phase']=='O' else j['id'])
    if r['job']!=j or r['runtime_sha256']!=sha(folder/'runtime.json'):raise ValueError('receipt runtime/job')
    actual={str(p.relative_to(folder)) for p in out.rglob('*') if p.is_file()}
    if not actual or actual!=set(r['files']):raise ValueError('receipt payload membership')
    for p,s in r['files'].items():
        q=(folder/p).resolve()
        if not q.is_relative_to(out.resolve()) or sha(q)!=s:raise ValueError('receipt payload SHA/path')
    validate_result(c,j,r['result'])
    if j['phase']!='O' and read(out/'result.json')!=r['result']:raise ValueError('receipt/result identity')
    if j['phase'] in ('S','P','C'):
        key=test_key(c,j,read(out/'source.json'));cap=dict(scope=SCOPE,job=j['id'],key=key,purpose=j['purpose'],mode='authorized_frozen_test')
        if read(folder/'claims'/(key+'.json'))!=dict(job=j,capability=cap,runtime_sha256=sha(folder/'runtime.json')):raise ValueError('claim/result identity')
    return r


def worker(config_path,runtime_path,job_id):
    c,folder=context(config_path,runtime_path);j=next((j for j in c['jobs'] if j['id']==job_id),None)
    if j is None:raise PermissionError('unknown job')
    def stop():
        if (folder/'STOP').exists():raise InterruptedError('owned STOP')
    stop()
    if j['phase']=='T':
        from .worker import train
        result=train(c,j,folder,stop)
    elif j['phase']=='O':
        for prior in c['jobs'][:-1]:checked(c,folder,prior)
        from .plots import completed
        result=completed(c,folder)
    else:
        from .worker import resolve,evaluate
        row=resolve(c,j,folder);key=test_key(c,j,row);cap=dict(scope=SCOPE,job=j['id'],key=key,purpose=j['purpose'],mode='authorized_frozen_test')
        write_new(folder/'claims'/(key+'.json'),dict(job=j,capability=cap,runtime_sha256=sha(runtime_path)))
        result=evaluate(c,j,row,folder,cap,stop)
    stop();validate_result(c,j,result);out=folder/('plots' if j['phase']=='O' else j['id'])
    write_new(folder/'receipts'/(job_id+'.json'),dict(job=j,result=result,files={str(p.relative_to(folder)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},runtime_sha256=sha(runtime_path)))


def owned_stop(folder):
    folder=Path(folder);r=read(folder/'runtime.json');c=load_ref(r['config_ref']);o=read(folder/'owner.json')
    if r['scope']!=SCOPE or folder.resolve()!=(Path(c['result_root'])/c['attempt']).resolve() or o['runtime_sha256']!=sha(folder/'runtime.json') or proc_start(o['pid'])!=o['start']:raise PermissionError('owned STOP identity')
    if not (folder/'STOP').exists():write_new(folder/'STOP',dict(owner=o,requested=time.time()))


def supervise(config_path,auth_path):
    c=preflight(config_path,auth_path);a=read(auth_path);folder=Path(c['result_root'])/c['attempt'];children=[]
    with open(c['global_lock'],'a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise PermissionError('GPU0 visibility required')
        folder.mkdir(parents=True,exist_ok=False)
        try:
            from m6_ch3_supplement_v10.gpu_identity import check_gpu_identity
            check_gpu_identity(c['gpu_uuid'],folder/'gpu-identity','supervisor-before-cuda',False);gpu=gpu_startup(a)
            runtime=folder/'runtime.json';write_new(runtime,dict(scope=SCOPE,commit=a['commit'],config_ref=dict(path=str(config_path),sha256=sha(config_path)),auth_ref=dict(path=str(auth_path),sha256=sha(auth_path)),environment_ref=c['environment_ref'],gpu=gpu,waves=c['waves']))
            write_new(folder/'owner.json',dict(pid=os.getpid(),start=proc_start(os.getpid()),runtime_sha256=sha(runtime)))
            for name in ('logs','receipts','claims'):(folder/name).mkdir()
            s=load_ref(c['sources_ref']);plan=load_ref(s['case_plan_ref']);pre=load_ref(s['old_preselection_ref'])
            from m6_ch3_test_reporting_v10.analysis import case_plan
            if any(case_plan(s['rows']['C-UrbanEV-'+m],pre)!=plan for m in ('AMD','TimeXer')):raise ValueError('common original week/sample identity')
            write_new(folder/'case-plan.json',plan)
            if sha(folder/'case-plan.json')!=s['case_plan_ref']['sha256']:raise ValueError('case plan serialization')
            for i,wave in enumerate(c['waves'],1):
                if (folder/'STOP').exists():raise InterruptedError('STOP')
                children=[]
                for name in wave:
                    with (folder/'logs'/(name+'.log')).open('x') as log:
                        p=subprocess.Popen([sys.executable,'-B','-m','m6_ch3_scope_amendment_v10.entry','worker','--config',str(config_path),'--runtime',str(runtime),'--job',name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    children.append((name,p))
                write_new(folder/f'children-wave-{i}.json',[dict(job=n,pid=p.pid,start=proc_start(p.pid)) for n,p in children])
                while any(p.poll() is None for _,p in children):
                    if any(p.poll() not in (None,0) for _,p in children) and not (folder/'STOP').exists():write_new(folder/'STOP',dict(reason='owned child technical failure'))
                    time.sleep(1)
                if any(p.returncode!=0 for _,p in children):raise RuntimeError('technical failure; later waves withheld')
                for name,_ in children:checked(c,folder,next(j for j in c['jobs'] if j['id']==name))
                write_new(folder/f'wave-{i}-complete.json',dict(jobs=wave))
            if (folder/'STOP').exists():raise InterruptedError('STOP before completion')
            if {p.stem for p in (folder/'receipts').glob('*.json')}!={j['id'] for j in c['jobs']} or len(list((folder/'claims').glob('*.json')))!=24:raise ValueError('complete receipt/claim coverage')
            for j in c['jobs']:checked(c,folder,j)
            write_new(folder/'complete.json',dict(technical_complete=True,jobs=41,waves=29,new_training=16,new_test_evaluations=24,sensitivity_cells=25,sensitivity_references=9,position_cells=15,case_models=5,case_nodes=10,case_plots=4,scientific_review='Pending; no main model/table replacement'))
        except BaseException:
            if any(p.poll() is None for _,p in children):
                if not (folder/'STOP').exists():write_new(folder/'STOP',dict(reason='supervisor failure'))
                for _,p in children:p.wait()
            write_new(folder/'failure.json',dict(traceback=traceback.format_exc()));raise


def status(folder):
    folder=Path(folder);rs=[read(p) for p in (folder/'receipts').glob('*.json')];progress={};evaluation={};active=[]
    for p in folder.glob('TR-*/history.jsonl'):
        lines=p.read_text().splitlines()
        if lines:
            try:progress[p.parent.name]={k:json.loads(lines[-1])[k] for k in ('epoch','steps','best_epoch','bad')}
            except json.JSONDecodeError:progress[p.parent.name]={'partial_last_line':True}
    for p in folder.glob('TR-*/updates.jsonl'):
        lines=p.read_text().splitlines()
        if lines:
            try:
                latest=json.loads(lines[-1]);progress.setdefault(p.parent.name,{}).update(current_epoch=latest['epoch'],current_step=latest['step'])
            except json.JSONDecodeError:progress.setdefault(p.parent.name,{})['partial_update_line']=True
    for p in folder.glob('*/access.jsonl'):
        done=[];last=None
        for line in p.read_text().splitlines():
            try:
                last=json.loads(line)
                if last.get('event')=='after_forward':done.append(last)
            except json.JSONDecodeError:pass  # A writer may have an unfinished last line.
        evaluation[p.parent.name]=dict(completed_batches=len(done),presentations=sum(x['samples'] for x in done),last_event=last)
    for p in folder.glob('children-wave-*.json'):
        for x in read(p):
            try:
                if proc_start(x['pid'])==x['start']:active.append(x)
            except (FileNotFoundError,ProcessLookupError):pass
    return dict(completed={k:sum(r['job']['phase']==k for r in rs) for k in ('P','C','T','S','O')},waves=len(list(folder.glob('wave-*-complete.json'))),total_waves=29,training_progress=progress,evaluation_progress=evaluation,active_owned_children=active,complete=(folder/'complete.json').exists(),failure=(folder/'failure.json').exists(),STOP=(folder/'STOP').exists())
