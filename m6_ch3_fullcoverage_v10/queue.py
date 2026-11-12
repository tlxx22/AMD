"""Forty fixed serial waves, no training, no automatic retry/resume."""
import os
import sys
import time
import fcntl
import subprocess
import traceback
import json
from pathlib import Path
from .contract import ROOT,SCOPE,read,sha,write_new,load_ref,permission,verify_sources,capability
from moderntcn_input_ablation_v10.launcher import proc_start,gpu_startup


def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True,timeout=30).strip()


def preflight(cp,ap):
    a=read(ap);c=permission(cp,a)
    from m6_ch3_supplement_v10.environment import verify
    verify(c);verify_sources(load_ref(c['sources_ref']))
    h=git('rev-parse','HEAD')
    if h!=a['commit'] or git('rev-parse','@{u}')!=h or git('status','--porcelain'):raise PermissionError('bound clean Git')
    if git('ls-remote','origin','refs/heads/'+git('branch','--show-current')).split()[0]!=h:raise PermissionError('actual remote')
    if (Path(c['result_root'])/c['attempt']).exists():raise FileExistsError('attempt already used; no automatic resume')
    return c


def context(cp,rp):
    r=read(rp);a=load_ref(r['auth_ref']);c=permission(cp,a);folder=Path(rp).parent;o=read(folder/'owner.json')
    if r['scope']!=SCOPE or r['commit']!=a['commit'] or r['config_ref']!=dict(path=str(cp),sha256=sha(cp)) or r['waves']!=c['waves']:raise PermissionError('runtime binding')
    if folder.resolve()!=(Path(c['result_root'])/c['attempt']).resolve() or o['runtime_sha256']!=sha(rp) or proc_start(o['pid'])!=o['start'] or os.getppid()!=o['pid']:raise PermissionError('owned PID/start/parent')
    if git('rev-parse','HEAD')!=r['commit'] or git('status','--porcelain'):raise PermissionError('local source changed')
    from m6_ch3_supplement_v10.environment import verify
    verify(c);return c,folder


def checked(c,folder,j):
    folder=Path(folder);out=folder/('data' if j['phase']=='O' else j['id']);r=read(folder/'receipts'/(j['id']+'.json'))
    if j not in c['jobs'] or r['job']!=j or r['runtime_sha256']!=sha(folder/'runtime.json'):raise ValueError('receipt job/runtime')
    if set(r['files'])!={str(p.relative_to(folder)) for p in out.rglob('*') if p.is_file()}:raise ValueError('payload membership')
    for p,h in r['files'].items():
        if not (folder/p).resolve().is_relative_to(out.resolve()) or sha(folder/p)!=h:raise ValueError('payload SHA/path')
    z=r['result']
    if z['job']!=j:raise ValueError('result job')
    if j['phase']=='O':
        if (z['logical_cells'],z['region_metric_cells'],z['position_cells'])!=(54,6600,30) or z['model_calls']!=0 or z['regions_selected']!=[]:raise ValueError('offline coverage/no selection')
    else:
        if read(out/'result.json')!=z or (z['calls'],z['presentations'],z['elements'])!=tuple(j['expected'][k] for k in ('batches','samples','elements')) or any(z[k]!=0 for k in ('training','backward','optimizer')):raise ValueError('evaluation coverage/freeze')
        cap=capability(c,j,read(out/'source.json'))
        if read(folder/'claims'/(cap['evaluation_key']+'.json'))!=dict(job=j,capability=cap,runtime_sha256=sha(folder/'runtime.json')):raise ValueError('claim identity')
    return r


def worker(cp,rp,name):
    c,folder=context(cp,rp);j=next((j for j in c['jobs'] if j['id']==name),None)
    if j is None:raise PermissionError('unknown job')
    def stop():
        if (folder/'STOP').exists():raise InterruptedError('owned STOP')
    stop()
    if j['phase']=='O':
        for prev in c['jobs'][:-1]:checked(c,folder,prev)
        from .reporting import completed
        result=completed(c,folder)
    else:
        row=load_ref(c['sources_ref'])['rows'][j['cell']];cap=capability(c,j,row)
        write_new(folder/'claims'/(cap['evaluation_key']+'.json'),dict(job=j,capability=cap,runtime_sha256=sha(rp)))
        from .worker import run
        result=run(c,j,folder,cap,stop)
    stop();out=folder/('data' if j['phase']=='O' else name)
    write_new(folder/'receipts'/(name+'.json'),dict(job=j,result=result,runtime_sha256=sha(rp),files={str(p.relative_to(folder)):sha(p) for p in out.rglob('*') if p.is_file()}))
    checked(c,folder,j)


def owned_stop(folder):
    folder=Path(folder);r=read(folder/'runtime.json');c=load_ref(r['config_ref']);o=read(folder/'owner.json')
    if r['scope']!=SCOPE or folder.resolve()!=(Path(c['result_root'])/c['attempt']).resolve() or o['runtime_sha256']!=sha(folder/'runtime.json') or proc_start(o['pid'])!=o['start']:raise PermissionError('owned STOP identity')
    if not (folder/'STOP').exists():write_new(folder/'STOP',dict(owner=o,requested=time.time()))


def supervise(cp,ap):
    c=preflight(cp,ap);a=read(ap);folder=Path(c['result_root'])/c['attempt'];child=None
    with open(c['global_lock'],'a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise PermissionError('fixed GPU visibility')
        folder.mkdir(parents=True,exist_ok=False)
        try:
            from m6_ch3_supplement_v10.gpu_identity import check_gpu_identity
            check_gpu_identity(c['gpu_uuid'],folder/'gpu-identity','before-cuda',False);gpu=gpu_startup(a)
            rp=folder/'runtime.json';write_new(rp,dict(scope=SCOPE,commit=a['commit'],config_ref=dict(path=str(cp),sha256=sha(cp)),auth_ref=dict(path=str(ap),sha256=sha(ap)),environment_ref=c['environment_ref'],waves=c['waves'],gpu=gpu))
            write_new(folder/'owner.json',dict(pid=os.getpid(),start=proc_start(os.getpid()),runtime_sha256=sha(rp)))
            for name in ('logs','claims','receipts'):(folder/name).mkdir()
            for i,w in enumerate(c['waves'],1):
                if (folder/'STOP').exists():raise InterruptedError('STOP')
                name=w[0]
                with (folder/'logs'/(name+'.log')).open('x') as log:child=subprocess.Popen([sys.executable,'-B','-m','m6_ch3_fullcoverage_v10.entry','worker','--config',str(cp),'--runtime',str(rp),'--job',name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                write_new(folder/f'children-wave-{i}.json',[dict(job=name,pid=child.pid,start=proc_start(child.pid))])
                if child.wait()!=0:raise RuntimeError('technical failure; later waves stopped')
                checked(c,folder,next(j for j in c['jobs'] if j['id']==name));write_new(folder/f'wave-{i}-complete.json',dict(jobs=w))
            if (folder/'STOP').exists() or {p.stem for p in (folder/'receipts').glob('*.json')}!={j['id'] for j in c['jobs']} or len(list((folder/'claims').glob('*.json')))!=39:raise ValueError('complete coverage')
            for j in c['jobs']:checked(c,folder,j)
            write_new(folder/'complete.json',dict(technical_complete=True,model_jobs=39,offline_jobs=1,waves=40,new_test_accesses=39,training=0,logical_cells=54,region_metric_cells=6600,region_selection='deferred to user/ChatGPT',scientific_review='Pending'))
        except BaseException:
            if child is not None and child.poll() is None:
                if not (folder/'STOP').exists():write_new(folder/'STOP',dict(reason='supervisor failure'))
                child.wait()
            write_new(folder/'failure.json',dict(traceback=traceback.format_exc()));raise


def status(folder):
    folder=Path(folder);rs=[read(p) for p in (folder/'receipts').glob('*.json')];active=[];progress={}
    for p in folder.glob('children-wave-*.json'):
        for x in read(p):
            try:
                if proc_start(x['pid'])==x['start']:active.append(x)
            except (FileNotFoundError,ProcessLookupError):pass
    for p in folder.glob('*/access.jsonl'):
        count=samples=0;last=None
        for line in p.read_text().splitlines():
            try:
                last=json.loads(line)
                if last.get('event')=='after_forward':count+=1;samples+=last['samples']
            except json.JSONDecodeError:pass
        progress[p.parent.name]=dict(completed_batches=count,presentations=samples,last_event=last)
    return dict(completed={phase:sum(r['job']['phase']==phase for r in rs) for phase in ('P','C','O')},waves=len(list(folder.glob('wave-*-complete.json'))),total_waves=40,active_owned_children=active,evaluation_progress=progress,training_jobs=0,complete=(folder/'complete.json').exists(),failure=(folder/'failure.json').exists(),STOP=(folder/'STOP').exists())
