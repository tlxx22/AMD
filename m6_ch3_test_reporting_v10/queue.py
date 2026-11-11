"""One serial frozen-test queue; no automatic resume or inherited test claims."""
import os
import sys
import time
import fcntl
import subprocess
import traceback
from pathlib import Path
from .contract import ROOT, SCOPE, PURPOSE, permission, read, sha, load_ref, write_new, selected, evaluation_key
from moderntcn_input_ablation_v10.launcher import proc_start, gpu_startup


def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True,timeout=30).strip()


def local_sources(c):
    ledger=load_ref(c['sources_ref'])
    for row in [*ledger['rows'].values(),*ledger['centers'].values()]:
        for k,r in row.items():
            if k.endswith('_ref') and isinstance(r,dict) and 'path' in r and k!='weights_ref':
                if sha(r['path'])!=r['sha256']:raise ValueError('source text changed: '+k)
        if Path(row['weights_ref']['path']).stat().st_size!=row['weights_stat']['bytes']:raise ValueError('weight file stat changed')
    return ledger


def preflight(config_path,auth_path):
    c=permission(config_path,read(auth_path))
    from m6_ch3_supplement_v10.environment import verify
    verify(c);local_sources(c)
    auth=read(auth_path);h=git('rev-parse','HEAD')
    if h!=auth['commit'] or git('rev-parse','@{u}')!=h or git('status','--porcelain'):raise PermissionError('clean bound Git required')
    if git('ls-remote','origin','refs/heads/'+git('branch','--show-current')).split()[0]!=h:raise PermissionError('remote mismatch')
    if (Path(c['result_root'])/c['attempt']).exists():raise FileExistsError('attempt already used')
    return c


def context(config_path,runtime_path):
    r=read(runtime_path);auth=load_ref(r['auth_ref']);c=permission(config_path,auth)
    if r['scope']!=SCOPE or r['config_ref']!=dict(path=str(config_path),sha256=sha(config_path)) or r['commit']!=auth['commit']:raise PermissionError('runtime binding')
    folder=Path(runtime_path).parent
    if folder.resolve()!=(Path(c['result_root'])/c['attempt']).resolve():raise PermissionError('attempt path')
    owner=read(folder/'owner.json')
    if owner['runtime_sha256']!=sha(runtime_path) or proc_start(owner['pid'])!=owner['start'] or os.getppid()!=owner['pid']:raise PermissionError('owned worker PID/start')
    if git('rev-parse','HEAD')!=r['commit'] or git('status','--porcelain'):raise PermissionError('local source state')
    from m6_ch3_supplement_v10.environment import verify
    verify(c);return c,folder


def claim(folder,config,job,runtime_path):
    key=evaluation_key(config,job)
    cap=dict(scope=SCOPE,purpose=PURPOSE,job=job['id'],evaluation_key=key,mode='authorized_frozen_test')
    write_new(Path(folder)/'claims'/(key+'.json'),dict(capability=cap,job=job,runtime_sha256=sha(runtime_path),new_test_access=True))
    return cap


def checked_receipt(folder,job,runtime_path):
    folder=Path(folder);r=read(folder/'receipts'/(job['id']+'.json'))
    if r['job']!=job or r['runtime_sha256']!=sha(runtime_path):raise ValueError('receipt identity')
    prefix=folder/('plots' if job['id']=='O' else job['id'])
    actual={str(p.relative_to(folder)) for p in prefix.rglob('*') if p.is_file()}
    if not actual or actual!=set(r['files']):raise ValueError('payload membership')
    for p,s in r['files'].items():
        target=(folder/p).resolve()
        if not target.is_relative_to(prefix.resolve()) or sha(target)!=s:raise ValueError('payload checksum/path')
    if job['id']!='O':
        x=r['result']
        if x['job']!=job or x['presentations']!=job['expected']['samples'] or x['calls']!=job['expected']['batches'] or x['elements']!=job['expected']['elements'] or not x['new_test_access'] or x['split']!='test' or any(x[k]!=0 for k in ('train','backward','optimizer')):raise ValueError('evaluation coverage')
        if read(prefix/'result.json')!=x:raise ValueError('sealed result/receipt differs')
        key=x['evaluation_key'];claim_record=read(folder/'claims'/(key+'.json'))
        expected_cap=dict(scope=SCOPE,purpose=PURPOSE,job=job['id'],evaluation_key=key,mode='authorized_frozen_test')
        if claim_record!=dict(capability=expected_cap,job=job,runtime_sha256=sha(runtime_path),new_test_access=True):raise ValueError('test claim/receipt identity')
    return r


def worker(config_path,runtime_path,job_id):
    c,folder=context(config_path,runtime_path)
    def stop():
        if (folder/'STOP').exists():raise InterruptedError('owned STOP')
    stop()
    job=next((j for j in c['jobs'] if j['id']==job_id),None)
    if job_id=='O':
        job=dict(id='O',phase='O')
        for j in c['jobs']:checked_receipt(folder,j,runtime_path)
        from .plots import completed
        result=completed(c,folder);prefix=folder/'plots'
    else:
        if job is None:raise PermissionError('unknown evaluation')
        cap=claim(folder,c,job,runtime_path)
        from .evaluate import run
        result=run(c,job,folder,cap,stop);prefix=folder/job_id
    stop();files={str(p.relative_to(folder)):sha(p) for p in sorted(prefix.rglob('*')) if p.is_file()}
    write_new(folder/'receipts'/(job_id+'.json'),dict(job=job,result=result,files=files,runtime_sha256=sha(runtime_path)))


def owned_stop(folder):
    folder=Path(folder);owner=read(folder/'owner.json');runtime=read(folder/'runtime.json');c=load_ref(runtime['config_ref'])
    if runtime['scope']!=SCOPE or folder.resolve()!=(Path(c['result_root'])/c['attempt']).resolve() or owner['runtime_sha256']!=sha(folder/'runtime.json') or proc_start(owner['pid'])!=owner['start']:raise PermissionError('owned STOP identity')
    if not (folder/'STOP').exists():write_new(folder/'STOP',dict(owner=owner,requested=time.time()))


def supervise(config_path,auth_path):
    c=preflight(config_path,auth_path);auth=read(auth_path);folder=Path(c['result_root'])/c['attempt']
    with open(c['global_lock'],'a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise PermissionError('visible GPU0 required')
        folder.mkdir(parents=True,exist_ok=False);child=None
        try:
            from m6_ch3_supplement_v10.gpu_identity import check_gpu_identity
            check_gpu_identity(c['gpu_uuid'],folder/'gpu-identity','supervisor-before-cuda',False)
            gpu=gpu_startup(auth)
            runtime_path=folder/'runtime.json'
            write_new(runtime_path,dict(scope=SCOPE,purpose=PURPOSE,commit=auth['commit'],config_ref=dict(path=str(config_path),sha256=sha(config_path)),auth_ref=dict(path=str(auth_path),sha256=sha(auth_path)),environment_ref=c['environment_ref'],gpu=gpu,waves=c['waves']))
            write_new(folder/'owner.json',dict(pid=os.getpid(),start=proc_start(os.getpid()),runtime_sha256=sha(runtime_path)))
            for name in ('logs','receipts','claims'):(folder/name).mkdir()
            # Metadata only: commit exact shared test week before ANY prediction.
            from .analysis import case_plan,preselect
            ledger=load_ref(c['sources_ref']);pre=load_ref(c['preselection_ref'])
            if preselect(load_ref(pre['source_ref']),pre['node_order'])!=pre['draw_indices']:raise ValueError('preselection not reproducible')
            plans=[case_plan(ledger['rows']['CT-UrbanEV-'+m],pre) for m in ('ModernTCN','TimeMixer','UV')]
            if plans[1:]!=[plans[0],plans[0]]:raise ValueError('common case week/sample identity')
            write_new(folder/'case-plan.json',plans[0])
            for i,wave in enumerate(c['waves'],1):
                if (folder/'STOP').exists():raise InterruptedError('STOP')
                j=wave[0]
                with (folder/'logs'/(j+'.log')).open('x') as log:
                    child=subprocess.Popen([sys.executable,'-B','-m','m6_ch3_test_reporting_v10.entry','worker','--config',str(config_path),'--runtime',str(runtime_path),'--job',j],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                write_new(folder/f'children-wave-{i}.json',[dict(job=j,pid=child.pid,start=proc_start(child.pid))])
                while child.poll() is None:time.sleep(1)
                if child.returncode:raise RuntimeError('technical failure; no later dispatch')
                job=dict(id='O',phase='O') if j=='O' else next(x for x in c['jobs'] if x['id']==j)
                checked_receipt(folder,job,runtime_path)
                write_new(folder/f'wave-{i}-complete.json',dict(jobs=wave))
            if (folder/'STOP').exists():raise InterruptedError('STOP before completion')
            expected={j['id'] for j in c['jobs']}|{'O'}
            if {p.stem for p in (folder/'receipts').glob('*.json')}!=expected or {p.stem for p in (folder/'claims').glob('*.json')}!={evaluation_key(c,j) for j in c['jobs']}:raise ValueError('complete coverage/claims')
            for j in c['jobs']+[dict(id='O',phase='O')]:checked_receipt(folder,j,runtime_path)
            write_new(folder/'complete.json',dict(technical_complete=True,model_evaluations=20,offline_jobs=1,waves=21,first_sensitivity_test=8,posthoc_repeated_test=12,center_references=2,case_presentations=5040,training=0,scientific_review='Pending; no main table replacement'))
        except BaseException:
            if child is not None and child.poll() is None:
                if not (folder/'STOP').exists():write_new(folder/'STOP',dict(reason='supervisor failed'))
                child.wait()
            write_new(folder/'failure.json',dict(traceback=traceback.format_exc()))
            raise


def status(folder):
    folder=Path(folder);rs=[read(p) for p in (folder/'receipts').glob('*.json')];active=[]
    for p in folder.glob('children-wave-*.json'):
        for r in read(p):
            try:
                if proc_start(r['pid'])==r['start']:active.append(r)
            except (FileNotFoundError,ProcessLookupError):pass
    return dict(completed={phase:sum(r['job']['phase']==phase for r in rs) for phase in ('S','P','C','O')},new_test_receipts=sum(r['job']['phase']!='O' for r in rs),center_references=2,waves_completed=len(list(folder.glob('wave-*-complete.json'))),active_owned_children=active,complete=(folder/'complete.json').exists(),failure=(folder/'failure.json').exists(),STOP=(folder/'STOP').exists())
