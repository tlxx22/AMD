"""Fixed train-all -> freeze-all -> test-once -> seal queue. No AUTH creation."""
import argparse
import importlib.metadata
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from .imports import sha
from .protocol import MODELS,validate_plan
from .dependencies import verify_closure,execution_seeds,execution_assets
from .runtime_io import read,write_new
from .runtime_io import proc_start
from . import lifecycle

ROOT=Path(__file__).resolve().parents[1]
SCOPE='m7-v10-spatial-216-v1'


def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True,timeout=30).strip()


def verify_files(binding):
    for path,expected in binding.items():
        if sha(path)!=expected:raise PermissionError('bound bytes changed: '+path)


def verify_environment(reference):
    if sha(reference['path'])!=reference['sha256']:raise PermissionError('environment identity changed')
    environment=read(reference['path'])
    # Do not resolve symlinks: the isolated venv interpreter shares a base binary.
    if os.path.abspath(sys.executable)!=environment['python'] or sys.version!=environment['version']:
        raise PermissionError('bound M7 interpreter required')
    for package,version in environment['libraries'].items():
        if importlib.metadata.version(package)!=version:raise PermissionError('environment package changed: '+package)
    verify_files(environment['core_files'])


def admission(config,auth_path,online=True):
    c=read(config);a=read(auth_path)
    if a.get('scope')!=SCOPE or a.get('review_approved') is not True or not a.get('commit'):raise PermissionError('post-review actual AUTH required')
    if c['preparation_blockers'] or not c.get('resource_frozen'):raise PermissionError('unclosed engineering/resource admission')
    if len(c['tasks'])!=216 or {t['model'] for t in c['tasks']}!=set(MODELS):raise ValueError('exact scope')
    validate_plan(c)
    for task in c['tasks']:
        meta_path=Path(task['data_path'])/'metadata.json';meta=read(meta_path)
        if sha(meta_path)!=task['metadata_sha256']:raise PermissionError('task metadata binding')
        if (task['N'],task['C'],task['node_order'],task['features'],task['endpoints'])!=(meta['nodes'],len(meta['features']),meta['node_order'],meta['features'],meta['endpoints']):raise ValueError('task data schema')
    if a['config_sha256']!=sha(config) or a['task_ids']!=[t['id'] for t in c['tasks']]:raise PermissionError('config/tasks binding')
    if git('rev-parse','HEAD')!=a['commit'] or git('status','--porcelain'):raise PermissionError('clean reviewed worktree required')
    if online:
        tracking=git('rev-parse','@{u}')
        remote=git('ls-remote','origin','refs/heads/'+git('branch','--show-current')).split()[0]
        if tracking!=a['commit'] or remote!=a['commit']:raise PermissionError('tracking/actual remote mismatch')
    source_root=ROOT.parent/'amd-execution-evidence/m7/M7-v10-spatial-implementation-r1/sources'
    verify_closure(c['source_binding'],execution_seeds(ROOT,source_root,read(ROOT/'ch4/temporal_sources.json')))
    for path,expected in execution_assets(ROOT,source_root,read(ROOT/'ch4/temporal_sources.json')).items():
        if c['source_binding'].get(path)!=expected:raise PermissionError('unbound/changed runtime asset: '+path)
    verify_files(c['source_binding']);verify_files(c['dependency_identity']['source_snapshots'])
    verify_environment(c['environment']);verify_files(a['resource_evidence'])
    if any(task['readiness_blockers'] for task in c['tasks']):raise PermissionError('task-level readiness remains blocked')
    if a['resource_plan']!=c['resource_plan']:raise PermissionError('reviewed waves/device/threads binding')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=a['resource_plan']['visible_devices']:raise PermissionError('legal GPU binding')
    ordered=[i for wave in a['resource_plan']['waves'] for i in wave]
    if ordered!=a['task_ids']:raise PermissionError('predeclared order/coverage')
    return c,a


def freeze_best(c,folder):
    folder=Path(folder);rows=[]
    for task in c['tasks']:
        result=read(folder/task['id']/'train-result.json')
        path=folder/task['id']/'best.pt'
        if result['task_id']!=task['id'] or not result['training_complete'] or result['test_accesses']!=0:raise ValueError('all training must complete before test')
        if sha(path)!=result['best_sha256']:raise ValueError('best checksum')
        rows.append(dict(task_id=task['id'],best_sha256=result['best_sha256'],training_binding=result['binding']))
    write_new(folder/'frozen-best.json',dict(tasks=rows,config_sha256=sha(folder/'config.json')))
    return rows


def claim_test(folder,task_id):
    folder=Path(folder);frozen=read(folder/'frozen-best.json')
    if frozen['config_sha256']!=sha(folder/'config.json'):raise PermissionError('frozen config changed')
    row=next(r for r in frozen['tasks'] if r['task_id']==task_id)
    # Existing claim is ambiguous/in progress/completed: never evaluate again.
    write_new(folder/'claims'/(task_id+'.json'),dict(task_id=task_id,freeze_sha256=sha(folder/'frozen-best.json'),created=time.time()))
    return row


def prepare_final_domain(task,folder):
    from .prepare_data import urban,charged,pems,EVIDENCE
    destination=Path(folder)/'test-data'/task['domain']
    if destination.exists():
        if not (destination/'complete.json').exists():raise RuntimeError('incomplete test data preparation; preserve and review')
        verify_files(read(destination/'complete.json')['files']);return destination
    if task['domain']=='UrbanEV':meta=urban(destination,_final=True)
    elif task['domain'].startswith('PEMS'):meta=pems(destination,task['domain'],_final=True)
    else:meta=charged(destination,task['domain'],EVIDENCE/'data/CHARGED-Hourly-r2.zip',_final=True)
    before=read(Path(task['data_path'])/'metadata.json')
    for key in ['node_order','features','endpoints']:
        if meta[key]!=before[key]:raise ValueError('test data identity differs from frozen training')
    for name in ('geo.npy','sem.npy','target_scale.npy'):
        if sha(destination/name)!=before['files'][name]:raise ValueError('train-fitted graph/scaler changed')
    write_new(destination/'complete.json',dict(files={str(p):sha(p) for p in destination.iterdir() if p.is_file()}))
    return destination


def worker(folder,name,phase):
    try:return _worker(folder,name,phase)
    except BaseException as error:
        lifecycle.record_exception(folder,name,phase,error)
        raise


def _worker(folder,name,phase):
    folder=Path(folder);runtime=read(folder/'runtime.json');owner=read(folder/'owner.json');c=read(folder/'config.json')
    if owner['pid']!=os.getppid() or proc_start(owner['pid'])!=owner['start']:raise PermissionError('owned live parent')
    if owner['runtime_sha256']!=sha(folder/'runtime.json') or runtime['config_sha256']!=sha(folder/'config.json'):raise PermissionError('runtime changed')
    verify_files(c['source_binding'])
    task=next(t for t in c['tasks'] if t['id']==name);out=folder/name
    def stop():
        if (folder/'STOP').exists():raise InterruptedError('owned STOP')
    stop()
    if phase=='train':
        out.mkdir(exist_ok=False);binding=dict(task_id=name,config_sha256=sha(folder/'config.json'),commit=runtime['commit'])
        final=None
    else:
        binding=claim_test(folder,name);final=prepare_final_domain(task,folder)
    from .worker import run
    result=run(task,out,phase,stop,binding,test_folder=final)
    write_new(folder/'receipts'/(name+'-'+phase+'.json'),dict(phase=phase,task_id=name,result=result,
              files={str(p.relative_to(folder)):sha(p) for p in out.rglob('*') if p.is_file()}))


def supervise(config,auth,folder):
    c,a=admission(config,auth);folder=Path(folder)
    if not folder.resolve().is_relative_to(ROOT.parent/'amd-execution-evidence/m7'):raise PermissionError('M7 evidence scope')
    with open(a['resource_plan']['global_lock'],'a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        folder.mkdir(parents=True,exist_ok=False);write_new(folder/'config.json',c)
        write_new(folder/'runtime.json',dict(scope=SCOPE,commit=a['commit'],config_sha256=sha(folder/'config.json'),
                   auth_path=str(Path(auth).resolve()),auth_sha256=sha(auth),session=os.environ.get('TMUX'),environment=c['environment']))
        write_new(folder/'owner.json',dict(pid=os.getpid(),start=proc_start(os.getpid()),runtime_sha256=sha(folder/'runtime.json')))
        for name in ('logs','claims','receipts'):(folder/name).mkdir()
        children=[];launched=[];failure_receipts=[]
        supervisor_started_at=lifecycle.stamp();supervisor_start=time.monotonic()
        def finalize(entry):
            if entry.get('finalized'):return
            child=entry['child']
            if child.poll() is None:return
            reference=lifecycle.finish(folder,entry['record'],child.returncode)
            entry['finalized']=True;entry['failure_receipt']=reference
            if reference:failure_receipts.append(reference)
        try:
            for phase,waves in [('train',a['resource_plan']['waves']),('test',[[t['id']] for t in c['tasks']])]:
                if phase=='test':freeze_best(c,folder)
                for wave_no,wave in enumerate(waves):
                    if (folder/'STOP').exists():raise InterruptedError('STOP before wave')
                    children=[]
                    wave_entries=[]
                    for name in wave:
                        log=(folder/'logs'/(name+'-'+phase+'.log')).open('x')
                        record=lifecycle.begin(folder,name,phase)
                        try:child=subprocess.Popen([sys.executable,'-B','-m','ch4.queue','worker','--folder',str(folder),'--task',name,'--phase',phase],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                        finally:log.close()
                        children.append(child);entry=dict(child=child,record=record);launched.append(entry);wave_entries.append(entry)
                        lifecycle.attach(folder,record,child)
                    write_new(folder/f'{phase}-wave-{wave_no}-children.json',[dict(task=e['record']['task_id'],pid=e['child'].pid,start=e['record']['proc_start']) for e in wave_entries])
                    while True:
                        for entry in wave_entries:finalize(entry)
                        if any(e.get('failure_receipt') for e in wave_entries):
                            if not (folder/'STOP').exists():write_new(folder/'STOP',dict(reason='technical failure'))
                        if all(e.get('finalized') for e in wave_entries):break
                        time.sleep(.5)
                    if any(e.get('failure_receipt') for e in wave_entries):raise RuntimeError('technical failure; later scheduling stopped')
                    for name in wave:
                        receipt=read(folder/'receipts'/(name+'-'+phase+'.json'))
                        verify_files({str(folder/p):h for p,h in receipt['files'].items()})
                    write_new(folder/f'{phase}-wave-{wave_no}-complete.json',dict(tasks=wave))
            from .reporting import seal
            seal(c,folder)
        except BaseException as error:
            failure_traceback=traceback.format_exc()
            if not (folder/'STOP').exists():write_new(folder/'STOP',dict(reason='supervisor failure'))
            for entry in launched:
                if entry['child'].poll() is None:entry['child'].wait()
                finalize(entry)
            write_new(folder/'failure.json',dict(supervisor_started_at=supervisor_started_at,
                      supervisor_ended_at=lifecycle.stamp(),supervisor_elapsed_seconds=time.monotonic()-supervisor_start,
                      failure_type=type(error).__module__+'.'+type(error).__qualname__,failure_message=str(error),
                      traceback=failure_traceback,failure_receipts=failure_receipts));raise


def owned_stop(folder):
    folder=Path(folder);owner=read(folder/'owner.json')
    if proc_start(owner['pid'])!=owner['start'] or owner['runtime_sha256']!=sha(folder/'runtime.json'):raise PermissionError('STOP owner mismatch')
    if not (folder/'STOP').exists():write_new(folder/'STOP',dict(owner=owner,requested=time.time()))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['preflight','start','worker','stop','status'])
    for key in ('config','auth','folder','task','phase'):p.add_argument('--'+key)
    a=p.parse_args()
    if a.command=='preflight':admission(a.config,a.auth)
    elif a.command=='start':supervise(a.config,a.auth,a.folder)
    elif a.command=='worker':worker(a.folder,a.task,a.phase)
    elif a.command=='stop':owned_stop(a.folder)
    else:
        folder=Path(a.folder)
        print(json.dumps(dict(receipts=len(list((folder/'receipts').glob('*.json'))),complete=(folder/'complete.json').exists(),failure=(folder/'failure.json').exists(),stop=(folder/'STOP').exists())))


if __name__=='__main__':main()
