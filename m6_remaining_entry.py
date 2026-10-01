"""One tmux controller, synchronous existing guarded model-group entries."""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from utils.ch3_contract import read_profiles, validate_manifest
from utils import ch3_remaining as scope
from ch3_runner import dump

@contextlib.contextmanager
def lock(path):
    with Path(path).open('a+')as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def identity(pid):
    p=Path('/proc')/str(pid)/'stat'
    try:
        v=p.read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,start_ticks=v[19],state=v[0])
    except FileNotFoundError:return None

def same(v):
    now=identity(v['pid']) if v else None
    return bool(now and now['start_ticks']==v['start_ticks']and now['state']!='Z')

def stop_marker(root):
    # Atomic persistent marker first; dispatch checks both sides of spawn.
    with (root/'STOP').open('a'):pass

def signal_owned(v):
    if same(v):os.kill(v['pid'],signal.SIGTERM);return True
    return False

def command(g,child,report):
    script='m6_extension_entry.py' if g['model']in scope.MODELS[:4] else 'ch3_runner.py'
    return [scope.PYTHON,'-B',str(scope.ROOT/script),'start','--model',g['model'],'--approval',child,'--probe-report',report]

def sequence(p,root,commands,before,check,*,popen=subprocess.Popen,queue_id=None):
    """No successor until real child exit AND technical gate. Fixture-testable."""
    completed=[];child=None;owned=None;active=None
    def stop(sig,frame):
        stop_marker(root)
        signal_owned(owned)
    old=signal.signal(signal.SIGTERM,stop)
    try:
        for g in p['groups']:
            active=g['model']
            if (root/'STOP').exists():raise InterruptedError('persistent queue STOP')
            before(g)
            with lock(root/'dispatch.lock'):
                if (root/'STOP').exists():raise InterruptedError('STOP at group handoff')
                log=(root/(active+'.log')).open('x')
                try:child=popen(commands[active],cwd=scope.ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
                except BaseException:log.close();raise
                owned=identity(child.pid)
                if not owned:child.wait();log.close();raise RuntimeError('child exited before ownership registration')
                dump(root/'current.json',dict(model=active,child=owned,command=commands[active],log=str(root/(active+'.log'))))
                dump(root/'progress.json',dict(current_model=active,completed_models=completed.copy(),pending_models=[x['model']for x in p['groups']if x['model']not in completed and x['model']!=active],phase='running'))
            try:
                sent=False
                while True:
                    if (root/'STOP').exists() and not sent:
                        sent=signal_owned(owned)
                    try:code=child.wait(timeout=.2);break
                    except subprocess.TimeoutExpired:continue
            finally:log.close()
            child=None
            if code!=0:raise RuntimeError(active+' exited '+str(code))
            if same(owned):raise RuntimeError('child did not exit')
            if (root/'STOP').exists():raise InterruptedError('STOP after group exit')
            receipt=check(g)
            if receipt.get('status')!='technical-complete' or receipt.get('result_review')!='pending':raise ValueError('technical receipt missing or falsely reviewed')
            dump(root/(active+'-handoff.json'),receipt)
            with lock(root/'dispatch.lock'):
                if (root/'STOP').exists():raise InterruptedError('STOP during technical handoff')
                completed.append(active)
                dump(root/'progress.json',dict(current_model=None,completed_models=completed.copy(),pending_models=[x['model']for x in p['groups']if x['model']not in completed],phase='handoff-complete'))
        with lock(root/'dispatch.lock'):
            if (root/'STOP').exists():raise InterruptedError('STOP before final complete')
            dump(root/'complete.json',dict(queue_id=queue_id or scope.QUEUE,status='technical-complete',result_review='pending',task_ids=p['task_ids'],groups=completed,plan=scope.digest(p),handoffs={m:scope.ref(root/(m+'-handoff.json'))for m in completed}))
    except BaseException as exc:
        if child is not None and child.poll()is None:
            signal_owned(owned)
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:pass
        dump(root/'failure.json',dict(error=type(exc).__name__+': '+str(exc),current_model=active,completed_models=completed,pending_models=[x['model']for x in p['groups']if x['model']not in completed and x['model']!=active],child=owned,child_still_live=same(owned),automatic_retry=False))
        raise
    finally:signal.signal(signal.SIGTERM,old)

def status(c,root):
    control=scope.read(root/'controller.json')if (root/'controller.json').exists()else{}
    progress=scope.read(root/'progress.json')if (root/'progress.json').exists()else{}
    current=scope.read(root/'current.json')if (root/'current.json').exists()else{}
    m=current.get('model');group_progress={};actual=dict(adam=0,forward=0,backward=0);done=[]
    if m:
        path=scope.group_control(c,m)/'progress.json'
        if path.exists():group_progress=scope.read(path)
    for model in scope.MODELS:
        for run in scope.model_ids(c,model):
            d=scope.result_path(c,scope.task_by_id(c,run))
            if (d/'result.json').exists():done.append(run)
            if (d/'budget.json').exists():
                try:b=scope.read(d/'budget.json')
                except json.JSONDecodeError:continue
                for k in actual:actual[k]+=b['counts'].get(k,0)
    return dict(controller=control,running=same(control),current=current,progress=progress,current_wave=group_progress,completed_task_files=done,actual_budget=actual,STOP=(root/'STOP').exists(),failure=scope.read(root/'failure.json')if (root/'failure.json').exists()else None,complete=(root/'complete.json').exists(),logs=[str(root.parent/'remaining-models-v1-launcher.log')]+[str(root/(m+'.log'))for m in scope.MODELS])

def cli(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['dry-run','preflight','start','status','logs','complete','safe-stop']);parser.add_argument('--approval')
    args=parser.parse_args(argv);c=validate_manifest(read_profiles());root=scope.control_root(c)
    if args.action in ('status','logs'):
        print(json.dumps(status(c,root),ensure_ascii=False,indent=2));return 0
    if args.action=='safe-stop':
        if not (root/'controller.json').exists():raise ValueError('no owned queue controller')
        stop_marker(root);control=scope.read(root/'controller.json');sent=signal_owned(control)
        if not sent and (root/'current.json').exists():sent=signal_owned(scope.read(root/'current.json')['child'])
        print(json.dumps(dict(STOP=True,signal_sent=sent)));return 0
    a=scope.read(args.approval)if args.approval else None
    if args.action=='complete':
        if not (root/'complete.json').exists():print(json.dumps(dict(complete=False,status=status(c,root))));return 2
        if not a:a=scope.bound(scope.read(root/'controller.json')['approval'])
        p,children=scope.authorization(c,a,check_existing=False);v=scope.read(root/'complete.json')
        if v.get('groups')!=list(scope.MODELS)or v.get('task_ids')!=p['task_ids']or v.get('plan')!=scope.digest(p)or (root/'STOP').exists()or (root/'failure.json').exists()or same(scope.read(root/'controller.json')):raise ValueError('queue completion identity/lifecycle')
        for g in p['groups']:
            old=scope.bound(v['handoffs'][g['model']]);now=scope.technical_handoff(c,g,children[g['model']])
            if old!=now:raise ValueError('handoff not reproducible')
        print(json.dumps(dict(complete=True,result_review='pending',tasks=228)));return 0
    reasons=[];p=None
    try:
        p,children=scope.authorization(c,a)
        from ch3_runner import GPULock
        with GPULock(c):pass
        sys.path.insert(0,str(scope.ROOT/'tools/restricted_regression'))
        from m5_formal_entry import gpu_sample,resource_assessment
        if not resource_assessment(gpu_sample([]),[])['admission']:raise RuntimeError('resource admission unavailable')
    except (OSError,ValueError,KeyError,PermissionError,RuntimeError)as exc:reasons.append(str(exc))
    if args.action in ('dry-run','preflight'):
        if p is None:p=scope.plan(c,scope.read(scope.PARENT)['decisions'])
        print(json.dumps(dict(queue=scope.QUEUE,plan=p,blocked=reasons),ensure_ascii=False,indent=2));return 2 if args.action=='preflight'and reasons else 0
    if reasons:raise PermissionError('; '.join(reasons))
    root.parent.mkdir(parents=True,exist_ok=True)
    with lock(root.parent/'.remaining-models-v1.lock'):
        scope.reject_existing(c,p);root.mkdir(exist_ok=False)
        dump(root/'controller.json',dict(**identity(os.getpid()),queue_id=scope.QUEUE,approval=scope.ref(args.approval),commit=a['commit'],protocol_sha=a['protocol_sha'],plan_sha256=a['plan']['sha256']))
        commands={g['model']:command(g,a['children'][g['model']]['path'],a['resource_report']['path'])for g in p['groups']}
        def before(g):
            if scope.sha(args.approval)!=scope.read(root/'controller.json')['approval']['sha256']:raise ValueError('total permit changed')
            scope.authorization(c,a,check_existing=False)
            if scope.group_control(c,g['model']).exists()or any(Path(v).exists()for v in g['result_paths'].values()):raise FileExistsError('current group retained output')
            with GPULock(c):pass
        def check(g):
            scope.authorization(c,a,check_existing=False)
            return scope.technical_handoff(c,g,children[g['model']])
        try:sequence(p,root,commands,before,check)
        finally:
            # Preserve observed counters on success, failure and interruption;
            # never refund or infer zero for workers still finishing cleanup.
            snapshot=status(c,root)
            dump(root/'execution-accounting.json',dict(snapshot=snapshot,budget_refund=False,checkpoint_reads=0))
    return 0

if __name__=='__main__':sys.exit(cli())
