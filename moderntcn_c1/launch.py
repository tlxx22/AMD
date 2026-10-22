"""Persistent owned 27→53 chain; no torch import or GPU query while waiting."""
import fcntl
import importlib.metadata
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from .contract import *
from .upstream import inspect,verify_completion,live,proc_identity,Blocked

CONTROL=EVIDENCE/'chain-control'
SESSION='m6-moderntcn-C1-round2-v1'
AUTH_PATH=EVIDENCE/'authorization/AUTH.json'

def environment():
    return dict(python=str(PYTHON),version=sys.version.split()[0],packages={n:importlib.metadata.version(n)for n in ('torch','numpy','pandas','scikit-learn')})

def preflight(config_path,auth,network=True,fresh=True):
    c=validate(read(config_path))
    import socket
    if socket.gethostname()!='b82673a648b4' or auth.get('hostname')!=socket.gethostname():raise PermissionError('C1 host identity')
    if auth.get('comparison_policy')!=POLICY:raise PermissionError('C1 comparison policy')
    if Path(config_path)!=CONFIG or auth.get('schema')!='moderntcn-C1-launch-v1' or auth.get('review_approved')is not True or auth.get('revision')!=ID:raise PermissionError('C1 explicit reviewed scope')
    if auth.get('resume_review_approved')is not False or auth.get('resume_task_ids')!=[]or auth.get('previous_runtime_ref')is not None:raise PermissionError('C1 fresh only')
    if auth['config_sha256']!=sha(CONFIG)or auth['source_binding']!=source_binding()or auth['task_ids']!=[t['id']for t in c['tasks']]or auth['waves']!=c['waves']or auth['upstream']!=UPSTREAM:raise PermissionError('C1 config/source/upstream/task binding')
    if auth['task_profile_bindings']!={t['id']:digest(profile(c,t))for t in c['tasks']}or auth['budget']!=dict(runs=53,run_epochs=860,optimizer_updates=2164920):raise PermissionError('C1 profiles/budget')
    if auth['result_root']!=str(RESULT_ROOT)or auth['attempt']!=ATTEMPT or auth['environment']!=environment():raise PermissionError('C1 paths/environment')
    if git('rev-parse','HEAD')!=auth['commit']or git('branch','--show-current')!=BRANCH or git('status','--porcelain','--untracked-files=all'):raise PermissionError('C1 clean actual commit')
    if network:
        if git('rev-parse','@{u}')!=auth['commit']or git('ls-remote','origin','refs/heads/'+BRANCH).split()[0]!=auth['commit']or git('rev-list','--left-right','--count','HEAD...@{u}')!='0\t0':raise PermissionError('C1 three ends')
    resource=auth['resource_plan']
    if resource['q_max']!=4 or resource['mandatory_probe']is not False or resource['waves']!=c['waves']or not resource['reviewed_evidence_refs']:raise PermissionError('C1 resource contract')
    for r in resource['reviewed_evidence_refs']:
        if sha(r['path'])!=r['sha256']:raise PermissionError('C1 resource reference changed')
    if auth['gpu_management_query']!=dict(timeout_seconds=60,max_attempts=3,retry_only='TimeoutExpired'):raise PermissionError('C1 GPU management policy')
    if auth['gpu_uuid']!=bound(UPSTREAM['runtime'])['gpu_identity']['uuid']:raise PermissionError('C1 reviewed device binding')
    if sha(MAPPING)!=MAPPING_SHA:raise PermissionError('C1 final round2 catalog changed')
    parents={r['path']:bound(r)for r in c['parent_protocols']}
    for t in c['tasks']:
        r=c['parents'][t['id']];pc=parents[r['protocol']['path']]
        if pc['resolved_profiles'][r['task']['id']]!=r['old_profile']or pc['datasets'][t['dataset']]!=c['datasets'][t['dataset']]:raise PermissionError('C1 parent scientific provenance')
        m=bound(r['manifest']);result=bound(r['result'])
        if m['profile']!=r['old_profile']or digest(m['metadata'])!=r['data_sha']or result['id']!=r['old_cell']['run_id']:raise PermissionError('C1 final parent result identity')
    for p,v in c['data_stat'].items():
        if {k:getattr(Path(p).stat(),k)for k in v}!=v:raise PermissionError('C1 data stat changed')
    src=c['sources']['ModernTCN']
    if subprocess.check_output(['git','-C',src['repository'],'rev-parse','HEAD'],text=True,timeout=30).strip()!=src['commit']:raise PermissionError('C1 author commit')
    for p,h in src['files'].items():
        if sha(p)!=h:raise PermissionError('C1 author source')
    if fresh:
        if (RESULT_ROOT/ATTEMPT).exists():raise FileExistsError('C1 actual attempt already exists')
        if (RESULT_ROOT/'registry').exists()and any((RESULT_ROOT/'registry'/t['id']).exists()for t in c['tasks']):raise FileExistsError('C1 success identity already exists')
    return dict(passed=True,commit=auth['commit'],task_count=53,waves=14,budget=auth['budget'],training_started=False,GPU_query=False,upstream_state=inspect(UPSTREAM)['state'])

def gpu_startup(auth):
    # Exactly one startup management check after technical completion and mutex acquisition.
    for n in range(3):
        try:
            output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.free','--format=csv,noheader,nounits'],text=True,timeout=60)
            row=next(x.split(',')for x in output.splitlines()if x.split(',')[0].strip()=='0');idx,uid,total,free=[x.strip()for x in row]
            if uid!=auth['gpu_uuid']or float(free)<max(8192,float(total)*.1):raise PermissionError('C1 GPU UUID/free memory')
            return dict(index=idx,uuid=uid,total_MiB=float(total),free_MiB=float(free))
        except subprocess.TimeoutExpired:
            if n==2:raise
            time.sleep(2)

def acquire_mutex(path):
    # Existing inode, no creation/truncation/replacement; locks are held by this chain only.
    f=open(path,'r+')
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:f.close();return None
    return f

def state(name,**extra):
    row=dict(state=name,time=time.time(),**extra);atomic(CONTROL/'status.json',row)
    print(json.dumps(row,ensure_ascii=False,sort_keys=True),flush=True)
def stop_requested():return (CONTROL/'STOP').exists()
def pause():
    for _ in range(45):
        if stop_requested():raise InterruptedError('owned C1 STOP')
        time.sleep(1)

def cleanup(children):
    for child in children:
        if child['process'].poll()is None and live(child):os.killpg(child['pid'],signal.SIGTERM)
    for child in children:
        try:child['process'].wait(timeout=15)
        except subprocess.TimeoutExpired:
            if live(child):os.killpg(child['pid'],signal.SIGKILL)
            child['process'].wait()

def chain():
    auth=read(AUTH_PATH);preflight(CONFIG,auth)
    CONTROL.mkdir()  # exclusive single arm; retained chain never silently resumed
    owner=dict(pid=os.getpid(),start=proc_identity(os.getpid())['start'],worktree=str(ROOT),auth_ref=ref(AUTH_PATH))
    write_new(CONTROL/'owner.json',owner);children=[];lock=None
    def interrupted(signum,frame):raise InterruptedError('C1 owned signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        while True:
            if stop_requested():raise InterruptedError('C1 STOP before upstream gate')
            status=inspect(UPSTREAM);state(status['state'],upstream=status)
            if status['ready']:break
            pause()
        proof=verify_completion(UPSTREAM);write_new(CONTROL/'upstream-completion.json',proof)
        while lock is None:
            if stop_requested():raise InterruptedError('C1 STOP before mutex')
            # Drift/STOP after completion also blocks this chain before resource acquisition.
            if not inspect(UPSTREAM)['ready']:raise Blocked('upstream changed after proof')
            state('WAIT_MUTEX_AVAILABLE');lock=acquire_mutex(LOCK)
            if lock is None:pause()
        state('FINAL_LIGHT_PREFLIGHT');preflight(CONFIG,auth);startup=gpu_startup(auth)
        if stop_requested():raise InterruptedError('C1 STOP before dispatch')
        c=read(CONFIG);folder=RESULT_ROOT/ATTEMPT;folder.mkdir(parents=True)
        for n in ('runs','staging','logs','receipts'): (folder/n).mkdir()
        write_new(folder/'owner.json',owner)
        runtime=dict(schema='moderntcn-C1-runtime-v1',revision=ID,commit=auth['commit'],owner=owner,authorization_ref=ref(AUTH_PATH),config_sha256=sha(CONFIG),source_binding=c['source_binding'],task_ids=auth['task_ids'],waves=c['waves'],gpu_identity=startup,upstream_completion_ref=ref(CONTROL/'upstream-completion.json'))
        write_new(folder/'runtime.json',runtime);runtime_ref=ref(folder/'runtime.json');completed=[]
        for wi,wave in enumerate(c['waves'],1):
            if stop_requested():raise InterruptedError('C1 STOP before next wave')
            state('RUN_53_FIXED_TASKS',wave=wi,task_ids=wave,completed=len(completed));children=[]
            for tid in wave:
                if stop_requested():raise InterruptedError('C1 STOP during dispatch')
                logfile=(folder/'logs'/(tid+'.log')).open('x')
                cmd=[str(PYTHON),'-B',str(ROOT/'moderntcn_c1_entry.py'),'worker','--task-id',tid,'--runtime',runtime_ref['path'],'--runtime-sha',runtime_ref['sha256']]
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4');env.pop('AMD_RR_CONFIG',None);env.pop('AMD_RR_CONFIG_SHA256',None);env.pop('PYTHONPATH',None)
                p=subprocess.Popen(cmd,cwd=ROOT,stdout=logfile,stderr=subprocess.STDOUT,env=env,start_new_session=True);logfile.close()
                ch=dict(process=p,pid=p.pid,start=proc_identity(p.pid)['start'],task_id=tid);children.append(ch)
                atomic(folder/'children.json',[{k:v for k,v in x.items()if k!='process'}for x in children])
            while any(x['process'].poll()is None for x in children):
                if stop_requested()or any(x['process'].poll()not in (None,0)for x in children):raise InterruptedError('C1 STOP/worker failure; no next wave')
                time.sleep(1)
            from .artifacts import audit
            for ch in children:
                if ch['process'].returncode!=0:raise RuntimeError('C1 failed child')
                t=next(t for t in c['tasks']if t['id']==ch['task_id']);r=read(folder/'receipts'/(t['id']+'.json'));actual=audit(folder/'runs'/t['id'],c,t,auth['commit'])
                if {k:r[k]for k in actual}!=actual or r['seal_ref']!=ref(folder/'runs'/t['id']/'seal.json'):raise ValueError('C1 child receipt/source mismatch')
                completed.append(t['id'])
            write_new(folder/('wave-'+str(wi)+'-complete.json'),dict(task_ids=wave,children=[{k:v for k,v in x.items()if k!='process'}for x in children]))
            children=[]
        if completed!=auth['task_ids']or list((folder/'staging').iterdir()):raise ValueError('C1 exact final coverage/staging')
        state('TECHNICAL_AUDIT_AND_SUMMARY');from .summary import summarize
        summary=summarize(c,auth,folder);write_new(folder/'summary.json',summary)
        write_new(folder/'complete.json',dict(technical_complete=True,task_ids=completed,waves=14,summary_ref=ref(folder/'summary.json'),result_review='Pending',effect='see saved comparisons'))
        state('COMPLETE',tasks=53,summary_ref=ref(folder/'summary.json'))
    except BaseException as exc:
        cleanup(children)
        state('STOPPED' if stop_requested()or isinstance(exc,InterruptedError)and 'signal' in str(exc)else 'BLOCKED',error=repr(exc),automatic_retry=False)
        write_new(CONTROL/'failure-or-stop.json',dict(error=repr(exc),time=time.time(),next_wave_dispatched=False))
        raise
    finally:
        if lock is not None:lock.close()
        write_new(CONTROL/'exit.json',dict(pid=os.getpid(),start=owner['start'],time=time.time(),own_children_reaped=not any(live(ch)for ch in children),own_mutex_released=True))

def arm():
    preflight(CONFIG,read(AUTH_PATH))
    if CONTROL.exists():raise FileExistsError('C1 chain already armed/retained; no implicit retry')
    subprocess.run(['tmux','has-session','-t','='+SESSION],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
    import shlex
    log=EVIDENCE/'launcher.log'
    if log.exists():raise FileExistsError('existing C1 launcher history')
    command='exec '+shlex.join([str(PYTHON),'-B',str(ROOT/'moderntcn_c1_entry.py'),'chain'])+' > '+shlex.quote(str(log))+' 2>&1'
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='');env.pop('AMD_RR_CONFIG',None);env.pop('PYTHONPATH',None)
    subprocess.run(['tmux','new-session','-d','-s',SESSION,'-c',str(ROOT),command],env=env,check=True)
    return dict(armed=True,session=SESSION,log=str(log),note='waiting does not own GPU mutex; no model until upstream verified')

def stop():
    owner=read(CONTROL/'owner.json')
    if owner['worktree']!=str(ROOT)or owner['auth_ref']!=ref(AUTH_PATH)or not live(owner):raise PermissionError('C1 expired/foreign owner')
    cmd=Path(f"/proc/{owner['pid']}/cmdline").read_bytes()
    if str(ROOT/'moderntcn_c1_entry.py').encode()not in cmd or b'chain'not in cmd:raise PermissionError('C1 process command mismatch')
    (CONTROL/'STOP').touch(exist_ok=True)
    return dict(stop_requested=True,scope='C1 chain only; upstream untouched')

def status():
    value=dict(armed=CONTROL.exists(),prepared=True)
    for n in ('status.json','owner.json','exit.json','failure-or-stop.json'):
        if (CONTROL/n).exists():value[n]=read(CONTROL/n)
    if 'owner.json'in value:value['owner_alive']=live(value['owner.json'])
    folder=RESULT_ROOT/ATTEMPT
    if folder.exists():
        value['completed']=len(list((folder/'receipts').glob('*.json')));value['children']=read(folder/'children.json')if (folder/'children.json').exists()else []
    return value
