"""M-only handoff: waits for explicit permits; never signs or edits them."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
from utils import ch3_m_tasks as s
from utils import ch3_m_execution as e
from utils import ch3_m_launch as launch
from utils.ch3_contract import digest
from ch3_runner import code_binding,environment_binding,git,dump
from m6_remaining_entry import identity,same,lock,stop_marker
ROOT=s.PACKAGE/'handoff-execution-v1'
STATES=('WAIT_OLD_QUEUE','OLD_BOUNDARY_SEALED','WAIT_PROBE_APPROVAL','PROBE_RUNNING','WAIT_FORMAL_REVIEW','FORMAL_RUNNING','COMPLETE')
POLL_SECONDS=60
OLD_QUEUE_START_ANCHOR={
 'path':'/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/production-start.json',
 'sha256':'829215cc1fd95dcdf9b6dbdf0e2d44f7974224784a49082ca9dd38cfc0f87ecd',
}

def anchor(c):
 start=s.bound(OLD_QUEUE_START_ANCHOR)
 row=start['queue']['controller.json']
 if start['commit']!=s.BASE or row['content']['queue_id']!='m6-remaining-models-v1':raise ValueError('old queue anchor identity')
 return row

def old_queue_state(c):
 root=Path(c['m_experiment']['old_queue'])
 if(root/'failure.json').exists()or(root/'STOP').exists():raise ValueError('old MS queue failure/STOP; no M handoff')
 row=anchor(c);p=root/'controller.json'
 if s.sha(p)!=row['sha256']or json.loads(p.read_text())!=row['content']or s.sha(p)!=row['sha256']:raise ValueError('old MS controller identity changed')
 if(root/'complete.json').exists():
  # Existing complete scope, handoffs and controller-exit gate; no model computation.
  e.old_boundary(c);return 'complete'
 if not same(row['content']):raise ValueError('old MS controller disappeared without complete')
 return 'wait'

def seal_boundary(c):
 b=e.old_boundary(c);p=s.PACKAGE/'old-queue-technical-boundary.json'
 if p.exists()or p.is_symlink():
  ref=s.ref(p)
  expected=hashlib.sha256((json.dumps(b,ensure_ascii=False,indent=2)+'\n').encode()).hexdigest()
  if ref['sha256']!=expected or s.bound(ref)!=b:raise ValueError('existing old boundary content/SHA conflict')
  return ref
 launch.exclusive_json(p,b)
 ref=s.ref(p)
 if s.bound(ref)!=b:raise ValueError('old boundary changed during exclusive seal')
 return ref

def static_binding(c):
 return dict(commit=git('rev-parse','HEAD'),protocol_sha=digest(c),code=code_binding(),environment=environment_binding())

def static_reasons(c):
 reasons=[]
 if git('branch','--show-current')!='m6/m-baselines-v1'or git('status','--porcelain','--untracked-files=all')or git('rev-parse','HEAD')==s.BASE:reasons.append('independent reviewed clean M closure required before handoff')
 return reasons

def preflight(c,*,starting=False,token=None):
 reasons=static_reasons(c);x=launch.spec('handoff');record=None
 if ROOT.exists()or ROOT.is_symlink()or launch.claim_path('handoff').exists():reasons.append('retained handoff controller/complete/failure; fresh repeat forbidden')
 if starting:
  try:record=launch.verify(c,'handoff',token)
  except(OSError,ValueError,PermissionError)as exc:reasons.append(str(exc))
 if (x['log'].exists()or x['log'].is_symlink()or launch.metadata_path('handoff').exists())and not record:reasons.append('retained handoff launcher; matching current identity required')
 try:
  if subprocess.run(['tmux','has-session','-t',x['session']],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0 and not record:reasons.append('retained handoff tmux session')
  old_queue_state(c)
 except(OSError,ValueError,RuntimeError,subprocess.CalledProcessError)as exc:reasons.append(str(exc))
 return list(dict.fromkeys(reasons))

def permit(c,probe):
 p=s.PACKAGE/('probe-review.json'if probe else'formal-review.json')
 if not p.exists():return None
 ref=s.ref(p);a=s.bound(ref)
 from m6_m_entry import readiness
 reasons=readiness(c,a,str(p),probe)
 if reasons:raise PermissionError('; '.join(reasons))
 if a['old_boundary']!=s.ref(s.PACKAGE/'old-queue-technical-boundary.json'):raise ValueError('handoff exact sealed boundary binding')
 if not probe:
  report=s.bound(a['resource_report'])
  if report['review']['original_report']!=s.ref(e.PROBE_ROOT/'complete.json'):raise ValueError('formal permission must review this actual M probe completion')
 return dict(path=str(p),sha256=ref['sha256'],value=a)

def signal_child(v,owner=None):
 if not v or not same(v):return False
 expected=owner or identity(os.getpid());declared=v.get('handoff_owner',{})
 if not expected or any(declared.get(k)!=expected.get(k)for k in ('pid','start_ticks')):raise ValueError('M child is not owned by this handoff identity')
 if v.get('m_kind')not in ('probe','formal'):raise ValueError('M child handoff scope absent')
 if same(expected):
  fields=(Path('/proc')/str(v['pid'])/'stat').read_text().rsplit(')',1)[1].split()
  if int(fields[1])!=expected['pid']:raise ValueError('M child parent identity mismatch')
 argv=(Path('/proc')/str(v['pid'])/'cmdline').read_bytes().decode().split('\0')
 if ('--probe'in argv)!=(v['m_kind']=='probe'):raise ValueError('M child probe/formal stop scope mismatch')
 from m6_m_entry import signal_M_owned
 return signal_M_owned(v)

def signal_supervisor(v):
 if not same(v):return False
 if not launch.command_has(v['pid'],s.ROOT/'m6_m_handoff_entry.py'):raise ValueError('handoff PID is not this owned supervisor')
 os.kill(v['pid'],signal.SIGTERM);return True

def run_owned(c,root,probe,bound_permit):
 """One synchronous child, same guarded start gate and exclusive launcher identity."""
 kind='probe'if probe else'formal';p=Path(bound_permit['path']);x=launch.spec(kind)
 if s.sha(p)!=bound_permit['sha256']:raise ValueError('permit changed before dispatch')
 if(root/'STOP').exists():raise InterruptedError('handoff STOP before child')
 # Same exclusive creation as shell noclobber; no stale file is removed.
 if x['log'].parent.resolve()!=x['log'].parent:raise ValueError('M owned launcher parent path changed')
 x['log'].parent.mkdir(parents=True,exist_ok=True)
 fd=os.open(str(x['log']),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
 token=launch.prepare(c,kind,os.getpid(),driver='handoff',approval=p)
 args=[c['execution']['python'],'-B',str(s.ROOT/'m6_m_entry.py'),'start','--approval',str(p)]+(['--probe']if probe else[])
 child=None;owned=None
 try:
  with x['log'].open('a')as log:
   child=subprocess.Popen(args,cwd=s.ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1',launch.TOKEN_ENV:token})
   child_identity=identity(child.pid)
   if not child_identity:child.wait();raise RuntimeError('M child exited before identity registration')
   owned=dict(child_identity,handoff_owner=identity(os.getpid()),m_kind=kind)
   dump(root/'current.json',dict(kind=kind,child=owned,command=args,permit=dict(path=str(p),sha256=bound_permit['sha256']),log=str(x['log'])))
   sent=False
   while True:
    if(root/'STOP').exists()and not sent:sent=signal_child(owned)
    try:code=child.wait(timeout=.2);break
    except subprocess.TimeoutExpired:continue
  if code!=0 or same(owned):raise RuntimeError(kind+' child failed/not exited: '+str(code))
  if(root/'STOP').exists():raise InterruptedError('handoff STOP after child')
  if s.sha(p)!=bound_permit['sha256']:raise ValueError('permit changed during child')
  from m6_m_entry import verify_completion
  report=verify_completion(c,e.PROBE_ROOT if probe else e.CONTROL,bound_permit['value'],probe)
  return dict(kind=kind,permit=dict(path=str(p),sha256=bound_permit['sha256']),child=owned,exit_code=code,complete=s.ref((e.PROBE_ROOT if probe else e.CONTROL)/'complete.json'),reviewed=False,technical_complete=True)
 except BaseException:
  if child is not None and child.poll()is None:
   signal_child(owned)
   try:child.wait(timeout=60)
   except subprocess.TimeoutExpired:pass
  raise

def run(c,root=ROOT,*,sleep=time.sleep,interval=POLL_SECONDS):
 base=static_binding(c);state='WAIT_OLD_QUEUE';receipts={};boundary=None
 def update(next_state):
  nonlocal state
  state=next_state;dump(root/'progress.json',dict(state=state,receipts=receipts,boundary=boundary,result_review='pending',permit_generation=False))
  with(root/'states.jsonl').open('a')as f:f.write(json.dumps(dict(state=state),ensure_ascii=False)+'\n')
 def interrupted(sig,frame):
  stop_marker(root)
  p=root/'current.json'
  if not p.exists()or not same(json.loads(p.read_text()).get('child')):raise InterruptedError('handoff stopped while waiting')
 previous=signal.signal(signal.SIGTERM,interrupted)
 try:
  update(state)
  while state!='COMPLETE':
   if(root/'STOP').exists():raise InterruptedError('persistent handoff STOP')
   if static_binding(c)!=base:raise ValueError('handoff source/config/commit/environment changed')
   if state=='WAIT_OLD_QUEUE':
    if old_queue_state(c)=='wait':sleep(interval);continue
    boundary=seal_boundary(c);update('OLD_BOUNDARY_SEALED');update('WAIT_PROBE_APPROVAL')
   elif state in ('WAIT_PROBE_APPROVAL','WAIT_FORMAL_REVIEW'):
    probe=state=='WAIT_PROBE_APPROVAL';a=permit(c,probe)
    if a is None:sleep(interval);continue
    update('PROBE_RUNNING'if probe else'FORMAL_RUNNING')
    receipt=run_owned(c,root,probe,a);receipts['probe'if probe else'formal']=receipt
    launch.exclusive_json(root/('probe-handoff.json'if probe else'formal-handoff.json'),receipt)
    update('WAIT_FORMAL_REVIEW'if probe else'COMPLETE')
   else:raise ValueError('unrecognized handoff state')
  dump(root/'complete.json',dict(scope=s.ID+'-handoff',task_ids=[t['id']for t in c['tasks']],technical_complete=True,result_review='pending',receipts=receipts,boundary=boundary,**base))
 except BaseException as exc:
  dump(root/'failure.json',dict(state=state,error=type(exc).__name__+': '+str(exc),receipts=receipts,boundary=boundary,automatic_retry=False,result_review='pending'));raise
 finally:signal.signal(signal.SIGTERM,previous)

def start(c,token):
 reasons=preflight(c,starting=True,token=token)
 if reasons:raise PermissionError('; '.join(reasons))
 with lock(s.PACKAGE/'.m-handoff-v1.lock'):
  if ROOT.exists():raise FileExistsError('duplicate handoff')
  launch.claim(c,'handoff',token);ROOT.mkdir(exist_ok=False)
  dump(ROOT/'controller.json',dict(**identity(os.getpid()),scope=s.ID+'-handoff',**static_binding(c)))
  run(c)

def status():
 value=dict(output=str(ROOT),running=False,prepared=True,start_called=ROOT.exists(),complete=(ROOT/'complete.json').exists(),STOP=(ROOT/'STOP').exists())
 for n in ('controller','current','progress','failure'):
  p=ROOT/(n+'.json')
  if p.exists():value[n]=json.loads(p.read_text())
 if value.get('controller'):value['running']=same(value['controller'])
 return value

def safe_stop():
 p=ROOT/'controller.json'
 if not p.exists():raise ValueError('no owned handoff controller')
 stop_marker(ROOT);control=json.loads(p.read_text());sent=signal_supervisor(control)
 if not sent and(ROOT/'current.json').exists():sent=signal_child(json.loads((ROOT/'current.json').read_text()).get('child'),owner=control)
 return dict(STOP=True,signal_sent=sent,old_MS_signal_sent=False)
