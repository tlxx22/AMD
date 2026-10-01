"""One guarded M queue or M probe tmux controller; no model import."""
import argparse,json,os,sys,time,signal
from pathlib import Path
from utils.ch3_contract import read_profiles,validate_manifest,digest
from utils import ch3_m_tasks as s
from utils import ch3_m_execution as e
from ch3_runner import dump,GPULock
from m6_remaining_entry import identity,same,sequence,signal_owned,stop_marker,lock
from utils import ch3_m_launch as launch

def readiness(c,app,approval,probe=False,*,starting=False,token=None):
 reasons=e.authorization_reasons(c,app,probe=probe);record=None
 if starting:
  try:record=launch.verify(c,'probe'if probe else'formal',token,approval=approval)
  except (OSError,ValueError,PermissionError)as exc:reasons.append(str(exc))
 reasons+=e.fresh_conflicts(c,probe,launch_token=token if starting else None,approval=approval)
 import subprocess
 session=launch.spec('probe'if probe else'formal')['session']
 try:
  present=subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
  if present and not(record and record['driver']=='tmux'and record['actual_tmux']['session']==session):reasons.append('M retained tmux session')
 except FileNotFoundError:reasons.append('tmux unavailable')
 if not reasons:
  try:
   with GPULock(c):pass
   sys.path.insert(0,str(s.ROOT/'tools/restricted_regression'));from m5_formal_entry import gpu_sample,resource_assessment
   if not resource_assessment(gpu_sample([]),[])['admission']:reasons.append('M resource admission unavailable')
  except (OSError,RuntimeError,ValueError)as exc:reasons.append(str(exc))
 return list(dict.fromkeys(reasons))

def verify_completion(c,root,app,probe=False):
 if not(root/'complete.json').exists():raise ValueError('M execution has no completion')
 r=json.loads((root/'complete.json').read_text())
 if probe:e.validate_probe_completion(c,r)
 else:
  from utils.ch3_m_summary import technical_group
  if r['task_ids']!=[t['id']for t in c['tasks']]or same(json.loads((root/'controller.json').read_text()))or(root/'STOP').exists()or(root/'failure.json').exists():raise ValueError('M total completion identity')
  for m in s.MODELS:
   if s.bound(r['handoffs'][m])!=technical_group(c,m,app):raise ValueError('M handoff replay')
 return r

def signal_M_owned(v):
 if not same(v):return False
 argv=(Path('/proc')/str(v['pid'])/'cmdline').read_bytes().decode().split('\0')
 if str(s.ROOT/'m6_m_entry.py')not in argv or not any(action in argv for action in ('start','group')):raise ValueError('PID/start_ticks match but process is not this M execution chain')
 return signal_owned(v)

def group(c,a,model):
 if model not in s.MODELS:raise ValueError('M model scope')
 root=e.CONTROL/model
 if root.exists():raise FileExistsError('M group retained complete/failed/staging')
 reasons=e.authorization_reasons(c,a)
 if reasons:raise PermissionError('; '.join(reasons))
 with GPULock(c):
  root.mkdir(parents=True,exist_ok=False);dump(root/'controller.json',dict(**identity(os.getpid()),model=model,commit=a['commit'],protocol_sha=digest(c)))
  sys.path.insert(0,str(s.ROOT/'tools/restricted_regression'));from m5_formal_entry import make_config,run_configs
  report=s.bound(a['resource_report']);ids=[];n=0
  try:
   for g in [x for x in c['groups']if x['model']==model]:
    for wave in e.wave_ids(g['task_ids'],report['decisions'][g['id']]['concurrency']):
     if(e.CONTROL/'STOP').exists():raise InterruptedError('M queue STOP before wave')
     if e.authorization_reasons(c,a):raise ValueError('M bindings changed')
     configs=[make_config(c,'ch3_formal',s.result_path(next(t for t in c['tasks']if t['id']==r)),task=r,approval=a)for r in wave]
     measured=run_configs(configs,root/('wave-'+str(n)),monitor=True)
     if not e.wave_passed(measured):raise RuntimeError('M formal wave failure; stop remaining')
     ids+=wave;dump(root/'progress.json',dict(wave=n,completed_task_ids=ids));n+=1
   dump(root/'complete.json',dict(task_ids=ids,protocol_sha=digest(c),commit=a['commit'],result_review='pending'))
  except BaseException as exc:dump(root/'failure.json',dict(error=repr(exc),completed=ids));raise

def status(root,c=None):
 d=dict(output=str(root),running=False,complete=(root/'complete.json').exists(),failure=None,start_called=root.exists())
 for n in ('controller','current','progress','failure'):
  p=root/(n+'.json')
  if p.exists():d[n]=json.loads(p.read_text())
 if d.get('controller'):d['running']=same(d['controller'])
 if c is not None and root==e.CONTROL:
  totals=dict(adam=0,forward=0,backward=0);done=[]
  for t in c['tasks']:
   out=s.result_path(t)
   if(out/'budget.json').exists():
    b=json.loads((out/'budget.json').read_text())
    for k in totals:totals[k]+=b['counts'][k]
   if(out/'result.json').exists():done.append(t['id'])
  d.update(actual_budget=totals,completed_task_ids=done,result_review='pending')
  m=d.get('current',{}).get('model');gp=root/m/'progress.json'if m else None
  if gp and gp.exists():d['current_wave']=json.loads(gp.read_text())
 return d

def cli():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['dry-run','preflight','start','logs','status','complete','safe-stop','group','summary','estimate','audit','seal-boundary','prepare-launch']);p.add_argument('--approval');p.add_argument('--probe',action='store_true');p.add_argument('--model');p.add_argument('--output');p.add_argument('--wrapper-pid',type=int);a=p.parse_args();c=validate_manifest(read_profiles());root=e.PROBE_ROOT if a.probe else e.CONTROL
 if a.action=='prepare-launch':
  print(launch.prepare(c,'probe'if a.probe else'formal',a.wrapper_pid,approval=a.approval));return 0
 if a.action=='seal-boundary':
  from utils.ch3_m_handoff import seal_boundary
  print(json.dumps(seal_boundary(c)));return 0
 if a.action in ('summary','estimate','audit'):
  from utils.ch3_m_summary import summarize,estimate,audit_checkpoints
  if not a.output:raise ValueError('external output required')
  out=Path(a.output)
  if not out.resolve().is_relative_to(s.PACKAGE):raise ValueError('M audit output outside package')
  if a.action in ('summary','estimate'):
   with out.open('x')as f:json.dump((summarize if a.action=='summary'else estimate)(c),f,ensure_ascii=False,indent=2);f.write('\n')
  else:
   if not a.approval:raise PermissionError('M checkpoint audit requires original reviewed formal permit')
   permit=json.loads(Path(a.approval).read_text());reasons=e.authorization_reasons(c,permit)
   if reasons:raise PermissionError('; '.join(reasons))
   if not(e.CONTROL/'complete.json').exists():raise ValueError('M 84-run total technical completion required before weight audit')
   from utils.ch3_m_summary import technical_group
   for model in s.MODELS:technical_group(c,model,permit)
   audit_checkpoints(c,out)
  return 0
 if a.action in ('logs','status'):
  d=status(root,c)
  if a.action=='logs':d['logs']=[str(e.launcher_log(a.probe))]+[str(p)for p in root.glob('*.log')]
  print(json.dumps(d,ensure_ascii=False,indent=2));return 0
 if a.action=='safe-stop':
  if not(root/'controller.json').exists():raise ValueError('no M owned controller')
  stop_marker(root);control=json.loads((root/'controller.json').read_text());sent=signal_M_owned(control)
  if not sent and(root/'current.json').exists():sent=signal_M_owned(json.loads((root/'current.json').read_text())['child'])
  print(json.dumps(dict(STOP=True,signal_sent=sent)));return 0
 app=json.loads(Path(a.approval).read_text())if a.approval else None
 if a.action=='group':group(c,app,a.model);return 0
 if a.action=='complete':
  if not(root/'complete.json').exists():print(json.dumps(dict(complete=False,status=status(root))));return 2
  verify_completion(c,root,app,a.probe)
  print(json.dumps(dict(execution_complete=True,result_review='pending',reviewed=False)));return 0
 token=os.environ.get(launch.TOKEN_ENV)if a.action=='start'else None
 reasons=readiness(c,app,a.approval,a.probe,starting=a.action=='start',token=token)
 if a.action in ('dry-run','preflight'):
  print(json.dumps(dict(scope=s.PROBE if a.probe else s.ID,plan=e.plan(c),blocked=reasons),ensure_ascii=False,indent=2));return 2 if a.action=='preflight'and reasons else 0
 if reasons:raise PermissionError('; '.join(reasons))
 root.parent.mkdir(parents=True,exist_ok=True)
 with lock(root.parent/('.'+root.name+'.lock')):
  if root.exists():raise FileExistsError('M duplicate start')
  launch.claim(c,'probe'if a.probe else'formal',token,approval=a.approval)
  root.mkdir();dump(root/'controller.json',dict(**identity(os.getpid()),commit=app['commit'],protocol_sha=digest(c),approval=s.ref(a.approval)))
  if a.probe:
   dump(root/'approval.json',app)
   with GPULock(c):e.run_probe(c,app)
  else:
   from utils.ch3_m_summary import technical_group
   plan=dict(groups=[dict(model=m)for m in s.MODELS],task_ids=[t['id']for t in c['tasks']]);commands={m:[c['execution']['python'],'-B',str(s.ROOT/'m6_m_entry.py'),'group','--model',m,'--approval',a.approval]for m in s.MODELS}
   def before(g):
    if e.authorization_reasons(c,app):raise ValueError('M dynamic binding changed')
    if any(s.result_path(t).exists()for t in c['tasks']if t['model']==g['model']):raise FileExistsError('M output already exists')
   def check(g):return technical_group(c,g['model'],app)
   try:
    sequence(plan,root,commands,before,check,queue_id=s.ID)
    done=json.loads((root/'complete.json').read_text());done.update(task_ids=[t['id']for t in c['tasks']],protocol_sha=digest(c),commit=app['commit']);dump(root/'complete.json',done)
   except BaseException:raise
 return 0
if __name__=='__main__':sys.exit(cli())
