"""Three fixed-LR technical groups; independent, explicitly budgeted scope.

No executable permit is shipped. Uses the existing restricted six-step worker,
monitor and per-domain numerical policy. Never calls a bare model/worker.
"""
import copy,os,signal,sys,json
from pathlib import Path
from utils.ch3_contract import digest,profile,task_by_id
from utils.ch3_extension_probe import PACKAGE,PRODUCTION,PARENT,Budget,read,sha
from utils.ch3_revision import REVISION,ids,changed_groups,boundary_reasons,validate_revision_profiles
from utils.ch3_extension import controller_live
SCOPE='timemixer-revision-numeric-v1'
ROOT=PACKAGE/'revision-numeric-probe-v1'
PLAN=PACKAGE/'timemixer-retire-v2/admission-plan.json'
CAPS={'adam':144,'forward':192,'backward':144}
ZERO={'adam':0,'forward':0,'backward':0}
WORKER={'adam':6,'forward':8,'backward':6,'validation':2}

def specification(c):
 validate_revision_profiles(c);p=read(PLAN)
 if p['protocol_sha']!=digest(c) or p['profile_shas']!=c['timemixer_revision']['effective_profile_hashes']:raise ValueError('revision plan profile/protocol mismatch')
 if p['changed_LR_numeric_groups']!=changed_groups() or p['proposed_numeric_scope']!={'workers':12,'serial_references':12,'concurrent_workers':12,'steps_each':6,'forward_each':8,'Adam':144,'backward':144,'forward':192,'waves':{'ETTh1':4,'Weather':4,'ECL':2}}:raise ValueError('exact revision plan/budget required')
 groups=[]
 for gid,q in zip(changed_groups(),[4,4,2]):
  g=next(x for x in c['groups']if x['id']==gid);runs=[r for r in ids()[:12]if r.startswith(gid+'-')]
  if g['representatives']!=runs or g['task_ids']!=runs:raise ValueError('exact 12 revision representatives')
  groups.append(dict(id=gid,representatives=runs,q=q,waves=[runs[i:i+q]for i in range(0,4,q)]))
 # Bind Exchange to the same immutable original admission evidence used by the extension.
 from utils.ch3_extension_probe import PLAN as extension_plan
 parent_sha=read(extension_plan)['parent_report_sha256']
 return dict(probe_scope=SCOPE,execution_revision=REVISION,protocol_sha=digest(c),plan_sha256=sha(PLAN),groups=groups,
   task_ids=ids()[:12],profile_shas={r:digest(profile(c,task_by_id(c,r)))for r in ids()[:12]},
   root=str(ROOT),fixture_root=str(ROOT/'fixture'),caps=CAPS,already_charged=ZERO,worker=WORKER,
   exchange_parent=dict(path=str(PARENT),sha256=parent_sha,group='TimeMixer-Exchange-MS',
     profile_shas={r:c['extension']['original_profile_hashes'][r]for r in ids()[12:]}))

def authorization_reasons(c,a):
 s=specification(c)
 if not a:return ['revision numeric scope needs separate budget approval/review/closure permit']
 reasons=[]
 if a.get('purpose')!='ch3_resource_probe' or a.get('probe_scope')!=SCOPE or a.get('extension_batch') is not None:reasons.append('foreign revision numeric purpose/scope')
 if any(a.get(k)is not True for k in ('reviewed','execution_permitted','budget_authorized')) or a.get('synthetic_fixture') is not False:reasons.append('template/fixture/unapproved numeric budget is not executable')
 for key,value in [('scope_sha',digest(s)),('protocol_sha',digest(c)),('plan_sha256',s['plan_sha256']),('caps',CAPS),('already_charged',ZERO),('authorized_task_ids',s['task_ids'])]:
  if a.get(key)!=value:reasons.append('revision numeric binding mismatch: '+key)
 for key in ('code','environment','hardware','commit'):
  if not a.get(key):reasons.append('actual closure binding missing: '+key)
 reasons+=boundary_reasons(c,a.get('safe_boundary',{}))
 for path in Path(c['execution']['evidence']).glob('formal-*/controller.json'):
  if controller_live(path):reasons.append('formal controller live: '+str(path))
 for path in (Path(c['timemixer_revision']['result_root'])/'controller.json',PACKAGE/'extension-probe-v1/controller.json'):
  if controller_live(path):reasons.append('conflicting controller live: '+str(path))
 # No revision_resource_report dependency: this scope produces that evidence.
 return reasons

def locations():return str(ROOT),str(ROOT/'fixture')

def worker_path(c,task,out):
 s=specification(c);g=next((g for g in s['groups']if task in g['representatives']),None)
 if g is None:raise ValueError('outside revision 12-task scope')
 out=Path(out)
 if str(out)!=str(out.resolve()) or out not in [ROOT/g['id']/stage/task for stage in ('serial','parallel')]:raise ValueError('exact revision technical output required')
 return g

def validate_worker(c,s):
 if s.get('probe_scope')!=SCOPE:raise ValueError('worker scope mismatch')
 reasons=authorization_reasons(c,s.get('approval'))
 if reasons:raise PermissionError('; '.join(reasons))
 worker_path(c,s['task'],s['output'])
 if (s['session_root'],s['fixture_root'])!=locations() or s.get('prefix_files') or s.get('resume') or s.get('artifact_root')!=s['output']:raise ValueError('revision synthetic-only worker boundary')
 if s['limits']!={'adam':6,'forward':8,'backward':6,'seconds':1800}:raise ValueError('exact six-step ceiling')
 if s.get('ids')!=[s['task']]:raise ValueError('exact worker ID required')

def validate_wave(c,configs,out):
 if not configs:raise ValueError('empty revision wave')
 for s in configs:validate_worker(c,s)
 runs=[s['task']for s in configs];g=worker_path(c,runs[0],configs[0]['output']);base=Path(configs[0]['output']).parent
 if any(Path(s['output']).parent!=base for s in configs):raise ValueError('mixed revision groups/stages')
 if base.name=='serial':legal=[[r]for r in g['representatives']]
 else:legal=g['waves']
 if runs not in legal or Path(out)!=base/('monitor-'+runs[0]):raise ValueError('fixed revision wave/monitor required')
 if (ROOT/'STOP').exists():raise InterruptedError('revision STOP before monitor/worker launch')
 return ROOT/'STOP'

def readiness(c,a):
 reasons=authorization_reasons(c,a)
 if ROOT.exists() or ROOT.is_symlink():reasons.append('retained revision numeric output; no retry')
 from ch3_runner import ROOT as repo
 if repo.resolve()!=PRODUCTION:reasons.append('candidate is not reviewed production')
 if not reasons:
  from ch3_runner import preflight
  reasons+=preflight(c,'TimeMixer',a,probe=True)
 return list(dict.fromkeys(reasons))

def run_fixed_groups(c,launch,compare,stopped):
 """Pure orchestration seam; production launch always uses bound run_configs."""
 decisions={}
 for g in specification(c)['groups']:
  serial=[];reference=[];measured=[];observed=[]
  for stage,waves in [('serial',[[r]for r in g['representatives']]),('parallel',g['waves'])]:
   for runs in waves:
    if stopped():raise InterruptedError('revision STOP before wave')
    m,t=launch(runs,ROOT/g['id']/stage)
    if m.get('failure') or any(m['returncodes']) or m.get('resource_admission')is not True:raise RuntimeError('revision wave failed; no fallback/refund/suffix')
    if len(t)!=len(runs):raise ValueError('missing worker trajectories')
    for r,v in zip(runs,t):
     if v.get('id')!=r or v.get('profile_sha')!=digest(profile(c,task_by_id(c,r))) or v.get('steps')!=6 or v.get('finite')is not True:raise ValueError('trajectory identity/finite')
     review=v.get('memory_review',{})
     if review.get('blocked')is not False or review.get('needs_long_window')is not False:raise ValueError('existing RSS criterion unresolved')
    if stage=='serial':serial.append(m);reference+=t
    else:measured.append(m);observed+=t
  checks=[compare(c,task_by_id(c,r),x,y)for r,x,y in zip(g['representatives'],reference,observed)]
  if len(checks)!=4 or not all(x['passed']is True for x in checks):raise ValueError('numerical failure; no fallback/refund/suffix')
  elapsed=sum(x['elapsed']for x in measured);speed=sum(x['elapsed']for x in serial)/elapsed if elapsed>0 else 0
  if speed<=1:raise ValueError('fixed concurrency short-package benefit not demonstrated')
  decisions[g['id']]=dict(status='Passed',concurrency=g['q'],representatives=g['representatives'],serial=serial,
    numerical_protocol_sha=digest(c),scope='six updates at new fixed LR; report still needs review',
    **{str(g['q']):dict(resource='Passed',numerical_comparisons=checks,makespan=elapsed,speedup=speed)})
 return decisions

def exchange_inheritance(c):
 binding=specification(c)['exchange_parent']
 if sha(PARENT)!=binding['sha256']:raise ValueError('Exchange parent evidence bytes changed')
 for r,h in binding['profile_shas'].items():
  if digest(profile(c,task_by_id(c,r)))!=h:raise ValueError('Exchange profile cannot inherit')
 parent=read(PARENT);v=parent['decisions'][binding['group']]
 if v.get('status')!='Passed' or v.get('concurrency')!=4:raise ValueError('Exchange q4 parent not accepted')
 return copy.deepcopy(v),binding

def execute(c,a):
 reasons=readiness(c,a)
 if reasons:raise PermissionError('; '.join(reasons))
 from ch3_runner import GPULock,dump,compare_probe_trajectories
 sys.path.insert(0,str(PRODUCTION/'tools/restricted_regression'))
 from m5_formal_entry import make_config,run_configs,gpu_sample,resource_assessment
 s=specification(c);exchange,parent=exchange_inheritance(c);budget=Budget(CAPS,ZERO);owned=False;peaks={}
 def launch(runs,directory):
  if (ROOT/'STOP').exists():raise InterruptedError('STOP')
  if len(runs)>1:
   sample=gpu_sample([]);reserve=max(8*1024**3,.1*sample['total'])
   if not resource_assessment(sample,[])['admission'] or any(r not in peaks for r in runs) or sample['free']-sum(peaks[r]for r in runs)<reserve:raise MemoryError('fixed revision wave lacks conservative headroom; no fallback')
  # Reserve BEFORE make_config/worker creation; exceptions never refund.
  ceilings=[dict(output=str(directory/r),limits=WORKER)for r in runs]
  budget.reserve(ceilings);dump(ROOT/'budget.json',vars(budget));configs=[]
  try:
   configs=[make_config(c,'ch3_probe',directory/r,task=r,approval=a)for r in runs]
   result=run_configs(configs,directory/('monitor-'+runs[0]),monitor=True)
   if result['failure'] or any(result['returncodes']) or not result['resource_admission']:return result,[]
   traces=[]
   for conf in configs:
    counts=read(conf['budget_file'])['counts']
    if any(counts.get(k)!=WORKER[k]for k in CAPS):raise ValueError('actual six-step meter incomplete')
    t=read(Path(conf['output'])/'trajectory.json')
    if t.get('synthetic_fixture'):raise ValueError('fixture cannot become real evidence')
    if directory.name=='serial':
     peaks[conf['task']]=max([t['reserved'],result['whole_card_peak']]+[v for v in result['process_peaks'].values()if v is not None])
    traces.append(t)
   return result,traces
  finally:
   budget.record(configs);dump(ROOT/'budget.json',vars(budget))
 def stop(sig,frame):raise InterruptedError('owned revision probe interrupted')
 old=signal.signal(signal.SIGTERM,stop)
 try:
  with GPULock(c):
   ROOT.mkdir(exist_ok=False);owned=True;(ROOT/'fixture').mkdir()
   dump(ROOT/'controller.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],scope_sha=digest(s)))
   decisions=run_fixed_groups(c,launch,compare_probe_trajectories,lambda:(ROOT/'STOP').exists())
   if budget.counts!=CAPS:raise ValueError('exact revision total debit missing')
   decisions[parent['group']]=exchange
   evidence={str(p.relative_to(ROOT)):sha(p)for p in sorted(ROOT.rglob('*.json'))}
   for trace_path in ROOT.glob('*/*/*/trajectory.json'):
    for row in read(trace_path).get('full_numeric_trace',[]):
     for key in ('schema_file','data_file'):
      p=Path(row[key])
      if not p.resolve().is_relative_to(ROOT):raise ValueError('numeric payload outside revision scope')
      evidence[str(p.relative_to(ROOT))]=sha(p)
   report=dict(purpose='revision_numeric_report',probe_scope=SCOPE,execution_revision=REVISION,execution_complete=True,
     reviewed=False,synthetic_fixture=False,protocol_sha=digest(c),scope_sha=digest(s),scope=s,
     task_ids=ids(),profile_shas=c['timemixer_revision']['effective_profile_hashes'],decisions=decisions,
     inheritance={parent['group']:parent},budget=vars(budget),evidence=evidence,
     **{k:a[k]for k in ('commit','code','environment','hardware')})
   dump(ROOT/'complete.json',report)
 except BaseException as exc:
  if owned:dump(ROOT/'failure.json',dict(error=repr(exc),budget=vars(budget),scope_sha=digest(s),reviewed=False))
  raise
 finally:signal.signal(signal.SIGTERM,old)

def validate_completion(c,r):
 s=specification(c)
 if r.get('purpose')!='revision_numeric_report' or r.get('probe_scope')!=SCOPE or r.get('scope_sha')!=digest(s) or r.get('scope')!=s or r.get('synthetic_fixture')is not False or r.get('execution_complete')is not True:raise ValueError('not actual revision numeric completion')
 if r.get('task_ids')!=ids() or r.get('profile_shas')!=c['timemixer_revision']['effective_profile_hashes'] or r.get('budget',{}).get('counts')!=CAPS:raise ValueError('revision completion coverage/budget')
 evidence=r.get('evidence',{})
 for g in s['groups']:
  for stage in ('serial','parallel'):
   for run in g['representatives']:
    for name in ('trajectory.json','budget.json','config.json'):
     key='/'.join((g['id'],stage,run,name))
     if key not in evidence:raise ValueError('completion missing actual worker source')
 for key,h in evidence.items():
  path=ROOT/key
  if not path.resolve().is_relative_to(ROOT) or sha(path)!=h:raise ValueError('completion evidence SHA/path mismatch')
 from ch3_runner import compare_probe_trajectories
 def replay(runs,directory):
  m=read(directory/('monitor-'+runs[0])/'process.json');traces=[]
  if str((directory/('monitor-'+runs[0])/'process.json').relative_to(ROOT))not in evidence:raise ValueError('unbound monitor')
  for run in runs:
   b=read(directory/run/'budget.json')
   if any(b['counts'].get(k)!=WORKER[k]for k in CAPS):raise ValueError('incomplete actual operation count')
   t=read(directory/run/'trajectory.json')
   for row in t.get('full_numeric_trace',[]):
    for key in ('schema_file','data_file'):
     path=Path(row[key])
     if not path.resolve().is_relative_to(ROOT) or str(path.relative_to(ROOT))not in evidence:raise ValueError('unbound numeric sidecar')
   traces.append(t)
  return m,traces
 measured=run_fixed_groups(c,replay,compare_probe_trajectories,lambda:False)
 exchange,binding=exchange_inheritance(c);measured[binding['group']]=exchange
 if r.get('decisions')!=measured or r.get('inheritance')!={binding['group']:binding}:raise ValueError('decision differs from measured/parent source')
 return r
