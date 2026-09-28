"""Exact nine-worker UrbanEV diagnosis. No numerical admission or executable permit."""
import os,signal,sys
from pathlib import Path
from utils.ch3_contract import digest,profile,task_by_id
from utils.ch3_extension_probe import PACKAGE,PRODUCTION,Budget,read,sha
from utils.ch3_extension import controller_live
SCOPE='urban-numeric-diagnostic-v1'
PACKAGE_ROOT=PACKAGE/SCOPE
PLAN=PACKAGE_ROOT/'plan.json'
ROOT=PACKAGE_ROOT/'execution'
CAPS={'adam':54,'forward':72,'backward':54}
ZERO={'adam':0,'forward':0,'backward':0}
DEBIT={'adam':90,'forward':138,'backward':90}
PARENT_CAPS={'adam':360,'forward':512,'backward':360}
WORKER={'adam':6,'forward':8,'backward':6}

def specification(c):
 p=read(PLAN);runs=[f'TimeMixer-UrbanEV-F4-f1-h{h}-s2024'for h in (3,6,9,12)]
 waves=[dict(stage='serial',task_ids=[r])for r in runs]+[dict(stage='repeat-h3',task_ids=[runs[0]]),dict(stage='q4',task_ids=runs)]
 profiles={r:profile(c,task_by_id(c,r))for r in runs}
 expected=dict(scope=SCOPE,protocol_sha=digest(c),task_ids=runs,profiles=profiles,profile_shas={r:digest(v)for r,v in profiles.items()},waves=waves,caps=CAPS,worker=WORKER,workers=9,already_charged=DEBIT,parent_caps=PARENT_CAPS,output=str(ROOT),no_retries=True,no_fallback=True)
 if any(p.get(k)!=v for k,v in expected.items()):raise ValueError('exact diagnostic plan/profile/budget mismatch')
 return dict(**expected,plan_sha256=sha(PLAN),source_receipt=p['source_receipt'],failure=p['failure'],partial=p['partial'],fixture_root=str(ROOT/'fixture'),capture='raw-state-only-v1; no gate/tolerance changes')

def authorization_reasons(c,a):
 s=specification(c);reasons=[]
 if not a:return ['diagnostic budget approval, reviewed implementation and clean closure permit required']
 if a.get('purpose')!='ch3_resource_probe'or a.get('probe_scope')!=SCOPE:reasons.append('foreign purpose/scope')
 if any(a.get(k)is not True for k in ('reviewed','execution_permitted','budget_authorized'))or a.get('synthetic_fixture')is not False:reasons.append('template/fixture/unapproved diagnostic budget')
 for k,v in dict(scope_sha=digest(s),plan_sha256=s['plan_sha256'],protocol_sha=digest(c),caps=CAPS,already_charged=DEBIT,parent_caps=PARENT_CAPS,authorized_task_ids=s['task_ids'],capture='raw-state-only-v1').items():
  if a.get(k)!=v:reasons.append('diagnostic binding mismatch: '+k)
 for k in ('code','environment','hardware','commit'):
  if not a.get(k):reasons.append('missing closure binding: '+k)
 for key in ('source_receipt','failure','partial'):
  b=s[key]
  if a.get(key)!=b or sha(b['path'])!=b['sha256']:reasons.append('source evidence changed: '+key)
 from utils.ch3_revision import boundary_reasons
 reasons+=boundary_reasons(c,a.get('safe_boundary',{}))
 paths=list(Path(c['execution']['evidence']).glob('formal-*/controller.json'))+[Path(c['timemixer_revision']['result_root'])/'controller.json',PACKAGE/'extension-probe-v1/controller.json',PACKAGE/'revision-numeric-probe-v1/controller.json']
 if any(controller_live(p)for p in paths):reasons.append('conflicting controller live')
 return reasons

def locations():return str(ROOT),str(ROOT/'fixture')
def worker_path(c,task,out):
 s=specification(c);p=Path(out)
 allowed=[ROOT/w['stage']/r for w in s['waves']for r in w['task_ids']if r==task]
 if str(p)!=str(p.resolve())or p not in allowed:raise ValueError('exact diagnostic task/stage/output required')
 return p.parent.name

def validate_worker(c,s):
 if s.get('probe_scope')!=SCOPE:raise ValueError('worker scope mismatch')
 reasons=authorization_reasons(c,s.get('approval'))
 if reasons:raise PermissionError('; '.join(reasons))
 worker_path(c,s['task'],s['output'])
 if (s['session_root'],s['fixture_root'])!=locations()or s.get('prefix_files')or s.get('resume')or s.get('artifact_root')!=s['output']:raise ValueError('synthetic diagnostic boundary')
 if s['limits']!={'adam':6,'forward':8,'backward':6,'seconds':1800}or s.get('ids')!=[s['task']]:raise ValueError('exact worker ceiling/ID')

def validate_wave(c,configs,out):
 if not configs:raise ValueError('empty diagnostic wave')
 for s in configs:validate_worker(c,s)
 base=Path(configs[0]['output']).parent;runs=[s['task']for s in configs]
 if any(Path(s['output']).parent!=base for s in configs):raise ValueError('mixed diagnostic stages')
 if dict(stage=base.name,task_ids=runs)not in specification(c)['waves']or Path(out)!=base/('monitor-'+runs[0]):raise ValueError('fixed diagnostic wave')
 if (ROOT/'STOP').exists():raise InterruptedError('diagnostic STOP')
 return ROOT/'STOP'

def readiness(c,a):
 reasons=authorization_reasons(c,a)
 if Path(__file__).resolve().parents[1]!=PRODUCTION:reasons.append('candidate not deployed/reviewed/closed; preparation only')
 if ROOT.exists()or ROOT.is_symlink():reasons.append('retained diagnostic output; no restart')
 if not reasons:
  from ch3_runner import preflight
  reasons+=preflight(c,'TimeMixer',a,probe=True)
 return list(dict.fromkeys(reasons))

def run_waves(c,launch,stopped):
 """No admission comparison, no q2/q1 fallback. Scalar/state differences are diagnostic output."""
 traces={};measurements=[]
 for w in specification(c)['waves']:
  if stopped():raise InterruptedError('diagnostic STOP before launch')
  m,rows=launch(w['task_ids'],ROOT/w['stage'])
  if m.get('failure')or any(m['returncodes'])or m.get('resource_admission')is not True:raise RuntimeError('diagnostic worker/resource failure; stop suffix, no retry')
  if len(rows)!=len(w['task_ids']):raise ValueError('incomplete diagnostic workers')
  for r,t in zip(w['task_ids'],rows):
   if t.get('id')!=r or t.get('profile_sha')!=digest(profile(c,task_by_id(c,r)))or t.get('steps')!=6 or t.get('finite')is not True:raise ValueError('trajectory identity/finite')
   if t.get('memory_review',{}).get('blocked')is not False or t['memory_review'].get('needs_long_window')is not False:raise ValueError('existing RSS gate unresolved; no added steps')
   if [v['step']for v in t.get('urban_diagnostic_trace',[])]!=list(range(7)):raise ValueError('missing state capture')
   traces[(w['stage'],r)]=t
  measurements.append(dict(stage=w['stage'],tasks=w['task_ids'],measurement=m))
 return traces,measurements

def comparisons(c,traces):
 from utils.ch3_urban_capture import compare_traces
 runs=specification(c)['task_ids'];h3=runs[0]
 return dict(serial_repeat_h3=compare_traces(traces['serial',h3],traces['repeat-h3',h3],ROOT),
   serial_vs_q4={r:compare_traces(traces['serial',r],traces['q4',r],ROOT)for r in runs})

def execute(c,a):
 reasons=readiness(c,a)
 if reasons:raise PermissionError('; '.join(reasons))
 from ch3_runner import GPULock,dump
 sys.path.insert(0,str(PRODUCTION/'tools/restricted_regression'))
 from m5_formal_entry import make_config,run_configs,gpu_sample,resource_assessment
 budget=Budget(CAPS,ZERO);owned=False;peaks={};spec=specification(c)
 def launch(runs,directory):
  if (ROOT/'STOP').exists():raise InterruptedError('STOP')
  if len(runs)>1:
   sample=gpu_sample([]);reserve=max(8*1024**3,.1*sample['total'])
   if not resource_assessment(sample,[])['admission']or any(r not in peaks for r in runs)or sample['free']-sum(peaks[r]for r in runs)<reserve:raise MemoryError('q4 lacks conservative headroom; no fallback')
  budget.reserve([dict(output=str(directory/r),limits=WORKER)for r in runs]);dump(ROOT/'budget.json',vars(budget));configs=[]
  try:
   configs=[make_config(c,'ch3_probe',directory/r,task=r,approval=a)for r in runs]
   m=run_configs(configs,directory/('monitor-'+runs[0]),monitor=True)
   if m['failure']or any(m['returncodes'])or not m['resource_admission']:return m,[]
   traces=[]
   for s in configs:
    if any(read(s['budget_file'])['counts'][k]!=WORKER[k]for k in CAPS):raise ValueError('worker meter mismatch')
    t=read(Path(s['output'])/'trajectory.json');traces.append(t)
    if directory.name=='serial':peaks[s['task']]=max([t['reserved'],m['whole_card_peak']]+[v for v in m['process_peaks'].values()if v is not None])
   return m,traces
  finally:budget.record(configs);dump(ROOT/'budget.json',vars(budget))
 def stop(sig,frame):raise InterruptedError('owned diagnostic interrupted')
 old=signal.signal(signal.SIGTERM,stop)
 try:
  with GPULock(c):
   ROOT.mkdir(exist_ok=False);owned=True;(ROOT/'fixture').mkdir()
   dump(ROOT/'controller.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],scope_sha=digest(spec)))
   traces,measurements=run_waves(c,launch,lambda:(ROOT/'STOP').exists())
   results=comparisons(c,traces)
   if budget.counts!=CAPS:raise ValueError('exact diagnostic debit missing')
   evidence={str(p.relative_to(ROOT)):sha(p)for p in sorted(ROOT.rglob('*'))if p.is_file()and p.suffix in ('.json','.bin','.jsonl')}
   report=dict(probe_scope=SCOPE,scope_sha=digest(spec),scope=spec,execution_complete=True,reviewed=False,admission_granted=False,
      synthetic_fixture=False,comparisons=results,measurements=measurements,budget=vars(budget),
      cumulative={k:DEBIT[k]+budget.counts[k]for k in CAPS},evidence=evidence,**{k:a[k]for k in ('commit','code','protocol_sha','environment','hardware')})
   dump(ROOT/'complete.json',report)
 except BaseException as exc:
  if owned:dump(ROOT/'failure.json',dict(error=repr(exc),budget=vars(budget),reviewed=False,admission_granted=False))
  raise
 finally:signal.signal(signal.SIGTERM,old)

def validate_completion(c,r):
 spec=specification(c)
 if r.get('scope')!=spec or r.get('scope_sha')!=digest(spec)or r.get('execution_complete')is not True or r.get('synthetic_fixture')is not False or r.get('admission_granted')is not False:raise ValueError('diagnostic report identity')
 if r.get('budget',{}).get('counts')!=CAPS:raise ValueError('diagnostic debit mismatch')
 ev=r['evidence']
 for n,h in ev.items():
  p=ROOT/n
  if not p.resolve().is_relative_to(ROOT)or sha(p)!=h:raise ValueError('evidence path/SHA')
 def replay(runs,directory):
  paths=[directory/('monitor-'+runs[0])/'process.json']+[directory/r/n for r in runs for n in ('config.json','budget.json','trajectory.json')]
  if any(str(p.relative_to(ROOT))not in ev for p in paths):raise ValueError('unbound execution evidence')
  rows=[]
  for run in runs:
   s=read(directory/run/'config.json');validate_worker(c,s)
   if read(directory/run/'budget.json')['counts']!={'adam':6,'forward':8,'backward':6}:raise ValueError('actual diagnostic worker debit')
   t=read(directory/run/'trajectory.json')
   for point in t['urban_diagnostic_trace']:
    for k in ('schema_file','data_file','meta_file'):
     if str(Path(point[k]).relative_to(ROOT))not in ev:raise ValueError('unbound diagnostic payload')
   rows.append(t)
  return read(paths[0]),rows
 traces,measurements=run_waves(c,replay,lambda:False)
 if measurements!=r['measurements']or comparisons(c,traces)!=r['comparisons']:raise ValueError('diagnostic summary not reproducible from evidence')
 return r
