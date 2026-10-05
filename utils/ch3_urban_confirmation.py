"""Exact eight-worker UrbanEV scoped numerical confirmation. No executable permit shipped."""
import os,signal,sys
from pathlib import Path
from utils.ch3_contract import digest,profile,task_by_id
from utils.ch3_extension_probe import PACKAGE,PRODUCTION,Budget,read,sha
from utils.ch3_extension import controller_live
SCOPE='urban-numeric-confirmation-v1'
PACKAGE_ROOT=PACKAGE/SCOPE
PLAN=PACKAGE_ROOT/'plan.json'
ROOT=PACKAGE_ROOT/'execution'
CAPS={'adam':48,'forward':80,'backward':48}
ZERO={'adam':0,'forward':0,'backward':0}
DEBIT={'adam':144,'forward':210,'backward':144}
PARENT_CAPS={'adam':360,'forward':512,'backward':360}
WORKER={'adam':6,'forward':10,'backward':6}

POLICY=dict(id='timemixer-urban-f4-confirmation-v1',model='TimeMixer',dataset='UrbanEV',input_variant='F4',horizons=[3,6,9,12],state_atol=1e-4,loss_atol=1e-6,metric_atol=1e-6,rtol=0,evaluation_steps=[2,6],nonfloating_and_step='exact',initialization_rng_batches='exact')

def policy(c,t):
 if (t['model'],t['dataset'],t['input_variant'],t['h'])!=('TimeMixer','UrbanEV','F4',t['h']) or t['h']not in POLICY['horizons']:return None
 p=profile(c,t)
 if c.get('type1_followup'):
  if (p['T'],p['pred_len'],p['C'],p['training']['batch'],p['training']['eval_batch'],p['training']['lr'])!=(12,1,11,128,128,1e-4):raise ValueError('type1 inherited UrbanEV computation')
  from utils.ch3_type1_tasks import numeric_policy
  return numeric_policy(c,t)
 if 'baseline_unified' in c:
  if (p['T'],p['pred_len'],p['C'],p['training']['batch'],p['training']['eval_batch'],p['training']['lr'])!=(12,1,11,128,128,.01):raise ValueError('unified UrbanEV inherited computation')
  from utils.ch3_baseline_unified_tasks import numeric_policy
  return numeric_policy(c,t)
 if (p['T'],p['pred_len'],p['C'],p['training']['batch'],p['training']['eval_batch'],p['training']['lr'])!=(12,1,11,128,128,.001):raise ValueError('policy requires frozen UrbanEV computation')
 if 'native_replacement' in c:
  from utils.ch3_native_tasks import numeric_policy
  return numeric_policy(c,t)
 # Complete structure/seed/precision identity is also bound by PLAN/profile SHA.
 return POLICY

def cached_endpoint(model,batches,evaluator,snapshot,batch_digest):
 """Caller supplies the SAME two CPU batches. No sampling, update or model call here."""
 if len(batches)!=2:raise ValueError('exactly two cached validation batches')
 modes=[(m,m.training)for m in model.modules()];before=snapshot();identities=[batch_digest(b)for b in batches]
 try:
  metrics=evaluator(batches)
 finally:
  for m,mode in modes:m.training=mode
 after=snapshot()
 if before!=after or identities!=[batch_digest(b)for b in batches]:raise ValueError('endpoint evaluation changed persistent state/RNG/batches')
 return dict(metrics=metrics,batch_ids=identities,before=before,after=after,mode_restored=all(m.training==v for m,v in modes))

def compare_measured(c,t,x,y,detail):
 """Decision over source-verified full-state measurements; no tolerance inferred from data."""
 import math
 rule=policy(c,t)
 if rule is None:raise ValueError('foreign numeric policy task')
 for k in ('id','profile_sha','initial','initial_rng','batch_ids','steps','final_rng','validation_tail','threads','affinity'):
  if x.get(k)!=y.get(k):raise ValueError('exact identity/RNG/shape mismatch: '+k)
 if x.get('id')!=t['id']or x.get('profile_sha')!=digest(profile(c,t))or x.get('steps')!=6 or x.get('finite')is not True or y.get('finite')is not True:raise ValueError('task/profile/finite')
 points=detail['points'];errors=[];state_max=0.
 if [p['step']for p in points]!=list(range(7)):raise ValueError('seven full-state points')
 for p in points:
  if not p.get('metadata_equal')or not p.get('rng_equal'):errors.append('state metadata/RNG at '+str(p['step']))
  if p['step']==0 and not p.get('identical'):errors.append('initial state not exact')
  if not p.get('tensors'):raise ValueError('empty full-state capture')
  for row in p['tensors']:
   if not row.get('structure_equal')or not row.get('finite'):errors.append('structure/finite '+row['name']);continue
   diff=row['max_abs_diff']
   if diff is None or not math.isfinite(diff)or diff<0:raise ValueError('invalid measurement')
   state_max=max(state_max,diff)
   if row['exact_required']:
    if not row['byte_equal']:errors.append('exact state '+row['name'])
   elif diff>rule['state_atol']:errors.append('float state '+row['name'])
 loss_diffs=[]
 if len(x['trajectory'])!=6 or len(y['trajectory'])!=6:raise ValueError('six loss records')
 for i,(a,b)in enumerate(zip(x['trajectory'],y['trajectory']),1):
  if a['step']!=i or b['step']!=i or not all(math.isfinite(v['loss'])for v in (a,b)):raise ValueError('loss step/finite')
  loss_diffs.append(abs(a['loss']-b['loss']))
 metric_diffs={}
 for side in (x,y):
  obs=side.get('urban_confirmation',{})
  if obs.get('policy_sha')!=digest(rule)or set(obs.get('evaluations',{}))!={'2','6'}:raise ValueError('two evaluation checkpoints/policy required')
  e2,e6=obs['evaluations']['2'],obs['evaluations']['6']
  if e2['batch_ids']!=e6['batch_ids']or len(e2['batch_ids'])!=2 or e6['before']!=e6['after']or e6.get('mode_restored')is not True:raise ValueError('cached batches or endpoint preservation')
  if e6['before']['rng']!=side['final_rng']or e6['before']['model']!=side['final']or e6['before']['optimizer']!=side['trajectory'][-1]['optimizer']:raise ValueError('endpoint not after original six-step state/RNG')
  if e2['metrics']!=side['validation']:raise ValueError('step2 is not original evaluation')
 for step in ('2','6'):
  a=x['urban_confirmation']['evaluations'][step];b=y['urban_confirmation']['evaluations'][step]
  if a['batch_ids']!=b['batch_ids']or a['target']!=b['target']:raise ValueError('evaluation batch/target identity')
  p=profile(c,t)
  if a['target']!=dict(index=p['target_idx'],pred_len=p['pred_len'],C=p['C'],metric_space='train-standardized target-only'):raise ValueError('target semantics do not match profile')
  u,v=a['metrics'],b['metrics'];n=u['elements']
  if not isinstance(n,int)or n<=0 or n!=v['elements']:raise ValueError('evaluation element count')
  if not all(math.isfinite(w[k])for w in (u,v)for k in ('mse','mae','sse','sae')):raise ValueError('evaluation finite')
  metric_diffs[step]={k:abs(u[k]-v[k])for k in ('mse','mae')}
  metric_diffs[step].update(normalized_sse=abs(u['sse']-v['sse'])/n,normalized_sae=abs(u['sae']-v['sae'])/n)
 if max(loss_diffs)>rule['loss_atol']:errors.append('training loss bound')
 if any(v>rule['metric_atol']for row in metric_diffs.values()for v in row.values()):errors.append('evaluation bound')
 return dict(passed=not errors,policy_sha=digest(rule),rtol=0,mode='urban_scoped_full_state',state_max_abs=state_max,loss_max_abs=max(loss_diffs),evaluation_diffs=metric_diffs,failures=errors,full_state=detail)

def compare_confirmation(c,t,x,y):
 from utils.ch3_urban_capture import compare_traces
 return compare_measured(c,t,x,y,compare_traces(x,y,ROOT))

def specification(c):
 p=read(PLAN);runs=[f'TimeMixer-UrbanEV-F4-f1-h{h}-s2024'for h in (3,6,9,12)]
 waves=[dict(stage='serial',task_ids=[r])for r in runs]+[dict(stage='q4',task_ids=runs)]
 profiles={r:profile(c,task_by_id(c,r))for r in runs}
 expected=dict(scope=SCOPE,protocol_sha=digest(c),task_ids=runs,profiles=profiles,profile_shas={r:digest(v)for r,v in profiles.items()},waves=waves,caps=CAPS,worker=WORKER,workers=8,already_charged=DEBIT,parent_caps=PARENT_CAPS,output=str(ROOT),no_retries=True,no_fallback=True,policy=POLICY,carry_forward_sha256=sha(PACKAGE_ROOT/'carry-forward.json'))
 if any(p.get(k)!=v for k,v in expected.items()):raise ValueError('exact diagnostic plan/profile/budget mismatch')
 return dict(**expected,plan_sha256=sha(PLAN),source_receipt=p['source_receipt'],failure=p['failure'],partial=p['partial'],fixture_root=str(ROOT/'fixture'),capture='full-state-and-cached-endpoint-v1')

def authorization_reasons(c,a,check_live=True):
 s=specification(c);reasons=[]
 if not a:return ['diagnostic budget approval, reviewed implementation and clean closure permit required']
 if a.get('purpose')!='ch3_resource_probe'or a.get('probe_scope')!=SCOPE:reasons.append('foreign purpose/scope')
 if any(a.get(k)is not True for k in ('reviewed','execution_permitted','budget_authorized'))or a.get('synthetic_fixture')is not False:reasons.append('template/fixture/unapproved diagnostic budget')
 for k,v in dict(scope_sha=digest(s),plan_sha256=s['plan_sha256'],protocol_sha=digest(c),caps=CAPS,already_charged=DEBIT,parent_caps=PARENT_CAPS,authorized_task_ids=s['task_ids'],capture='full-state-and-cached-endpoint-v1',policy_sha=digest(POLICY),carry_forward_sha256=s['carry_forward_sha256']).items():
  if a.get(k)!=v:reasons.append('diagnostic binding mismatch: '+k)
 for k in ('code','environment','hardware','commit'):
  if not a.get(k):reasons.append('missing closure binding: '+k)
 for key in ('source_receipt','failure','partial'):
  b=s[key]
  if a.get(key)!=b or sha(b['path'])!=b['sha256']:reasons.append('source evidence changed: '+key)
 from utils.ch3_revision import boundary_reasons
 reasons+=boundary_reasons(c,a.get('safe_boundary',{}))
 paths=list(Path(c['execution']['evidence']).glob('formal-*/controller.json'))+[Path(c['timemixer_revision']['result_root'])/'controller.json',PACKAGE/'extension-probe-v1/controller.json',PACKAGE/'revision-numeric-probe-v1/controller.json',PACKAGE/'urban-numeric-diagnostic-v1/execution/controller.json']
 if check_live and any(controller_live(p)for p in paths):reasons.append('conflicting controller live')
 return reasons

def locations():return str(ROOT),str(ROOT/'fixture')
def worker_path(c,task,out):
 s=specification(c);p=Path(out)
 allowed=[ROOT/w['stage']/r for w in s['waves']for r in w['task_ids']if r==task]
 if str(p)!=str(p.resolve())or p not in allowed:raise ValueError('exact diagnostic task/stage/output required')
 return p.parent.name

def validate_worker(c,s,*,replay=False):
 if s.get('probe_scope')!=SCOPE:raise ValueError('worker scope mismatch')
 reasons=authorization_reasons(c,s.get('approval'),check_live=not replay)
 if reasons:raise PermissionError('; '.join(reasons))
 worker_path(c,s['task'],s['output'])
 if (s['session_root'],s['fixture_root'])!=locations()or s.get('prefix_files')or s.get('resume')or s.get('artifact_root')!=s['output']:raise ValueError('synthetic diagnostic boundary')
 if s['limits']!={'adam':6,'forward':10,'backward':6,'seconds':1800}or s.get('ids')!=[s['task']]:raise ValueError('exact worker ceiling/ID')

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
  from utils.ch3_admission_merge import inherited_sources
  inherited_sources(c)
  from ch3_runner import preflight
  reasons+=preflight(c,'TimeMixer',a,probe=True)
 return list(dict.fromkeys(reasons))

def run_waves(c,launch,stopped):
 """Fixed dispatch; any worker/resource failure stops the suffix, no fallback."""
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
 runs=specification(c)['task_ids']
 return {r:compare_confirmation(c,task_by_id(c,r),traces['serial',r],traces['q4',r])for r in runs}

def decision(c,traces,measurements):
 checks=comparisons(c,traces)
 if not all(x['passed']for x in checks.values()):raise ValueError('UrbanEV scoped numerical bound failed; no fallback/retry')
 serial=sum(m['measurement']['elapsed']for m in measurements if m['stage']=='serial')
 parallel=sum(m['measurement']['elapsed']for m in measurements if m['stage']=='q4')
 if parallel<=0 or serial/parallel<=1:raise ValueError('same-task captured confirmation benefit not demonstrated')
 return dict(status='Passed',concurrency=4,representatives=specification(c)['task_ids'],
   scope='six updates plus cached endpoint evaluation; captured-package timing, not formal throughput',
   numerical_protocol_sha=digest(c),policy_sha=digest(POLICY),
   **{'4':dict(resource='Passed',numerical_comparisons=list(checks.values()),makespan=parallel,speedup=serial/parallel)})

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
   dump(ROOT/'comparisons.json',results)
   measured_decision=decision(c,traces,measurements)
   if budget.counts!=CAPS:raise ValueError('exact diagnostic debit missing')
   evidence={str(p.relative_to(ROOT)):sha(p)for p in sorted(ROOT.rglob('*'))if p.is_file()and p.suffix in ('.json','.bin','.jsonl')}
   report=dict(probe_scope=SCOPE,scope_sha=digest(spec),scope=spec,execution_complete=True,reviewed=False,admission_granted=False,
      synthetic_fixture=False,comparisons=results,measurements=measurements,budget=vars(budget),
      decisions={'TimeMixer-UrbanEV-F4':measured_decision},policy=POLICY,
      cumulative={k:DEBIT[k]+budget.counts[k]for k in CAPS},evidence=evidence,**{k:a[k]for k in ('commit','code','protocol_sha','environment','hardware')})
   dump(ROOT/'complete.json',report)
   from utils.ch3_admission_merge import build_merged
   dump(PACKAGE_ROOT/'merged-admission.json',build_merged(c,dict(path=str(ROOT/'complete.json'),sha256=sha(ROOT/'complete.json'))))
 except BaseException as exc:
  if owned:dump(ROOT/'failure.json',dict(error=repr(exc),budget=vars(budget),reviewed=False,admission_granted=False))
  raise
 finally:signal.signal(signal.SIGTERM,old)

def validate_completion(c,r):
 spec=specification(c)
 if r.get('probe_scope')!=SCOPE or r.get('scope')!=spec or r.get('scope_sha')!=digest(spec)or r.get('execution_complete')is not True or r.get('synthetic_fixture')is not False or r.get('admission_granted')is not False:raise ValueError('diagnostic report identity')
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
   s=read(directory/run/'config.json');validate_worker(c,s,replay=True)
   if any(s['approval'].get(k)!=r.get(k)for k in ('commit','code','protocol_sha','environment','hardware')):raise ValueError('worker/report source mismatch')
   if read(directory/run/'budget.json')['counts']!={'adam':6,'forward':10,'backward':6}:raise ValueError('actual diagnostic worker debit')
   t=read(directory/run/'trajectory.json')
   for point in t['urban_diagnostic_trace']:
    for k in ('schema_file','data_file','meta_file'):
     if str(Path(point[k]).relative_to(ROOT))not in ev:raise ValueError('unbound diagnostic payload')
   rows.append(t)
  return read(paths[0]),rows
 from ch3_runner import code_binding
 if r.get('code')!=code_binding()or r.get('protocol_sha')!=digest(c):raise ValueError('confirmation source binding')
 traces,measurements=run_waves(c,replay,lambda:False)
 if measurements!=r['measurements']or comparisons(c,traces)!=r['comparisons']:raise ValueError('confirmation summary not reproducible from evidence')
 if r.get('policy')!=POLICY or r.get('decisions')!={'TimeMixer-UrbanEV-F4':decision(c,traces,measurements)}:raise ValueError('unmeasured confirmation decision')
 return r
