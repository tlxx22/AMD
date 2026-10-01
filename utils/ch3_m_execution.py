"""M-specific authorization/config/waves using the existing restricted worker."""
import json,math,os,copy,time,subprocess,sys
from pathlib import Path
from utils.ch3_contract import ROOT,digest,profile,task_by_id,step_arithmetic,validate_manifest
from utils import ch3_m_tasks as scope
PROBE_ROOT=scope.PACKAGE/'probe-execution-v1'
CONTROL=scope.RESULT/'queues/m-baselines-v1'

def launcher_log(probe):return scope.PACKAGE/'m-baselines-probe-v1-launcher.log'if probe else CONTROL.parent/'m-baselines-v1-launcher.log'
def fresh_conflicts(c,probe=False,*,launch_token=None,approval=None):
 root=PROBE_ROOT if probe else CONTROL;reasons=[]
 from utils import ch3_m_launch as launch
 if root.exists()or root.is_symlink():reasons.append('M retained execution/controller/log; fresh repeat forbidden')
 kind='probe'if probe else'formal'
 if launch.claim_path(kind).exists()or launch.claim_path(kind).is_symlink():reasons.append('M launch already claimed; duplicate start forbidden')
 if launcher_log(probe).exists()or launcher_log(probe).is_symlink()or launch.metadata_path(kind).exists()or launch.metadata_path(kind).is_symlink():
  try:launch.verify(c,'probe'if probe else'formal',launch_token,approval=approval)
  except (OSError,ValueError,PermissionError):reasons.append('M retained launcher or invalid current launch identity; fresh repeat forbidden')
 if not probe and any(scope.result_path(t).exists()or scope.result_path(t).is_symlink()for t in c['tasks']):reasons.append('M task output conflict')
 return reasons

def gpu_environment_reasons(env=None):
 values=os.environ if env is None else env
 return []if values.get('CUDA_VISIBLE_DEVICES')in (None,'0')else['M GPU0 mapping unavailable/changed; CPU CUDA mask cannot authorize workers']

def old_boundary(c):
 """Inspect old queue controls only; never stop it or load its weights/results."""
 root=Path(c['m_experiment']['old_queue']);plan=scope.bound(c['m_experiment']['old_queue_plan'])
 if (root/'STOP').exists()or(root/'failure.json').exists():raise ValueError('old MS queue failure/STOP requires its own audit')
 if not(root/'complete.json').exists():raise ValueError('old MS queue including TimeXer/N/S not yet complete')
 complete=json.loads((root/'complete.json').read_text())
 if complete.get('task_ids')!=plan['task_ids']or complete.get('groups')!=plan['order']or complete.get('plan')!=digest(plan):raise ValueError('old queue complete exact scope')
 from m6_remaining_entry import same
 controller=json.loads((root/'controller.json').read_text())
 if same(controller):raise ValueError('old queue controller original instance active')
 sources={'complete':scope.ref(root/'complete.json'),'controller':scope.ref(root/'controller.json')}
 from utils.ch3_remaining import technical_handoff
 parent=json.loads(subprocess.check_output(['git','-C',str(ROOT),'show',scope.BASE+':configs/ch3_formal_profiles.json']))
 # Existing JSON-only technical gate; result review remains pending. No weights/test.
 for g in plan['groups']:
  hand=scope.bound(complete['handoffs'][g['model']]);actual=technical_handoff(parent,g,scope.bound(scope.bound(controller['approval'])['children'][g['model']]))
  if actual!=hand:raise ValueError('old group technical handoff changed')
  sources[g['model']]=complete['handoffs'][g['model']]
 return dict(kind='old_MS_queue_technical_complete',result_review='pending',task_ids=plan['task_ids'],sources=sources)

def plan(c):
 ids=[t['id']for t in c['tasks']]
 return dict(id=scope.ID,probe_scope=scope.PROBE,protocol_sha=digest(c),order=list(scope.MODELS),domains=list(scope.DOMAINS),task_ids=ids,profile_shas={t['id']:digest(profile(c,t))for t in c['tasks']},run_budget=dict(runs=84,run_epochs=840),max_optimizer_steps=sum(step_arithmetic(c,t)['max_optimizer_steps']for t in c['tasks']),groups=[dict(id=g['id'],task_ids=g['task_ids'],serial=[[r]for r in g['task_ids']],q4=[g['task_ids']],q2=[g['task_ids'][:2],g['task_ids'][2:]],fallback=['q4','q2','q1'],resource_failure_only=True)for g in c['groups']],probe_caps=scope.CAPS,nominal=dict(workers=168,adam=1008,forward=1344,backward=1008),all_q2_extra=dict(workers=84,adam=504,forward=672,backward=504),cpu_forward_cap=64,numeric_policies=c['m_experiment']['numeric_policies'])

def approval_template(c,probe):
 from ch3_runner import code_binding,environment_binding,hardware_binding
 b=scope.bound(c['m_experiment']['data_binding_artifact'])
 return dict(purpose='ch3_resource_probe'if probe else'ch3_formal',m_scope=scope.PROBE if probe else scope.ID,reviewed=False,execution_permitted=False,budget_authorized=False,synthetic_fixture=False,commit=None,code=code_binding(),protocol_sha=digest(c),environment=environment_binding(),hardware=hardware_binding(),plan=None,authorized_task_ids=[t['id']for t in c['tasks']],profile_shas={t['id']:digest(profile(c,t))for t in c['tasks']},parent_MS_profiles={t['id']:profile(c,t)['parent_MS_profile']for t in c['tasks']},data_binding_artifact=c['m_experiment']['data_binding_artifact'],data_bindings=b['data_bindings'],source_states=b['source_states'],from_scratch=True,task='M',metric_scope='all_channels',seed_list=[2024],additional_search=0,old_boundary=None,resource_report=None,run_budget=dict(runs=84,run_epochs=840),max_optimizer_steps=plan(c)['max_optimizer_steps'],caps=scope.CAPS if probe else None,already_charged=dict(adam=0,forward=0,backward=0),source='Non-executable template; actual-byte review, independent clean closure, matching permit and old full queue completion required')

def authorization_reasons(c,a,probe=False,worker=False):
 from ch3_runner import code_binding,environment_binding,hardware_binding,git
 reasons=[]
 expected=scope.PROBE if probe else scope.ID
 if not a:return ['matching M reviewed authorization absent']
 machine=a.get('review_mode')=='preauthorized_machine_gate'
 if not machine and a.get('review_mode')is not None:reasons.append('unknown M review mode; exact manual or machine authorization required')
 if machine:
  try:
   from utils.ch3_m_auto import validate_permit
   validate_permit(c,a,probe,worker=worker)
  except(OSError,KeyError,ValueError,TypeError,subprocess.CalledProcessError)as exc:reasons.append(str(exc))
 reasons+=gpu_environment_reasons()
 if not a.get('plan'):reasons.append('bound exact M plan required')
 if not a.get('old_boundary'):reasons.append('old MS queue including TimeXer/N/S completion boundary required')
 if not probe and not a.get('resource_report'):reasons.append('reviewed new M probe report required before formal training')
 if not probe and(a.get('structure_frozen')is not True or a.get('m6_authorized')is not True):reasons.append('existing M6 structure/stage authorization required')
 for ok,why in [(a.get('m_scope')==expected,'M scope mismatch'),(a.get('purpose')==('ch3_resource_probe'if probe else'ch3_formal'),'M purpose'),((a.get('reviewed')is True or machine)and a.get('execution_permitted')is True,'M template not executable'),(a.get('budget_authorized')is True,'M independent budget approval'),(a.get('synthetic_fixture')is False,'fixture permission forbidden'),(a.get('from_scratch')is True,'M fresh-only'),(a.get('task')=='M'and a.get('metric_scope')=='all_channels','M supervision identity')]:
  if not ok:reasons.append(why)
 if git('branch','--show-current')!='m6/m-baselines-v1'or git('status','--porcelain','--untracked-files=all')or a.get('commit')!=git('rev-parse','HEAD')or a.get('commit')==scope.BASE:reasons.append('independent reviewed clean M closure required')
 for k,v in dict(code=code_binding(),protocol_sha=digest(c),environment=environment_binding(),hardware=hardware_binding(),authorized_task_ids=[t['id']for t in c['tasks']],profile_shas={t['id']:digest(profile(c,t))for t in c['tasks']},parent_MS_profiles={t['id']:profile(c,t)['parent_MS_profile']for t in c['tasks']},data_binding_artifact=c['m_experiment']['data_binding_artifact'],run_budget=dict(runs=84,run_epochs=840),max_optimizer_steps=plan(c)['max_optimizer_steps'],seed_list=[2024],additional_search=0).items():
  if a.get(k)!=v:reasons.append('M '+k+' mismatch')
 try:
  if scope.bound(a['plan'])!=plan(c):raise ValueError('exact M plan')
  b=scope.bound(c['m_experiment']['data_binding_artifact'])
  if a['data_bindings']!=b['data_bindings']or a['source_states']!=b['source_states']:raise ValueError('M derived data bindings changed')
  from utils.ch3_data import verify_source_state
  for d in c['datasets'].values():verify_source_state(d)
  for path,expected_state in b['source_states'].items():
   st=Path(path).stat();actual=dict(device=st.st_dev,inode=st.st_ino,size=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns)
   if actual!=expected_state:raise ValueError('M source stat changed')
  if not worker:
   if not a.get('old_boundary'):raise ValueError('old MS queue including TimeXer/N/S completion boundary required')
   if scope.bound(a['old_boundary'])!=old_boundary(c):raise ValueError('old queue completion boundary changed')
  else:
   boundary=scope.bound(a['old_boundary'])
   if boundary['kind']!='old_MS_queue_technical_complete':raise ValueError('worker M boundary kind')
  if probe:
   if a.get('caps')!=scope.CAPS or a.get('already_charged')!=dict(adam=0,forward=0,backward=0):raise ValueError('M independent caps/debit')
  else:
   if not a.get('resource_report'):raise ValueError('reviewed new M probe report required before formal training')
   report=scope.bound(a['resource_report'])
   if machine:
    from utils.ch3_m_auto import check_admission
    check_admission(c,report,worker=worker)
   elif worker:
    if report.get('reviewed')is not True or report.get('purpose')!='M_resource_admission'or report.get('protocol_sha')!=digest(c)or report.get('code')!=a['code']:raise ValueError('M worker reviewed resource binding')
   else:validate_probe_report(c,report)
 except (OSError,KeyError,ValueError,TypeError)as e:reasons.append(str(e))
 for src in c['sources'].values():
  if subprocess.check_output(['git','-C',src['repository'],'rev-parse','HEAD'],text=True).strip()!=src['commit']:reasons.append('author commit changed')
  if any(scope.sha(p)!=h for p,h in src['files'].items()):reasons.append('author SHA changed')
 return list(dict.fromkeys(reasons))

def validate_probe_report(c,r):
 if r.get('purpose')=='M_pre_authorized_technical_admission_v1':
  from utils.ch3_m_auto import check_admission
  return check_admission(c,r)
 if r.get('purpose')!='M_resource_admission'or r.get('reviewed')is not True:raise ValueError('reviewed new M probe report required')
 raw=scope.bound(r['review']['original_report'])
 if {k:v for k,v in r.items()if k not in ('reviewed','review')}!={k:v for k,v in raw.items()if k!='reviewed'}or not r['review'].get('source'):raise ValueError('M report review source')
 validate_probe_completion(c,raw)
 return r

def exact_path(c,t,purpose,phase,out):
 expected=scope.result_path(t)if purpose=='ch3_formal'else PROBE_ROOT/t['group']/phase/t['id']
 if Path(out)!=expected or Path(out).is_symlink()or expected.resolve()!=expected:raise ValueError('exact isolated M worker output')
 return expected

def validate_worker(c,s):
 from m5_formal_entry import repository_files
 if s.get('purpose')not in ('ch3_probe','ch3_formal')or s.get('resume')is not False:raise ValueError('M exact fresh purpose')
 if s.get('kernel_probe',False)is not False or s.get('device')!='cuda:0':raise ValueError('M GPU0 only; no backend/operator replay')
 t=task_by_id(c,s['task']);probe=s['purpose']=='ch3_probe';a=s['approval']
 if s['m_scope']!=a['m_scope']or s['protocol_sha']!=digest(c)or s['bound_files']!=repository_files()or s['ids']!=[t['id']]:raise ValueError('M config/task/source identity')
 exact_path(c,t,s['purpose'],s.get('m_phase'),s['output'])
 if s['artifact_root']!=s['output']:raise ValueError('M fresh artifact root')
 if probe and s['m_phase']not in ('serial','q4','q2'):raise ValueError('M phase')
 roots=(str(PROBE_ROOT),str(scope.PACKAGE/'fixtures'))if probe else(str(scope.RESULT),str(scope.PACKAGE/'fixtures'))
 if (s['session_root'],s['fixture_root'])!=roots or os.environ.get('TMPDIR')!=roots[1]or not Path(roots[1]).is_dir():raise ValueError('M fixture/environment scope')
 expected_prefix={}if probe else {c['datasets'][t['dataset']]['path']:c['datasets'][t['dataset']]['endpoints'][2]}
 if s['prefix_files']!=expected_prefix:raise ValueError('M numeric data/test capability boundary')
 if probe and s['limits']!=dict(adam=6,forward=8,backward=6,seconds=1800):raise ValueError('M six-step exact budget')
 if not probe and s['limits']!=dict(adam=None,forward=None,backward=None,seconds=None):raise ValueError('M fixed formal meter/epoch contract')
 src=c['sources'].get(t['model']);expected=src['files']if src else{}
 if s['author_files']!=expected:raise ValueError('M own author source')
 if s.get('metadata_files')!=metadata_files(c,a):raise ValueError('M metadata allowlist binding')
 reasons=authorization_reasons(c,a,probe=probe,worker=True)
 if reasons:raise PermissionError('; '.join(reasons))

def metadata_files(c,a):
 refs=[a.get('plan'),a.get('old_boundary'),a.get('resource_report'),c['m_experiment']['data_binding_artifact']]
 result={r['path']:r['sha256']for r in refs if r}
 if a.get('resource_report'):
  r=scope.bound(a['resource_report'])
  refs2=[r['original_report'],r['probe_permit']]if a.get('review_mode')=='preauthorized_machine_gate'else[r['review']['original_report']]+list(r.get('evidence',{}).values())
  if a.get('review_mode')=='preauthorized_machine_gate':refs2.append(scope.bound(r['original_report'])['approval'])
  result.update({r['path']:r['sha256']for r in refs2})
 if a.get('review_mode')=='preauthorized_machine_gate':
  from utils.ch3_m_auto import SOURCES
  result.update({r['path']:r['sha256']for r in SOURCES.values()})
 return result

def make_config(c,purpose,out,*,task,approval,artifact_root=None,resume=False):
 from ch3_runner import dump
 from m5_formal_entry import repository_files
 t=task_by_id(c,task);probe=purpose=='ch3_probe';out=Path(out);phase=out.parent.name if probe else None
 exact_path(c,t,purpose,phase,out)
 reasons=authorization_reasons(c,approval,probe=probe)
 if reasons:raise PermissionError('; '.join(reasons))
 if resume:raise PermissionError('M no automatic resume')
 out.mkdir(parents=True,exist_ok=False);src=c['sources'].get(t['model'])
 limits=dict(adam=6,forward=8,backward=6,seconds=1800)if probe else dict(adam=None,forward=None,backward=None,seconds=None)
 s=dict(version='restricted-regression-minimal-v3',repo=str(ROOT),tool_root=str(ROOT/'tools/restricted_regression'),purpose=purpose,task=task,case=None,ids=[task],protocol_sha=digest(c),m_scope=approval['m_scope'],m_phase=phase,session_root=str(PROBE_ROOT if probe else scope.RESULT),fixture_root=str(scope.PACKAGE/'fixtures'),audit_log=str(out/'audit.jsonl'),budget_file=str(out/'budget.json'),output=str(out),limits=limits,bound_files=repository_files(),author_roots=[s['repository']for s in c['sources'].values()],author_files=src['files']if src else{},prefix_files={}if probe else{c['datasets'][t['dataset']]['path']:c['datasets'][t['dataset']]['endpoints'][2]},metadata_files=metadata_files(c,approval),forbidden_roots=[],device='cuda:0',approval=approval,artifact_root=str(out),resume=False,kernel_probe=False)
 for dirname in ('cache/torch/kernels','mpl','cuda-cache'):(out/dirname).mkdir(parents=True,exist_ok=True)
 dump(out/'config.json',s)
 from resource_budget import initialize
 initialize(s['budget_file'],purpose,limits)
 return s

def validate_wave(c,configs,out):
 if not configs:raise ValueError('empty M wave')
 purpose=configs[0]['purpose'];probe=purpose=='ch3_probe';ids=[s['task']for s in configs];ts=[task_by_id(c,r)for r in ids];g=next(g for g in c['groups']if g['id']==ts[0]['group'])
 if len(ids)!=len(set(ids))or any(t['group']!=g['id']for t in ts)or any(s['approval']!=configs[0]['approval']for s in configs):raise ValueError('mixed M wave/scope')
 if probe:
  phase=configs[0]['m_phase'];valid=[[r]for r in g['task_ids']]if phase=='serial'else[g['task_ids']]if phase=='q4'else[g['task_ids'][:2],g['task_ids'][2:]]if phase=='q2'else[]
  stop=PROBE_ROOT/'STOP';root=PROBE_ROOT
 else:
  report=scope.bound(configs[0]['approval']['resource_report']);q=report['decisions'][g['id']]['concurrency'];valid=wave_ids(g['task_ids'],q);stop=CONTROL/'STOP';root=scope.RESULT
 if ids not in valid or not Path(out).resolve().is_relative_to(root):raise ValueError('M fixed wave/output mismatch')
 for s,t in zip(configs,ts):exact_path(c,t,purpose,s.get('m_phase'),s['output'])
 return stop

def wave_ids(ids,q):
 if q not in (1,2,4):raise ValueError('M concurrency')
 return [ids[i:i+q]for i in range(0,len(ids),q)]

def debit(budget,workers):
 request={k:workers*v for k,v in dict(adam=6,forward=8,backward=6).items()}
 if any(budget['reserved'][k]+request[k]>scope.CAPS[k]for k in request):raise ValueError('M GPU budget exhausted before dispatch')
 for k,v in request.items():budget['reserved'][k]+=v
 return request

def resource_fallback(measured):return measured.get('failure_kind')=='resource'
def wave_passed(row):return row.get('failure')is None and all(v==0 for v in row.get('returncodes',[]))and bool(row.get('returncodes'))and row.get('resource_admission')is True

def compare(c,t,x,y):
 from ch3_runner import compare_probe_trajectories,_compare_full_numeric_files
 row=compare_probe_trajectories(c,t,x,y)
 if len(x.get('M_full_state_trace',[]))!=6 or len(y.get('M_full_state_trace',[]))!=6:raise ValueError('M six full gradient/buffer/optimizer snapshots required')
 for trace in (x['M_full_state_trace'],y['M_full_state_trace']):
  for point in trace:
   if not Path(point['schema_file']).resolve().is_relative_to(PROBE_ROOT)or not Path(point['data_file']).resolve().is_relative_to(PROBE_ROOT):raise ValueError('M full numeric payload outside scope')
 if scope.numeric_policy(c,t)is None:
  for a,b in zip(x['M_full_state_trace'],y['M_full_state_trace']):
   z=_compare_full_numeric_files(dict(state_atol=0),a,b)
   if not z['passed']:row['passed']=False;row['reason']='M full gradient/buffer/optimizer exact mismatch'
 return row

def run_probe(c,a,root=PROBE_ROOT):
 from ch3_runner import dump
 from m5_formal_entry import make_config,run_configs
 budget=dict(caps=scope.CAPS,reserved=dict(adam=0,forward=0,backward=0),actual=dict(adam=0,forward=0,backward=0),refund=False);decisions={};evidence={}
 def wave(group,phase,ids,n):
  if (root/'STOP').exists():raise InterruptedError('M probe STOP')
  debit(budget,len(ids));dump(root/'budget.json',budget)
  cfg=[]
  try:
   for r in ids:cfg.append(make_config(c,'ch3_probe',root/group/phase/r,task=r,approval=a))
   measured=run_configs(cfg,root/group/phase/('wave-'+str(n)),monitor=True)
   evidence[group+'/'+phase+'/'+str(n)]=scope.ref(root/group/phase/('wave-'+str(n))/'process.json')
  finally:
   for s in cfg:
    b=json.loads(Path(s['budget_file']).read_text())
    for k in budget['actual']:budget['actual'][k]+=b['counts'][k]
   dump(root/'budget.json',budget)
  return measured
 try:
  for g in c['groups']:
   traces={};serial=[]
   for n,r in enumerate(g['task_ids']):
    m=wave(g['id'],'serial',[r],n);serial.append(m)
    if not wave_passed(m):raise RuntimeError('M serial reference/resource failed; no fallback')
    traces['serial',r]=json.loads((root/g['id']/'serial'/r/'trajectory.json').read_text())
   chosen=None;parallel=[];comparisons=[];attempts=[]
   for phase,q in [('q4',4),('q2',2)]:
    parallel=[];comparisons=[];res_failure=False
    for n,ids in enumerate(wave_ids(g['task_ids'],q)):
     m=wave(g['id'],phase,ids,n);parallel.append(m)
     if not wave_passed(m):
      if resource_fallback(m):res_failure=True;break
      raise RuntimeError('M nonresource/unknown worker failure; stop')
     for r in ids:
      traces[phase,r]=json.loads((root/g['id']/phase/r/'trajectory.json').read_text());row=compare(c,task_by_id(c,r),traces['serial',r],traces[phase,r]);comparisons.append(row)
      if not row['passed']:raise ValueError('M numeric comparison failed; no concurrency fallback')
    attempts.append(dict(phase=phase,waves=parallel,resource_failed=res_failure))
    if res_failure:
     if q==2:chosen=1
     continue
    serial_time=sum(x['elapsed']for x in serial);parallel_time=sum(x['elapsed']for x in parallel)
    # Existing short-package benefit line (gain, exact same four tasks). No scientific search.
    if parallel_time>=serial_time:
     raise ValueError('M short-package benefit failed; no nonresource concurrency fallback')
    chosen=q;break
   if chosen is None:raise RuntimeError('M concurrency unresolved')
   decisions[g['id']]=dict(status='Passed',concurrency=chosen,serial=serial,parallel=parallel,numerical_comparisons=comparisons,attempts=attempts,policy_sha=digest(c['m_experiment']['numeric_policies'][g['id']]),makespan_scope='captured six-step synthetic technical package',q1_only_if_serial_valid=True)
   dump(root/'progress.json',dict(decisions=decisions,budget=budget))
  artifacts={}
  for key in evidence:
   group,phase,n=key.split('/');g=next(g for g in c['groups']if g['id']==group);ids=([[x]for x in g['task_ids']]if phase=='serial'else wave_ids(g['task_ids'],int(phase[1:])))[int(n)]
   for run in ids:
    d=root/group/phase/run
    for name in ('config.json','budget.json','trajectory.json','audit.jsonl','runtime.json'):
     p=d/name;artifacts[str(p)]=scope.ref(p)if p.exists()else None
    if(d/'trajectory.json').exists():
     tr=json.loads((d/'trajectory.json').read_text())
     for point in tr.get('M_full_state_trace',[]):
      for key in ('schema_file','data_file'):
       p=Path(point[key])
       if not p.resolve().is_relative_to(d)or p.is_symlink():raise ValueError('M payload outside its exact worker')
       artifacts[str(p)]=scope.ref(p)
   p=root/group/phase/('wave-'+n)/'memory.jsonl';artifacts[str(p)]=scope.ref(p)if p.exists()else None
  report=dict(purpose='M_resource_admission',m_scope=scope.PROBE,reviewed=False,admission_granted=False,protocol_sha=digest(c),commit=a['commit'],code=a['code'],environment=a['environment'],hardware=a['hardware'],approval=scope.ref(root/'approval.json'),decisions=decisions,budget=budget,evidence=evidence,artifacts=artifacts,execution_complete=True)
  validate_probe_completion(c,report);dump(root/'complete.json',report);return report
 except BaseException as exc:
  dump(root/'failure.json',dict(error=repr(exc),decisions=decisions,budget=budget,reviewed=False));raise

def validate_probe_completion(c,r):
 from ch3_runner import code_binding,environment_binding,hardware_binding,git,_compare_full_numeric_files
 if r.get('m_scope')!=scope.PROBE or r.get('protocol_sha')!=digest(c)or set(r.get('decisions',{}))!={g['id']for g in c['groups']}or r.get('execution_complete')is not True:raise ValueError('M probe complete exact 21 scope')
 if r.get('code')!=code_binding()or r.get('environment')!=environment_binding()or r.get('hardware')!=hardware_binding():raise ValueError('M probe source/environment/hardware')
 if r.get('commit')!=git('rev-parse','HEAD')or r.get('purpose')!='M_resource_admission':raise ValueError('M probe closure/purpose')
 root=PROBE_ROOT
 if(root/'STOP').exists()or(root/'failure.json').exists():raise ValueError('M retained STOP/failure cannot be admission complete')
 if r.get('approval',{}).get('path')!=str(root/'approval.json'):raise ValueError('M actual probe approval path')
 approval=scope.bound(r['approval'])
 for k in ('code','commit','protocol_sha','environment','hardware'):
  if approval.get(k)!=r.get(k):raise ValueError('M probe approval/source mismatch')
 if json.loads((root/'budget.json').read_text())!=r['budget']or r['budget'].get('caps')!=scope.CAPS:raise ValueError('M actual probe budget/caps')
 expected_keys=set()
 for g in c['groups']:
  d=r['decisions'][g['id']];q=d['concurrency'];attempts=d.get('attempts',[])
  if d['status']!='Passed'or type(q)is not int or q not in (1,2,4):raise ValueError('M resource decision')
  if [a['phase']for a in attempts]!=(['q4']if q==4 else['q4','q2']):raise ValueError('M resource-only fallback history missing/order')
  expected_keys.update(g['id']+'/serial/'+str(n)for n in range(4))
  for attempt in attempts:
   phase=attempt['phase'];waves=attempt['waves'];width=int(phase[1:])
   if not waves or len(waves)>4//width:raise ValueError('M fallback wave count')
   failed=not wave_passed(waves[-1])
   if type(attempt['resource_failed'])is not bool or attempt['resource_failed']!=failed or not all(wave_passed(v)for v in waves[:-1]):raise ValueError('M fallback wave/failed classification')
   if failed and not resource_fallback(waves[-1]):raise ValueError('M nonresource failure cannot fallback')
   if not failed and len(waves)!=4//width:raise ValueError('M incomplete successful parallel attempt')
   if (phase=='q4'and q!=4 and not failed)or(phase=='q2'and (q==1)!=failed):raise ValueError('M fallback chosen q inconsistent')
   expected_keys.update(g['id']+'/'+phase+'/'+str(n)for n in range(len(waves)))
  if d['parallel']!=attempts[-1]['waves']:raise ValueError('M final attempt measurement mismatch')
 if set(r['evidence'])!=expected_keys:raise ValueError('M exact attempted evidence coverage')
 for path,ref in r['artifacts'].items():
  p=Path(path)
  if not p.resolve().is_relative_to(root)or p.is_symlink()or(ref is None and p.exists())or(ref is not None and ref!=scope.ref(p)):raise ValueError('M worker artifact binding changed')
 reserved=dict(adam=0,forward=0,backward=0);actual=dict(reserved)
 for key,ref in r['evidence'].items():
  group,phase,n=key.split('/');g=next(g for g in c['groups']if g['id']==group);ids=([[x]for x in g['task_ids']]if phase=='serial'else wave_ids(g['task_ids'],int(phase[1:])))[int(n)]
  process=root/group/phase/('wave-'+n)/'process.json'
  if ref!=scope.ref(process):raise ValueError('M process path/source identity')
  v=scope.bound(ref)
  if v.get('failure_kind')not in (None,'resource')or(not wave_passed(v)and not resource_fallback(v)):raise ValueError('M unknown/nonresource failure')
  for run in ids:
   d=root/group/phase/run;cfg=json.loads((d/'config.json').read_text());b=json.loads((d/'budget.json').read_text())
   if set(b.get('counts',{}))!={'adam','forward','backward'}or any(type(b['counts'][k])is not int or not 0<=b['counts'][k]<=cap for k,cap in dict(adam=6,forward=8,backward=6).items()):raise ValueError('M worker actual counter type/range')
   if cfg['task']!=run or cfg['m_phase']!=phase or cfg['protocol_sha']!=digest(c)or cfg['approval']!=approval or cfg['m_scope']!=scope.PROBE or cfg['output']!=str(d)or cfg['prefix_files']or cfg['limits']!=dict(adam=6,forward=8,backward=6,seconds=1800):raise ValueError('M process/task/permit identity')
   for name in ('config.json','budget.json'):
    if r['artifacts'].get(str(d/name))!=scope.ref(d/name):raise ValueError('M required worker binding missing')
   if wave_passed(v):
    if b['counts']!=dict(adam=6,forward=8,backward=6):raise ValueError('M exact successful six-step costs')
    for name in ('trajectory.json','audit.jsonl','runtime.json'):
     if r['artifacts'].get(str(d/name))!=scope.ref(d/name):raise ValueError('M successful worker artifact missing')
    tr=json.loads((d/'trajectory.json').read_text());t=task_by_id(c,run)
    if not compare(c,t,tr,tr)['passed']:raise ValueError('M self state/finite/identity check')
    for point in tr['M_full_state_trace']:
     for key in ('schema_file','data_file'):
      p=Path(point[key])
      if not p.resolve().is_relative_to(d)or p.is_symlink()or r['artifacts'].get(str(p))!=scope.ref(p):raise ValueError('M full state payload binding')
     if not _compare_full_numeric_files(dict(state_atol=0),point,point)['passed']:raise ValueError('M full state finite self check')
   for k in reserved:reserved[k]+=dict(adam=6,forward=8,backward=6)[k];actual[k]+=b['counts'][k]
 if r['budget']['actual']!=actual or r['budget']['reserved']!=reserved or r['budget']['refund']is not False:raise ValueError('M replay actual/precharged accounting')
 for g in c['groups']:
  d=r['decisions'][g['id']]
  if d['status']!='Passed'or d['concurrency']not in (1,2,4):raise ValueError('M resource decision')
  serial=[scope.bound(r['evidence'][g['id']+'/serial/'+str(n)])for n in range(4)]
  if d['serial']!=serial or not all(wave_passed(v)for v in serial)or d['policy_sha']!=digest(c['m_experiment']['numeric_policies'][g['id']]):raise ValueError('M four serial process/policy bindings')
  if d['concurrency']>1:
   phase='q'+str(d['concurrency']);parallel=[scope.bound(r['evidence'][g['id']+'/'+phase+'/'+str(n)])for n in range(4//d['concurrency'])]
   if d['parallel']!=parallel or not all(wave_passed(v)for v in parallel)or sum(v['elapsed']for v in parallel)>=sum(v['elapsed']for v in serial):raise ValueError('M chosen concurrency process/benefit mismatch')
  final_comparisons=[]
  for attempt in d['attempts']:
   phase=attempt['phase'];width=int(phase[1:]);comparisons=[]
   replay=[scope.bound(r['evidence'][g['id']+'/'+phase+'/'+str(n)])for n in range(len(attempt['waves']))]
   if replay!=attempt['waves']:raise ValueError('M fallback process measurements changed')
   for n,v in enumerate(replay):
    if not wave_passed(v):continue
    for run in wave_ids(g['task_ids'],width)[n]:
     x=json.loads((root/g['id']/'serial'/run/'trajectory.json').read_text());y=json.loads((root/g['id']/phase/run/'trajectory.json').read_text());row=compare(c,task_by_id(c,run),x,y)
     if not row['passed']:raise ValueError('M failed numeric gate in fallback history')
     comparisons.append(row)
   final_comparisons=comparisons
  if d['numerical_comparisons']!=final_comparisons:raise ValueError('M numeric summary not reproducible')
  for run in g['task_ids']:
   t=task_by_id(c,run);sp=root/g['id']/'serial'/run;tr=json.loads((sp/'trajectory.json').read_text());b=json.loads((sp/'budget.json').read_text())
   if b['counts']!=dict(adam=6,forward=8,backward=6)or tr['profile_sha']!=digest(profile(c,t))or not tr['finite']:raise ValueError('M serial evidence/counter')
   if d['concurrency']>1:
    phase='q'+str(d['concurrency']);pp=root/g['id']/phase/run;other=json.loads((pp/'trajectory.json').read_text())
    if json.loads((pp/'budget.json').read_text())['counts']!=dict(adam=6,forward=8,backward=6)or not compare(c,t,tr,other)['passed']:raise ValueError('M repeated-state replay failed')
 for ref in r['evidence'].values():
  v=scope.bound(ref)
  # Failed resource attempts remain evidence and cannot become successful updates.
  if v.get('failure_kind')not in (None,'resource'):raise ValueError('M business failure in evidence')
 if any(type(r['budget'][field][k])is not int or r['budget'][field][k]<0 for field in ('actual','reserved')for k in scope.CAPS)or any(r['budget']['reserved'][k]>scope.CAPS[k]or r['budget']['actual'][k]>r['budget']['reserved'][k]for k in scope.CAPS):raise ValueError('M actual/reserved/cap')
 return r

CPU_ROOT=scope.PACKAGE/'cpu-shapes-v1'
CPU_ID='test_m6_m_tasks.MCPUShapeTests.test_native_all_channel_output'
def cpu_tasks(c,model):
 if model not in scope.MODELS:raise ValueError('CPU model scope')
 hs=scope.HS if model in ('AMD','TimeXer')else(96,720)
 return [t['id']for t in c['tasks']if t['model']==model and t['h']in hs]
def validate_cpu(c,s):
 from m5_formal_entry import repository_files
 t=task_by_id(c,s['task']);model=s['case'];source=c['sources'].get(model)
 if s['purpose']!='ch3_m_cpu_shapes'or s['m_scope']!='m-baselines-cpu-shapes-v1'or s['ids']!=[CPU_ID]or s['cpu_tasks']!=cpu_tasks(c,model)or s['task']!=s['cpu_tasks'][0]or s['bound_files']!=repository_files()or s['protocol_sha']!=digest(c):raise ValueError('pre-fixture CPU exact source/IDs/model cases')
 if s['output']!=str(CPU_ROOT/model)or s['session_root']!=str(CPU_ROOT)or s['fixture_root']!=str(scope.PACKAGE/'fixtures')or s['budget_file']!=str(CPU_ROOT/'shared-budget.json'):raise ValueError('CPU exact output/global budget')
 if s['limits']!=dict(adam=0,forward=64,backward=0,seconds=3600,rss=4*1024**3)or s['prefix_files']or s['author_files']!=(source['files']if source else{}):raise ValueError('CPU exact64/read/source scope')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=''or os.environ.get('TMPDIR')!=s['fixture_root']:raise ValueError('CPU CUDA/fixture environment')
 if any(os.environ.get(k)!='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')):raise ValueError('CPU single thread')

def make_cpu_config(c,model):
 from ch3_runner import dump
 from m5_formal_entry import repository_files
 ids=cpu_tasks(c,model);out=CPU_ROOT/model;out.mkdir(exist_ok=False);source=c['sources'].get(model)
 config=dict(version='restricted-regression-minimal-v3',repo=str(ROOT),tool_root=str(ROOT/'tools/restricted_regression'),purpose='ch3_m_cpu_shapes',case=model,task=ids[0],cpu_tasks=ids,ids=[CPU_ID],protocol_sha=digest(c),m_scope='m-baselines-cpu-shapes-v1',session_root=str(CPU_ROOT),fixture_root=str(scope.PACKAGE/'fixtures'),audit_log=str(out/'audit.jsonl'),budget_file=str(CPU_ROOT/'shared-budget.json'),output=str(out),limits=dict(adam=0,forward=64,backward=0,seconds=3600,rss=4*1024**3),bound_files=repository_files(),author_roots=[s['repository']for s in c['sources'].values()],author_files=source['files']if source else{},prefix_files={},metadata_files={},forbidden_roots=[],device='cpu',approval=None,artifact_root=str(out),resume=False,kernel_probe=False)
 for dirname in ('cache/torch/kernels','mpl','cuda-cache'):(out/dirname).mkdir(parents=True,exist_ok=True)
 dump(out/'config.json',config);return config
