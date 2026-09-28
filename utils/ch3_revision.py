"""Fixed, fresh TimeMixer repeat after explicit user retirement; no auto recovery."""
import copy,json,hashlib,os
from pathlib import Path
from utils.ch3_contract import digest,profile
REVISION='timemixer-fixedlr-v2'
DOMAINS=('ETTh1','Weather','ECL','Exchange')
def ids():return [f'TimeMixer-{d}-MS-f1-h{h}-s2024'for d in DOMAINS for h in (96,192,336,720)]
def changed_groups():return ['TimeMixer-'+d+'-MS'for d in DOMAINS[:3]]
def root(c):return Path(c['execution']['evidence'])/'revisions'/REVISION/'formal-TimeMixer'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate_revision_profiles(c):
 r=c.get('timemixer_revision')
 if not r:return
 if r['id']!=REVISION or r['attempt']!=2 or r['task_ids']!=ids()or r['changed_group_ids']!=changed_groups()or r['result_root']!=str(root(c)):raise ValueError('exact revision identity/queue')
 if (r['fresh_runs'],r['run_epoch_cap'],r['coverage_runs'],r['coverage_epochs'],r['attempt_runs'],r['attempt_epoch_cap'])!=(16,200,552,6240,568,6440):raise ValueError('fixed repeated-attempt budget')
 for t in c['tasks']:
  if t['id']not in ids():continue
  p=profile(c,t);old=copy.deepcopy(p);expected_lr=.0003 if t['dataset']=='Exchange'else .001
  if p['training']['lr']!=expected_lr:raise ValueError('one fixed LR, no fallback/search')
  if t['dataset']!='Exchange':old['training']['lr']=.01
  if digest(old)!=c['extension']['original_profile_hashes'][t['id']]or digest(old)!=r['original_profile_hashes'][t['id']]or digest(p)!=r['effective_profile_hashes'][t['id']]:raise ValueError('revision changed more than authorized LR')

def boundary_reasons(c,b):
 if not c.get('timemixer_revision'):
  return []if b.get('audited_complete')is True and b.get('original_TimeMixer_16_complete')is True and b.get('audit_sha256')else['original safe-end audit missing']
 if b.get('kind')!='user_authorized_retirement':return ['retirement boundary required, not checkpoint integrity audit']
 path=b.get('receipt_path');h=b.get('receipt_sha256')
 if not path or not h or not Path(path).is_file():return ['actual deletion receipt missing']
 if sha(path)!=h:return ['retirement receipt SHA mismatch']
 d=read(path);E=Path(c['execution']['evidence']);expected=[str(E/'formal-TimeMixer'),str(E/'model-TimeMixer-controller-1790524409463376147.log'),str(E.parent/'m6-timemixer-launch-poaodauo')]
 if d.get('status')!='deleted'or d.get('remaining_paths')!=[]or d.get('authorized_paths')!=expected:return ['old artifacts not fully retired']
 if d.get('boundary_kind')!='user_authorized_retirement'or d.get('checkpoint_integrity_audited')is not False or d.get('backup_created')is not False:return ['retirement must not claim checkpoint integrity or backup']
 if any(Path(p).exists()or Path(p).is_symlink()for p in expected):return ['retired path still exists']
 return []


def completion_audit_reasons(c,ref):
 """41-run catchup waits for an explicitly bound attempt2 completion audit.
 Only current revision JSON identities are read, never retired artifacts/weights.
 """
 if not ref or not ref.get('path') or not ref.get('sha256'):return ['TimeMixer revision completion audit required before 41-run supplement']
 try:
  p=Path(ref['path'])
  if sha(p)!=ref['sha256']:raise ValueError('completion audit SHA mismatch')
  a=read(p)
  if (a.get('purpose')!='m6_revision_completion_audit' or a.get('reviewed')is not True
      or a.get('execution_revision')!=REVISION or a.get('attempt')!=2
      or a.get('protocol_sha')!=digest(c) or a.get('task_ids')!=ids()):raise ValueError('completion audit identity/review mismatch')
  commit=a.get('commit','')
  if len(commit)!=40 or any(x not in '0123456789abcdef'for x in commit):raise ValueError('audit actual commit required')
  complete=root(c)/'complete.json';binding=a.get('complete',{})
  if binding.get('path')!=str(complete) or sha(complete)!=binding.get('sha256'):raise ValueError('revision complete binding mismatch')
  d=read(complete)
  if d.get('execution_revision')!=REVISION or d.get('attempt')!=2 or d.get('protocol_sha')!=digest(c) or d.get('task_ids')!=ids():raise ValueError('foreign revision completion')
  sources=a.get('runs',{})
  if set(sources)!=set(ids()):raise ValueError('completion audit requires exact 16 actual run bindings')
  tasks={t['id']:t for t in c['tasks']}
  for run in ids():
   b=sources[run]
   for filename,key in [('manifest.json','manifest'),('result.json','result')]:
    path=root(c)/run/filename;v=b.get(key,{})
    if v.get('path')!=str(path) or sha(path)!=v.get('sha256'):raise ValueError('completion source bytes mismatch '+run+'/'+filename)
   m=read(root(c)/run/'manifest.json')['identity']
   if (m.get('run_id')!=run or m.get('execution_revision')!=REVISION or m.get('attempt')!=2
       or m.get('profile_sha')!=digest(profile(c,tasks[run])) or m.get('protocol_sha')!=digest(c)
       or m.get('commit')!=commit or m.get('data_sha')!=b.get('data_sha') or len(m.get('data_sha',''))!=64):raise ValueError('completion manifest actual identity mismatch')
  return []
 except (OSError,ValueError,KeyError,TypeError)as exc:return ['revision completion audit: '+str(exc)]

def revision_report_reasons(c,ref,approval=None):
 if not c.get('timemixer_revision'):return []
 if not ref or not ref.get('path')or not Path(ref['path']).is_file():return ['fixed-LR revision numerical/resource admission not completed']
 if sha(ref['path'])!=ref.get('sha256'):return ['revision resource SHA mismatch']
 d=read(ref['path']);r=c['timemixer_revision']
 if d.get('purpose')=='revision_numeric_report':
  from utils.ch3_revision_probe import validate_completion
  try:validate_completion(c,d)
  except (ValueError,OSError,KeyError,TypeError)as exc:return ['revision measured evidence: '+str(exc)]
 if d.get('execution_revision')!=REVISION or d.get('protocol_sha')!=digest(c)or d.get('profile_shas')!=r['effective_profile_hashes']or d.get('task_ids')!=ids():return ['revision resource profile/scope mismatch']
 if d.get('reviewed')is not True:return ['revision resource evidence not reviewed']
 if not approval or any(not approval.get(k) or d.get(k)!=approval[k] for k in ('code','environment','hardware')):return ['revision resource source/environment/hardware not bound to authorization']
 for domain,q in zip(DOMAINS,[4,4,2,4]):
  v=d.get('decisions',{}).get('TimeMixer-'+domain+'-MS',{})
  if v.get('status')!='Passed'or v.get('concurrency')!=q:return ['exact revision waves lack admission']
  if domain!='Exchange'and v.get('numerical_protocol_sha')!=digest(c):return ['changed LR cannot inherit old numeric trajectory']
 return []

def authorization_reasons(c,a):
 if not a:return ['explicit revision review/closure authorization missing']
 reasons=[];r=c['timemixer_revision']
 if a.get('execution_revision')!=REVISION or a.get('attempt')!=2 or a.get('authorized_task_ids')!=ids()or a.get('fresh')is not True or a.get('protocol_sha')!=digest(c):reasons.append('old/foreign revision authorization')
 if a.get('reviewed')is not True or a.get('execution_permitted')is not True:reasons.append('revision template is not executable')
 if a.get('run_budget')!={'runs':16,'run_epochs':200}or a.get('parent_protocol_sha')!=r['parent_protocol_sha']:reasons.append('revision budget/parent mismatch')
 reasons+=boundary_reasons(c,a.get('safe_boundary',{}));reasons+=revision_report_reasons(c,a.get('revision_resource_report'),a)
 return reasons

def validate_spawn(c,run,out,artifact,a,resume=False):
 if run not in c.get('timemixer_revision',{}).get('task_ids',[]):return
 if resume:raise PermissionError('fresh revision only; resume requires separate reviewed recovery implementation')
 if Path(out)!=root(c)/run or Path(artifact)!=root(c)/run:raise PermissionError('old/arbitrary TimeMixer artifact path forbidden')
 reasons=authorization_reasons(c,a)
 if reasons:raise PermissionError('; '.join(reasons))

def identity_fields(c,t,a):
 if t['id']not in c.get('timemixer_revision',{}).get('task_ids',[]):return {}
 if a.get('execution_revision')!=REVISION or a.get('attempt')!=2:raise PermissionError('revision identity missing')
 return dict(execution_revision=REVISION,attempt=2,parent_protocol_sha=c['timemixer_revision']['parent_protocol_sha'],fresh=True)

def waves(c):
 validate_revision_profiles(c);bank=ids();return [bank[:4],bank[4:8],bank[8:10],bank[10:12],bank[12:16]]
def reject_existing(c):
 if root(c).exists()or root(c).is_symlink():raise FileExistsError('revision success/failure/staging retained; no automatic retry')
def readiness(c,a,report_ref=None):
 reasons=authorization_reasons(c,a)
 from ch3_runner import ROOT
 if ROOT.resolve()!=Path('/public/home/yueweiting/大论文/AMD'):reasons.append('isolated candidate not deployed/closed')
 if root(c).exists():reasons.append('revision output retained; duplicate start forbidden')
 if not reasons:
  from ch3_runner import preflight
  reasons+=preflight(c,'TimeMixer',a)
 return list(dict.fromkeys(reasons))
def execute(c,a,execute_wave=None):
 reasons=readiness(c,a)
 if reasons:raise PermissionError('; '.join(reasons))
 from ch3_runner import GPULock,dump
 from utils.ch3_extension import execute_waves
 import signal
 out=root(c)
 def stop(signum,frame):raise InterruptedError('owned revision stop')
 prior=signal.signal(signal.SIGTERM,stop)
 try:
  with GPULock(c):
   reject_existing(c);out.mkdir(parents=True,exist_ok=False)
   dump(out/'controller.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],model='TimeMixer',execution_revision=REVISION,attempt=2))
   execute_waves(c,waves(c),out,a,execute=execute_wave)
   dump(out/'complete.json',dict(execution_revision=REVISION,attempt=2,protocol_sha=digest(c),task_ids=ids(),result_review='Pending'))
 finally:signal.signal(signal.SIGTERM,prior)
