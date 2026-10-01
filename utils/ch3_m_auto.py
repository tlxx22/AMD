"""Fixed M preauthorization; machine gates never claim human result review."""
import copy,json,subprocess
from pathlib import Path
from utils import ch3_m_tasks as s
from utils import ch3_m_execution as e
from utils import ch3_m_launch as launch
from utils.ch3_contract import digest,profile
from ch3_runner import code_binding,environment_binding,hardware_binding,git

BASE='9696876670b1f3e5c69f7a4607ea34edc80d995b'
VERSION='m-baselines-preauthorized-machine-gate-v1'
MODE='preauthorized_machine_gate'
CONFIG_SHA='1a2b79c1ddb52478b3897ec2e4294197aa592dd25cc082043a4702c68590b75a'
PROTOCOL_SHA='f43a1335c70c1f82b76e6d09ae13d8aaa625b389b0b722117ff9a442ca241068'
SOURCES={
 'plan':dict(path=str(s.PACKAGE/'plan.json'),sha256='d7c42b2cdeaed22183062cc13b4368fafe893966b84d5bbf5f0b9d5be38750a8'),
 'data':dict(path=str(s.PACKAGE/'data-bindings.json'),sha256='4f975637feef6834aeed085c305305dffe4747b0ccc4f75c1c391a68c1b62ac8'),
 'provenance':dict(path=str(s.PACKAGE/'source-provenance.json'),sha256='3687b21224123fc3383029f3f4bfbfff46a5fa2fd292a7f8f4232e6345ceda18'),
}
CHANGED_CODE={'ch3_runner.py','utils/ch3_m_handoff.py','utils/ch3_m_execution.py','m6_m_handoff_entry.py','tests/test_m6_m_handoff.py','utils/ch3_m_auto.py','tests/test_m6_m_auto.py'}

def permit_path(probe):return s.PACKAGE/('auto-probe-permit.json'if probe else'auto-formal-permit.json')
def admission_path():return s.PACKAGE/'auto-probe-admission.json'

def policy(c):
 return dict(version=VERSION,base_commit=BASE,protocol_sha=PROTOCOL_SHA,config_sha256=CONFIG_SHA,sources=copy.deepcopy(SOURCES),task_ids=[t['id']for t in c['tasks']],profile_shas={t['id']:digest(profile(c,t))for t in c['tasks']},plan_sha256=SOURCES['plan']['sha256'],probe_caps=s.CAPS,run_budget=dict(runs=84,run_epochs=840),max_optimizer_steps=294790,additional_search=0,review_mode=MODE,manual_review=False)

def binding(c):
 """Closure-owned code plus immutable, previously checked preparation sources."""
 head=git('rev-parse','HEAD')
 if git('branch','--show-current')!='m6/m-baselines-v1'or git('status','--porcelain','--untracked-files=all')or head==BASE:raise ValueError('follow-up clean closure required')
 if subprocess.run(['git','-C',str(s.ROOT),'merge-base','--is-ancestor',BASE,head],capture_output=True).returncode:raise ValueError('M preauthorization closure ancestry')
 if s.sha(s.ROOT/'configs/ch3_formal_profiles.json')!=CONFIG_SHA or digest(c)!=PROTOCOL_SHA:raise ValueError('M frozen configuration/protocol changed')
 code=code_binding()
 for path,value in code.items():
  import hashlib
  raw=subprocess.check_output(['git','-C',str(s.ROOT),'show',head+':'+path])
  if hashlib.sha256(raw).hexdigest()!=value:raise ValueError('M code not bound to closure: '+path)
 prior=s.bound(SOURCES['provenance']);data=s.bound(SOURCES['data'])
 if s.bound(SOURCES['plan'])!=e.plan(c)or c['m_experiment']['data_binding_artifact']!=SOURCES['data']:raise ValueError('M preauthorized plan/data source')
 if {k:v for k,v in code.items()if k not in CHANGED_CODE}!={k:v for k,v in prior['code'].items()if k not in CHANGED_CODE}:raise ValueError('M computation/source inheritance changed')
 env=environment_binding();hw=hardware_binding()
 if env!=prior['environment']or hw!=prior['hardware']:raise ValueError('M preauthorized environment/hardware changed')
 return dict(commit=head,code=code,protocol_sha=digest(c),environment=env,hardware=hw,plan=copy.deepcopy(SOURCES['plan']),data_binding_artifact=copy.deepcopy(SOURCES['data']),data_bindings=data['data_bindings'],source_states=data['source_states'])

def stop_check(root):
 root=Path(root)
 if(root/'STOP').exists()or(root/'failure.json').exists():raise InterruptedError('M auto admission STOP/failure')

def check_admission(c,r,*,worker=False):
 if r.get('purpose')!='M_pre_authorized_technical_admission_v1'or r.get('review_mode')!=MODE or r.get('manual_review')is not False or r.get('reviewed')is not False or r.get('technical_admission')is not True:raise ValueError('M machine admission format')
 context=binding(c)
 for k in ('commit','code','protocol_sha','environment','hardware'):
  if r.get(k)!=context[k]:raise ValueError('M machine admission '+k)
 if r.get('policy')!=policy(c)or r.get('policy_sha')!=digest(policy(c)):raise ValueError('M machine policy binding')
 if r.get('original_report',{}).get('path')!=str(e.PROBE_ROOT/'complete.json'):raise ValueError('M machine actual probe path')
 raw=s.bound(r['original_report'])
 if any(r.get(k)!=raw.get(k)for k in ('decisions','budget','commit','code','protocol_sha','environment','hardware')):raise ValueError('M machine admission changed measurements')
 probe=s.bound(r['probe_permit'])
 validate_permit(c,probe,True,worker=worker)
 if s.bound(raw['approval'])!=probe or raw.get('commit')!=probe['commit']:raise ValueError('M probe actual execution permit')
 if r.get('old_boundary')!=probe['old_boundary']:raise ValueError('M machine boundary mismatch')
 if not worker:e.validate_probe_completion(c,raw)
 return r

def validate_permit(c,a,probe,*,worker=False):
 expected=permit_path(probe)
 if a.get('review_mode')!=MODE or a.get('manual_review')is not False or a.get('reviewed')is not False or a.get('synthetic_fixture')is not False:raise ValueError('M exact machine authorization format')
 exact=dict(purpose='ch3_resource_probe'if probe else'ch3_formal',m_scope=s.PROBE if probe else s.ID,execution_permitted=True,budget_authorized=True,authorized_task_ids=[t['id']for t in c['tasks']],profile_shas={t['id']:digest(profile(c,t))for t in c['tasks']},parent_MS_profiles={t['id']:profile(c,t)['parent_MS_profile']for t in c['tasks']},from_scratch=True,task='M',metric_scope='all_channels',seed_list=[2024],additional_search=0,run_budget=dict(runs=84,run_epochs=840),max_optimizer_steps=294790,caps=s.CAPS if probe else None,already_charged=dict(adam=0,forward=0,backward=0))
 if any(a.get(k)!=v or(type(v)is bool and type(a.get(k))is not bool)for k,v in exact.items()):raise ValueError('M machine exact scope/budget/training grant')
 if a.get('old_boundary',{}).get('path')!=str(s.PACKAGE/'old-queue-technical-boundary.json')or s.bound(a['old_boundary']).get('kind')!='old_MS_queue_technical_complete':raise ValueError('M machine exact sealed boundary')
 if a.get('permit_path')!=str(expected)or a.get('policy')!=policy(c)or a.get('policy_sha')!=digest(policy(c)):raise ValueError('M machine permit path/policy')
 if a.get('authorization_basis')!=('preauthorized-by-reviewed-policy'if probe else'user pre-authorized full training iff preregistered M technical admission passes'):raise ValueError('M machine authorization basis')
 context=binding(c)
 for k,v in context.items():
  if a.get(k)!=v:raise ValueError('M machine '+k+' mismatch')
 if probe:
  if a.get('technical_admission')is not False or a.get('resource_report')is not None or a.get('probe_complete')is not None:raise ValueError('M probe permit cannot pregrant formal admission')
 else:
  if a.get('structure_frozen')is not True or a.get('m6_authorized')is not True:raise ValueError('M machine frozen M6 authorization')
  if a.get('technical_admission')is not True or a.get('resource_report',{}).get('path')!=str(admission_path()):raise ValueError('M actual machine admission required')
  r=check_admission(c,s.bound(a['resource_report']),worker=worker)
  if a.get('probe_complete')!=r['original_report']or a.get('probe_decisions')!=r['decisions']or a.get('probe_budget')!=r['budget']or a.get('old_boundary')!=r['old_boundary']:raise ValueError('M formal actual probe binding')
 return a

def create_permit(c,probe,boundary,root,admission=None):
 """Preflight first; O_EXCL issuance; no template or human-review impersonation."""
 stop_check(root);context=binding(c);path=permit_path(probe)
 if path.exists()or path.is_symlink():raise FileExistsError('retained auto M permit; no reissue')
 if boundary!=s.ref(s.PACKAGE/'old-queue-technical-boundary.json')or s.bound(boundary)!=e.old_boundary(c):raise ValueError('M auto exact old queue boundary')
 a=dict(**context,purpose='ch3_resource_probe'if probe else'ch3_formal',m_scope=s.PROBE if probe else s.ID,reviewed=False,manual_review=False,review_mode=MODE,execution_permitted=True,budget_authorized=True,synthetic_fixture=False,policy=policy(c),policy_sha=digest(policy(c)),permit_path=str(path),authorization_basis='preauthorized-by-reviewed-policy'if probe else'user pre-authorized full training iff preregistered M technical admission passes',authorized_task_ids=[t['id']for t in c['tasks']],profile_shas={t['id']:digest(profile(c,t))for t in c['tasks']},parent_MS_profiles={t['id']:profile(c,t)['parent_MS_profile']for t in c['tasks']},from_scratch=True,task='M',metric_scope='all_channels',seed_list=[2024],additional_search=0,old_boundary=boundary,resource_report=admission,run_budget=dict(runs=84,run_epochs=840),max_optimizer_steps=294790,caps=s.CAPS if probe else None,already_charged=dict(adam=0,forward=0,backward=0),technical_admission=not probe,probe_complete=None,source='Explicit user preauthorization of this closure-owned policy; technical gates only, scientific result review pending')
 if not probe:
  r=check_admission(c,s.bound(admission));a.update(structure_frozen=True,m6_authorized=True,probe_complete=r['original_report'],probe_decisions=r['decisions'],probe_budget=r['budget'])
 from m6_m_entry import readiness
 reasons=readiness(c,a,str(path),probe)
 if reasons:raise PermissionError('; '.join(reasons))
 stop_check(root);launch.exclusive_json(path,a);stop_check(root)
 return dict(path=str(path),sha256=s.sha(path),value=s.bound(s.ref(path)))

def audit_probe(c,root,probe_ref):
 """Replay actual saved state/monitor evidence; never run a model or sign a review."""
 stop_check(root);path=admission_path()
 if path.exists()or path.is_symlink():raise FileExistsError('retained M automatic admission')
 if probe_ref['path']!=str(permit_path(True)):raise ValueError('M auto audit actual permit path')
 permit=s.bound(probe_ref);validate_permit(c,permit,True)
 raw_ref=s.ref(e.PROBE_ROOT/'complete.json');raw=s.bound(raw_ref)
 controller=s.bound(s.ref(e.PROBE_ROOT/'controller.json'))
 from m6_remaining_entry import same
 if same(controller)or controller.get('approval')!=probe_ref:raise ValueError('M probe controller live/permit mismatch')
 e.validate_probe_completion(c,raw)
 if s.bound(raw['approval'])!=permit or any(raw.get(k)!=permit[k]for k in ('commit','code','protocol_sha','environment','hardware')):raise ValueError('M actual probe execution source mismatch')
 r=dict(purpose='M_pre_authorized_technical_admission_v1',review_mode=MODE,manual_review=False,reviewed=False,technical_admission=True,admission_granted=False,policy=policy(c),policy_sha=digest(policy(c)),original_report=raw_ref,probe_permit=probe_ref,old_boundary=permit['old_boundary'],result_review='pending',**{k:raw[k]for k in ('decisions','budget','commit','code','protocol_sha','environment','hardware')})
 check_admission(c,r);stop_check(root);launch.exclusive_json(path,r);stop_check(root)
 return s.ref(path)
