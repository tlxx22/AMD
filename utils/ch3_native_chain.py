"""One preauthorized successor, four guarded stages; scientific review stays pending."""
import ast
import hashlib
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path
from utils.ch3_contract import ROOT,read_profiles,validate_manifest,digest,profile
from utils import ch3_m_tasks as source
from utils import ch3_native_tasks as scope
from utils import ch3_native_execution as execution
from utils.ch3_m_launch import exclusive_json,file_identity,ancestors,command_has,tmux_view
from utils.ch3_ms_retirement import boundary as retirement_boundary
from m6_remaining_entry import identity,same,lock,stop_marker,signal_owned,sequence
from ch3_runner import git,code_binding,environment_binding,hardware_binding,dump,GPULock

BASE=scope.BASE
ROOT_CONTROL=scope.PACKAGE/'chain-execution-v1'
SESSION='ch3-native-time-mark-chain-v4'
LOG=scope.PACKAGE/'native-chain-launcher.log'
TOKEN_ENV='CH3_NATIVE_CHAIN_TOKEN'
STATES=('WAIT_OLD_MS_RETIREMENT','OLD_MS_RETIREMENT_BOUNDARY_SEALED','TMARK_PROBE_RUNNING','TMARK_AUTO_AUDIT','TMARK_FORMAL_RUNNING','TMARK_BOUNDARY_SEALED','M_PROBE_RUNNING','M_AUTO_PROBE_AUDIT','M_FORMAL_RUNNING','COMPLETE')
MODE='preauthorized_machine_gate'
# Fixed preparation references are filled from the reviewed plan, never self-signed at runtime.
PREPARATION={'data_tmark': {'path': '/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/native-time-mark-chain-v1/replacement-data-bindings.json', 'sha256': '3eaca493d25a258fa3dc85104cf497f1c88affa3c593bb79cf46347011950ef4'}, 'data_m': {'path': '/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/native-time-mark-chain-v1/m-data-bindings.json', 'sha256': '1e4e8aa4d3e4ecb64b4f846a610559244fc557185206d483fac6c2cb836f19c7'}, 'author_marks': {'path': '/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/native-time-mark-chain-v1/author-timefeatures.json', 'sha256': '41609472a02806db3ab0691b4a129e010e52ad06021d9d262db3ed5ef87acadb'}, 'profile_diff': {'path': '/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/native-time-mark-chain-v1/parent-profile-diff.json', 'sha256': '52132cc12851952ca9c099bdd05343976202e91810bfef21203a69c5d4d9b1e6'}, 'environment_source': {'path': '/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/source-provenance.json', 'sha256': '3687b21224123fc3383029f3f4bfbfff46a5fa2fd292a7f8f4232e6345ceda18'}, 'retirement': {'path': '/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/native-time-mark-chain-v2/retirement/receipt.json', 'sha256': '973898058035a04ddd0bae72ab6df2df6842cc924b3fbc83e79d420fa8d1f1a1'}, 'plan_tmark': {'path': '/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/native-time-mark-chain-v4/replacement-plan.json', 'sha256': 'b36f1f59fa37016e41e17a2ff249fbb759d161098103c6ef162b0d80a76627bd'}, 'plan_m': {'path': '/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/native-time-mark-chain-v4/m-plan.json', 'sha256': 'f69b7673cf747b5e0b06bbf729a00eddbca68bf2f3ffaa33e76599932a1d07b8'}, 'prior_execution': {'path': '/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/native-time-mark-chain-v4/prior-executions.json', 'sha256': 'fc14a447090b4096190d0152d44b64d21ff2af8b9afe04b168ac90f955bc54a5'}, 'policy_basis': {'path': '/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/native-time-mark-chain-v4/historical-policy-basis.json', 'sha256': '7876103621f0402e0e8deb19c0f512622eac051909ccd42e4ddda0dda211e333'}, 'v3_observation': {'path': '/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/native-time-mark-chain-v4/v3-quantitative-basis.json', 'sha256': '0c29f495d8c509aca24d33eda528194f4fa8ef144e5260c7cfabfaea12574ec8'}}
PREPARATION['source_proof']={'path': '/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/native-time-mark-chain-v4/source-inheritance.json', 'sha256': 'ad96dec879dbeed41c7469744a4efd5a32802d292941f2f29590862bf20d084a'}


def configs():
    return {stage:validate_manifest(read_profiles(path)) for stage,path in (('tmark',scope.PROFILE_FILE),('m',ROOT/'configs/ch3_formal_profiles.json'))}


def stop_check(root=None):
    root=ROOT_CONTROL if root is None else root
    if (root/'STOP').exists() or (root/'failure.json').exists():raise InterruptedError('successor persistent STOP/failure')


def policy_code_sha256(path=None):
    """Bind every policy statement; normalize only the proof's recursive SHA literal."""
    tree=ast.parse(Path(path or __file__).read_text());found=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1:
            target=node.targets[0]
            if isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name) and target.value.id=='PREPARATION' and isinstance(target.slice,ast.Constant) and target.slice.value=='source_proof':found.append(node)
    if len(found)!=1 or not isinstance(found[0].value,ast.Dict):raise ValueError('exact fixed policy source-proof declaration')
    fields=[value for key,value in zip(found[0].value.keys,found[0].value.values) if isinstance(key,ast.Constant) and key.value=='sha256']
    if len(fields)!=1 or not isinstance(fields[0],ast.Constant) or not isinstance(fields[0].value,str) or len(fields[0].value)!=64:raise ValueError('exact recursive SHA literal')
    fields[0].value='closure-bound-source-proof-SHA'
    return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()


def binding(c,worker=False):
    head=git('rev-parse','HEAD')
    if git('branch','--show-current')!='m6/m-baselines-v1' or git('status','--porcelain','--untracked-files=all') or git('rev-parse','HEAD^')!=BASE:raise ValueError('reviewed direct successor clean closure required')
    if not PREPARATION:raise ValueError('closure-bound preparation references absent')
    for ref in PREPARATION.values():source.bound(ref)
    plan_key='plan_'+scope.context(c)['stage']
    if source.bound(PREPARATION[plan_key])!=scope.plan(c):raise ValueError('bound exact successor plan changed')
    data_ref=PREPARATION['data_'+scope.context(c)['stage']];data=source.bound(data_ref)
    if c['native_time_mark']['data_binding_artifact']!=data_ref:raise ValueError('successor exact data derivation')
    code=code_binding()
    proof=source.bound(PREPARATION['source_proof'])
    if proof.get('policy_code_sha256')!=policy_code_sha256():raise ValueError('reviewed successor policy execution logic changed')
    if {k:v for k,v in code.items() if k!='utils/ch3_native_chain.py'}!=proof['after_code']:
        raise ValueError('reviewed successor computation/source fingerprint changed')
    for path,value in code.items():
        raw=subprocess.check_output(['git','-C',str(ROOT),'show',head+':'+path])
        if hashlib.sha256(raw).hexdigest()!=value:raise ValueError('unclosed successor source '+path)
    prior=source.bound(PREPARATION['environment_source']);env=environment_binding();hw=hardware_binding()
    if env!=prior['environment'] or hw!=prior['hardware']:raise ValueError('approved environment/hardware changed')
    if not worker:
        for src in c['sources'].values():
            if subprocess.check_output(['git','-C',src['repository'],'rev-parse','HEAD'],text=True).strip()!=src['commit'] or any(source.sha(p)!=sha for p,sha in src['files'].items()):raise ValueError('author source identity changed')
    from utils.ch3_m6 import source_states
    if source_states(c)!=data['source_states']:raise ValueError('bound data metadata/stat changed')
    if not worker:
        for row in source.bound(PREPARATION['author_marks'])['files']:
            if source.sha(row['path'])!=row['sha256']:raise ValueError('author timeF evidence changed')
    return dict(commit=head,code=code,protocol_sha=digest(c),environment=env,hardware=hw,
                plan=PREPARATION[plan_key],data_binding_artifact=data_ref,data_bindings=data['data_bindings'],source_states=data['source_states'])


def policy(c):
    ctx=scope.context(c)
    value=dict(version=scope.ID,base_commit=BASE,stage=ctx['stage'],protocol_sha=digest(c),
                task_ids=[t['id'] for t in c['tasks']],profile_shas={t['id']:digest(profile(c,t)) for t in c['tasks']},
                probe_caps=ctx['caps'],run_budget=dict(runs=ctx['runs'],run_epochs=ctx['epochs']),
                max_optimizer_steps=ctx['optimizer_steps'],additional_search=0,review_mode=MODE,manual_review=False,
                old_boundary_kind='old_MS_user_authorized_retirement_boundary_v1',original_old_batch_technical_complete=False)
    if ctx['stage']=='tmark':
        value['replacement_budget_lineage']=dict(prior_failed_executions=PREPARATION['prior_execution'],
            v2_consumed=dict(adam=192,forward=272,backward=192),v3_consumed=dict(adam=144,forward=208,backward=144),budget_refund=False,
            v4_starting_debit=dict(adam=0,forward=0,backward=0),v4_caps=ctx['caps'],
            historical_actual_plus_v4_max=dict(adam=774,forward=1088,backward=774))
    return value


def approval_template(c,probe):
    """Preparation only. No future commit, result SHA or machine admission invented."""
    ctx=scope.context(c);data_ref=PREPARATION['data_'+ctx['stage']];data=source.bound(data_ref)
    return dict(purpose='ch3_resource_probe' if probe else 'ch3_formal',successor_scope=ctx['probe_scope'] if probe else ctx['formal_scope'],reviewed=False,manual_review=False,review_mode=MODE,execution_permitted=False,budget_authorized=False,synthetic_fixture=False,commit=None,code=code_binding(),protocol_sha=digest(c),environment=source.bound(PREPARATION['environment_source'])['environment'],hardware=source.bound(PREPARATION['environment_source'])['hardware'],plan=PREPARATION['plan_'+ctx['stage']],data_binding_artifact=data_ref,data_bindings=data['data_bindings'],source_states=data['source_states'],authorized_task_ids=[t['id'] for t in c['tasks']],profile_shas={t['id']:digest(profile(c,t)) for t in c['tasks']},run_budget=dict(runs=ctx['runs'],run_epochs=ctx['epochs']),max_optimizer_steps=ctx['optimizer_steps'],caps=ctx['caps'] if probe else None,already_charged=dict(adam=0,forward=0,backward=0),from_scratch=True,seed_list=[2024],additional_search=0,task='MS' if ctx['stage']=='tmark' else 'M',metric_scope='target_only' if ctx['stage']=='tmark' else 'all_channels',old_boundary=None,replacement_boundary=None,resource_report=None,technical_admission=False,policy=policy(c),policy_sha=digest(policy(c)),permit_path=str(permit_path(c,probe)),source='Non-executable preparation. New actual-byte review/closure and saved technical gates required; no human review is impersonated')


def permit_path(c,probe):return scope.PACKAGE/(scope.context(c)['stage']+('-probe-permit.json' if probe else '-formal-permit.json'))
def admission_path(c):return scope.PACKAGE/(scope.context(c)['stage']+'-technical-admission.json')
def old_path():return scope.PACKAGE/'old-ms-retirement-boundary.json'
def replacement_path():return scope.PACKAGE/'native-time-mark-replacement-boundary.json'


def check_boundary(c,ref,worker=False):
    if ref is None or ref.get('path')!=str(old_path()):raise ValueError('exact old MS boundary required')
    raw=source.bound(ref)
    if raw!=retirement_boundary(worker=worker):raise ValueError('exact user-authorized retirement boundary changed')


def replacement_boundary(c,ref,worker=False):
    if not ref or ref.get('path')!=str(replacement_path()):raise ValueError('new M requires exact replacement boundary')
    b=source.bound(ref);tc=configs()['tmark']
    if b.get('task_ids')!=[t['id'] for t in tc['tasks']] or b.get('technical_complete')is not True or b.get('result_review')!='pending' or b.get('protocol_sha')!=digest(tc) or b.get('commit')!=git('rev-parse','HEAD'):raise ValueError('87 replacement sealed identity')
    if not worker and b!=build_replacement_boundary(tc):raise ValueError('replacement boundary not reproducible')
    return b


def validate_admission(c,r,worker=False):
    ctx=scope.context(c)
    if r.get('purpose')!='native_successor_machine_admission_v1' or r.get('review_mode')!=MODE or r.get('manual_review')is not False or r.get('reviewed')is not False or r.get('technical_admission')is not True or r.get('policy')!=policy(c):raise ValueError('exact machine admission format')
    if r['probe_complete']['path']!=str(ctx['probe_root']/'complete.json') or r['probe_permit']['path']!=str(permit_path(c,True)):raise ValueError('exact actual admission paths')
    raw=source.bound(r['probe_complete']);permit=source.bound(r['probe_permit'])
    for k in ('commit','protocol_sha','code','environment','hardware','decisions','budget'):
        if r[k]!=raw[k]:raise ValueError('machine admission changed actual measurements')
    if source.bound(raw['approval'])!=permit:raise ValueError('actual probe permit source')
    validate_permit(c,permit,True,worker=worker)
    if not worker:execution.validate_probe_completion(c,raw)
    return r


def validate_permit(c,a,probe,worker=False):
    if not a:raise PermissionError('matching successor permit absent')
    ctx=scope.context(c);expected=ctx['probe_scope'] if probe else ctx['formal_scope'];bound=binding(c,worker=worker)
    exact=dict(purpose='ch3_resource_probe' if probe else 'ch3_formal',successor_scope=expected,
               reviewed=False,manual_review=False,review_mode=MODE,execution_permitted=True,budget_authorized=True,
               synthetic_fixture=False,from_scratch=True,seed_list=[2024],additional_search=0,
               authorized_task_ids=[t['id'] for t in c['tasks']],profile_shas={t['id']:digest(profile(c,t)) for t in c['tasks']},
               run_budget=dict(runs=ctx['runs'],run_epochs=ctx['epochs']),max_optimizer_steps=ctx['optimizer_steps'],
               caps=ctx['caps'] if probe else None,already_charged=dict(adam=0,forward=0,backward=0),
               permit_path=str(permit_path(c,probe)),policy=policy(c),policy_sha=digest(policy(c)),
               technical_admission=not probe,task='MS' if ctx['stage']=='tmark' else 'M',
               metric_scope='target_only' if ctx['stage']=='tmark' else 'all_channels')
    for k,v in {**bound,**exact}.items():
        if a.get(k)!=v or (type(v)is bool and type(a.get(k))is not bool):raise ValueError('successor permit '+k+' mismatch')
    if a.get('authorization_basis')!='user preauthorized successor from exact N/S retirement boundary iff all preregistered technical gates pass':raise ValueError('exact successor authorization basis')
    check_boundary(c,a.get('old_boundary'),worker)
    if ctx['stage']=='m':
        b=replacement_boundary(c,a.get('replacement_boundary'),worker)
        if b['old_boundary']!=a['old_boundary']:raise ValueError('M two-boundary lineage')
    elif a.get('replacement_boundary')is not None:raise ValueError('tmark cannot use M/foreign replacement boundary')
    if probe:
        if a.get('resource_report')is not None:raise ValueError('probe cannot pregrant formal')
    else:
        if a.get('structure_frozen')is not True or a.get('m6_authorized')is not True or not a.get('resource_report') or a['resource_report']['path']!=str(admission_path(c)):raise ValueError('formal exact machine technical grant')
        report=validate_admission(c,source.bound(a['resource_report']),worker)
        if a.get('probe_complete')!=report['probe_complete'] or a.get('probe_decisions')!=report['decisions'] or a.get('probe_budget')!=report['budget']:raise ValueError('formal actual probe binding')
    return a


def readiness(c,a,probe=False):
    reasons=execution.authorization_reasons(c,a,probe)
    root=execution.roots(c,probe)
    if root.exists() or root.is_symlink():reasons.append('retained successor stage; no fresh retry')
    if not probe and any(scope.result_path(c,t).exists() or scope.result_path(c,t).is_symlink() for t in c['tasks']):reasons.append('retained exact successor task output')
    if not reasons:
        try:
            with GPULock(c):pass
            sys.path.insert(0,str(ROOT/'tools/restricted_regression'))
            from m5_formal_entry import gpu_sample,resource_assessment
            if not resource_assessment(gpu_sample([]),[])['admission']:raise ValueError('successor GPU resource admission')
        except (OSError,RuntimeError,ValueError) as exc:reasons.append(str(exc))
    return list(dict.fromkeys(reasons))


def create_permit(c,probe,boundary,replacement=None,admission=None):
    stop_check();path=permit_path(c,probe)
    if path.exists() or path.is_symlink():raise FileExistsError('one-shot successor permit retained')
    ctx=scope.context(c)
    a=dict(**binding(c),purpose='ch3_resource_probe' if probe else 'ch3_formal',successor_scope=ctx['probe_scope'] if probe else ctx['formal_scope'],reviewed=False,manual_review=False,review_mode=MODE,execution_permitted=True,budget_authorized=True,synthetic_fixture=False,from_scratch=True,seed_list=[2024],additional_search=0,authorized_task_ids=[t['id'] for t in c['tasks']],profile_shas={t['id']:digest(profile(c,t)) for t in c['tasks']},run_budget=dict(runs=ctx['runs'],run_epochs=ctx['epochs']),max_optimizer_steps=ctx['optimizer_steps'],caps=ctx['caps'] if probe else None,already_charged=dict(adam=0,forward=0,backward=0),permit_path=str(path),policy=policy(c),policy_sha=digest(policy(c)),technical_admission=not probe,task='MS' if ctx['stage']=='tmark' else 'M',metric_scope='target_only' if ctx['stage']=='tmark' else 'all_channels',old_boundary=boundary,replacement_boundary=replacement,resource_report=admission,authorization_basis='user preauthorized successor from exact N/S retirement boundary iff all preregistered technical gates pass')
    if not probe:
        report=validate_admission(c,source.bound(admission));a.update(structure_frozen=True,m6_authorized=True,probe_complete=report['probe_complete'],probe_decisions=report['decisions'],probe_budget=report['budget'])
    reasons=readiness(c,a,probe)
    if reasons:raise PermissionError('; '.join(reasons))
    stop_check();exclusive_json(path,a);stop_check();return source.ref(path)


def audit_probe(c,permit_ref):
    stop_check();ctx=scope.context(c);path=admission_path(c)
    if path.exists():raise FileExistsError('retained automatic successor admission')
    permit=source.bound(permit_ref);validate_permit(c,permit,True)
    root=ctx['probe_root'];control=json.loads((root/'controller.json').read_text())
    if same(control) or control.get('approval')!=permit_ref:raise ValueError('probe original still live/permit identity')
    raw_ref=source.ref(root/'complete.json');raw=source.bound(raw_ref);execution.validate_probe_completion(c,raw)
    r=dict(purpose='native_successor_machine_admission_v1',review_mode=MODE,manual_review=False,reviewed=False,technical_admission=True,admission_granted=False,result_review='pending',policy=policy(c),probe_complete=raw_ref,probe_permit=permit_ref,**{k:raw[k] for k in ('commit','code','protocol_sha','environment','hardware','decisions','budget')})
    validate_admission(c,r);stop_check();exclusive_json(path,r);stop_check();return source.ref(path)


def build_replacement_boundary(c):
    ctx=scope.context(c);control=json.loads((ctx['control']/'controller.json').read_text())
    if same(control) or (ctx['control']/'STOP').exists() or (ctx['control']/'failure.json').exists():raise ValueError('replacement original not complete')
    a=source.bound(control['approval']);validate_permit(c,a,False)
    done=json.loads((ctx['control']/'complete.json').read_text())
    if done['task_ids']!=[t['id'] for t in c['tasks']] or done['groups']!=list(ctx['models']):raise ValueError('87 exact technical completion')
    sources={}
    for model in ctx['models']:
        if source.bound(done['handoffs'][model])!=execution.technical_group(c,model,a):raise ValueError('replacement technical handoff changed')
        sources[model]=done['handoffs'][model]
    return dict(kind='native_time_mark_replacement_technical_boundary_v1',task_ids=[t['id'] for t in c['tasks']],results={t['id']:dict(result=source.ref(scope.result_path(c,t)/'result.json'),manifest=source.ref(scope.result_path(c,t)/'manifest.json')) for t in c['tasks']},technical_complete=True,result_review='pending',old_boundary=a['old_boundary'],sources=sources,complete=source.ref(ctx['control']/'complete.json'),**{k:a[k] for k in ('commit','protocol_sha','code','environment','hardware','data_binding_artifact')})


def seal(path,value):
    path=Path(path);expected=hashlib.sha256((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()).hexdigest()
    if path.exists() or path.is_symlink():
        if source.ref(path)['sha256']!=expected or source.bound(source.ref(path))!=value:raise ValueError('sealed boundary conflict; never overwrite')
    else:exclusive_json(path,value)
    return source.ref(path)


def launch_paths(kind):
    if kind not in ('handoff','tmark-probe','tmark-formal','m-probe','m-formal'):raise ValueError('exact native chain launch kind')
    log=LOG if kind=='handoff' else scope.PACKAGE/(kind+'-launcher.log')
    return dict(log=log,metadata=Path(str(log)+'.launch.json'),claim=Path(str(log)+'.claimed.json'))


def prepare_launch(kind,owner_pid,permit=None):
    x=launch_paths(kind);owner=identity(owner_pid)
    if not owner or not same(owner):raise ValueError('native launch owner not live')
    driver=ROOT/'scripts/ch3/start_native_time_mark_chain.sh' if kind=='handoff' else ROOT/'m6_native_chain_entry.py'
    if not command_has(owner_pid,driver) or not any(v['pid']==owner_pid and v['start_ticks']==owner['start_ticks'] for v in ancestors(os.getpid())):raise ValueError('exact native wrapper/chain owner')
    if x['log'].stat().st_size!=0 or x['claim'].exists():raise ValueError('native launcher stale/claimed')
    token=secrets.token_hex(32)
    value=dict(kind=kind,owner=owner,log=str(x['log']),log_identity=file_identity(x['log']),commit=git('rev-parse','HEAD'),token_sha256=hashlib.sha256(token.encode()).hexdigest(),permit=source.ref(permit) if permit else None)
    exclusive_json(x['metadata'],value);return token


def verify_launch(kind,token,permit=None):
    x=launch_paths(kind)
    if not isinstance(token,str) or len(token)!=64:raise PermissionError('native wrapper token required')
    file_identity(x['metadata']);v=json.loads(x['metadata'].read_text())
    if v.get('kind')!=kind or v.get('log')!=str(x['log']) or v.get('commit')!=git('rev-parse','HEAD') or v.get('permit')!=(source.ref(permit) if permit else None) or v.get('log_identity')!=file_identity(x['log']) or not secrets.compare_digest(v.get('token_sha256',''),hashlib.sha256(token.encode()).hexdigest()) or x['claim'].exists():raise ValueError('native launcher exact identity mismatch/stale')
    if kind=='handoff':tmux_view(SESSION)
    elif os.getppid()!=v['owner']['pid'] or not same(v['owner']) or not command_has(v['owner']['pid'],ROOT/'m6_native_chain_entry.py'):raise ValueError('stage not owned by this current chain')
    else:tmux_view(SESSION)
    return v


def signal_stage(v,owner):
    if not v or not same(v):return False
    if v.get('owner')!=dict(pid=owner['pid'],start_ticks=owner['start_ticks']) or v.get('kind')not in ('tmark-probe','tmark-formal','m-probe','m-formal'):raise ValueError('not an owned successor stage')
    argv=(Path('/proc')/str(v['pid'])/'cmdline').read_bytes().decode().split('\0')
    if str(ROOT/'m6_native_chain_entry.py')not in argv or 'stage-start'not in argv:raise ValueError('foreign PID despite matching ticks')
    if same(owner):
        fields=(Path('/proc')/str(v['pid'])/'stat').read_text().rsplit(')',1)[1].split()
        if int(fields[1])!=owner['pid']:raise ValueError('owned child actual parent mismatch')
    return signal_owned(v)


def run_owned(c,probe,permit_ref):
    stop_check();ctx=scope.context(c);kind=ctx['stage']+('-probe' if probe else '-formal');p=Path(permit_ref['path']);a=source.bound(permit_ref);x=launch_paths(kind)
    if readiness(c,a,probe):raise PermissionError('owned stage preflight blocked')
    fd=os.open(str(x['log']),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
    token=prepare_launch(kind,os.getpid(),p);args=[c['execution']['python'],'-B',str(ROOT/'m6_native_chain_entry.py'),'stage-start','--stage',ctx['stage'],'--approval',str(p)]+(['--probe'] if probe else [])
    child=None;owned=None;owner=identity(os.getpid())
    try:
        stop_check()
        with x['log'].open('a') as log:
            child=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1',TOKEN_ENV:token})
            v=identity(child.pid)
            if not v:child.wait();raise RuntimeError('stage vanished before identity registration')
            owned=dict(v,kind=kind,owner=dict(pid=owner['pid'],start_ticks=owner['start_ticks']))
            dump(ROOT_CONTROL/'current.json',dict(child=owned,permit=permit_ref,command=args,log=str(x['log'])))
            sent=False
            while True:
                if (ROOT_CONTROL/'STOP').exists() and not sent:sent=signal_stage(owned,owner)
                try:code=child.wait(timeout=.2);break
                except subprocess.TimeoutExpired:continue
        stop_check()
        if code!=0 or same(owned):raise RuntimeError('owned stage failed/not exited: '+str(code))
        verify_stage(c,a,probe)
        return dict(child=owned,exit_code=code,permit=permit_ref,complete=source.ref(execution.roots(c,probe)/'complete.json'),technical_complete=True,result_review='pending')
    finally:
        if child is not None and child.poll()is None:
            signal_stage(owned,owner)
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:raise RuntimeError('owned chain did not exit in bounded cleanup; retained evidence')


def verify_stage(c,a,probe):
    root=execution.roots(c,probe)
    if same(json.loads((root/'controller.json').read_text())) or (root/'STOP').exists() or (root/'failure.json').exists():raise ValueError('stage original live/STOP/failure')
    r=json.loads((root/'complete.json').read_text())
    if probe:return execution.validate_probe_completion(c,r)
    ctx=scope.context(c)
    if r['task_ids']!=[t['id'] for t in c['tasks']] or r['groups']!=list(ctx['models']) or r.get('protocol_sha')!=digest(c) or r.get('commit')!=a['commit']:raise ValueError('formal stage exact completion')
    for model in ctx['models']:
        if source.bound(r['handoffs'][model])!=execution.technical_group(c,model,a):raise ValueError('formal group handoff changed')
    return r


def stage_start(c,a,approval,probe,token):
    ctx=scope.context(c);kind=ctx['stage']+('-probe' if probe else '-formal');root=execution.roots(c,probe)
    verify_launch(kind,token,approval);stop_check()
    reasons=readiness(c,a,probe)
    if reasons:raise PermissionError('; '.join(reasons))
    with lock(scope.PACKAGE/('.'+kind+'.lock')):
        if root.exists():raise FileExistsError('retained successor stage')
        exclusive_json(launch_paths(kind)['claim'],dict(consumer=identity(os.getpid()),launch=source.ref(launch_paths(kind)['metadata'])))
        root.mkdir(parents=True,exist_ok=False);dump(root/'controller.json',dict(**identity(os.getpid()),approval=source.ref(approval),commit=a['commit'],protocol_sha=digest(c)))
        if probe:
            dump(root/'approval.json',a)
            with GPULock(c):execution.run_probe(c,a)
        else:
            commands={model:[c['execution']['python'],'-B',str(ROOT/'m6_native_chain_entry.py'),'group','--stage',ctx['stage'],'--model',model,'--approval',approval] for model in ctx['models']}
            def before(g):
                stop_check()
                reasons=execution.authorization_reasons(c,a)
                if reasons:raise PermissionError('; '.join(reasons))
                if any(scope.result_path(c,t).exists() for t in c['tasks'] if t['model']==g['model']):raise FileExistsError('retained formal task; no skip/retry')
            sequence(dict(groups=[dict(model=m) for m in ctx['models']],task_ids=[t['id'] for t in c['tasks']]),root,commands,before,lambda g:execution.technical_group(c,g['model'],a),queue_id=ctx['formal_scope'])
            r=json.loads((root/'complete.json').read_text());r.update(protocol_sha=digest(c),commit=a['commit'],result_review='pending');dump(root/'complete.json',r)


def preflight(starting=False,token=None):
    reasons=[];record=None
    try:
        cs=configs()
        for c in cs.values():binding(c)
        retirement_boundary()
    except (OSError,ValueError,KeyError,subprocess.CalledProcessError) as exc:reasons.append(str(exc))
    if starting:
        try:record=verify_launch('handoff',token)
        except (OSError,ValueError,PermissionError) as exc:reasons.append(str(exc))
    if ROOT_CONTROL.exists() or ROOT_CONTROL.is_symlink():reasons.append('retained successor controller/failure/complete')
    for kind in ('handoff','tmark-probe','tmark-formal','m-probe','m-formal'):
        x=launch_paths(kind)
        if x['claim'].exists() or ((x['log'].exists() or x['metadata'].exists()) and not (kind=='handoff' and record)):reasons.append('retained successor launch '+kind)
    for p in [old_path(),replacement_path()]+[scope.PACKAGE/(stage+name) for stage in ('tmark','m') for name in ('-probe-permit.json','-formal-permit.json','-technical-admission.json','-probe-execution-v1')]:
        if p.exists() or p.is_symlink():reasons.append('retained successor evidence '+str(p))
    for path in (scope.REPLACEMENT,scope.M_RESULT):
        if path.exists() or path.is_symlink():reasons.append('retained exact new result namespace '+str(path))
    if subprocess.run(['tmux','has-session','-t',SESSION],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0 and not record:reasons.append('retained successor tmux')
    return list(dict.fromkeys(reasons))


def run(root=ROOT_CONTROL,sleep=time.sleep,interval=60):
    cs=configs();initial={stage:binding(c) for stage,c in cs.items()};state=STATES[0];receipts={};old=None;replacement=None
    def update(next_state):
        nonlocal state
        state=next_state;dump(root/'progress.json',dict(state=state,receipts=receipts,result_review='pending',manual_review=False))
        with (root/'states.jsonl').open('a') as f:f.write(json.dumps(dict(state=state))+'\n')
    def stopped(sig,frame):
        stop_marker(root)
        if not (root/'current.json').exists() or not same(json.loads((root/'current.json').read_text()).get('child')):raise InterruptedError('successor stopped without active child')
    previous=signal.signal(signal.SIGTERM,stopped)
    try:
        update(state)
        while state!='COMPLETE':
            stop_check(root)
            if any(binding(c)!=initial[stage] for stage,c in cs.items()):raise ValueError('successor binding changed')
            if state=='WAIT_OLD_MS_RETIREMENT':
                old=seal(old_path(),retirement_boundary());update('OLD_MS_RETIREMENT_BOUNDARY_SEALED')
            elif state=='OLD_MS_RETIREMENT_BOUNDARY_SEALED':
                receipts['tmark_probe_permit']=create_permit(cs['tmark'],True,old);update('TMARK_PROBE_RUNNING')
            elif state=='TMARK_PROBE_RUNNING':
                receipts['tmark_probe']=run_owned(cs['tmark'],True,receipts['tmark_probe_permit']);update('TMARK_AUTO_AUDIT')
            elif state=='TMARK_AUTO_AUDIT':
                receipts['tmark_admission']=audit_probe(cs['tmark'],receipts['tmark_probe_permit']);stop_check(root)
                receipts['tmark_formal_permit']=create_permit(cs['tmark'],False,old,admission=receipts['tmark_admission']);update('TMARK_FORMAL_RUNNING')
            elif state=='TMARK_FORMAL_RUNNING':
                receipts['tmark_formal']=run_owned(cs['tmark'],False,receipts['tmark_formal_permit']);stop_check(root)
                replacement=seal(replacement_path(),build_replacement_boundary(cs['tmark']));update('TMARK_BOUNDARY_SEALED')
            elif state=='TMARK_BOUNDARY_SEALED':
                receipts['m_probe_permit']=create_permit(cs['m'],True,old,replacement);update('M_PROBE_RUNNING')
            elif state=='M_PROBE_RUNNING':
                receipts['m_probe']=run_owned(cs['m'],True,receipts['m_probe_permit']);update('M_AUTO_PROBE_AUDIT')
            elif state=='M_AUTO_PROBE_AUDIT':
                receipts['m_admission']=audit_probe(cs['m'],receipts['m_probe_permit']);stop_check(root)
                receipts['m_formal_permit']=create_permit(cs['m'],False,old,replacement,receipts['m_admission']);update('M_FORMAL_RUNNING')
            elif state=='M_FORMAL_RUNNING':
                receipts['m_formal']=run_owned(cs['m'],False,receipts['m_formal_permit']);update('COMPLETE')
            else:raise ValueError('unrecognized successor state')
        dump(root/'complete.json',dict(scope=scope.ID,technical_complete=True,original_old_batch_technical_complete=False,old_retirement_accepted=True,result_review='pending',manual_review=False,old_boundary=old,replacement_boundary=replacement,receipts=receipts,bindings=initial))
    except BaseException as exc:dump(root/'failure.json',dict(state=state,error=repr(exc),receipts=receipts,automatic_retry=False,result_review='pending'));raise
    finally:signal.signal(signal.SIGTERM,previous)


def start(token):
    reasons=preflight(True,token)
    if reasons:raise PermissionError('; '.join(reasons))
    with lock(scope.PACKAGE/'.native-chain.lock'):
        if ROOT_CONTROL.exists():raise FileExistsError('duplicate successor chain')
        exclusive_json(launch_paths('handoff')['claim'],dict(consumer=identity(os.getpid()),launch=source.ref(launch_paths('handoff')['metadata'])))
        ROOT_CONTROL.mkdir();dump(ROOT_CONTROL/'controller.json',dict(**identity(os.getpid()),scope=scope.ID,commit=git('rev-parse','HEAD')));run()


def safe_stop():
    control=json.loads((ROOT_CONTROL/'controller.json').read_text());stop_marker(ROOT_CONTROL);sent=False
    if same(control):
        if not command_has(control['pid'],ROOT/'m6_native_chain_entry.py'):raise ValueError('foreign successor supervisor identity')
        os.kill(control['pid'],signal.SIGTERM);sent=True
    elif (ROOT_CONTROL/'current.json').exists():sent=signal_stage(json.loads((ROOT_CONTROL/'current.json').read_text()).get('child'),control)
    return dict(STOP=True,signal_sent=sent,old_MS_signal_sent=False)


def status():
    value=dict(root=str(ROOT_CONTROL),running=False,complete=(ROOT_CONTROL/'complete.json').exists(),STOP=(ROOT_CONTROL/'STOP').exists(),manual_review=False,result_review='pending')
    for name in ('controller','progress','current','failure'):
        p=ROOT_CONTROL/(name+'.json')
        if p.exists():value[name]=json.loads(p.read_text())
    if value.get('controller'):value['running']=same(value['controller'])
    return value
