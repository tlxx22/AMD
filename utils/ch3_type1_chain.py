"""Single supervised MS -> M chain. Only reviewed closure/start records execute."""
import contextlib,hashlib,hmac,json,os,secrets,signal,subprocess,sys,time
from pathlib import Path
from utils.ch3_contract import ROOT,digest,profile
from utils import ch3_type1_tasks as s
from utils.ch3_native_recovery_records import bound,ref,exclusive,manifest_projection,process_refs,scan_manifest,compact_decisions
from m6_remaining_entry import identity,same,lock,signal_owned

PYTHON='/public/home/yueweiting/大论文/amd-execution-envs/m5-source-smoke-8sr2d3d_/bin/python'
CONTROL=s.RESULT/'queue'/'controller'
LOG=s.PACKAGE/'followup-launcher.log'
SESSION='ch3-baseline-type1-followup-v3'
TOKEN='CH3_TYPE1_LAUNCH_TOKEN'
SECRET='CH3_TYPE1_RUNTIME_SECRET'
ENVIRONMENT_REF=dict(path=str(s.PACKAGE/'environment-hardware.json'),sha256='e2fa17dc103fcc70acb9d7c49db75171126afdedf84f0cb5db0bb9fb944576c2')
STAGE_STATES={'M_AMEND':('AMEND_RESOURCE_NUMERIC_PROBE','AMEND_AUTO_AUDIT','AMEND_FORMAL_ALL_BASELINES','SEAL_AMEND_BOUNDARY'),
    'URBAN_SUBSET':('URBAN_RESOURCE_NUMERIC_PROBE','URBAN_AUTO_AUDIT','URBAN_FORMAL_ALL_BASELINES','SEAL_URBAN_BOUNDARY'),
    'EPF_ALL':('EPF_RESOURCE_NUMERIC_PROBE','EPF_AUTO_AUDIT','EPF_FORMAL_ALL_BASELINES','SEAL_EPF_BOUNDARY'),
    'M_ALL':('M_RESOURCE_NUMERIC_PROBE','M_AUTO_AUDIT','M_FORMAL_ALL_BASELINES','SEAL_M_BOUNDARY')}
STATES=('WAIT_V3_COMPLETE_AND_RELEASED','FOLLOWUP_PROTOCOL_PREFLIGHT',*STAGE_STATES['M_AMEND'],'SEAL_ROUND2_REVISED_BOUNDARY',*STAGE_STATES['URBAN_SUBSET'],*STAGE_STATES['EPF_ALL'],*STAGE_STATES['M_ALL'],'COMPLETE')


def upstream_anchors():
    from utils.ch3_type1_upstream import anchors
    return anchors()

def upstream_status():
    from utils.ch3_type1_upstream import status
    return status()

def wait_upstream():
    from utils.ch3_type1_upstream import status
    while True:
        stop_check();v=status(full=True)
        if v['READY_FOR_GPU_EXECUTION']:
            v.update(handoff_scope=s.ID,successor_owner=owner())
            secret=os.environ.get(SECRET)
            if not secret:raise PermissionError('controlled handoff lifecycle absent')
            v['mac']=hmac.new(secret.encode(),digest(v).encode(),hashlib.sha256).hexdigest()
            return exclusive(CONTROL/'upstream-technical-boundary.json',v)
        time.sleep(30)

def configs():return {stage:s.validate(json.loads(s.file(stage).read_text())) for stage in s.STAGES}


def owner(pid=None):
    row=identity(os.getpid() if pid is None else pid)
    if not row:raise ValueError('owned instance absent')
    return {k:row[k] for k in ('pid','start_ticks')}


def stop_check():
    if (CONTROL/'STOP').exists():raise InterruptedError('persistent unified STOP; no retry')
    if (CONTROL/'failure.json').exists():raise RuntimeError('retained unified failure')


@contextlib.contextmanager
def dispatch_guard():
    with lock(CONTROL/'dispatch.lock'):
        stop_check();yield;stop_check()


def dynamic(c,worker=False):
    from ch3_runner import git,code_binding,environment_binding,hardware_binding
    from utils.ch3_m6 import source_states
    data=bound(c['baseline_unified']['data_ref'])
    env=bound(ENVIRONMENT_REF)
    recipe=bound(s.AUTHOR_RECIPE)
    for value in c['baseline_unified'].get('extension_refs',{}).values():bound(value)
    if c['baseline_unified']['stage']=='M_AMEND':
        from utils.ch3_round2_amendment import RECIPE_REF
        onecycle=bound(RECIPE_REF)
        for value in onecycle['source_refs']:
            if __import__('utils.ch3_native_recovery_records',fromlist=['sha']).sha(value['path'])!=value['sha256']:raise ValueError('frozen TimeMixer OneCycle source changed')
    if any(__import__('utils.ch3_native_recovery_records',fromlist=['sha']).sha(p)!=h for p,h in recipe['files'].items()):raise ValueError('locked TimeXer type1 source recipe changed')
    if env['environment']!=environment_binding() or env['hardware']!=hardware_binding():raise ValueError('frozen unified environment/hardware changed')
    if source_states(c)!=data['source_states']:raise ValueError('data file identity changed')
    for name,rows in data['metadata'].items():
        if {t['id'] for t in c['tasks'] if t['dataset']==name}!=set(rows) or any(digest(m)!=data['data_bindings'][name][r] for r,m in rows.items()):raise ValueError('exact data metadata projection coverage')
    for src in (() if worker else c['sources'].values()):
        if subprocess.check_output(['git','-C',src['repository'],'rev-parse','HEAD'],text=True).strip()!=src['commit'] or any(__import__('utils.ch3_native_recovery_records',fromlist=['sha']).sha(p)!=h for p,h in src['files'].items()):raise ValueError('locked author source changed')
    return dict(commit=git('rev-parse','HEAD'),protocol_sha=digest(c),code=code_binding(),environment=environment_binding(),hardware=hardware_binding(),source_states=data['source_states'])


def closure():
    from ch3_runner import git
    head=git('rev-parse','HEAD')
    if head==s.BASE or git('rev-parse','HEAD^')!=s.BASE or git('branch','--show-current')!='m6/type1-followup-v1' or git('status','--porcelain') or git('rev-parse','@{u}')!=head or git('rev-list','--left-right','--count','HEAD...@{u}').split()!=['0','0']:raise ValueError('reviewed direct-successor clean followup closure required')
    return head


def start_template():
    cs=configs()
    return dict(purpose='baseline_type1_start_authorization_v1',scope=s.ID,scientific_protocol=s.PROTOCOL,
        reviewed=False,execution_permitted=False,closure_commit=None,structure_frozen=False,m6_authorized=False,
        config_refs={stage:ref(s.file(stage)) for stage in cs},plan_refs={stage:ref(s.package(stage)/(stage.lower()+'-plan.json')) for stage in cs},
        upstream_anchors=upstream_anchors(),author_recipe_ref=s.AUTHOR_RECIPE,environment_hardware_ref=ENVIRONMENT_REF,
        formal_caps={stage:s.formal_budget(c)['total'] for stage,c in cs.items()},probe_caps={stage:s.probe_budget(c)['caps'] for stage,c in cs.items()},
        budget_authorized=False,additional_search=0,seed=2024,from_scratch=True,result_review='pending',authorization_basis='non-executable preparation template; bind actual reviewed closure and the user-authorized 112 round-two amendment plus 231 third-round tasks and derived fixed caps')


def validate_start(a):
    head=closure();expected=start_template()
    if not a or a.get('reviewed')is not True or a.get('execution_permitted')is not True or a.get('structure_frozen')is not True or a.get('m6_authorized')is not True or a.get('budget_authorized')is not True or a.get('closure_commit')!=head:raise PermissionError('reviewed unified start authorization and independent new budget required')
    mutable={'reviewed','execution_permitted','structure_frozen','m6_authorized','budget_authorized','closure_commit','authorization_basis'}
    if {k:v for k,v in a.items() if k not in mutable}!={k:v for k,v in expected.items() if k not in mutable}:raise PermissionError('exact unified authorization scope/plan/budget/refs')
    if not a.get('authorization_basis') or 'non-executable' in a['authorization_basis']:raise PermissionError('actual authorization basis required')
    return a


def validate_permit(c,a,probe=False,worker=False):
    stop_check();ctx=s.context(c);start=validate_start(bound(a['start_authorization_ref']))
    expected=dynamic(c,worker)
    if a.get('purpose')!=('baseline_type1_probe_permit_v1' if probe else 'baseline_type1_formal_permit_v1') or a.get('type1_scope')!=s.ID or a.get('successor_scope')!=(ctx['probe_scope'] if probe else ctx['formal_scope']) or a.get('execution_permitted')is not True or a.get('manual_review')is not False or a.get('reviewed')is not False or a.get('review_mode')!='preauthorized_machine_gate':raise PermissionError('exact machine-gate permit only')
    for k,v in expected.items():
        if a.get(k)!=v:raise ValueError('permit current binding changed: '+k)
    if a.get('authorized_task_ids')!=[t['id'] for t in c['tasks']] or a.get('profile_shas')!={t['id']:digest(profile(c,t)) for t in c['tasks']} or a.get('data_binding_ref')!=c['baseline_unified']['data_ref'] or a.get('additional_search')!=0 or a.get('from_scratch')is not True:raise ValueError('exact fresh scientific profile scope')
    if a.get('caps')!=(s.probe_budget(c)['caps'] if probe else s.formal_budget(c)['total']) or a.get('budget_refund')is not False:raise ValueError('independent authorized caps')
    required=s.STAGES[:s.STAGES.index(ctx['stage'])]
    if set(a.get('predecessor_boundaries',{}))!=set(required):raise ValueError('exact new predecessor rings required')
    for stage in required:validate_boundary_light(a['predecessor_boundaries'][stage],stage)
    validate_upstream_boundary_light(a['upstream_boundary_ref'])
    if ctx['stage']!='M_AMEND':validate_round2_boundary_light(a['round2_boundary_ref'])
    elif a.get('round2_boundary_ref') is not None:raise ValueError('amendment must precede the revised round-two boundary')
    if not probe:validate_summary_light(c,a['summary_ref'])
    return a


def validate_upstream_boundary_light(value):
    v=bound(value);body={k:x for k,x in v.items()if k!='mac'};secret=os.environ.get(SECRET,'')
    if value['path']!=str(CONTROL/'upstream-technical-boundary.json') or not secret or not hmac.compare_digest(v.get('mac',''),hmac.new(secret.encode(),digest(body).encode(),hashlib.sha256).hexdigest()):raise PermissionError('handoff boundary not sealed by this controlled lifecycle')
    if v.get('handoff_scope')!=s.ID or v.get('anchors')!=upstream_anchors() or v.get('READY_FOR_GPU_EXECUTION')is not True or v.get('successor_owner')!=json.loads((CONTROL/'controller.json').read_text())['owner'] or not same(v['successor_owner']):raise ValueError('fixed upstream and current successor owner required')
    return v

def validate_summary_light(c,value):
    v=bound(value);ctx=s.context(c)
    secret=os.environ.get(SECRET,'');body={k:x for k,x in v.items() if k!='mac'}
    if value['path']!=str(ctx['control']/'admission-summary.json') or not secret or not hmac.compare_digest(v.get('mac',''),hmac.new(secret.encode(),digest(body).encode(),hashlib.sha256).hexdigest()) or not same(v.get('owner')):raise PermissionError('admission not sealed by this current owned lifecycle')
    if v['owner']!=json.loads((CONTROL/'controller.json').read_text())['owner'] or v['complete_ref']['path']!=str(ctx['probe_root']/'complete.json') or v['manifest_ref']['path']!=str(ctx['control']/'probe-artifact-manifest.json'):raise ValueError('exact AUTO_AUDIT source refs')
    if v.get('purpose')!='baseline_type1_technical_admission_v1' or v.get('scope')!=ctx['probe_scope'] or v.get('technical_admission')is not True or v.get('manual_review')is not False or v.get('reviewed')is not False or v.get('result_review')!='pending':raise ValueError('sealed machine technical admission')
    if v['protocol_sha']!=digest(c) or v['policy_sha']!=digest(c['baseline_unified']['numeric_policies']) or v['profile_shas']!={t['id']:digest(profile(c,t)) for t in c['tasks']}:raise ValueError('admission exact policy/profile')
    groups=s.probe_groups(c)
    if set(v['decisions'])!={g['id'] for g in groups}:raise ValueError('admission exact group coverage')
    for g in groups:
        d=v['decisions'][g['id']]
        if d['status']!='Passed' or d['concurrency']not in (1,2,4) or d['concurrency']>g['planned_q'] or d['coverage']!=g['coverage']:raise ValueError('measured q/coverage summary')
    if v['budget']['caps']!=ctx['caps'] or v['budget']['refund']is not False or any(v['budget']['actual'][k]>v['budget']['reserved'][k] or v['budget']['reserved'][k]>ctx['caps'][k] for k in ctx['caps']):raise ValueError('actual probe budget')
    return v


def validate_boundary_light(value,stage='URBAN_SUBSET'):
    b=bound(value);c=configs()[stage]
    if value['path']!=str(s.context(c)['control']/'technical-boundary.json'):raise ValueError('exact sealed ring boundary path')
    if b.get('purpose')!='baseline_type1_'+stage+'_boundary_v1' or b.get('scope')!=s.ID or b.get('technical_complete')is not True or b.get('result_review')!='pending' or b['task_ids']!=[t['id'] for t in c['tasks']] or b['protocol_sha']!=digest(c):raise ValueError('sealed MS boundary scope')
    return b

def validate_round2_boundary_light(value):
    from utils import ch3_round2_amendment as amend
    v=bound(value);secret=os.environ.get(SECRET,'');body={k:x for k,x in v.items() if k!='mac'}
    if value['path']!=str(amend.RESULT/'queue/round2-boundary.json') or not secret or not hmac.compare_digest(v.get('mac',''),hmac.new(secret.encode(),digest(body).encode(),hashlib.sha256).hexdigest()):raise PermissionError('round-two amendment boundary not sealed by this lifecycle')
    if v.get('owner')!=owner(json.loads((CONTROL/'controller.json').read_text())['owner']['pid']) or not same(v['owner']) or v.get('scope')!=s.ID or v.get('technical_complete')is not True or v.get('result_review')!='pending' or v.get('effective_counts')!={'MS':203,'M':168,'total':371} or v.get('planned_formal_executions')!=399 or v.get('effective_task_set_sha')!=digest(sorted(amend.expected_cells())):raise ValueError('exact revised 371-source boundary and owned lifecycle')
    return v

def seal_round2_boundary(amendment_ref):
    from utils import ch3_round2_amendment as amend
    upstream_ref=ref(CONTROL/'upstream-technical-boundary.json');upstream=validate_upstream_boundary_light(upstream_ref)
    validate_boundary_light(amendment_ref,'M_AMEND')
    main=amend.build_index(upstream['boundaries'],amendment_ref,configs()['M_AMEND'])
    main_ref=exclusive(amend.RESULT/'queue/round2-main-results.json',main)
    body=dict(purpose='round2_revised_technical_boundary_v1',scope=s.ID,owner=owner(),effective_counts=main['effective_counts'],planned_formal_executions=399,effective_task_set_sha=digest(sorted(amend.expected_cells())),main_index_ref=main_ref,original_upstream_ref=upstream_ref,amendment_ref=amendment_ref,technical_complete=True,result_review='pending')
    secret=os.environ.get(SECRET)
    if not secret:raise PermissionError('owned revision seal absent')
    body['mac']=hmac.new(secret.encode(),digest(body).encode(),hashlib.sha256).hexdigest()
    return exclusive(amend.RESULT/'queue/round2-boundary.json',body)


def create_permit(c,start_ref,probe,summary_ref=None,boundary_ref=None,round2_ref=None):
    stop_check();validate_start(bound(start_ref));ctx=s.context(c)
    from utils.ch3_native_recovery import resource_check
    resource_check(c)
    a=dict(purpose='baseline_type1_probe_permit_v1' if probe else 'baseline_type1_formal_permit_v1',type1_scope=s.ID,
        successor_scope=ctx['probe_scope'] if probe else ctx['formal_scope'],**dynamic(c),start_authorization_ref=start_ref,
        reviewed=False,manual_review=False,review_mode='preauthorized_machine_gate',execution_permitted=True,
        authorized_task_ids=[t['id'] for t in c['tasks']],profile_shas={t['id']:digest(profile(c,t)) for t in c['tasks']},
        data_binding_ref=c['baseline_unified']['data_ref'],caps=s.probe_budget(c)['caps'] if probe else s.formal_budget(c)['total'],
        budget_refund=False,additional_search=0,from_scratch=True,result_review='pending',upstream_boundary_ref=ref(CONTROL/'upstream-technical-boundary.json'),authorization_basis='user pre-authorized full training iff preregistered technical gates pass')
    a['predecessor_boundaries']=boundary_ref or {}
    a['round2_boundary_ref']=round2_ref
    if not probe:
        summary=validate_summary_light(c,summary_ref);a['summary_ref']=summary_ref;a['manifest_ref']=summary['manifest_ref'];a['technical_admission']=True
    validate_permit(c,a,probe)
    return exclusive(ctx['control']/('probe-permit.json' if probe else 'formal-permit.json'),a)


def audit_probe(c):
    """One saved-evidence full audit at AUTO_AUDIT, never in formal paths."""
    from utils.ch3_native_execution import validate_probe_completion
    ctx=s.context(c);complete_ref=ref(ctx['probe_root']/'complete.json');report=bound(complete_ref)
    validate_probe_completion(c,report);stop_check()
    manifest=manifest_projection(report,complete_ref,ctx['probe_root'],process_refs(report))
    manifest_ref=exclusive(ctx['control']/'probe-artifact-manifest.json',manifest)
    summary=dict(purpose='baseline_type1_technical_admission_v1',scope=ctx['probe_scope'],technical_admission=True,
        reviewed=False,manual_review=False,review_mode='preauthorized_machine_gate',result_review='pending',
        complete_ref=complete_ref,manifest_ref=manifest_ref,protocol_sha=digest(c),policy_sha=digest(c['baseline_unified']['numeric_policies']),
        profile_shas={t['id']:digest(profile(c,t)) for t in c['tasks']},decisions=compact_decisions(report),budget=report['budget'],
        owner=owner(),**{k:report[k] for k in ('commit','code','environment','hardware')})
    secret=os.environ.get(SECRET)
    if not secret:raise PermissionError('AUTO_AUDIT controlled lifecycle absent')
    summary['mac']=hmac.new(secret.encode(),digest(summary).encode(),hashlib.sha256).hexdigest()
    return exclusive(ctx['control']/'admission-summary.json',summary)


def seal_runtime(c,permit_ref):
    ctx=s.context(c);a=validate_permit(c,bound(permit_ref));summary=validate_summary_light(c,a['summary_ref']);manifest=bound(a['manifest_ref'])
    # Exactly one integrity scan per stage supervisor lifecycle. Child processes consume HMAC-bound refs only.
    scan_manifest(manifest,summary['complete_ref'],ctx['probe_root']);stop_check()
    body=dict(purpose='baseline_type1_runtime_v1',scope=s.ID,stage=ctx['stage'],owner=owner(),permit_ref=permit_ref,
        summary_ref=a['summary_ref'],manifest_ref=a['manifest_ref'],protocol_sha=digest(c),commit=a['commit'],code=a['code'],
        integrity_scan_passed=True,full_scans=1,result_review='pending',upstream_boundary_ref=a['upstream_boundary_ref'],science_protocol=c['baseline_unified']['id'])
    secret=os.environ.get(SECRET)
    if not secret:raise PermissionError('owned runtime secret absent')
    body['mac']=hmac.new(secret.encode(),digest(body).encode(),hashlib.sha256).hexdigest()
    return exclusive(ctx['control']/'runtime-admission.json',body)


def validate_runtime(c,value,permit_ref):
    stop_check();v=bound(value);body={k:x for k,x in v.items() if k!='mac'};secret=os.environ.get(SECRET,'')
    if not secret or not hmac.compare_digest(v.get('mac',''),hmac.new(secret.encode(),digest(body).encode(),hashlib.sha256).hexdigest()):raise PermissionError('runtime not from current controlled chain')
    if v.get('scope')!=s.ID or v.get('stage')!=s.context(c)['stage'] or v['permit_ref']!=permit_ref or v['protocol_sha']!=digest(c) or v['integrity_scan_passed']is not True or v['full_scans']!=1 or not same(v['owner']):raise ValueError('current formal lifecycle binding')
    if v['owner']!=json.loads((CONTROL/'controller.json').read_text())['owner'] or v.get('science_protocol')!=c['baseline_unified']['id']:raise ValueError('different supervisor lifecycle/protocol')
    return v


def readiness(a=None,launch=False):
    reasons=[]
    if not a or a.get('reviewed')is not True or a.get('execution_permitted')is not True:reasons.append('reviewed followup start record not materialized')
    if not a or a.get('budget_authorized')is not True:reasons.append('user-authorized fixed caps await reviewed closure/start-record binding')
    try:
        configs();upstream_status();closure();validate_start(a)
    except (OSError,KeyError,ValueError,PermissionError,subprocess.CalledProcessError) as exc:reasons.append(str(exc))
    for stage,c in configs().items():
        try:dynamic(c)
        except (OSError,KeyError,ValueError,subprocess.CalledProcessError) as exc:reasons.append(str(exc))
    if s.RESULT.exists() or s.RESULT.is_symlink() or (not launch and (LOG.exists() or LOG.is_symlink() or Path(str(LOG)+'.launch.json').exists() or Path(str(LOG)+'.claimed.json').exists())):reasons.append('retained unified execution/output/launcher; fresh repeat forbidden')
    from utils.ch3_round2_amendment import RESULT as amendment_result
    if amendment_result.exists() or amendment_result.is_symlink():reasons.append('retained amendment result root; fresh repeat forbidden')
    if not launch and subprocess.run(['tmux','has-session','-t',SESSION],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:reasons.append('retained unified tmux')
    return list(dict.fromkeys(reasons))


def readiness_report(a=None):
    blocked=readiness(a)
    try:v=upstream_status()
    except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError) as exc:v=dict(state='UPSTREAM_BLOCKED',error=str(exc),READY_FOR_GPU_EXECUTION=False)
    return dict(blocked=blocked,READY_TO_ARM_HANDOFF=not blocked,READY_FOR_GPU_EXECUTION=False,upstream=v,arming_does_not_require_old_completion=True,execution_requires_full_203_MS_84_M_and_owned_exit=True)


def wrapper_command(pid):
    proc=Path('/proc')/str(pid);args=proc.joinpath('cmdline').read_bytes().decode().split('\0');cwd=proc.joinpath('cwd').resolve()
    return any(a in ('start','arm')for a in args) and any(a and (Path(a) if Path(a).is_absolute()else cwd/a)==ROOT/'scripts/ch3/start_type1_followup.sh'for a in args)

def prepare_launch(approval_ref,pid):
    from utils.ch3_m_launch import ancestors,file_identity,command_has
    if pid not in [v['pid'] for v in ancestors(os.getpid())] or not same(owner(pid)) or not wrapper_command(pid):raise PermissionError('actual unified wrapper ancestor required')
    if s.RESULT.exists() or LOG.is_symlink() or LOG.stat().st_size!=0:raise ValueError('new noclobber launcher required')
    token=secrets.token_hex(32)
    exclusive(str(LOG)+'.launch.json',dict(scope=s.ID,approval=approval_ref,log=str(LOG),log_identity=file_identity(LOG),wrapper=owner(pid),token_sha=hashlib.sha256(token.encode()).hexdigest()))
    return token


def verify_launch(approval_ref,token):
    from utils.ch3_m_launch import file_identity,tmux_view
    if not isinstance(token,str) or len(token)!=64:raise PermissionError('wrapper token required; no internal direct start')
    v=json.loads(Path(str(LOG)+'.launch.json').read_text())
    if v['scope']!=s.ID or v['approval']!=approval_ref or v['log']!=str(LOG) or v['log_identity']!=file_identity(LOG) or not secrets.compare_digest(v['token_sha'],hashlib.sha256(token.encode()).hexdigest()):raise PermissionError('launcher identity mismatch')
    if Path(str(LOG)+'.claimed.json').exists():raise FileExistsError('launch already claimed')
    v['tmux']=tmux_view(SESSION);return v


def safe_stop(signal_supervisor=True):
    if not (CONTROL/'controller.json').exists():raise ValueError('no owned unified supervisor')
    record=json.loads((CONTROL/'controller.json').read_text())
    if record.get('scope')!=s.ID:raise PermissionError('only this new scope may be stopped')
    with (CONTROL/'STOP').open('a') as f:f.flush();os.fsync(f.fileno())
    controller=record['owner']
    current=json.loads((CONTROL/'current.json').read_text()) if (CONTROL/'current.json').exists() else {}
    child=current.get('child');sent=False
    from utils.ch3_m_launch import command_has
    if child and same(child):
        args=Path('/proc',str(child['pid']),'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        if current.get('owner')!=controller or args!=current.get('command') or str(ROOT/'m6_type1_followup_entry.py')not in args or not any(action in args for action in ('probe-child','group-child')) or int(Path('/proc',str(child['pid']),'stat').read_text().rsplit(')',1)[1].split()[1])!=controller['pid']:raise PermissionError('owned child parent changed')
        sent=signal_owned(child)
    if same(controller) and not command_has(controller['pid'],ROOT/'m6_type1_followup_entry.py'):raise PermissionError('foreign supervisor')
    return dict(STOP=True,child_signal_sent=sent,supervisor_signal_sent=signal_owned(controller) if signal_supervisor and same(controller) else False,old_chain_signal_sent=False)


def wait_owned(stage,permit_ref,probe,model=None,runtime_ref=None):
    stop_check();label=stage+('-probe' if probe else '-'+model);args=[PYTHON,'-B',str(ROOT/'m6_type1_followup_entry.py'),'probe-child' if probe else 'group-child','--stage',stage,'--approval',permit_ref['path'],'--approval-sha',permit_ref['sha256']]
    if model:args+=['--model',model,'--runtime',runtime_ref['path'],'--runtime-sha',runtime_ref['sha256']]
    child=None;instance=None
    try:
        with (CONTROL/(label+'.log')).open('x') as log:
            with dispatch_guard():
                child=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
                instance=owner(child.pid)
                from ch3_runner import dump
                dump(CONTROL/'current.json',dict(owner=owner(),child=instance,permit=permit_ref,command=args))
            while child.poll()is None:
                if (CONTROL/'STOP').exists():signal_owned(instance)
                time.sleep(.1)
        stop_check()
        if child.returncode!=0 or same(instance):raise RuntimeError('owned child technical failure: '+label)
        return dict(child=instance,exit_code=child.returncode,log=str(CONTROL/(label+'.log')))
    finally:
        if child is not None and child.poll()is None:
            signal_owned(instance)
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:raise RuntimeError('owned cleanup timeout; retained evidence')


def validate_child(permit_ref):
    v=json.loads((CONTROL/'current.json').read_text())
    if v['child']!=owner() or v['owner']!=owner(os.getppid()) or v['permit']!=permit_ref or not same(v['owner']):raise PermissionError('actual synchronous owned child required')


def seal_boundary(c,receipts):
    stop_check()
    for m in s.MODELS:
        r=bound(receipts[m])
        if r.get('technical_complete')is not True or r['model']!=m or r['task_ids']!=[t['id'] for t in c['tasks'] if t['model']==m]:raise ValueError('exact successful model receipt')
    return exclusive(s.context(c)['control']/'technical-boundary.json',dict(purpose='baseline_type1_'+s.context(c)['stage']+'_boundary_v1',scope=s.ID,
        task_ids=[t['id'] for t in c['tasks']],protocol_sha=digest(c),receipts=receipts,technical_complete=True,result_review='pending',**dynamic(c)))


def drive(actions,update,check):
    """Sequential synchronous state machine, deliberately independent of effect metrics."""
    receipts={}
    for state in STATES:
        check();update(state)
        if state!='COMPLETE':receipts[state]=actions[state](receipts)
        check()
    return receipts


def run(start_ref):
    from ch3_runner import dump
    cs=configs()
    def probe(stage,r):
        c=cs[stage];ctx=s.context(c);ctx['control'].mkdir(parents=True,exist_ok=False)
        predecessors={k:r[STAGE_STATES[k][3]]for k in s.STAGES[:s.STAGES.index(stage)]}
        pr=create_permit(c,start_ref,True,boundary_ref=predecessors,round2_ref=r.get('SEAL_ROUND2_REVISED_BOUNDARY'))
        wait_owned(stage,pr,True);return pr
    def formal(stage,r):
        c=cs[stage];summary=r[STAGE_STATES[stage][1]]
        predecessors={k:r[STAGE_STATES[k][3]]for k in s.STAGES[:s.STAGES.index(stage)]}
        pr=create_permit(c,start_ref,False,summary,predecessors,r.get('SEAL_ROUND2_REVISED_BOUNDARY'))
        runtime=seal_runtime(c,pr);receipts={}
        for model in s.MODELS:
            stop_check();wait_owned(stage,pr,False,model,runtime)
            receipts[model]=ref(s.context(c)['control']/('group-'+model)/'complete.json')
        return receipts
    actions={'WAIT_V3_COMPLETE_AND_RELEASED':lambda r:wait_upstream(), 'FOLLOWUP_PROTOCOL_PREFLIGHT':lambda r:validate_start(bound(start_ref)),
        'SEAL_ROUND2_REVISED_BOUNDARY':lambda r:seal_round2_boundary(r['SEAL_AMEND_BOUNDARY'])}
    for stage,(probe_state,audit_state,formal_state,seal_state) in STAGE_STATES.items():
        actions[probe_state]=lambda r,stage=stage:probe(stage,r)
        actions[audit_state]=lambda r,stage=stage:audit_probe(cs[stage])
        actions[formal_state]=lambda r,stage=stage:formal(stage,r)
        actions[seal_state]=lambda r,stage=stage:seal_boundary(cs[stage],r[STAGE_STATES[stage][2]])
    receipts=drive(actions,lambda state:dump(CONTROL/'progress.json',dict(state=state,scope=s.ID,result_review='pending')),stop_check)
    stop_check()
    exclusive(CONTROL/'complete.json',dict(scope=s.ID,technical_complete=True,result_review='pending',AMEND_boundary=receipts['SEAL_AMEND_BOUNDARY'],round2_boundary=receipts['SEAL_ROUND2_REVISED_BOUNDARY'],URBAN_boundary=receipts['SEAL_URBAN_BOUNDARY'],EPF_boundary=receipts['SEAL_EPF_BOUNDARY'],M_boundary=receipts['SEAL_M_BOUNDARY'],total_runs=343,third_round_runs=231,round2_effective_runs=371))


def start(start_ref,token):
    launch=verify_launch(start_ref,token);reasons=readiness(bound(start_ref),True)
    if reasons:raise PermissionError('; '.join(reasons))
    with lock(s.PACKAGE/'queue.lock'):
        if s.RESULT.exists():raise FileExistsError('retained execution')
        exclusive(str(LOG)+'.claimed.json',dict(launch=ref(str(LOG)+'.launch.json'),consumer=owner(),tmux=launch['tmux']))
        CONTROL.mkdir(parents=True,exist_ok=False)
        for stage in s.STAGES:(s.package(stage)/'fixtures').mkdir(exist_ok=True)
        exclusive(CONTROL/'controller.json',dict(scope=s.ID,owner=owner(),launch=ref(str(LOG)+'.claimed.json'),authorization=start_ref))
        os.environ[SECRET]=secrets.token_hex(32)
        signal.signal(signal.SIGTERM,lambda *_:safe_stop(signal_supervisor=False))
        try:run(start_ref)
        except BaseException as exc:
            exclusive(CONTROL/'failure.json',dict(error=repr(exc),scope=s.ID,automatic_retry=False,result_review='pending'));raise


def status():
    r=dict(scope=s.ID,state='Prepared',running=False,result_review='pending',log=str(LOG))
    if (CONTROL/'controller.json').exists():
        r['controller']=json.loads((CONTROL/'controller.json').read_text());r['running']=same(r['controller']['owner'])
        if (CONTROL/'progress.json').exists():r.update(json.loads((CONTROL/'progress.json').read_text()))
    r.update(STOP=(CONTROL/'STOP').exists(),failure=(CONTROL/'failure.json').exists(),complete=(CONTROL/'complete.json').exists())
    return r
