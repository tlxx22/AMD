"""Single supervised MS -> M chain. Only reviewed closure/start records execute."""
import contextlib,hashlib,hmac,json,os,secrets,signal,subprocess,sys,time
from pathlib import Path
from utils.ch3_contract import ROOT,digest,profile
from utils import ch3_baseline_unified_tasks as s
from utils.ch3_native_recovery_records import bound,ref,exclusive,manifest_projection,process_refs,scan_manifest,compact_decisions
from m6_remaining_entry import identity,same,lock,signal_owned

PYTHON='/public/home/yueweiting/大论文/amd-execution-envs/m5-source-smoke-8sr2d3d_/bin/python'
CONTROL=s.RESULT/'queue'/'controller'
LOG=s.PACKAGE/'unified-launcher.log'
SESSION='ch3-baseline-unified96-oc01-v3'
TOKEN='CH3_UNIFIED_LAUNCH_TOKEN'
SECRET='CH3_UNIFIED_RUNTIME_SECRET'
ENVIRONMENT_REF=dict(path=str(s.PACKAGE/'environment-hardware.json'),sha256='e2fa17dc103fcc70acb9d7c49db75171126afdedf84f0cb5db0bb9fb944576c2')
STATES=('BASELINE_PROTOCOL_PREFLIGHT','MS_RESOURCE_NUMERIC_PROBE','MS_AUTO_AUDIT','MS_FORMAL_ALL_BASELINES','SEAL_MS_BOUNDARY','M_RESOURCE_NUMERIC_PROBE','M_AUTO_AUDIT','M_FORMAL_ALL_BASELINES','COMPLETE')


def configs():return {stage:s.validate(json.loads(s.file(stage).read_text())) for stage in ('MS','M')}


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
    if head==s.BASE or git('rev-parse','HEAD^')!=s.BASE or git('branch','--show-current')!='m6/m-baselines-v1' or git('status','--porcelain') or git('rev-parse','@{u}')!=head or git('rev-list','--left-right','--count','HEAD...@{u}').split()!=['0','0']:raise ValueError('reviewed direct-successor clean unified closure required')
    return head


def start_template():
    cs=configs()
    return dict(purpose='baseline_unified_start_authorization_v1',scope=s.ID,scientific_protocol=s.PROTOCOL,
        reviewed=False,execution_permitted=False,closure_commit=None,structure_frozen=False,m6_authorized=False,
        config_refs={stage:ref(s.file(stage)) for stage in cs},plan_refs={stage:ref(s.PACKAGE/(stage.lower()+'-plan.json')) for stage in cs},
        retirement_ref=s.RETIREMENT,author_recipe_ref=s.AUTHOR_RECIPE,environment_hardware_ref=ENVIRONMENT_REF,
        formal_caps={stage:s.formal_budget(c)['total'] for stage,c in cs.items()},probe_caps={stage:s.probe_budget(c)['caps'] for stage,c in cs.items()},
        budget_authorized=False,additional_search=0,seed=2024,from_scratch=True,result_review='pending',authorization_basis='non-executable preparation template; bind actual reviewed closure and explicit new budget authorization')


def validate_start(a):
    head=closure();expected=start_template()
    if not a or a.get('reviewed')is not True or a.get('execution_permitted')is not True or a.get('structure_frozen')is not True or a.get('m6_authorized')is not True or a.get('budget_authorized')is not True or a.get('closure_commit')!=head:raise PermissionError('reviewed unified start authorization and independent new budget required')
    mutable={'reviewed','execution_permitted','structure_frozen','m6_authorized','budget_authorized','closure_commit','authorization_basis'}
    if {k:v for k,v in a.items() if k not in mutable}!={k:v for k,v in expected.items() if k not in mutable}:raise PermissionError('exact unified authorization scope/plan/budget/refs')
    if not a.get('authorization_basis') or 'non-executable' in a['authorization_basis']:raise PermissionError('actual authorization basis required')
    receipt=bound(s.RETIREMENT)
    if receipt.get('budget_refund')is not False or receipt.get('scientific_failure')is not False:raise ValueError('exact prior protocol retirement')
    return a


def validate_permit(c,a,probe=False,worker=False):
    stop_check();ctx=s.context(c);start=validate_start(bound(a['start_authorization_ref']))
    expected=dynamic(c,worker)
    if a.get('purpose')!=('baseline_unified_probe_permit_v1' if probe else 'baseline_unified_formal_permit_v1') or a.get('unified_scope')!=s.ID or a.get('successor_scope')!=(ctx['probe_scope'] if probe else ctx['formal_scope']) or a.get('execution_permitted')is not True or a.get('manual_review')is not False or a.get('reviewed')is not False or a.get('review_mode')!='preauthorized_machine_gate':raise PermissionError('exact machine-gate permit only')
    for k,v in expected.items():
        if a.get(k)!=v:raise ValueError('permit current binding changed: '+k)
    if a.get('authorized_task_ids')!=[t['id'] for t in c['tasks']] or a.get('profile_shas')!={t['id']:digest(profile(c,t)) for t in c['tasks']} or a.get('data_binding_ref')!=c['baseline_unified']['data_ref'] or a.get('additional_search')!=0 or a.get('from_scratch')is not True:raise ValueError('exact fresh scientific profile scope')
    if a.get('caps')!=(s.probe_budget(c)['caps'] if probe else s.formal_budget(c)['total']) or a.get('budget_refund')is not False:raise ValueError('independent authorized caps')
    if ctx['stage']=='M':validate_boundary_light(a['ms_boundary_ref'])
    if not probe:validate_summary_light(c,a['summary_ref'])
    return a


def validate_summary_light(c,value):
    v=bound(value);ctx=s.context(c)
    secret=os.environ.get(SECRET,'');body={k:x for k,x in v.items() if k!='mac'}
    if value['path']!=str(ctx['control']/'admission-summary.json') or not secret or not hmac.compare_digest(v.get('mac',''),hmac.new(secret.encode(),digest(body).encode(),hashlib.sha256).hexdigest()) or not same(v.get('owner')):raise PermissionError('admission not sealed by this current owned lifecycle')
    if v['owner']!=json.loads((CONTROL/'controller.json').read_text())['owner'] or v['complete_ref']['path']!=str(ctx['probe_root']/'complete.json') or v['manifest_ref']['path']!=str(ctx['control']/'probe-artifact-manifest.json'):raise ValueError('exact AUTO_AUDIT source refs')
    if v.get('purpose')!='baseline_unified_technical_admission_v1' or v.get('scope')!=ctx['probe_scope'] or v.get('technical_admission')is not True or v.get('manual_review')is not False or v.get('reviewed')is not False or v.get('result_review')!='pending':raise ValueError('sealed machine technical admission')
    if v['protocol_sha']!=digest(c) or v['policy_sha']!=digest(c['baseline_unified']['numeric_policies']) or v['profile_shas']!={t['id']:digest(profile(c,t)) for t in c['tasks']}:raise ValueError('admission exact policy/profile')
    groups=s.probe_groups(c)
    if set(v['decisions'])!={g['id'] for g in groups}:raise ValueError('admission exact group coverage')
    for g in groups:
        d=v['decisions'][g['id']]
        if d['status']!='Passed' or d['concurrency']not in (1,2,4) or d['concurrency']>g['planned_q'] or d['coverage']!=g['coverage']:raise ValueError('measured q/coverage summary')
    if v['budget']['caps']!=ctx['caps'] or v['budget']['refund']is not False or any(v['budget']['actual'][k]>v['budget']['reserved'][k] or v['budget']['reserved'][k]>ctx['caps'][k] for k in ctx['caps']):raise ValueError('actual probe budget')
    return v


def validate_boundary_light(value):
    b=bound(value);c=configs()['MS']
    if b.get('purpose')!='baseline_unified_MS_boundary_v1' or b.get('scope')!=s.ID or b.get('technical_complete')is not True or b.get('result_review')!='pending' or b['task_ids']!=[t['id'] for t in c['tasks']] or b['protocol_sha']!=digest(c):raise ValueError('sealed MS boundary scope')
    return b


def create_permit(c,start_ref,probe,summary_ref=None,boundary_ref=None):
    stop_check();validate_start(bound(start_ref));ctx=s.context(c)
    from utils.ch3_native_recovery import resource_check
    resource_check(c)
    a=dict(purpose='baseline_unified_probe_permit_v1' if probe else 'baseline_unified_formal_permit_v1',unified_scope=s.ID,
        successor_scope=ctx['probe_scope'] if probe else ctx['formal_scope'],**dynamic(c),start_authorization_ref=start_ref,
        reviewed=False,manual_review=False,review_mode='preauthorized_machine_gate',execution_permitted=True,
        authorized_task_ids=[t['id'] for t in c['tasks']],profile_shas={t['id']:digest(profile(c,t)) for t in c['tasks']},
        data_binding_ref=c['baseline_unified']['data_ref'],caps=s.probe_budget(c)['caps'] if probe else s.formal_budget(c)['total'],
        budget_refund=False,additional_search=0,from_scratch=True,result_review='pending',authorization_basis='user pre-authorized full training iff preregistered technical gates pass')
    if boundary_ref:a['ms_boundary_ref']=boundary_ref
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
    summary=dict(purpose='baseline_unified_technical_admission_v1',scope=ctx['probe_scope'],technical_admission=True,
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
    body=dict(purpose='baseline_unified_runtime_v1',scope=s.ID,stage=ctx['stage'],owner=owner(),permit_ref=permit_ref,
        summary_ref=a['summary_ref'],manifest_ref=a['manifest_ref'],protocol_sha=digest(c),commit=a['commit'],code=a['code'],
        integrity_scan_passed=True,full_scans=1,result_review='pending')
    secret=os.environ.get(SECRET)
    if not secret:raise PermissionError('owned runtime secret absent')
    body['mac']=hmac.new(secret.encode(),digest(body).encode(),hashlib.sha256).hexdigest()
    return exclusive(ctx['control']/'runtime-admission.json',body)


def validate_runtime(c,value,permit_ref):
    stop_check();v=bound(value);body={k:x for k,x in v.items() if k!='mac'};secret=os.environ.get(SECRET,'')
    if not secret or not hmac.compare_digest(v.get('mac',''),hmac.new(secret.encode(),digest(body).encode(),hashlib.sha256).hexdigest()):raise PermissionError('runtime not from current controlled chain')
    if v.get('scope')!=s.ID or v.get('stage')!=s.context(c)['stage'] or v['permit_ref']!=permit_ref or v['protocol_sha']!=digest(c) or v['integrity_scan_passed']is not True or v['full_scans']!=1 or not same(v['owner']):raise ValueError('current formal lifecycle binding')
    if v['owner']!=json.loads((CONTROL/'controller.json').read_text())['owner']:raise ValueError('different supervisor lifecycle')
    return v


def readiness(a=None,launch=False):
    reasons=[]
    if not a or a.get('reviewed')is not True or a.get('execution_permitted')is not True:reasons.append('reviewed unified start authorization absent')
    if not a or a.get('budget_authorized')is not True:reasons.append('independent unified formal/probe execution budget authorization required')
    try:
        configs();bound(s.RETIREMENT);closure();validate_start(a)
    except (OSError,KeyError,ValueError,PermissionError,subprocess.CalledProcessError) as exc:reasons.append(str(exc))
    for stage,c in configs().items():
        try:dynamic(c)
        except (OSError,KeyError,ValueError,subprocess.CalledProcessError) as exc:reasons.append(str(exc))
    if s.RESULT.exists() or s.RESULT.is_symlink() or (not launch and (LOG.exists() or LOG.is_symlink() or Path(str(LOG)+'.launch.json').exists() or Path(str(LOG)+'.claimed.json').exists())):reasons.append('retained unified execution/output/launcher; fresh repeat forbidden')
    if not launch and subprocess.run(['tmux','has-session','-t',SESSION],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:reasons.append('retained unified tmux')
    if a:
        try:
            from utils.ch3_native_recovery import resource_check
            resource_check(configs()['MS'])
        except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError) as exc:reasons.append(str(exc))
    return list(dict.fromkeys(reasons))


def prepare_launch(approval_ref,pid):
    from utils.ch3_m_launch import ancestors,file_identity,command_has
    if pid not in [v['pid'] for v in ancestors(os.getpid())] or not same(owner(pid)) or not command_has(pid,ROOT/'scripts/ch3/start_baseline_unified.sh'):raise PermissionError('actual unified wrapper ancestor required')
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
    with (CONTROL/'STOP').open('a') as f:f.flush();os.fsync(f.fileno())
    record=json.loads((CONTROL/'controller.json').read_text());controller=record['owner']
    current=json.loads((CONTROL/'current.json').read_text()) if (CONTROL/'current.json').exists() else {}
    child=current.get('child');sent=False
    from utils.ch3_m_launch import command_has
    if child and same(child):
        args=Path('/proc',str(child['pid']),'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        if current.get('owner')!=controller or args!=current.get('command') or str(ROOT/'m6_baseline_unified_entry.py')not in args or not any(action in args for action in ('probe-child','group-child')) or int(Path('/proc',str(child['pid']),'stat').read_text().rsplit(')',1)[1].split()[1])!=controller['pid']:raise PermissionError('owned child parent changed')
        sent=signal_owned(child)
    if same(controller) and not command_has(controller['pid'],ROOT/'m6_baseline_unified_entry.py'):raise PermissionError('foreign supervisor')
    return dict(STOP=True,child_signal_sent=sent,supervisor_signal_sent=signal_owned(controller) if signal_supervisor else False,old_chain_signal_sent=False)


def wait_owned(stage,permit_ref,probe,model=None,runtime_ref=None):
    stop_check();label=stage+('-probe' if probe else '-'+model);args=[PYTHON,'-B',str(ROOT/'m6_baseline_unified_entry.py'),'probe-child' if probe else 'group-child','--stage',stage,'--approval',permit_ref['path'],'--approval-sha',permit_ref['sha256']]
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
    return exclusive(s.context(c)['control']/'technical-boundary.json',dict(purpose='baseline_unified_'+s.context(c)['stage']+'_boundary_v1',scope=s.ID,
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
        pr=create_permit(c,start_ref,True,boundary_ref=r.get('SEAL_MS_BOUNDARY'))
        wait_owned(stage,pr,True);return pr
    def formal(stage,r):
        c=cs[stage];summary=r[stage+'_AUTO_AUDIT'];pr=create_permit(c,start_ref,False,summary,r.get('SEAL_MS_BOUNDARY'))
        runtime=seal_runtime(c,pr);receipts={}
        for model in s.MODELS:
            stop_check();wait_owned(stage,pr,False,model,runtime)
            receipts[model]=ref(s.context(c)['control']/('group-'+model)/'complete.json')
        if stage=='M':return seal_boundary(c,receipts)
        return receipts
    actions={'BASELINE_PROTOCOL_PREFLIGHT':lambda r:validate_start(bound(start_ref)),
        'MS_RESOURCE_NUMERIC_PROBE':lambda r:probe('MS',r),'MS_AUTO_AUDIT':lambda r:audit_probe(cs['MS']),
        'MS_FORMAL_ALL_BASELINES':lambda r:formal('MS',r),'SEAL_MS_BOUNDARY':lambda r:seal_boundary(cs['MS'],r['MS_FORMAL_ALL_BASELINES']),
        'M_RESOURCE_NUMERIC_PROBE':lambda r:probe('M',r),'M_AUTO_AUDIT':lambda r:audit_probe(cs['M']),
        'M_FORMAL_ALL_BASELINES':lambda r:formal('M',r)}
    receipts=drive(actions,lambda state:dump(CONTROL/'progress.json',dict(state=state,scope=s.ID,result_review='pending')),stop_check)
    mb=receipts['M_FORMAL_ALL_BASELINES'];stop_check()
    exclusive(CONTROL/'complete.json',dict(scope=s.ID,technical_complete=True,result_review='pending',MS_boundary=receipts['SEAL_MS_BOUNDARY'],M_boundary=mb,total_runs=287))


def start(start_ref,token):
    launch=verify_launch(start_ref,token);reasons=readiness(bound(start_ref),True)
    if reasons:raise PermissionError('; '.join(reasons))
    with lock(s.PACKAGE/'queue.lock'):
        if s.RESULT.exists():raise FileExistsError('retained execution')
        exclusive(str(LOG)+'.claimed.json',dict(launch=ref(str(LOG)+'.launch.json'),consumer=owner(),tmux=launch['tmux']))
        CONTROL.mkdir(parents=True,exist_ok=False);(s.PACKAGE/'fixtures').mkdir(exist_ok=True)
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
