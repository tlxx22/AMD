"""Bound execution repair; historical probe is never replayed by formal paths."""
import ast
import contextlib
import contextvars
import copy
import json
import hashlib
import hmac
import os
import secrets
import signal
import subprocess
import time
from pathlib import Path
from utils.ch3_contract import ROOT, digest, profile, task_by_id, step_arithmetic
from utils import ch3_native_recovery_records as records
from utils import ch3_native_tasks as scope
from utils import ch3_m_tasks as source
from m6_remaining_entry import identity, same, lock, stop_marker, signal_owned

BASE = records.BASE
ID = records.ID
PACKAGE = scope.PACKAGE.parent/'native-time-mark-chain-v4-recovery1'
CONTROL = PACKAGE/'chain-execution-v1'
SESSION = 'ch3-native-tmark-chain-v4-recovery1'
LOG = PACKAGE/'recovery-launcher.log'
TMARK_RESULT = scope.EVIDENCE/'revisions/native-time-mark-v4-recovery1'
M_RESULT = scope.EVIDENCE/'m-tasks/m-baselines-native-time-mark-v4-recovery1'
TOKEN_ENV = 'CH3_NATIVE_RECOVERY_TOKEN'
RUNTIME_SECRET_ENV = 'CH3_NATIVE_RECOVERY_RUNTIME_SECRET'
STATES = ('VERIFY_ORIGINAL_ADMISSION','VERIFY_RECOVERY_INVENTORY','SEAL_RUNTIME_ADMISSION',
          'REMAINING_TMARK_FORMAL','SEAL_87_REPLACEMENT_BOUNDARY','M_PROBE','M_AUTO_AUDIT','M_FORMAL','COMPLETE')
TRUST = {
    'probe_permit': dict(path=str(scope.PACKAGE/'tmark-probe-permit.json'),sha256='6049df2dab0249ae0aef9af31b4e348afb44fb98ec3c54c667582d3c00d1a8ce'),
    'complete': dict(path=str(scope.PACKAGE/'tmark-probe-execution-v1/complete.json'),sha256='cd4f33c14fbeca0584a1918ed2eb4a357ac2944e70d3fc8e0a499bd88bbeacdf'),
    'admission': dict(path=str(scope.PACKAGE/'tmark-technical-admission.json'),sha256='102c286cb4bbf99391f9916fd52b8a4f376214d2fcc8e1139891aed2bbfd89d4'),
    'formal_permit': dict(path=str(scope.PACKAGE/'tmark-formal-permit.json'),sha256='3f22c2b25ff252a56a0e3d5c216434843b339425bb74cb0aaff21a29bb05643c')}
# Filled by preparation from exclusive, actual evidence; never current self-signing.
PREPARED = {
    'inventory': dict(path=str(PACKAGE/'recovery-inventory.json'),sha256='697b3f8ae5f46a948035d1e4f90acb15b8ac857ec723836499d8149e1f39d50a'),
    'stop_after': dict(path=str(PACKAGE/'stop-after.json'),sha256='2fc353c4f8bf01b96d2de5dd9bb5597042faca16754b1ed0fcaa48f36d975b76'),
    'effective_map': dict(path=str(PACKAGE/'effective-result-map.json'),sha256='622814bb9f4b214435aa2f6308eb398a671657d8a95229d9d3ffb2dc99ead098'),
    'manifest': dict(path=str(PACKAGE/'probe-artifact-manifest.json'),sha256='5df6457076ec2369d00821208989f76f3605cd9ecfffc4ee6e6993cd5f2a9b43'),
    'process_manifest': dict(path=str(PACKAGE/'probe-process-manifest.json'),sha256='8cb4b2e3922d70e84eb8e8ea60ebb576388c04b43792622e642b621e3c2a1184'),
    'summary': dict(path=str(PACKAGE/'admission-summary.json'),sha256='bfeec24868bf9560fa73392fba57b216f724ffb2b75e2bbbe7f10c5f5ea3ee9f')}
PROOF = dict(path=str(PACKAGE/'science-inheritance.review-candidate4.json'), sha256='8ba2e979f756a0f61568737840c2f096fd4877c8ae7291ad73fd17d663654af0')
OLD_RETIREMENT = dict(path=str(scope.PACKAGE/'old-ms-retirement-boundary.json'),
                     sha256='19ba70adb4b344cbce2e740bf873d5ad4170e32f02def659f976391f124836ae')
_ACTIVE = contextvars.ContextVar('native_recovery_execution_context', default=None)


def original_configs():
    from utils.ch3_contract import read_profiles, validate_manifest
    return {k:validate_manifest(read_profiles(p)) for k,p in (
        ('tmark',scope.PROFILE_FILE),('m',ROOT/'configs/ch3_formal_profiles.json'))}


def active_context(c):
    active = _ACTIVE.get()
    return active if active and active['protocol_sha']==digest(c) else None


@contextlib.contextmanager
def activate(stage):
    c=original_configs()[stage]
    # Get the historical context before installing the new one.
    token=_ACTIVE.set(None)
    try: ctx=copy.deepcopy(scope.context(c))
    finally: _ACTIVE.reset(token)
    ctx.update(id=ID, probe_scope=ID+'-'+stage+'-probe', formal_scope=ID+'-'+stage+'-formal',
               protocol_sha=digest(c), result_root=TMARK_RESULT if stage=='tmark' else M_RESULT,
               control=CONTROL/stage, probe_root=scope.PACKAGE/'tmark-probe-execution-v1' if stage=='tmark' else PACKAGE/'m-probe-execution-v1',
               fixture=PACKAGE/'fixtures', stage=stage)
    token=_ACTIVE.set(ctx)
    try: yield c
    finally: _ACTIVE.reset(token)


def prepared(name):
    if name not in PREPARED: raise ValueError('closure-bound recovery preparation absent: '+name)
    return records.bound(PREPARED[name])


def stop_check(root=None, drain=False):
    root=CONTROL if root is None else root
    if any((Path(root)/n).exists() for n in ('STOP','failure.json')):
        raise InterruptedError('persistent recovery STOP/failure')
    if drain and (Path(root)/'DRAIN_STOP').exists():
        raise DrainStop('planned recovery drain')


class DrainStop(InterruptedError): pass


def policy_ast_sha(path=None):
    tree=ast.parse(Path(path or __file__).read_text())
    found=False
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='PROOF' for x in node.targets):
            for kw in node.value.keywords:
                if kw.arg=='sha256':kw.value=ast.Constant('reviewed-science-proof-reference');found=True
    if not found:raise ValueError('exact normalized proof literal absent')
    return digest(ast.dump(tree,include_attributes=False))


def controller_binding():
    from ch3_runner import git, code_binding
    head=git('rev-parse','HEAD')
    if git('branch','--show-current')!='m6/m-baselines-v1' or head==BASE or git('rev-parse','HEAD^')!=BASE or git('status','--porcelain','--untracked-files=all'):
        raise ValueError('reviewed direct-successor clean recovery closure required')
    proof=records.bound(PROOF);code=code_binding()
    if proof['science_baseline_commit']!=BASE or proof['policy_ast_sha']!=policy_ast_sha():
        raise ValueError('reviewed recovery policy bytes changed')
    if {k:v for k,v in code.items() if k!='utils/ch3_native_recovery.py'}!=proof['after_code']:
        raise ValueError('reviewed recovery source bytes changed')
    for name,expected in proof['protected_bytes'].items():
        if records.sha(ROOT/name)!=expected:raise ValueError('science protected byte changed: '+name)
    for path,expected in proof['author_sources'].items():
        if records.sha(path)!=expected:raise ValueError('approved author source byte changed')
    for name, expected in code.items():
        raw=subprocess.check_output(['git','-C',str(ROOT),'show',head+':'+name])
        if __import__('hashlib').sha256(raw).hexdigest()!=expected:raise ValueError('source not closed')
    return dict(controller_execution_commit=head,worker_execution_commit=head,
                science_baseline_commit=BASE,science_computation_fingerprint=proof['science_computation_fingerprint'],code=code)


def dynamic_binding(c, reference):
    from ch3_runner import git, environment_binding, hardware_binding,code_binding
    if git('rev-parse','HEAD')!=reference['controller_execution_commit'] or git('status','--porcelain','--untracked-files=all'):
        raise ValueError('dynamic controller/source identity')
    if git('branch','--show-current')!='m6/m-baselines-v1':raise ValueError('recovery branch changed')
    if code_binding()!=reference['code']:raise ValueError('dynamic closed source bytes changed')
    if environment_binding()!=reference['environment'] or hardware_binding()!=reference['hardware']:
        raise ValueError('dynamic environment/hardware changed')
    from utils.ch3_m6 import source_states
    if source_states(c)!=reference['source_states']:raise ValueError('source/data state changed')
    for src in c['sources'].values():
        if subprocess.check_output(['git','-C',src['repository'],'rev-parse','HEAD'],text=True).strip()!=src['commit'] or any(records.sha(p)!=h for p,h in src['files'].items()):
            raise ValueError('author source binding changed')


def validate_summary_light(c, summary):
    stage='tmark' if 'native_replacement' in c else 'm'
    if summary.get('purpose')!='native_recovery_admission_summary_v1' or summary.get('scope')!=ID or summary.get('protocol_sha')!=digest(c):
        raise ValueError('compact admission scope/protocol')
    if summary.get('technical_admission')is not True or summary.get('manual_review')is not False or summary.get('reviewed')is not False or summary.get('result_review')!='pending' or summary.get('review_mode')!=records.MODE:
        raise ValueError('exact machine technical admission; no scientific review')
    if stage=='tmark' and (summary['original_complete_ref']!=TRUST['complete'] or summary['original_admission_ref']!=TRUST['admission'] or summary['original_probe_permit_ref']!=TRUST['probe_permit']):
        raise ValueError('original v4 trust chain mismatch')
    groups=scope.probe_groups(c)
    if set(summary['decisions'])!={g['id'] for g in groups}:raise ValueError('exact compact decision coverage')
    for g in groups:
        d=summary['decisions'][g['id']]
        if d['status']!='Passed' or d['concurrency'] not in (1,2,4) or d['concurrency']>g['planned_q'] or d['coverage']!=g.get('coverage'):
            raise ValueError('unapproved compact concurrency')
    return summary


def validate_retirement_light():
    b=records.bound(OLD_RETIREMENT)
    if b.get('kind')!='old_MS_user_authorized_retirement_boundary_v1' or b.get('technical_complete')is not False or b.get('retirement_accepted')is not True or b.get('scientific_failure')is not False or b.get('budget_refund')is not False or b.get('result_review')!='pending':
        raise ValueError('exact original user-authorized retirement boundary')
    return b


def validate_boundary_light(value):
    b=records.bound(value)
    if b.get('purpose')!='native_recovery_replacement_boundary_v1' or b.get('scope')!=ID or b.get('technical_complete')is not True or b.get('result_review')!='pending':
        raise ValueError('exact sealed replacement boundary')
    c=original_configs()['tmark']
    if b['task_ids']!=[t['id'] for t in c['tasks']] or b['protocol_sha']!=digest(c) or b['science_baseline_commit']!=BASE:
        raise ValueError('sealed 87-task science identity')
    if b.get('old_retirement_boundary_ref')!=OLD_RETIREMENT:
        raise ValueError('sealed boundary original retirement lineage')
    return b


def validate_permit_light(c, a, probe=False, worker=False):
    stage='tmark' if 'native_replacement' in c else 'm';stop_check()
    if probe and stage!='m':raise PermissionError('recovery never reruns tmark probe')
    expected=dict(purpose='ch3_resource_probe' if probe else 'ch3_formal',recovery_scope=ID,
                  successor_scope=ID+'-'+stage+('-probe' if probe else '-formal'),reviewed=False,
                  manual_review=False,review_mode=records.MODE,execution_permitted=True,
                  task='MS' if stage=='tmark' else 'M',metric_scope='target_only' if stage=='tmark' else 'all_channels',
                  science_baseline_commit=BASE,seed_list=[2024],additional_search=0,
                  protocol_sha=digest(c),authorized_task_ids=[t['id'] for t in c['tasks']],
                  profile_shas={t['id']:digest(profile(c,t)) for t in c['tasks']})
    ctx=scope.context(c)
    expected.update(run_budget=dict(runs=ctx['runs'],run_epochs=ctx['epochs']),max_optimizer_steps=ctx['optimizer_steps'],
                    numeric_policy_shas=policy_shas(c),structure_frozen=True,m6_authorized=True,
                    technical_admission=not probe,result_review='pending')
    if any(a.get(k)!=v for k,v in expected.items()):raise ValueError('exact compact recovery permit')
    if a['commit']!=a['controller_execution_commit'] or a['worker_execution_commit']!=a['controller_execution_commit']:
        raise ValueError('actual execution commit fields')
    if a.get('old_retirement_boundary_ref')!=OLD_RETIREMENT:
        raise ValueError('fixed original retirement reference required')
    if stage=='m':validate_boundary_light(a['replacement_boundary_ref'])
    if not probe:
        if stage=='tmark' and (a['technical_admission_ref']!=PREPARED['summary'] or a['probe_manifest_ref']!=PREPARED['manifest']):raise ValueError('fixed projected original admission/manifest required')
        if stage=='tmark' and a.get('probe_metadata_manifest_ref')!=PREPARED['process_manifest']:raise ValueError('fixed resource/approval projection required')
        summary=validate_summary_light(c,records.bound(a['technical_admission_ref']))
        if a['compact_decisions']!=summary['decisions'] or a['probe_manifest_ref']!=summary['probe_manifest_ref']:
            raise ValueError('formal compact decision/source mismatch')
    elif a['caps']!=scope.context(c)['caps'] or a['already_charged']!=dict.fromkeys(records.KINDS,0):
        raise ValueError('independent unchanged M probe cap/debit')
    dynamic_binding(c,a)
    from utils.ch3_native_chain import PREPARATION
    data=records.bound(PREPARATION['data_'+stage])
    if a['data_binding_ref']!=PREPARATION['data_'+stage] or a['data_bindings']!=data['data_bindings'] or a['source_states']!=data['source_states']:raise ValueError('permit accepted data lineage')
    authorization=records.bound(a['start_authorization_ref'])
    if authorization.get('reviewed')is not True or authorization.get('execution_permitted')is not True or authorization.get('scope')!=ID or authorization.get('controller_execution_commit')!=a['controller_execution_commit'] or authorization.get('science_proof_ref')!=PROOF:raise ValueError('actual reviewed repair start grant')
    if stage=='tmark':
        if a['inventory_ref']!=PREPARED['inventory'] or a['effective_map_ref']!=PREPARED['effective_map'] or a['authorized_execution_caps']!=authorization['authorized_execution_caps']:raise ValueError('exact recovery inventory/budget grant')
    return a


def validate_runtime_light(c, reference, permit_ref, require_owner=True):
    r=records.bound(reference)
    if r.get('purpose')!='native_recovery_runtime_admission_v1' or r.get('scope')!=ID or r['formal_permit_ref']!=permit_ref or r.get('integrity_scan_passed')is not True:
        raise ValueError('runtime admission exact chain identity')
    if r['stage']!=('tmark' if 'native_replacement' in c else 'm') or r['protocol_sha']!=digest(c):raise ValueError('runtime stage/protocol')
    expected_path=PACKAGE/'lifecycles'/r['lifecycle']/r['stage']/'runtime-admission.json'
    if reference['path']!=str(expected_path) or r['launch_identity']!=r['lifecycle']:raise ValueError('exact lifecycle runtime path')
    secret=os.environ.get(RUNTIME_SECRET_ENV)
    body={k:v for k,v in r.items() if k!='owned_chain_mac'}
    if not secret or not hmac.compare_digest(r.get('owned_chain_mac',''),hmac.new(secret.encode(),digest(body).encode(),hashlib.sha256).hexdigest()):raise PermissionError('runtime admission not authenticated by this owned chain')
    if require_owner:
        from utils.ch3_m_launch import ancestors, command_has
        chain={v['pid']:v for v in ancestors(os.getpid())}
        if not same(r['supervisor']) or chain.get(r['supervisor']['pid'],{}).get('start_ticks')!=r['supervisor']['start_ticks'] or not command_has(r['supervisor']['pid'],ROOT/'m6_native_recovery_entry.py'):
            raise ValueError('runtime record not owned by current controlled launch chain')
    dynamic_binding(c,r);stop_check()
    return r


def seal_runtime(c, permit_ref, summary_ref, manifest_ref, lifecycle, owner):
    """Exactly one scan per stage/lifecycle, including across group subprocesses."""
    stage='tmark' if 'native_replacement' in c else 'm';path=PACKAGE/'lifecycles'/lifecycle/stage/'runtime-admission.json'
    if path.exists():raise FileExistsError('retained lifecycle runtime record; no overwrite/reseal')
    a=validate_permit_light(c,records.bound(permit_ref));summary=validate_summary_light(c,records.bound(summary_ref))
    validate_retirement_light()
    manifest=records.bound(manifest_ref)
    if a.get('probe_metadata_manifest_ref'):
        manifest=records.merge_manifests([manifest,records.bound(a['probe_metadata_manifest_ref'])])
    probe_root=scope.PACKAGE/'tmark-probe-execution-v1' if stage=='tmark' else PACKAGE/'m-probe-execution-v1'
    scan=records.scan_manifest(manifest,summary['original_complete_ref'],probe_root)
    stop_check(drain=True)
    r=dict(purpose='native_recovery_runtime_admission_v1',scope=ID,stage=stage,lifecycle=lifecycle,
           supervisor=owner,launch_identity=lifecycle,created_at=time.time(),formal_permit_ref=permit_ref,
           technical_admission_ref=summary_ref,probe_manifest_ref=manifest_ref,
           old_retirement_boundary_ref=OLD_RETIREMENT,
           probe_metadata_manifest_ref=a.get('probe_metadata_manifest_ref'),
           inventory_ref=a.get('inventory_ref'),effective_map_ref=a.get('effective_map_ref'),
           integrity_scan_passed=True,scan_count=1,scan=scan,result_review='pending',
           **{k:a[k] for k in ('protocol_sha','controller_execution_commit','worker_execution_commit','science_baseline_commit','science_computation_fingerprint','environment','hardware','source_states','code')})
    secret=os.environ.get(RUNTIME_SECRET_ENV)
    if not secret:raise PermissionError('actual controlled runtime secret absent')
    r['owned_chain_mac']=hmac.new(secret.encode(),digest(r).encode(),hashlib.sha256).hexdigest()
    return records.exclusive(path,r)


def worker_permit(s):
    a=records.bound(s['formal_permit_ref'])
    if a.get('recovery_scope')!=ID:raise ValueError('worker foreign recovery permit')
    return a


def metadata_files(c,a,runtime_ref=None,resume_row=None):
    refs=[a.get('technical_admission_ref'),a.get('probe_manifest_ref'),a.get('probe_metadata_manifest_ref'),a.get('replacement_boundary_ref'),
          a.get('inventory_ref'),a.get('effective_map_ref'),a.get('start_authorization_ref'),a.get('data_binding_ref'),runtime_ref]
    result={r['path']:r['sha256'] for r in refs if r}
    # These are compact summaries only. Never follow original full-report refs.
    if resume_row:
        for name in ('manifest.json','last.pt','best.pt','history.jsonl'):
            r=resume_row['files'][name];result[r['path']]=r['sha256']
    return result


def approval_template():
    return dict(purpose='native_recovery_start_authorization_v1',scope=ID,
                reviewed=False,execution_permitted=False,science_baseline_commit=BASE,
                controller_execution_commit=None,original_formal_optimizer_max=3687530,
                authorized_execution_caps=None,additional_search=0,
                authorization_basis='user-authorized execution repair; actual byte review and budget required')


def readiness(approval=None,launch=None):
    reasons=[]
    try:
        c=original_configs()['tmark'];inv=records.validate_inventory(c,prepared('inventory'))
        budget=records.budget_projection(c,inv)
        cap=(approval or {}).get('authorized_execution_caps') or dict(adam=3687530,backward=3687530,forward=None)
        if any(cap.get(k)is not None and budget['total_execution_worst_case'][k]>cap[k] for k in records.KINDS):
            reasons.append('additional execution budget authorization required')
        if any(r['classification']=='E' for r in inv['rows']):reasons.append('inventory E')
        for r in inv['rows']:
            for v in r['files'].values():
                if records.sha(v['path'])!=v['sha256']:raise ValueError('original inventory artifact changed')
        stop=prepared('stop_after')
        if not stop['all_owned_exited'] or any(same(r) for r in stop['owned_instances']):raise ValueError('original owned instance alive')
        for value in TRUST.values():
            if records.sha(value['path'])!=value['sha256']:raise ValueError('fixed original trust reference changed')
        validate_retirement_light()
        summary=validate_summary_light(c,prepared('summary'))
        if summary['probe_manifest_ref']!=PREPARED['manifest']:raise ValueError('bound projected manifest reference')
        mapping=records.effective_map(c,inv,TMARK_RESULT)
        if mapping!=prepared('effective_map'):raise ValueError('fixed effective-result selection map')
        if any(Path(r['effective_root']).exists() for r in mapping['rows'].values() if r['classification']!='A') or M_RESULT.exists():raise ValueError('retained new result namespace')
    except (OSError,ValueError,KeyError,TypeError) as exc:reasons.append(str(exc))
    try:
        binding=controller_binding()
        if approval:
            validate_start_authorization(approval,binding)
    except (OSError,ValueError,KeyError,TypeError,subprocess.CalledProcessError) as exc:reasons.append(str(exc))
    try:
        data_binding(original_configs()['tmark']);data_binding(original_configs()['m'])
        resource_check(original_configs()['tmark'])
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,subprocess.CalledProcessError) as exc:reasons.append(str(exc))
    if not approval or approval.get('purpose')!='native_recovery_start_authorization_v1' or approval.get('scope')!=ID or approval.get('reviewed')is not True or approval.get('execution_permitted')is not True:
        reasons.append('reviewed recovery start authorization absent')
    if CONTROL.exists() or CONTROL.is_symlink() or ((LOG.exists() or LOG.is_symlink()) and not launch):reasons.append('retained recovery execution/launcher; audited resume required')
    if subprocess.run(['tmux','has-session','-t',SESSION],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0 and not launch:reasons.append('retained recovery tmux session')
    return list(dict.fromkeys(reasons))


def safe_stop(drain=False):
    if not (CONTROL/'controller.json').exists():raise ValueError('no actual owned recovery supervisor')
    write_marker(CONTROL/('DRAIN_STOP' if drain else 'STOP'))
    if drain:return dict(DRAIN_STOP=True,signal_sent=False,resume_eligible=True,unconditional_resume=False)
    controller=json.loads((CONTROL/'controller.json').read_text()) if (CONTROL/'controller.json').exists() else None
    child=json.loads((CONTROL/'current.json').read_text()).get('child') if (CONTROL/'current.json').exists() else None
    if child and controller and child.get('owner')==controller and same(child):
        if str(ROOT/'m6_native_recovery_entry.py') not in child.get('command',[]):raise ValueError('foreign child command')
        fields=(Path('/proc')/str(child['pid'])/'stat').read_text().rsplit(')',1)[1].split()
        if int(fields[1])!=controller['pid']:raise ValueError('owned child parent changed')
        signal_owned(child)
    if controller and same(controller):
        from utils.ch3_m_launch import command_has
        if not command_has(controller['pid'],ROOT/'m6_native_recovery_entry.py'):raise ValueError('foreign supervisor command')
    sent=signal_owned(controller) if controller else False
    return dict(STOP=True,signal_sent=sent,old_v4_signal_sent=False)


def status():
    if not (CONTROL/'controller.json').exists():return dict(scope=ID,running=False,state='Prepared',result_review='pending')
    controller=json.loads((CONTROL/'controller.json').read_text())
    progress=json.loads((CONTROL/'progress.json').read_text()) if (CONTROL/'progress.json').exists() else {}
    return dict(scope=ID,running=same(controller),controller=controller,progress=progress,
                STOP=(CONTROL/'STOP').exists(),DRAIN_STOP=(CONTROL/'DRAIN_STOP').exists(),
                failure=(CONTROL/'failure.json').exists(),complete=(CONTROL/'complete.json').exists())


@contextlib.contextmanager
def dispatch_guard():
    # A durable DRAIN before this lock prevents dispatch; once this check
    # succeeds the whole fixed wave is owned and may drain normally.
    with lock(PACKAGE/'.dispatch.lock'):
        stop_check(drain=True)
        yield


def write_marker(path):
    path=Path(path)
    if path.is_symlink():raise ValueError('marker symlink')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600) if not path.exists() else None
    if fd is not None:
        try:os.fsync(fd)
        finally:os.close(fd)


def resource_check(c):
    from ch3_runner import GPULock
    from utils.ch3_m_execution import gpu_environment_reasons
    reasons=gpu_environment_reasons()
    if reasons:raise ValueError('; '.join(reasons))
    with GPULock(c):pass
    import sys
    sys.path.insert(0,str(ROOT/'tools/restricted_regression'))
    from m5_formal_entry import gpu_sample,resource_assessment
    if not resource_assessment(gpu_sample([]),[])['admission']:raise ValueError('current GPU resource/headroom unavailable')


def data_binding(c):
    from utils.ch3_native_chain import PREPARATION
    from utils.ch3_m6 import source_states
    from ch3_runner import environment_binding,hardware_binding
    stage='tmark' if 'native_replacement' in c else 'm'
    value=PREPARATION['data_'+stage];data=records.bound(value)
    prior=records.bound(PREPARATION['environment_source'])
    if c['native_time_mark']['data_binding_artifact']!=value or data['source_states']!=source_states(c):raise ValueError('accepted data lineage/file state changed')
    expected={name:{t['id'] for t in c['tasks'] if t['dataset']==name} for name in c['datasets']}
    if {name:set(rows) for name,rows in data['data_bindings'].items()}!=expected or {name:set(rows) for name,rows in data['metadata'].items()}!=expected:
        raise ValueError('accepted exact task data metadata coverage')
    for name,rows in data['metadata'].items():
        if any(digest(metadata)!=data['data_bindings'][name][run] for run,metadata in rows.items()):
            raise ValueError('accepted per-task data metadata digest')
    if prior['environment']!=environment_binding() or prior['hardware']!=hardware_binding():raise ValueError('unchanged approved environment/hardware required')
    return dict(data_bindings=data['data_bindings'],data_binding_ref=value,source_states=data['source_states'],environment=prior['environment'],hardware=prior['hardware'])


def validate_start_authorization(a,binding=None):
    binding=binding or controller_binding()
    required=dict(purpose='native_recovery_start_authorization_v1',scope=ID,reviewed=True,execution_permitted=True,
                  science_baseline_commit=BASE,controller_execution_commit=binding['controller_execution_commit'],
                  original_formal_optimizer_max=3687530,additional_search=0,
                  inventory_ref=PREPARED['inventory'],effective_map_ref=PREPARED['effective_map'],
                  science_proof_ref=PROOF,probe_summary_ref=PREPARED['summary'],probe_manifest_ref=PREPARED['manifest'])
    if any(a.get(k)!=v for k,v in required.items()):raise ValueError('exact reviewed recovery authorization required')
    caps=a.get('authorized_execution_caps');projection=records.budget_projection(original_configs()['tmark'],prepared('inventory'))
    if not caps or set(caps)!=set(records.KINDS) or any(type(caps[k])is not int or caps[k]<projection['total_execution_worst_case'][k] for k in records.KINDS):raise ValueError('additional execution budget authorization required')
    if any(caps[k]>projection['original_formal_max'][k] for k in caps) and not a.get('additional_budget_authorization'):
        raise ValueError('explicit additional execution budget authorization ref required')
    if a.get('additional_budget_authorization'):
        extra=records.bound(a['additional_budget_authorization'])
        if extra.get('scope')!=ID or extra.get('reviewed')is not True or extra.get('authorized_execution_caps')!=caps or extra.get('inventory_ref')!=PREPARED['inventory']:raise ValueError('exact user additional budget grant')
    return binding


def create_permit(c,start_ref,probe=False,summary_ref=None,boundary_ref=None):
    stop_check(drain=True);authorization=records.bound(start_ref);binding=validate_start_authorization(authorization)
    stage=scope.context(c)['stage'];data=data_binding(c);ctx=scope.context(c)
    if probe and stage!='m':raise PermissionError('historical tmark probe cannot be rerun')
    path=PACKAGE/(stage+('-probe-permit.json' if probe else '-formal-permit.json'))
    if path.exists() or path.is_symlink():raise FileExistsError('one-shot actual recovery permit')
    old=records.bound(TRUST['formal_permit'])['old_boundary']
    if old!=OLD_RETIREMENT:raise ValueError('original fixed permit retirement lineage')
    validate_retirement_light()
    a=dict(**binding,**data,commit=binding['worker_execution_commit'],recovery_scope=ID,
           purpose='ch3_resource_probe' if probe else 'ch3_formal',successor_scope=ctx['probe_scope'] if probe else ctx['formal_scope'],
           reviewed=False,manual_review=False,review_mode=records.MODE,execution_permitted=True,
           authorization_basis='user-authorized execution repair; unchanged science; preauthorized technical gates',
           start_authorization_ref=start_ref,old_retirement_boundary_ref=old,protocol_sha=digest(c),
           seed_list=[2024],additional_search=0,authorized_task_ids=[t['id'] for t in c['tasks']],
           profile_shas={t['id']:digest(profile(c,t)) for t in c['tasks']},
           numeric_policy_shas=policy_shas(c),
           task='MS' if stage=='tmark' else 'M',metric_scope='target_only' if stage=='tmark' else 'all_channels',
           run_budget=dict(runs=ctx['runs'],run_epochs=ctx['epochs']),max_optimizer_steps=ctx['optimizer_steps'],
           structure_frozen=True,m6_authorized=True,from_scratch=stage=='m',result_review='pending',
           caps=ctx['caps'] if probe else None,already_charged=dict.fromkeys(records.KINDS,0),
           technical_admission=not probe,replacement_boundary_ref=boundary_ref)
    if stage=='tmark':
        a.update(inventory_ref=PREPARED['inventory'],effective_map_ref=PREPARED['effective_map'],
                 probe_metadata_manifest_ref=PREPARED['process_manifest'],
                 authorized_execution_caps=authorization['authorized_execution_caps'])
    if not probe:
        summary=validate_summary_light(c,records.bound(summary_ref))
        a.update(technical_admission_ref=summary_ref,probe_manifest_ref=summary['probe_manifest_ref'],
                 compact_decisions=summary['decisions'],probe_budget=summary['budget'])
    validate_permit_light(c,a,probe);resource_check(c);stop_check(drain=True)
    return records.exclusive(path,a)


def prepare_launch(approval_ref,wrapper_pid):
    from utils.ch3_m_launch import file_identity,ancestors
    if not same(identity(wrapper_pid)) or wrapper_pid not in [v['pid'] for v in ancestors(os.getpid())]:raise PermissionError('actual current wrapper ancestor required')
    args=(Path('/proc')/str(wrapper_pid)/'cmdline').read_bytes().decode().split('\0')
    if str(ROOT/'scripts/ch3/start_native_time_mark_recovery.sh') not in args or 'start' not in args:raise PermissionError('exact recovery wrapper start required')
    if CONTROL.exists() or LOG.stat().st_size!=0 or LOG.is_symlink():raise ValueError('exact newly created launcher required')
    token=secrets.token_hex(32)
    records.exclusive(str(LOG)+'.launch.json',dict(scope=ID,log=str(LOG),log_identity=file_identity(LOG),
         approval_ref=approval_ref,wrapper=identity(wrapper_pid),token_sha=__import__('hashlib').sha256(token.encode()).hexdigest()))
    return token


def verify_launch(approval_ref,token):
    from utils.ch3_m_launch import file_identity,tmux_view
    if not isinstance(token,str) or len(token)!=64:raise PermissionError('single-use wrapper launch token required')
    launch=json.loads(Path(str(LOG)+'.launch.json').read_text())
    if launch.get('scope')!=ID or launch['approval_ref']!=approval_ref or launch['log']!=str(LOG) or launch['log_identity']!=file_identity(LOG) or not secrets.compare_digest(launch['token_sha'],__import__('hashlib').sha256(token.encode()).hexdigest()):raise ValueError('current launcher identity mismatch')
    if Path(str(LOG)+'.claimed.json').exists():raise FileExistsError('launcher already claimed')
    launch['tmux']=tmux_view(SESSION)
    return launch


def wait_owned(args,label):
    """One synchronous tracked child. Child entries own their guarded workers."""
    from ch3_runner import dump
    stop_check(drain=True);owner=identity(os.getpid());child=None;owned=None
    logpath=CONTROL/(label+'.log')
    try:
        with logpath.open('x') as output:
            with dispatch_guard():
                child=subprocess.Popen(args,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
                v=identity(child.pid)
                if not v:child.wait();raise RuntimeError('child ended before owned registration')
                owned=dict(v,owner=owner,command=args,kind=label)
                dump(CONTROL/'current.json',dict(child=owned))
            while child.poll()is None:
                if (CONTROL/'STOP').exists():signal_owned(owned)
                time.sleep(.1)
        if child.returncode==3 and (CONTROL/'DRAIN_STOP').exists():raise DrainStop('owned child drained')
        stop_check(drain=True)
        if child.returncode!=0 or same(owned):raise RuntimeError('owned '+label+' failed: '+str(child.returncode))
        return dict(child=owned,exit_code=child.returncode,log=str(logpath),result_review='pending')
    finally:
        if child is not None and child.poll()is None:
            signal_owned(owned)
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:raise RuntimeError('owned child cleanup timeout; retained evidence')


def child_command(stage,permit_ref,probe=False,model=None,runtime_ref=None):
    c=original_configs()[stage]
    args=[c['execution']['python'],'-B',str(ROOT/'m6_native_recovery_entry.py'),'probe-child' if probe else 'group-child',
          '--stage',stage,'--approval',permit_ref['path'],'--approval-sha',permit_ref['sha256']]
    if model:args+=['--model',model,'--runtime',runtime_ref['path'],'--runtime-sha',runtime_ref['sha256']]
    return args


def validate_child_parent(permit_ref):
    controller=json.loads((CONTROL/'controller.json').read_text());current=json.loads((CONTROL/'current.json').read_text())['child']
    v=identity(os.getpid())
    if not same(controller) or controller['pid']!=os.getppid() or current['pid']!=v['pid'] or current['start_ticks']!=v['start_ticks'] or current['owner']!=controller:
        raise PermissionError('exact live tracked recovery child/parent required')
    if permit_ref['path'] not in current['command'] or permit_ref['sha256'] not in current['command']:raise PermissionError('owned child exact permit reference')


def run_formal(stage,permit_ref,summary_ref,lifecycle,owner):
    from utils import ch3_native_recovery_execution as workers
    with activate(stage) as c:
        a=records.bound(permit_ref);runtime_ref=seal_runtime(c,permit_ref,summary_ref,a['probe_manifest_ref'],lifecycle,owner)
        for model in scope.context(c)['models']:
            stop_check(drain=True);resource_check(c)
            receipt=wait_owned(child_command(stage,permit_ref,model=model,runtime_ref=runtime_ref),stage+'-'+model)
            group=scope.context(c)['control']/model
            complete=records.bound(records.ref(group/'complete.json'))
            expected=[t['id'] for t in c['tasks'] if t['model']==model]
            if complete.get('task_ids')!=expected or complete.get('formal_permit_ref')!=permit_ref or complete.get('runtime_admission_ref')!=runtime_ref or complete.get('protocol_sha')!=digest(c) or complete.get('technical_complete')is not True or complete.get('result_review')!='pending' or same(json.loads((group/'controller.json').read_text())):
                raise ValueError('group exited but exact technical handoff absent')
        return runtime_ref


def audit_m(permit_ref):
    """Only M_AUTO_AUDIT calls the final saved-evidence audit, once."""
    from utils import ch3_native_execution as execution
    stop_check(drain=True)
    with activate('m') as c:
        root=scope.context(c)['probe_root'];raw_ref=records.ref(root/'complete.json');raw=records.bound(raw_ref)
        controller=json.loads((root/'controller.json').read_text())
        if same(controller) or controller['approval']!=permit_ref:raise ValueError('M probe child not exited / wrong permit')
        if (PACKAGE/'m-admission-summary.json').exists():raise FileExistsError('retained M auto audit; no replay')
        execution.validate_probe_completion(c,raw)
        manifest_ref=records.exclusive(PACKAGE/'m-probe-artifact-manifest.json',records.manifest_projection(raw,raw_ref,root,records.process_refs(raw)))
        admission_ref=records.exclusive(PACKAGE/'m-technical-admission.json',dict(purpose='native_recovery_M_machine_admission_v1',probe_complete_ref=raw_ref,probe_permit_ref=permit_ref,
            decisions=records.compact_decisions(raw),budget=raw['budget'],technical_admission=True,reviewed=False,manual_review=False,result_review='pending'))
        summary=records.project_summary(raw,raw_ref,admission_ref,permit_ref);summary['probe_manifest_ref']=manifest_ref
        stop_check(drain=True)
        return records.exclusive(PACKAGE/'m-admission-summary.json',summary)


def drive(actions,update,check):
    """Ordered state transitions shared by production and bounded fixtures."""
    result={}
    for state in STATES[:-1]:
        check();update(state,result);result[state]=actions[state](result);check()
    check();update('COMPLETE',result)
    return result


def run(approval_ref,lifecycle):
    from ch3_runner import dump
    from utils import ch3_native_recovery_execution as workers
    owner=identity(os.getpid());state=STATES[0];receipts={};start=records.bound(approval_ref)
    def update(value,receipt):
        nonlocal state
        state=value;dump(CONTROL/'progress.json',dict(scope=ID,state=state,receipts=receipt,result_review='pending'))
    def verify_inventory(receipts):
        inv=records.validate_inventory(original_configs()['tmark'],prepared('inventory'))
        validate_start_authorization(start)
        return PREPARED['inventory']
    def permit(receipts):
        with activate('tmark') as c:return create_permit(c,approval_ref,summary_ref=PREPARED['summary'])
    def close(receipts):
        with activate('tmark') as c:return workers.closeout(c,records.bound(receipts['SEAL_RUNTIME_ADMISSION']),PREPARED['effective_map'],lifecycle)
    def m_probe(receipts):
        with activate('m') as c:
            p=create_permit(c,approval_ref,True,boundary_ref=receipts['SEAL_87_REPLACEMENT_BOUNDARY'])
        return dict(permit=p,execution=wait_owned(child_command('m',p,probe=True),'m-probe'))
    def m_formal(receipts):
        with activate('m') as c:p=create_permit(c,approval_ref,summary_ref=receipts['M_AUTO_AUDIT'],boundary_ref=receipts['SEAL_87_REPLACEMENT_BOUNDARY'])
        return run_formal('m',p,receipts['M_AUTO_AUDIT'],lifecycle,owner)
    actions={STATES[0]:lambda _:PREPARED['summary'],STATES[1]:verify_inventory,STATES[2]:permit,
             STATES[3]:lambda r:run_formal('tmark',r['SEAL_RUNTIME_ADMISSION'],PREPARED['summary'],lifecycle,owner),
             STATES[4]:close,STATES[5]:m_probe,STATES[6]:lambda r:audit_m(r['M_PROBE']['permit']),STATES[7]:m_formal}
    def stopped(sig,frame):
        write_marker(CONTROL/'STOP')
        if (CONTROL/'current.json').exists():
            child=json.loads((CONTROL/'current.json').read_text()).get('child')
            if child and child.get('owner')==owner:signal_owned(child)
        raise InterruptedError('user-authorized owned recovery safe-stop')
    previous=signal.signal(signal.SIGTERM,stopped)
    try:
        receipts=drive(actions,update,lambda:stop_check(drain=True))
        with dispatch_guard():
            records.exclusive(CONTROL/'complete.json',dict(scope=ID,technical_complete=True,result_review='pending',
                 science_baseline_commit=BASE,controller_execution_commit=start['controller_execution_commit'],receipts=receipts))
    except DrainStop:
        records.exclusive(CONTROL/'drain-boundary.json',dict(scope=ID,state='DRAINED',last_state=state,technical_failure=False,
             resume_eligible=True,unconditional_resume=False,receipts=receipts,result_review='pending'))
        update('DRAINED',receipts)
    except BaseException as exc:
        records.exclusive(CONTROL/'failure.json',dict(state=state,error=repr(exc),automatic_retry=False,scientific_failure=False,
             result_review='pending',receipts=receipts));raise
    finally:signal.signal(signal.SIGTERM,previous)


def start(approval_ref,token):
    launch=verify_launch(approval_ref,token);a=records.bound(approval_ref);reasons=readiness(a,launch)
    if reasons:raise PermissionError('; '.join(reasons))
    with lock(PACKAGE/'.supervisor.lock'):
        if CONTROL.exists():raise FileExistsError('retained recovery execution; legal resume audit required')
        lifecycle=secrets.token_hex(16)
        records.exclusive(str(LOG)+'.claimed.json',dict(launch=records.ref(str(LOG)+'.launch.json'),consumer=identity(os.getpid()),lifecycle=lifecycle))
        CONTROL.mkdir(exist_ok=False);records.exclusive(CONTROL/'controller.json',identity(os.getpid()))
        records.exclusive(CONTROL/'authorization.json',dict(approval_ref=approval_ref,lifecycle=lifecycle,controller_execution_commit=a['controller_execution_commit']))
        (PACKAGE/'fixtures').mkdir(exist_ok=True)
        os.environ[RUNTIME_SECRET_ENV]=secrets.token_hex(32)
        run(approval_ref,lifecycle)


def policy_shas(c):
    from utils.ch3_contract import numeric_probe_policy
    return {g['id']:digest(numeric_probe_policy(c,task_by_id(c,g['representatives'][0]))) for g in scope.probe_groups(c)}
