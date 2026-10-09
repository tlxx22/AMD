"""One reviewed continuation after round2-371; explicit depth and fold additions."""
import ast
import copy
import hashlib
import hmac
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from utils.ch3_contract import ROOT, digest, profile, step_arithmetic, numeric_probe_policy
from utils.ch3_native_recovery_records import bound, exclusive, ref, sha
from utils import ch3_amend_ett_identity_recovery as prefix

PACKAGE = prefix.numeric.POLICY_PACKAGE / 'patchtst-depth-urban6-repair-v1'
REVISION_PACKAGE = PACKAGE / 'patchtst-enc2-adoption-v1'
PRIOR_OBSERVATION_REF = dict(path=str(REVISION_PACKAGE / 'patch-enc1-observation-repair-v1' / 'observation-source.json'), sha256='99ee6307fe9888988447ef52ece1fe74eab913aeb746aa865db63928a084dab1')
NAMESPACE_OBSERVATION_REF = dict(path=str(REVISION_PACKAGE / 'patch-enc1-namespace-exit-repair-v1' / 'observation-source.json'), sha256='e3df66225062d89a330a3daf67f9c5a6654d1566a6ba730b5ae31832dc7fdf45')
OBSERVATION_PACKAGE = REVISION_PACKAGE / 'patch-enc1-identity-settle-repair-v1'
OBSERVATION_REF = dict(path=str(OBSERVATION_PACKAGE / 'observation-source.json'), sha256='0939eecb524ad65d71002ada47296b879f3967d16bc05a6150198df9882c0cd3')
WHOLE_CARD_PACKAGE = OBSERVATION_PACKAGE / 'whole-card-monitor-v1'
WORKER_GUARD_PACKAGE = WHOLE_CARD_PACKAGE / 'worker-guard-binding-fix-v1'
WORKER_FAILURE_REF = dict(path=str(WORKER_GUARD_PACKAGE / 'worker-guard-failure-source.json'), sha256='62b999f8deb896508cdaf3e35d2e0d1a2fa0cc8aed889e68807b66f58a8c19a1')
LEGACY_RESOURCE_CONTRACT_REF = dict(path=str(WHOLE_CARD_PACKAGE / 'resource-contract.json'), sha256='4b43a16d1fd3003ce35e4474141437635ba21a50d72f3f825c711750283b1f46')
EVENT_PACKAGE = WORKER_GUARD_PACKAGE / 'runtime-no-telemetry-v1'
RESOURCE_CONTRACT_REF = dict(path=str(EVENT_PACKAGE/'resource-contract.json'),sha256='e7f51568521cf9940b8adc287e6480a22baae197439ab2432bccab9829dbacd3')
R5_SOURCE_REF = dict(path=str(EVENT_PACKAGE/'r5-source.json'),sha256='2de463ec1e427c30657d170caaa37a8f2633ca41eee7ce46b9231299fd60e439')
OLD_RESULT = prefix.RESULT
RESULT = OLD_RESULT.with_name('baseline-unified-v3-ms-seal-m128-recovery1-patchtst-depth-urban6-r6-no-telemetry')
ATTEMPT = 'PATCHTST-depth-Urban6-exit-r6-no-telemetry'
SESSION = 'ch3-m6-patchtst-depth-urban6-r6-no-telemetry'
LOG = EVENT_PACKAGE / 'followup-launcher.log'
ENTRY = ROOT / 'm6_patchtst_depth_urban6_recovery_entry.py'
WRAPPER = ROOT / 'scripts/ch3/start_patchtst_depth_urban6_recovery.sh'
PRODUCER = 'b97392a179b55b4bffb2bf0e3b85267bb0b07750'
SOURCE_REF = dict(path=str(PACKAGE / 'source-anchors.json'), sha256='887db711ee4976b8d803a04d33f2e2545ecd8f080cf2da1ff630fcc0f6ed7914')
REUSE_REF = dict(path=str(PACKAGE / 'urban-probe-reuse.json'), sha256='1d0893f0be0a76713864abc731188a287a404a406e8a657f76c24f8d9a92a7d7')
CONTRACT_REF = dict(path=str(REVISION_PACKAGE / 'contract.json'), sha256='2cf33003a1615cf7a231a7ac9790fa60078e509af1131bdc19bf3df098b44032')
PREFIX_REF = dict(path=str(PACKAGE / 'prefix-verification.json'), sha256='8e21317dcae611a43982bf5610949d2c04548a33c369d0f1363c1433c318d1a0')
COMPLETED_PREFIX = COMPLETED_ROUND2 = True
COMPLETED_STAGES = ('M_BASE', 'M_AMEND')
SEED_STAGES = ('PATCH_ENC1','URBAN_SUBSET')
ACTIVE = False
SERIAL_CHECK = False
SERIAL_RESULT = RESULT.with_name(RESULT.name+'-serial-check')
SERIAL_LOG = EVENT_PACKAGE / 'serial-check-launcher.log'
SERIAL_SESSION = SESSION+'-serial-check'
SERIAL_ENTRY = ROOT / 'm6_patch_enc1_serial_check_entry.py'
SERIAL_TASK = 'PatchTST-ETTh1-M-round2-enc1-v1-f1-h96-s2024'
M_DATASETS = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Exchange')
MAIN_ADOPTION_STATE = 'SEAL_ROUND2_FIXED_PATCHTST_ENC2_BOUNDARY'


def fixed_encoder_policy():
    value=bound(contract()['fixed_encoder_policy_ref'])
    if (value.get('purpose')!='patchtst_M_fixed_encoder2_pre_results_v1' or value.get('chosen_encoder')!=2
        or value.get('datasets')!=list(M_DATASETS) or value.get('horizons')!=[96,192,336,720]
        or value.get('metric_based_reselection')is not False or value.get('decision_before_enc1_enc2_results')is not True
        or value.get('MS_profiles_unchanged')is not True or value.get('paper_additional_sections_required')is not False):
        raise ValueError('exact user-frozen encoder2 policy, no outcome-based reselection')
    return value


def encoder2_type1_task(prior):
    if prior['model']!='PatchTST':return copy.deepcopy(prior)
    group=prior['group']+'-enc2-v1'
    return dict(prior,group=group,id=group+'-f'+str(prior['fold'])+'-h'+str(prior['h'])+'-s2024',profile=group+'-h'+str(prior['h']),variant='encoder-2')


def contract():
    value = bound(CONTRACT_REF)
    if value['source_ref'] != SOURCE_REF or value['reuse_ref'] != REUSE_REF or value['old_config_refs'] != prefix.config_refs():
        raise ValueError('exact completed round2 and approved revision parents')
    return value


def config_refs():
    value = contract()
    return dict(value['old_config_refs'], **value['new_config_refs'])


def validate_revision(c):
    """Frozen parents plus only approved structure/fold deltas, no loose whitelist."""
    from utils import ch3_type1_tasks as s
    stage = c['baseline_unified']['stage']; spec = contract()
    if stage not in spec['new_config_refs'] or c != bound(spec['new_config_refs'][stage]):
        raise ValueError('exact new depth/Urban configuration revision')
    oldcs = {stage:bound(spec['old_config_refs'][stage])} if stage=='URBAN_SUBSET' else {}
    data = bound(c['baseline_unified']['data_ref'])
    seen = set();parents={};parent_data={}
    for t in c['tasks']:
        parent_ref = c['baseline_unified']['parent_refs'][t['id']]
        key=parent_ref['config_ref']['path']
        if key not in parents:parents[key]=bound(parent_ref['config_ref'])
        parent=parents[key]
        prior = next(x for x in parent['tasks'] if x['id'] == parent_ref['task_id'])
        before = profile(parent, prior); actual = profile(c, t)
        if parent_ref['profile_sha'] != digest(before):raise ValueError('exact parent profile digest')
        if stage.startswith('PATCH_ENC'):
            depth = int(stage[-1]); expected = copy.deepcopy(before)
            if before['structure']['e_layers'] != 3:raise ValueError('original PatchTST encoder3 parent')
            expected['structure']['e_layers'] = depth
            group = f'PatchTST-{t["dataset"]}-M-round2-enc{depth}-v1'
            expected_task = dict(prior, group=group, id=f'{group}-f1-h{t["h"]}-s2024', profile=f'{group}-h{t["h"]}', variant=f'encoder-{depth}')
            expected_parent = spec['old_config_refs']['M_BASE' if t['dataset'] in ('ETTh1','Exchange') else 'M_AMEND']
            if parent_ref['config_ref'] != expected_parent or t != expected_task or t['model'] != 'PatchTST':raise ValueError('exact enc1/2 task/parent variant identity')
            if t['dataset'] == 'Weather' and (expected['training']['epochs'],expected['training']['patience']) != (20,10):raise ValueError('Weather uses final20/10 source')
        elif stage=='M_ALL':
            if parent_ref['config_ref']!=spec['old_config_refs']['M_ALL']:raise ValueError('third-round encoder2 derives from its own type1 parent')
            expected=copy.deepcopy(before)
            if prior['model']=='PatchTST':
                if before['structure']['e_layers']!=3:raise ValueError('frozen third-round encoder3 parent')
                expected['structure']['e_layers']=2
            if t!=encoder2_type1_task(prior):raise ValueError('exact third-round variant/task identity')
        else:
            if parent_ref['config_ref'] != s.PARENT_REF or prior['dataset'] != 'UrbanEV' or prior['h'] not in (3,12) or prior['fold'] not in range(1,7):raise ValueError('six real frozen fold parents, two H only')
            expected = s.inherited_profile(prior)
            if t != s.new_task(prior):raise ValueError('original type1 task identities retained')
        if actual != expected or c['datasets'][t['dataset']]['path'] != parent['datasets'][t['dataset']]['path'] or c['sources'] != parent['sources']:raise ValueError('only approved structure or fold/profile derivations may differ')
        if c['baseline_unified']['numeric_policies'][t['model']+'-'+t['dataset']] != numeric_probe_policy(c,t):raise ValueError('uniform numeric policy remains exact')
        m = data['metadata'][t['dataset']][t['id']]
        if key not in parent_data:parent_data[key]=bound(parent['baseline_unified']['data_ref'])
        original_data=parent_data[key]
        if m != original_data['metadata'][prior['dataset']][prior['id']] or data['data_bindings'][t['dataset']][t['id']] != digest(m) or data['mapping'][t['id']] != dict(task_id=prior['id'],data_ref=parent['baseline_unified']['data_ref']):raise ValueError('true fold/domain scaler, metadata and window source')
        a = step_arithmetic(c,t)
        if m['window_counts'] != dict(train=a['train_windows'],validation=a['validation_windows']) or actual['training']['scheduler']['steps_per_epoch'] != a['train_batches']:raise ValueError('new batch/window scheduler arithmetic')
        seen.add((t['model'],t['dataset'],t['fold'],t['h']))
    expected_set = ({('PatchTST',d,1,h) for d in M_DATASETS for h in (96,192,336,720)} if stage.startswith('PATCH_ENC') else
        {(m,d,1,h)for m in s.MODELS for d in M_DATASETS for h in (96,192,336,720)} if stage=='M_ALL' else
        {(m,'UrbanEV',f,h) for m in s.MODELS for f in range(1,7) for h in (3,12)})
    if seen != expected_set or len(seen) != len(c['tasks']):raise ValueError('unique exact new task coverage')
    if stage == 'URBAN_SUBSET':
        old = oldcs[stage]
        if any(c['resolved_profiles'][t['id']] != profile(old,t) for t in old['tasks']) or any(c[k] != old[k] for k in ('datasets','sources','urban_folds','urban_input_variants')):raise ValueError('original28 profiles and split policy unchanged')
    if stage=='M_ALL':
        parent=bound(spec['old_config_refs'][stage]);fixed_encoder_policy()
        if c['tasks']!=[encoder2_type1_task(t)for t in parent['tasks']] or c['baseline_unified'].get('fixed_encoder_policy_ref')!=spec['fixed_encoder_policy_ref'] or c['baseline_unified']['numeric_policies']!=parent['baseline_unified']['numeric_policies']:raise ValueError('only24 third-round structure revisions; order/policy preserved')
    return c


def groups(c):
    stage = c['baseline_unified']['stage']; result = []
    from utils.ch3_type1_tasks import MODELS
    for model in ('PatchTST',) if stage.startswith('PATCH_ENC') else MODELS:
        banks = [[t for t in c['tasks'] if t['dataset'] == d] for d in M_DATASETS] if stage.startswith('PATCH_ENC') else [[t for t in c['tasks'] if t['model'] == model]]
        for rows in banks:
            label = stage+'-'+rows[0]['dataset']+'-four-H' if stage.startswith('PATCH_ENC') else 'cross-fold-f1-f6-H3-H12'
            result.append(dict(id=model+'-'+label,model=model,representatives=[t['id'] for t in rows],planned_q=4,coverage={t['id']:[t['id']] for t in rows},identities={t['id']:digest(profile(c,t)) for t in rows},equivalence='own profile; actual serial and bounded-q waves; no inherited encoder3 concurrency'))
    return result


def activate():
    global ACTIVE, DELTA_REF
    if ACTIVE:return
    prefix.activate()
    from utils import ch3_type1_tasks as s, ch3_type1_chain as q, ch3_round2_amendment as amend, ch3_ms_seal_recovery as ms
    old_file,old_package,old_validate,old_selected,old_groups,old_context,old_plan,old_budget = s.file,s.package,s.validate,s.selected,s.probe_groups,s.context,s.plan,s.probe_budget
    files = config_refs(); packages = {k:old_package(k) for k in s.STAGES}; spec = contract()
    s.STAGES = tuple(spec['stage_order'])
    s.file = lambda stage:Path(files[stage]['path'])
    s.package = lambda stage:EVENT_PACKAGE if stage=='PATCH_ENC1' else REVISION_PACKAGE if stage=='M_ALL' else PACKAGE if stage in spec['new_config_refs'] else packages[stage]
    s.validate = lambda c:validate_revision(c) if c['baseline_unified']['stage'] in spec['new_config_refs'] else old_validate(c)
    s.selected = lambda stage:[t for t in s.parent()['tasks'] if t['dataset']=='UrbanEV' and t['h'] in (3,12)] if stage=='URBAN_SUBSET' else old_selected(stage)
    s.probe_groups = lambda c:groups(c) if c['baseline_unified']['stage'] in ('PATCH_ENC1','PATCH_ENC2','URBAN_SUBSET') else old_groups(c)
    def budget(c):
        v=old_budget(c)
        if c['baseline_unified']['stage']=='PATCH_ENC1':
            actual=bound(OBSERVATION_REF)['retained_actual']
            for k in actual:v['nominal'][k]+=actual[k]
            v['nominal_workers']+=bound(OBSERVATION_REF)['retained_failed_workers']
            v.update(retained_historical_actual=actual,new_nominal_workers=48,observation_failure_cost_retained=True)
            v.update(parent_worker_failure_ref=WORKER_FAILURE_REF,retained_zero_compute_failure_workers=1)
        if c['baseline_unified']['stage']=='URBAN_SUBSET':
            failed=bound(REUSE_REF)['failed_actual']
            for k in failed:v['nominal'][k]+=failed[k];v['caps'][k]+=failed[k]
            v['nominal_workers']+=1;v['max_workers']+=1
            v.update(retained_historical_actual=bound(REUSE_REF)['historical_actual'],new_nominal_workers=167,tail_failure_cost_retained=True)
        return v
    s.probe_budget=budget
    s.RESULT=ms.RESULT=RESULT;amend.RESULT=RESULT/'round2-amendment'
    def context(c):
        v=old_context(c)
        return dict(v,models=('PatchTST',) if c['baseline_unified']['stage'].startswith('PATCH_ENC') else s.MODELS,fixture=EVENT_PACKAGE/c['baseline_unified']['stage']/'fixtures')
    s.context=context
    def plan(c):
        v=old_plan(c);models=list(context(c)['models']);v['models']=models;v['formal_waves']={m:v['formal_waves'][m] for m in models};return v
    s.plan=plan
    for stage in ('PATCH_ENC1','PATCH_ENC2'):q.STAGE_STATES[stage]=tuple(stage+suffix for suffix in ('_PROBE','_AUTO_AUDIT','_FORMAL','_BOUNDARY'))
    q.STATES=('VERIFY_IMPORT_MS203_AND_SEAL','FOLLOWUP_PROTOCOL_PREFLIGHT',*q.STAGE_STATES['M_BASE'],'SEAL_BASE_287_BOUNDARY',*q.STAGE_STATES['M_AMEND'],'SEAL_ROUND2_REVISED_BOUNDARY',*(state for stage in s.STAGES[2:] for state in q.STAGE_STATES[stage]),'COMPLETE')
    position=q.STATES.index(q.STAGE_STATES['PATCH_ENC2'][3])+1
    q.STATES=q.STATES[:position]+(MAIN_ADOPTION_STATE,)+q.STATES[position:]
    q.CONTROL=RESULT/'queue/controller';q.LOG,q.SESSION,q.ENTRY,q.WRAPPER=LOG,SESSION,ENTRY,WRAPPER
    q.PROBE_RECOVERY=sys.modules[__name__];q.QUEUE_LOCK=prefix.numeric.PACKAGE/'queue.lock';DELTA_REF=delta_ref();ACTIVE=True


def code_binding():
    return {str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'m6_patchtst_depth_urban6_recovery_entry.py',SERIAL_ENTRY,WRAPPER,ROOT/'tests/test_m6_patchtst_depth_urban6.py',ROOT/'tests/test_m6_patch_enc1_exit_observation.py',ROOT/'tests/test_m6_whole_card_monitor.py',ROOT/'tests/test_m6_worker_guard_binding.py')}


def activate_serial_check():
    """The same guards, separate lifecycle, no formal or second worker path."""
    global SERIAL_CHECK,RESULT,LOG,SESSION,ENTRY,ATTEMPT
    activate()
    from utils import ch3_type1_chain as q,ch3_type1_tasks as s,ch3_round2_amendment as amend,ch3_ms_seal_recovery as ms
    SERIAL_CHECK=True;RESULT=SERIAL_RESULT;LOG=SERIAL_LOG;SESSION=SERIAL_SESSION;ENTRY=SERIAL_ENTRY;ATTEMPT='PATCHTST-depth-Urban6-exit-r6-no-telemetry-serial-check'
    s.RESULT=ms.RESULT=RESULT;amend.RESULT=RESULT/'round2-amendment'
    q.CONTROL=RESULT/'queue/controller';q.LOG,q.SESSION,q.ENTRY=LOG,SESSION,ENTRY
    q.run=run_serial_check


def run_serial_check(start_ref):
    from utils import ch3_type1_chain as q
    from ch3_runner import dump
    if not SERIAL_CHECK:raise PermissionError('short acceptance requires its precise entry')
    refs=adopt_prefix();c=q.configs()['PATCH_ENC1'];ctx=q.s.context(c);ctx['control'].mkdir(exist_ok=False,parents=True)
    dump(q.CONTROL/'progress.json',dict(state='PATCH_ENC1_SINGLE_SERIAL_CHECK',scope=q.s.ID,result_review='pending'))
    permit=q.create_permit(c,start_ref,True,boundary_ref={k:refs[q.STAGE_STATES[k][3]]for k in COMPLETED_STAGES},round2_ref=refs['SEAL_ROUND2_REVISED_BOUNDARY'])
    q.wait_owned('PATCH_ENC1',permit,True)
    receipt=ref(ctx['probe_root']/'serial-check.json');body=bound(receipt)
    if body['task_id']!=SERIAL_TASK or body['approval']!=permit or body['serial_check_passed']is not True:raise ValueError('precise serial acceptance result required')
    exclusive(q.CONTROL/'serial-check-complete.json',dict(purpose='PATCH_ENC1_single_serial_acceptance_v1',receipt_ref=receipt,closure_commit=q.closure(),owner=q.owner(),execution_attempt=ATTEMPT,formal_runs=0,whole_stage_admission=False))
    dump(q.CONTROL/'progress.json',dict(state='SERIAL_CHECK_COMPLETE_AWAITING_MAIN_ARM',scope=q.s.ID,result_review='pending'))


def guard_serial_task(c,purpose,task,phase):
    if SERIAL_CHECK and(c['baseline_unified']['stage']!='PATCH_ENC1' or purpose!='ch3_probe' or task!=SERIAL_TASK or phase!='serial'):raise PermissionError('exact single serial task/phase only')


def serial_check_evidence(c,required=False):
    if not SERIAL_CHECK:return None
    """Once at adoption/audit: exact prior lifecycle, never an old MAC grant."""
    from utils import ch3_type1_chain as q
    from utils.ch3_native_execution import wave_passed
    root=SERIAL_RESULT;control=root/'queue/controller';done=control/'serial-check-complete.json'
    if not done.exists():
        if required or root.exists():raise ValueError('real single serial acceptance must complete before main arm; failed check is retained')
        return None
    v=bound(ref(done));report=bound(v['receipt_ref']);a=bound(report['approval']);owner=bound(ref(control/'controller.json'));auth=bound(owner['authorization'])
    expected_attempt='PATCHTST-depth-Urban6-exit-r6-no-telemetry-serial-check';expected_root=root/'probe/PATCH_ENC1'
    expected_auth=q.start_template()
    expected_auth['upstream_anchors']=dict(expected_auth['upstream_anchors'],execution_attempt=expected_attempt,execution_mode='single_serial_check')
    mutable={'reviewed','execution_permitted','structure_frozen','m6_authorized','budget_authorized','closure_commit','authorization_basis'}
    if ({k:v for k,v in auth.items()if k not in mutable}!={k:v for k,v in expected_auth.items()if k not in mutable}
        or any(auth.get(k)is not True for k in mutable-{'closure_commit','authorization_basis'})
        or not auth.get('authorization_basis') or 'non-executable'in auth['authorization_basis']):raise PermissionError('precise prior short-check authorization required')
    if (v['purpose']!='PATCH_ENC1_single_serial_acceptance_v1' or v['closure_commit']!=q.closure() or v['execution_attempt']!=expected_attempt
        or v['formal_runs']!=0 or v['whole_stage_admission']is not False or v['owner']!=owner['owner']
        or report['task_id']!=SERIAL_TASK or report['purpose']!='native_single_serial_acceptance_v1' or report['serial_check_passed']is not True
        or v['receipt_ref']['path']!=str(expected_root/'serial-check.json') or report['approval']!=ref(root/'queue/PATCH_ENC1/probe-permit.json')
        or a['execution_attempt']!=expected_attempt or auth['closure_commit']!=q.closure() or a['start_authorization_ref']!=owner['authorization']
        or auth['upstream_anchors'].get('single_serial_task')!=SERIAL_TASK or auth['upstream_anchors'].get('execution_mode')!='single_serial_check'
        or a['protocol_sha']!=digest(c) or report['protocol_sha']!=digest(c)):raise ValueError('single serial lifecycle/config/closure binding mismatch')
    current=q.dynamic(c)
    for field,expected in resource_binding().items():
        if report.get(field)!=expected or a.get(field)!=expected:raise ValueError('single serial exclusive resource binding')
    for k in ('commit','code','environment','hardware','source_states','protocol_sha'):
        if a[k]!=current[k] or report[k]!=a[k]:raise ValueError('single serial current production applicability: '+k)
    if any((control/x).exists()for x in ('failure.json','STOP')) or (expected_root/'failure.json').exists() or (expected_root/'STOP').exists():raise ValueError('retained short-check failure/STOP')
    if set(report['evidence'])!={groups(c)[0]['id']+'/serial/0'} or report['decisions'] or not isinstance(report.get('serial_self_check'),dict) or report['serial_self_check'].get('passed')is not True:raise ValueError('one valid serial wave is not whole-group admission')
    key=next(iter(report['evidence']));entry=report['evidence'][key]
    if entry['task_ids']!=[SERIAL_TASK] or not wave_passed(bound(entry['process'])):raise ValueError('single serial resource/exit receipt required')
    from utils.ch3_native_execution import wave_resource_identities
    identities=wave_resource_identities(bound(entry['process']),a,Path(entry['process']['path']).with_name('memory.jsonl'),[SERIAL_TASK])
    if identities.get(str(report['worker_identity']['pid']))!=report['worker_identity']:raise ValueError('single serial owned control receipt')
    for path,item in report['artifacts'].items():
        if not Path(path).resolve().is_relative_to(expected_root) or ref(path)!=item:raise ValueError('exact read-only single serial artifact binding')
    budget=report['budget'];historical=bound(OBSERVATION_REF)['retained_actual'];counts=dict(adam=6,backward=6,forward=8)
    if budget['historical_actual']!=historical or budget['new_actual']!=counts or budget['actual']!={k:historical[k]+counts[k]for k in counts} or budget['caps']!=q.s.probe_budget(c)['caps']:raise ValueError('single serial historical/new cost binding')
    from utils.ch3_type1_upstream import assert_owned_exited
    assert_owned_exited([owner['owner'],bound(ref(expected_root/'controller.json'))['owner'],report['worker_identity']])
    if subprocess.run(['tmux','has-session','-t',SERIAL_SESSION],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:raise ValueError('single serial lifecycle still active')
    return dict(report=report,key=key,entry=entry,producer=a,artifacts=report['artifacts'],counts=counts)


def delta_ref():return ref(EVENT_PACKAGE/'producer-delta-proof.json')


def resource_binding():
    value=bound(RESOURCE_CONTRACT_REF)
    expected=dict(purpose='M6_exclusive_gpu_event_resource_contract_v1',resource_mode='exclusive_gpu_event_driven_v1',
        exclusive_condition='user_declared_exclusive_server_GPU_single_experiment_queue',tool_proved_exclusive=False,
        device='cuda:0',GPU_index=0,reserve_bytes=8*1024**3,reserve_fraction=.1,query_timeout_seconds=10,
        startup_only=True,runtime_gpu_queries=0,worker_GPU_attribution='not_collected',runtime_memory_telemetry='not_collected',scientific_numeric_budget_changes=False)
    if value!=expected:raise PermissionError('exact user exclusive startup/event contract required')
    return dict(resource_mode=value['resource_mode'],resource_contract_ref=RESOURCE_CONTRACT_REF)


def monitor_binding(configs):
    from utils import ch3_type1_chain as q
    expected=resource_binding()
    if not ACTIVE or q.PROBE_RECOVERY is not sys.modules[__name__]:raise PermissionError('bound exclusive recovery context required')
    permits=[]
    for cfg in configs:
        a=cfg.get('approval')if cfg.get('purpose')=='ch3_probe'else bound(cfg['formal_permit_ref'])
        if cfg.get('type1_scope')!=q.s.ID or cfg.get('probe_schema_recovery_ref')!=REUSE_REF or any(cfg.get(k)!=v or a.get(k)!=v for k,v in expected.items()):raise PermissionError('worker/permit exclusive resource binding mismatch')
        permits.append(a)
    if any(a!=permits[0]or cfg['unified_stage']!=configs[0]['unified_stage']or cfg['purpose']!=configs[0]['purpose']for cfg,a in zip(configs,permits)):raise PermissionError('one exact resource wave/permit required')
    q.validate_permit(q.configs()[configs[0]['unified_stage']],permits[0],configs[0]['purpose']=='ch3_probe')
    gpu=permits[0].get('hardware',{}).get('gpu')
    if not isinstance(gpu,str)or not gpu.strip():raise PermissionError('fixed permit GPU identity absent')
    from utils.ch3_event_resources import validate_startup
    validate_startup(permits[0]['startup_hardware_ref'],permits[0],live=True)
    if any(cfg.get('startup_hardware_ref')!=permits[0]['startup_hardware_ref']for cfg in configs):raise PermissionError('exact wave hardware grant')
    return dict(expected,gpu_uuid=gpu.splitlines()[0].split(',')[0].strip())


def worker_metadata(c):
    values=[R5_SOURCE_REF,LEGACY_RESOURCE_CONTRACT_REF,CONTRACT_REF,contract()['fixed_encoder_policy_ref'],PRIOR_OBSERVATION_REF,NAMESPACE_OBSERVATION_REF,OBSERVATION_REF,WORKER_FAILURE_REF,RESOURCE_CONTRACT_REF]
    values += list(config_refs().values())
    for value in config_refs().values():
        parent=bound(value);values.append(parent['baseline_unified']['data_ref'])
        values += list(parent['baseline_unified'].get('extension_refs',{}).values())
        for name in ('numeric_revision_ref','weather20_patience_ref'):
            if parent['baseline_unified'].get(name):values.append(parent['baseline_unified'][name])
    values += [ref(prefix.WEATHER_PACKAGE/'producer-delta-proof.json'),prefix.SOURCE_REF,prefix.REUSE_REF,prefix.WEATHER_REF,prefix.ALL_M_POLICY_REF]
    parents={p['config_ref']['path']:p['config_ref'] for p in c['baseline_unified']['parent_refs'].values()}
    values += list(parents.values())
    for value in parents.values():values.append(bound(value)['baseline_unified']['data_ref'])
    return values


def verify_production_inheritance():
    proof=bound(delta_ref())
    if proof.get('purpose')!='event_driven_depth_urban6_exact_delta_v1' or proof.get('resource_contract_ref')!=RESOURCE_CONTRACT_REF or proof['producer_commit']!=PRODUCER or proof['contract_ref']!=CONTRACT_REF or proof['new_code']!=code_binding():raise ValueError('reviewed explicit continuation delta')
    for name,row in proof['changes'].items():
        old=subprocess.check_output(['git','-C',str(ROOT),'show',PRODUCER+':'+name])
        if hashlib.sha256(old).hexdigest()!=row['before_sha256'] or sha(ROOT/name)!=row['after_sha256']:raise ValueError('exact producer delta bytes: '+name)
        if name.endswith('.py'):
            def functions(raw):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
            before,after=functions(old),functions((ROOT/name).read_bytes())
            changed=sorted(k for k in set(before)|set(after) if before.get(k)!=after.get(k))
            if changed!=row['changed_symbols']:raise ValueError('exact reviewed function delta')
    for name,expected in proof.get('new_files',{}).items():
        if sha(ROOT/name)!=expected:raise ValueError('new event resource/fixture byte binding')
    for name,expected in proof.get('unchanged_training_ast',{}).items():
        node=next(n for n in ast.parse((ROOT/'ch3_runner.py').read_bytes()).body if getattr(n,'name',None)==name)
        if hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()!=expected:raise ValueError('training/evaluation math changed: '+name)
    # Computation producers and author sources are byte-identical; the changed
    # restricted entry only supplies monitoring and orchestration, not math.
    for name,expected in proof['unchanged_producers'].items():
        if sha(ROOT/name)!=expected:raise ValueError('training/evaluation producer changed: '+name)
    source=bound(SOURCE_REF);oldcode=bound(source['refs']['AMEND'])['code']
    if {name for name,old in oldcode.items() if sha(ROOT/name)!=old}!=set(proof['producer_changed_files']):raise ValueError('unregistered production inheritance delta')
    from tools.restricted_regression.run_restricted import verify_bundle
    if verify_bundle()!=proof['new_bundle_sha']:raise ValueError('new monitor bundle seal')
    return proof


def verify_source_light():
    verify_observation_source()
    verify_worker_guard_failure_source()
    verify_r5_source()
    source=bound(SOURCE_REF)
    if source['old_result']!=str(OLD_RESULT) or source['producer_commit']!=PRODUCER or source['config_refs']!=contract()['old_config_refs']:raise ValueError('specific completed371 producer')
    rows={k:bound(v) for k,v in source['refs'].items() if k!='launcher_log'}
    if ref(source['refs']['launcher_log']['path'])!=source['refs']['launcher_log']:raise ValueError('old launcher evidence changed')
    if rows['failure']['error']!="RuntimeError('owned child technical failure: URBAN_SUBSET-probe')" or rows['probe_failure']['error']!="RuntimeError('serial technical gate failed; no fallback')":raise ValueError('only registered Urban exit observation failure')
    reuse=bound(REUSE_REF);failed=bound(reuse['retained_failed']['process_ref'])
    if failed['returncodes']!=[0] or failed['resource_admission']is not False or failed['failure']!='whole-card sampling failed: owned exit observation did not settle within 3 seconds':raise ValueError('exact original monitor failure remains failed')
    if rows['authorization']['closure_commit']!=PRODUCER or rows['controller']['authorization']!=source['refs']['authorization'] or rows['probe_permit']['commit']!=PRODUCER:raise ValueError('old lifecycle identity')
    for stage,count in (('M_BASE',84),('AMEND',112)):
        b=rows[stage];c=bound(source['config_refs']['M_AMEND' if stage=='AMEND' else stage])
        if b['technical_complete']is not True or b['protocol_sha']!=digest(c) or b['task_ids']!=[t['id'] for t in c['tasks']] or len(set(b['task_ids']))!=count:raise ValueError('exact completed task prefix')
        from utils.ch3_type1_tasks import MODELS
        if set(b['receipts'])!=set(MODELS):raise ValueError('completed model receipt set')
    if rows['base']['counts']!=dict(MS=203,M=84,total=287) or rows['base']['boundaries']['M']!=source['refs']['M_BASE'] or rows['round2']['amendment_ref']!=source['refs']['AMEND'] or rows['round2']['main_index_ref']!=source['refs']['main'] or rows['round2']['original_upstream_ref']!=source['refs']['base']:raise ValueError('exact completed base and revised371 lineage')
    from utils.ch3_round2_amendment import validate_index
    validate_index(rows['main'])
    from utils.ch3_type1_upstream import assert_owned_exited
    assert_owned_exited(source['exit_instances'])
    instances=[]
    for row in list(reuse['accepted'].values())+[reuse['retained_failed']]:
        process=row.get('process',row.get('process_ref'));location=Path(process['path'])
        # These two bounded memory logs preserve the actual namespace/starttime;
        # numeric sidecars and checkpoints are not consulted by preflight.
        samples=[json.loads(line) for line in location.with_name('memory.jsonl').read_text().splitlines()]
        bank={}
        for sample in samples:
            for pid,m in sample.get('owned_pid_metadata',{}).items():
                if m.get('start_ticks')is not None:
                    identity=dict(pid=int(pid),start_ticks=str(m['start_ticks']))
                    if pid in bank and bank[pid]!=identity:raise ValueError('old memory process identity conflicts')
                    bank[pid]=identity
        if len(bank)!=1:raise ValueError('each old serial must have exactly one proven process identity')
        instances.extend(bank.values())
    assert_owned_exited(instances)
    for suffix in ('queue/controller/STOP','probe/URBAN_SUBSET/STOP'):
        if (OLD_RESULT/suffix).exists():raise ValueError('STOP is not this registered technical failure')
    if any((OLD_RESULT/stage).exists() for stage in ('URBAN_SUBSET','EPF_ALL','M_ALL')):raise ValueError('unexpected old third-round formal output; preserve and reassess affected scope')
    if source['old_probe_counts']!=dict(adam=12,backward=12,forward=16) or rows['old_budget']['actual']!=source['old_probe_counts']:raise ValueError('retained actual historical cost')
    return rows


def verify_worker_guard_failure_source():
    """The r4 access failure is negative evidence, never a valid resource wave."""
    value=bound(WORKER_FAILURE_REF);root=OLD_RESULT.with_name('baseline-unified-v3-ms-seal-m128-recovery1-patchtst-depth-urban6-r4-serial-check')
    if (value.get('purpose')!='exact_PATCH_ENC1_worker_guard_binding_failure_v1' or value['producer_commit']!='805656e9da7d780330cddbf79995f234deac90d6'
        or value['old_attempt']!='PATCHTST-depth-Urban6-exit-r4-serial-check' or value['old_result']!=str(root)
        or value['prior_observation_ref']!=OBSERVATION_REF or value['config_ref']!=config_refs()['PATCH_ENC1']
        or value['resource_contract_ref']!=LEGACY_RESOURCE_CONTRACT_REF or value['task_id']!=SERIAL_TASK):raise ValueError('exact r4 worker-guard failure source')
    refs=value['refs']
    for item in refs.values():
        if ref(item['path'])!=item:raise ValueError('r4 negative evidence changed: '+item['path'])
    process=bound(refs['process']);cfg=bound(refs['worker_config']);permit=bound(refs['permit']);auth=bound(refs['authorization'])
    zero=dict(adam=0,backward=0,forward=0);historical=bound(OBSERVATION_REF)['retained_actual'];budget=bound(refs['probe_budget'])
    if (process['returncodes']!=[1] or process['resource_admission']is not False or process['failure_kind']!='business'
        or process['resource_mode']!='exclusive_gpu_whole_card_v1' or process['resource_contract_ref']!=LEGACY_RESOURCE_CONTRACT_REF
        or process['sample_count']!=14 or value['sample_count']!=14 or value['attempt_actual']!=zero
        or value['retained_actual']!=historical or budget['new_actual']!=zero or budget['actual']!=historical
        or bound(refs['worker_budget'])['counts']!=zero or value['old_caps']!=dict(adam=432,backward=432,forward=576)
        or value['new_formal_runs']!=0 or permit['commit']!=value['producer_commit'] or auth['closure_commit']!=value['producer_commit']
        or permit['execution_attempt']!=value['old_attempt'] or cfg['approval']!=permit
        or cfg['protocol_sha']!=digest(bound(value['config_ref'])) or cfg['unified_stage']!='PATCH_ENC1'
        or any(cfg.get(k)!=v or permit.get(k)!=v for k,v in dict(resource_mode='exclusive_gpu_whole_card_v1',resource_contract_ref=LEGACY_RESOURCE_CONTRACT_REF).items())
        or cfg['metadata_files'].get(LEGACY_RESOURCE_CONTRACT_REF['path'])!=LEGACY_RESOURCE_CONTRACT_REF['sha256']
        or value['old_evidence_denied_path'] in cfg['metadata_files']):raise ValueError('r4 failure identity/cost/resource binding')
    denied=[json.loads(line) for line in Path(refs['audit']['path']).read_text().splitlines() if json.loads(line)['event']=='denied']
    if len(denied)!=1 or denied[0]['path']!=value['old_evidence_denied_path'] or denied[0]['reason']!='old evidence forbidden':raise ValueError('exact r4 denied metadata access')
    runtime=bound(refs['runtime'])
    if runtime['error']!="ForbiddenAccess('access refused: "+value['old_evidence_denied_path']+": old evidence forbidden')":raise ValueError('r4 worker failure preserved')
    if bound(refs['controller_failure'])['error']!="RuntimeError('owned child technical failure: PATCH_ENC1-probe')" or bound(refs['probe_failure'])['error']!="RuntimeError('shared guard failure: stop all execution')":raise ValueError('r4 failure remains failure')
    samples=[json.loads(line) for line in Path(refs['memory']['path']).read_text().splitlines()]
    if len(samples)!=14 or any(s.get('resource_mode')!='exclusive_gpu_whole_card_v1' or 'owned_pid_metadata' in s or 'nvml_processes' in s for s in samples):raise ValueError('r4 whole-card negative wave samples')
    from utils.ch3_type1_upstream import assert_owned_exited
    assert_owned_exited(value['exit_instances'])
    if subprocess.run(['tmux','has-session','-t','ch3-m6-patchtst-depth-urban6-r4-serial-check'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:raise ValueError('r4 short lifecycle still active')
    return value


def verify_prior_observation_source():
    """Specific r1 observation failure; never an arbitrary ignore-failure path."""
    value=bound(PRIOR_OBSERVATION_REF)
    old_result=OLD_RESULT.with_name('baseline-unified-v3-ms-seal-m128-recovery1-patchtst-depth-urban6-r1')
    if (value.get('purpose')!='exact_PATCH_ENC1_active_identity_observation_failure_v1' or value['producer_commit']!='e051b96e416433b8940d9ef6b71192033fa07a89'
        or value['old_attempt']!='PATCHTST-depth-Urban6-exit-r1' or value['old_result']!=str(old_result)
        or value['config_ref']!=config_refs()['PATCH_ENC1'] or value['completed_round2_source_ref']!=SOURCE_REF):raise ValueError('exact parent observation failure/config/source required')
    refs=value['refs']
    for item in refs.values():
        if ref(item['path'])!=item:raise ValueError('r1 negative observation evidence changed: '+item['path'])
    failure=bound(refs['queue/controller/failure.json']);probe=bound(refs['probe/PATCH_ENC1/failure.json']);process=bound(refs['process'])
    auth=bound(refs['start_authorization']);permit=bound(refs['probe/PATCH_ENC1/approval.json'])
    if (failure.get('error')!="RuntimeError('owned child technical failure: PATCH_ENC1-probe')" or probe.get('error')!="RuntimeError('serial technical gate failed; no fallback')"
        or process.get('failure')!='whole-card sampling failed: active owned process identity unavailable' or process.get('failure_kind')!='observation'
        or process.get('returncodes')!=[0] or process.get('resource_admission')is not False or probe.get('decisions')
        or value.get('retained_actual')!=dict(adam=6,backward=6,forward=8) or probe['budget']['actual']!=value['retained_actual']
        or value.get('new_formal_runs')!=0 or auth['closure_commit']!=value['producer_commit'] or permit['commit']!=value['producer_commit']
        or permit.get('execution_attempt')!=value['old_attempt'] or permit.get('start_authorization_ref')!=refs['start_authorization']):raise ValueError('exact r1 observation lifecycle and true failed cost')
    if bound(refs['runtime'])['error']is not None or bound(refs['worker_budget'])['counts']!=value['retained_actual']:raise ValueError('registered six-step worker identity/accounting')
    from utils.ch3_type1_upstream import assert_owned_exited
    assert_owned_exited(value['exit_instances'])
    if subprocess.run(['tmux','has-session','-t',value['old_session']],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:raise ValueError('old observation attempt session still exists')
    if any((old_result/stage).exists()for stage in ('PATCH_ENC1','PATCH_ENC2','URBAN_SUBSET','EPF_ALL','M_ALL')) or (old_result/'probe/PATCH_ENC1/complete.json').exists():raise ValueError('unexpected completed work in registered failed r1')
    return value


def verify_namespace_exit_source():
    """Two exact failed lifecycles, retained cost only; neither is a Passed seed."""
    prior=verify_prior_observation_source();value=bound(NAMESPACE_OBSERVATION_REF)
    old_result=OLD_RESULT.with_name('baseline-unified-v3-ms-seal-m128-recovery1-patchtst-depth-urban6-r2')
    if (value.get('purpose')!='exact_PATCH_ENC1_namespace_exit_observation_failure_v1' or value['producer_commit']!='d852398c7949a2feea9b572ee793c1e793ecd06c'
        or value['old_attempt']!='PATCHTST-depth-Urban6-exit-r2' or value['old_result']!=str(old_result)
        or value['prior_observation_ref']!=PRIOR_OBSERVATION_REF or value['config_ref']!=config_refs()['PATCH_ENC1']
        or value['completed_round2_source_ref']!=SOURCE_REF or value['task_id']!=prior['task_id']):raise ValueError('exact r2 namespace failure and prior r1/source binding')
    refs=value['refs']
    for item in refs.values():
        if ref(item['path'])!=item:raise ValueError('r2 negative observation evidence changed: '+item['path'])
    failure=bound(refs['queue/controller/failure.json']);probe=bound(refs['probe/PATCH_ENC1/failure.json']);process=bound(refs['process'])
    auth=bound(refs['start_authorization']);permit=bound(refs['probe/PATCH_ENC1/approval.json']);snapshot=bound(refs['observation_failure'])
    expected=dict(adam=12,backward=12,forward=16);attempt=dict(adam=6,backward=6,forward=8)
    if (failure.get('error')!="RuntimeError('owned child technical failure: PATCH_ENC1-probe')" or probe.get('error')!="RuntimeError('serial technical gate failed; no fallback')"
        or process.get('failure')!='whole-card sampling failed: owned process identity read/parse failed: 22760; PermissionError; namespace'
        or process.get('failure_kind')!='observation' or process.get('returncodes')!=[0] or process.get('resource_admission')is not False
        or process.get('observation_failure_ref')!=refs['observation_failure'] or probe.get('decisions')
        or value['retained_actual']!=expected or value['attempt_actual']!=attempt or value['retained_failed_workers']!=2
        or probe['budget']['actual']!=expected or probe['budget']['historical_actual']!=prior['retained_actual'] or probe['budget']['new_actual']!=attempt
        or value['old_caps']!=prior['old_caps'] or value['new_formal_runs']!=0 or auth['closure_commit']!=value['producer_commit']
        or permit['commit']!=value['producer_commit'] or permit.get('execution_attempt')!=value['old_attempt']
        or permit.get('start_authorization_ref')!=refs['start_authorization']):raise ValueError('exact r2 failed lifecycle, unchanged caps and cumulative true cost')
    identity=value['worker_identity'];pid=str(identity['pid']);before=snapshot['context']['before_metadata'][pid];after=snapshot['sample']['owned_pid_metadata'][pid]
    known=snapshot['state']['known'][pid]
    if (identity!=dict(pid=22760,host_pid=13636,start_ticks='211667661',namespace='pid:[4026534934]')
        or snapshot['context']['active_before']!=[identity['pid']] or snapshot['context']['active_after']!=[]
        or any(before.get(k)!=v for k,v in identity.items()) or known!=dict(host_pid=str(identity['host_pid']),start_ticks=identity['start_ticks'],namespace=identity['namespace'])
        or after.get('state')!='metadata_unavailable' or after.get('error_kind')!='PermissionError' or after.get('phase')!='namespace'
        or after.get('start_ticks_before')!=identity['start_ticks'] or after.get('start_ticks_after')is not None
        or snapshot['sample']['nvml_processes']!={} or snapshot['sample']['process_table_reliable']is not True
        or snapshot['current_owned'][0]['returncode']!=0 or snapshot['current_owned'][0]['metadata']['state']!='exited_during_sample'):
        raise ValueError('exact registered failed namespace sample, never a fresh exit grant')
    runtime=bound(refs['runtime'])
    if runtime['pid']!=identity['pid'] or runtime['error']is not None or bound(refs['worker_budget'])['counts']!=attempt:raise ValueError('registered r2 worker accounting')
    from utils.ch3_type1_upstream import assert_owned_exited
    assert_owned_exited(value['exit_instances'])
    if subprocess.run(['tmux','has-session','-t',value['old_session']],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:raise ValueError('old r2 session still exists')
    if any((old_result/stage).exists()for stage in ('PATCH_ENC1','PATCH_ENC2','URBAN_SUBSET','EPF_ALL','M_ALL')) or (old_result/'probe/PATCH_ENC1/complete.json').exists():raise ValueError('unexpected completed work in failed r2')
    return value


def verify_observation_source():
    """Three exact negative lifecycles: immutable costs, never Passed seeds."""
    prior=verify_namespace_exit_source();value=bound(OBSERVATION_REF)
    old_result=OLD_RESULT.with_name('baseline-unified-v3-ms-seal-m128-recovery1-patchtst-depth-urban6-r3')
    if (value.get('purpose')!='exact_PATCH_ENC1_namespace_identity_pending_failure_v1'
        or value['producer_commit']!='0abc95542f28c000d0366ad43a29e0b85867e38c'
        or value['old_attempt']!='PATCHTST-depth-Urban6-exit-r3' or value['old_result']!=str(old_result)
        or value['prior_observation_ref']!=NAMESPACE_OBSERVATION_REF or value['config_ref']!=config_refs()['PATCH_ENC1']
        or value['completed_round2_source_ref']!=SOURCE_REF or value['task_id']!=prior['task_id']):raise ValueError('exact registered r3 source required')
    refs=value['refs']
    for item in refs.values():
        if ref(item['path'])!=item:raise ValueError('r3 negative observation evidence changed: '+item['path'])
    process=bound(refs['process']);snap=bound(refs['observation_failure']);failed=bound(refs['probe/PATCH_ENC1/failure.json'])
    permit=bound(refs['probe/PATCH_ENC1/approval.json']);auth=bound(refs['start_authorization']);attempt=dict(adam=6,backward=6,forward=8)
    identity=value['worker_identity'];pid=str(identity['pid'])
    if (bound(refs['queue/controller/failure.json'])['error']!="RuntimeError('owned child technical failure: PATCH_ENC1-probe')"
        or process['returncodes']!=[0] or process['resource_admission']is not False or process['failure_kind']!='observation'
        or process['failure']!='whole-card sampling failed: owned process identity read/parse failed: 41212; PermissionError; namespace'
        or process['observation_failure_ref']!=refs['observation_failure'] or failed['decisions']
        or failed['budget']['historical_actual']!=prior['retained_actual'] or failed['budget']['new_actual']!=attempt
        or failed['budget']['actual']!=value['retained_actual'] or value['retained_actual']!=dict(adam=18,backward=18,forward=24)
        or value['attempt_actual']!=attempt or value['retained_failed_workers']!=3 or value['old_caps']!=prior['old_caps']
        or value['new_formal_runs']!=0 or auth['closure_commit']!=value['producer_commit'] or permit['commit']!=value['producer_commit']
        or permit.get('execution_attempt')!=value['old_attempt'] or permit.get('start_authorization_ref')!=refs['start_authorization']):raise ValueError('r3 failed lifecycle/cost/config mismatch')
    m=snap['sample']['owned_pid_metadata'][pid]
    if (identity!=dict(pid=41212,host_pid=49401,start_ticks='212094890',namespace='pid:[4026534934]')
        or snap['context']['active_before']!=[identity['pid']] or snap['context']['active_after']!=[identity['pid']]
        or any(snap['context']['before_metadata'][pid].get(k)!=v for k,v in identity.items())
        or m.get('error_kind')!='PermissionError' or m.get('phase')!='namespace' or m.get('start_ticks_before')!=identity['start_ticks']
        or snap['current_owned'][0]['returncode']is not None or snap['sample']['owned_host_pids']
        or set(snap['sample']['nvml_processes'])!={str(identity['host_pid'])}):raise ValueError('exact saved r3 unresolved identity sample required')
    if bound(refs['runtime'])['pid']!=identity['pid'] or bound(refs['runtime'])['error']is not None or bound(refs['worker_budget'])['counts']!=attempt:raise ValueError('r3 original six-step accounting')
    from utils.ch3_type1_upstream import assert_owned_exited
    assert_owned_exited(value['exit_instances'])
    if subprocess.run(['tmux','has-session','-t',value['old_session']],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:raise ValueError('old r3 session still exists')
    if any((old_result/stage).exists()for stage in ('PATCH_ENC1','PATCH_ENC2','URBAN_SUBSET','EPF_ALL','M_ALL')) or (old_result/'probe/PATCH_ENC1/complete.json').exists():raise ValueError('unexpected old r3 completed work')
    return value


def verify_prefix():
    """Once on arm: saved JSON/histories, receipts and checkpoint STAT only."""
    rows=verify_source_light();evidence=bound(PREFIX_REF)
    if evidence['source_ref']!=SOURCE_REF:raise ValueError('reviewed prefix verification identity')
    for name,value in evidence['metadata_refs'].items():
        if ref(name)!=value:raise ValueError('completed metadata changed: '+name)
    for name,state in evidence['checkpoint_stats'].items():
        p=Path(name)
        if p.is_symlink() or not p.is_file() or dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)!=state:raise ValueError('saved checkpoint identity changed; never reload/retest')
    return rows


def anchors():
    return dict(recovery='specific_PATCH_ENC1_observation_after_completed_round2_371',execution_attempt=ATTEMPT,execution_mode='single_serial_check'if SERIAL_CHECK else'fixed_remaining_chain',single_serial_task=SERIAL_TASK,single_serial_result=str(SERIAL_RESULT),parent_technical_failure_ref=R5_SOURCE_REF,prior_observation_failure_ref=OBSERVATION_REF,r5_numeric_reference_ref=R5_SOURCE_REF,timing_supplement=bound(R5_SOURCE_REF)['timing_supplement'],prior_Urban_failure_ref=SOURCE_REF,completed_round2_ref=SOURCE_REF,completed_prefix_ref=PREFIX_REF,probe_reuse_ref=REUSE_REF,contract_ref=CONTRACT_REF,fixed_encoder_policy_ref=contract()['fixed_encoder_policy_ref'],producer_delta_ref=delta_ref(),candidate_config_refs=config_refs(),completed_new_formal=196,expected_remaining_formal=335,approved_total_new_formal=531,third_round_runs=287,depth_supplement_runs=48,automatic_main_table_replacement=True,main_table_replacement_cells=24,chosen_M_encoder=2,metric_based_reselection=False,**resource_binding())


def status():
    verify_source_light()
    try:verify_production_inheritance()
    except (ImportError,RuntimeError)as exc:raise ValueError('completed371 producer/bundle verification failed: '+str(exc)[:300])from exc
    if not SERIAL_CHECK:verify_r5_source()
    return dict(state='COMPLETED_ROUND2_371_ADOPTION_AWAITING_START',READY_FOR_GPU_EXECUTION=False,anchors=anchors(),remaining_formal_runs=335,result_review='pending')


def completion_counts():return dict(total_runs=531,third_round_runs=287,adopted_new_formal_runs=196,executed_new_formal_runs=335,depth_supplement_runs=48)


def sign(body):return prefix.sign(body)


def adopt_prefix():
    from utils import ch3_type1_chain as q
    q.stop_check();verify_production_inheritance();rows=verify_prefix();source=bound(SOURCE_REF);commit=q.closure();q.stop_check()
    upstream=dict(state='COMPLETED_ROUND2_371_OLD_SOURCES_RELEASED',READY_FOR_GPU_EXECUTION=True,anchors=anchors(),boundaries=rows['upstream']['boundaries'],handoff_scope=q.s.ID,successor_owner=q.owner(),adoption_commit=commit,completed_round2_source_ref=SOURCE_REF,result_review='pending')
    refs={'VERIFY_IMPORT_MS203_AND_SEAL':exclusive(q.CONTROL/'upstream-technical-boundary.json',sign(upstream))}
    for stage,source_key in (('M_BASE','M_BASE'),('M_AMEND','AMEND')):
        lifecycle={'mac','adopted_source_ref','prefix_verification_ref','adoption_commit','adoption_owner'}
        old=rows[source_key]
        body=dict({k:v for k,v in old.items() if k not in lifecycle},prior_adoption_fields={k:v for k,v in old.items() if k in lifecycle},adopted_source_ref=source['refs'][source_key],prefix_verification_ref=PREFIX_REF,adoption_commit=commit,adoption_owner=q.owner())
        refs[q.STAGE_STATES[stage][3]]=exclusive(q.s.context(q.configs()[stage])['control']/'technical-boundary.json',sign(body))
    refs['SEAL_BASE_287_BOUNDARY']=q.seal_base287_boundary(refs['SEAL_M128_BOUNDARY'])
    body=dict(rows['round2'],original_upstream_ref=refs['SEAL_BASE_287_BOUNDARY'],amendment_ref=refs['SEAL_AMEND_BOUNDARY'],owner=q.owner(),adoption_commit=commit,adopted_source_ref=source['refs']['round2'])
    refs['SEAL_ROUND2_REVISED_BOUNDARY']=exclusive(q.s.RESULT/'round2-amendment/queue/round2-boundary.json',sign({k:v for k,v in body.items() if k!='mac'}))
    q.validate_round2_boundary_light(refs['SEAL_ROUND2_REVISED_BOUNDARY']);return refs


def validate_adopted_boundary(value):
    from utils import ch3_type1_chain as q
    body=bound(value);controller=bound(ref(q.CONTROL/'controller.json'));content={k:v for k,v in body.items() if k!='mac'}
    stage=body['purpose'].removeprefix('baseline_type1_').removesuffix('_boundary_v1')
    expected=bound(SOURCE_REF)['refs']['AMEND' if stage=='M_AMEND' else'M_BASE']
    if stage not in COMPLETED_STAGES or body.get('adopted_source_ref')!=expected or body.get('prefix_verification_ref')!=PREFIX_REF or body.get('adoption_owner')!=controller['owner'] or body.get('adoption_commit')!=q.closure():raise PermissionError('exact completed source/new adoption lifecycle')
    if os.environ.get(q.SECRET) and (not hmac.compare_digest(sign(content)['mac'],body.get('mac','')) or not q.same(controller['owner'])):raise PermissionError('current adoption MAC/owner')
    old=bound(expected)
    lifecycle={'mac','adopted_source_ref','prefix_verification_ref','adoption_commit','adoption_owner'}
    if {k:body[k] for k in old if k not in lifecycle}!={k:v for k,v in old.items() if k not in lifecycle} or body.get('prior_adoption_fields')!={k:v for k,v in old.items() if k in lifecycle}:raise ValueError('retained original production and prior lifecycle fields')


def validate_round2_adoption(body):
    expected=bound(SOURCE_REF)['refs']
    if body.get('adopted_source_ref')!=expected['round2'] or body['main_index_ref']!=expected['main']:raise ValueError('original371 main index only; depth never replaces main')


def validate_permit_link(c,a,probe):
    stage=c['baseline_unified']['stage']
    if stage in COMPLETED_STAGES:raise PermissionError('completed MS/M_BASE/M_AMEND cannot be redispatched')
    if SERIAL_CHECK and (stage!='PATCH_ENC1' or not probe):raise PermissionError('single serial mode cannot dispatch formal/other stages')
    if a.get('execution_attempt')!=ATTEMPT or a.get('probe_recovery_ref')!=REUSE_REF or c!=bound(config_refs()[stage]):raise PermissionError('exact new continuation attempt/profile/variant')
    if stage in ('URBAN_SUBSET','EPF_ALL','M_ALL') and a.get('round2_boundary_ref',{}).get('path')!=str(main_boundary_path()):raise PermissionError('third round requires the completed fixed-encoder2 main boundary')


def second_round():
    if main_boundary_path().exists():
        boundary=bound(ref(main_boundary_path()));value=validate_fixed_main_index(bound(boundary['main_index_ref']))
        if value['enc2_boundary_ref']!=boundary['enc2_boundary_ref']or value['adoption_policy_ref']!=boundary['adoption_policy_ref']:raise ValueError('fixed main index/boundary adoption mismatch')
        return value
    return dict(original_second_round(),fixed_encoder_adoption='pending_complete24_enc2',intended_M_encoder=2,adoption_policy_ref=contract()['fixed_encoder_policy_ref'],main_index_status='original371_historical_sources_only')


def original_second_round():
    from utils.ch3_round2_amendment import validate_index
    return validate_index(bound(bound(SOURCE_REF)['refs']['main']))


def main_index_path():return RESULT/'round2-amendment/queue/round2-main-results.patchtst-enc2-v1.json'
def main_boundary_path():return RESULT/'round2-amendment/queue/round2-boundary.patchtst-enc2-v1.json'


def encoder2_main_cells(boundary_ref):
    """Read sealed metadata, never compare outcomes or load best checkpoints."""
    from utils import ch3_type1_chain as q
    from utils.ch3_round2_amendment import artifact_rows,key
    c=q.configs()['PATCH_ENC2'];b=bound(boundary_ref)
    if (boundary_ref['path']!=str(q.s.context(c)['control']/'technical-boundary.json') or b.get('technical_complete')is not True
        or b.get('purpose')!='baseline_type1_PATCH_ENC2_boundary_v1' or b.get('scope')!=q.s.ID
        or b.get('protocol_sha')!=digest(c) or b.get('task_ids')!=[t['id']for t in c['tasks']]
        or set(b.get('receipts',{}))!={'PatchTST'}):raise ValueError('all24 encoder2 tasks must be technically sealed before adoption')
    files=artifact_rows(boundary_ref)
    group=bound(b['receipts']['PatchTST'])
    if group.get('technical_complete')is not True or group.get('model')!='PatchTST' or group.get('task_ids')!=b['task_ids'] or set(files)!=set(b['task_ids']):raise ValueError('exact encoder2 group completion and unique task coverage')
    old={row['cell_id']:row for row in original_second_round()['cells']};rows={}
    for t in c['tasks']:
        f=files[t['id']];p=profile(c,t);root=RESULT/'PATCH_ENC2'/'formal-PatchTST'/t['id']
        for name in ('result.json','manifest.json','runtime.json'):
            if f[name]['path']!=str(root/name):raise ValueError('encoder2 source must use the independent current variant root')
        result,manifest,runtime=[bound(f[name])for name in ('result.json','manifest.json','runtime.json')]
        if (manifest.get('task')!=t or manifest.get('profile')!=p or result.get('id')!=t['id']
            or result.get('profile_sha')!=digest(p) or manifest['identity'].get('profile_sha')!=digest(p)
            or result.get('protocol_sha')!=digest(c) or result.get('scientific_protocol')!=c['baseline_unified']['id']
            or manifest['identity'].get('protocol_sha')!=digest(c) or manifest['identity'].get('run_id')!=t['id']
            or manifest['identity'].get('scientific_protocol')!=c['baseline_unified']['id']
            or result.get('input_variant')!='M' or result.get('task')!='M' or result.get('metric_scope')!='all_channels'
            or manifest['identity'].get('task')!='M' or manifest['identity'].get('metric_scope')!='all_channels'
            or result.get('from_scratch')is not True or manifest['identity'].get('from_scratch')is not True
            or result.get('scheduler_sha')!=digest(p['training']['scheduler'])
            or result.get('commit')!=b['commit'] or manifest['identity'].get('commit')!=b['commit']
            or runtime.get('task')!=t['id'] or runtime.get('error')is not None
            or result.get('final_test',{}).get('calls')!=1 or result['final_test'].get('selected')!='best.pt'
            or result['final_test'].get('sha256')!=f['best.pt']['sha256']
            or not isinstance(result.get('best_epoch'),int) or not 1<=result['best_epoch']<=p['training']['epochs']
            or result['final_test'].get('epoch')!=result['best_epoch']
            or result.get('seed')!=2024 or not all(math.isfinite(result[k])and result[k]>=0 for k in ('mse','mae'))):
            raise ValueError('exact encoder2 result/manifest/producer/validation-best/test-once source')
        cell=key(t);previous=old[cell]
        rows[cell]=dict(cell_id=cell,task='M',model='PatchTST',dataset=t['dataset'],fold=t['fold'],H=t['h'],seed=2024,
            origin='fixed_encoder2_supplement',scientific_protocol=c['baseline_unified']['id'],execution_commit=b['commit'],
            profile_sha=result['profile_sha'],protocol_sha=result['protocol_sha'],result_ref=f['result.json'],manifest_ref=f['manifest.json'],runtime_ref=f['runtime.json'],
            mse=result['mse'],mae=result['mae'],supersedes=previous['result_ref'],encoder=2,variant='encoder-2',std='N/A')
    expected={row['cell_id']for row in old.values()if row['model']=='PatchTST'and row['task']=='M'}
    if set(rows)!=expected or len(rows)!=24:raise ValueError('fixed six-domain/four-H adoption; no per-cell layer choice')
    return rows


def build_fixed_main_index(enc2_boundary_ref):
    fixed_encoder_policy();old=original_second_round();new=encoder2_main_cells(enc2_boundary_ref)
    policy=contract()['fixed_encoder_policy_ref'];original=bound(SOURCE_REF)['refs']['main']
    adoption_id=digest(dict(policy_ref=policy,original_main_ref=original,enc2_boundary_ref=enc2_boundary_ref))
    adopted={row['cell_id']:dict(previous_cell=copy.deepcopy(row),new_cell=copy.deepcopy(new[row['cell_id']]))for row in old['cells']if row['cell_id']in new}
    return dict(purpose='round2_fixed_patchtst_enc2_main_results_v1',revision='round2-main-patchtst-enc2-v1',
        effective_counts=old['effective_counts'],planned_formal_executions=399,depth_supplement_executions=48,
        cells=[copy.deepcopy(new.get(row['cell_id'],row))for row in old['cells']],technical_complete=True,result_review='pending',
        original_main_ref=original,enc2_boundary_ref=enc2_boundary_ref,adoption_policy_ref=policy,adoption_id=adoption_id,adopted_cells=adopted,
        replaced_cells=24,retained_cells=347,chosen_PatchTST_M_encoder=2,metric_based_reselection=False,
        source_rule='user-frozen encoder2 before results; all24 PatchTST M switched together; all other347 sources retained',history_test_seen=True,**resource_binding())


def validate_fixed_main_index(value):
    if value.get('purpose')!='round2_fixed_patchtst_enc2_main_results_v1':raise ValueError('distinct fixed-encoder2 main-index version required')
    expected=build_fixed_main_index(value['enc2_boundary_ref'])
    if value!=expected:raise ValueError('exact fixed24 adoption,347 unchanged,original/new provenance and policy binding')
    return value


def seal_fixed_main(receipts):
    from utils import ch3_type1_chain as q
    q.stop_check();prior=receipts['SEAL_ROUND2_REVISED_BOUNDARY'];q.validate_round2_boundary_light(prior)
    enc2=receipts[q.STAGE_STATES['PATCH_ENC2'][3]];q.validate_boundary_light(enc2,'PATCH_ENC2')
    main=build_fixed_main_index(enc2);index=exclusive(main_index_path(),main)
    body=dict(purpose='round2_fixed_patchtst_enc2_boundary_v1',scope=q.s.ID,owner=q.owner(),commit=q.closure(),
        original_boundary_ref=prior,original_main_ref=main['original_main_ref'],enc2_boundary_ref=enc2,main_index_ref=index,
        adoption_policy_ref=main['adoption_policy_ref'],adoption_id=main['adoption_id'],effective_counts=main['effective_counts'],
        replaced_cells=24,retained_cells=347,chosen_PatchTST_M_encoder=2,metric_based_reselection=False,technical_complete=True,result_review='pending',**resource_binding())
    value=exclusive(main_boundary_path(),sign(body));validate_main_boundary(value);return value


def validate_main_boundary(value):
    """Compact sealed link for workers; the 371-cell adoption audit runs once."""
    from utils import ch3_type1_chain as q
    b=bound(value);content={k:v for k,v in b.items()if k!='mac'};secret=os.environ.get(q.SECRET)
    if any(b.get(k)!=v for k,v in resource_binding().items()):raise ValueError('fixed main resource contract')
    if value['path']!=str(main_boundary_path())or not secret or not hmac.compare_digest(sign(content)['mac'],b.get('mac','')):raise PermissionError('fixed main source requires the current adoption lifecycle MAC')
    enc2=q.validate_boundary_light(b['enc2_boundary_ref'],'PATCH_ENC2')
    q.validate_round2_boundary_light(b['original_boundary_ref'])
    controller=bound(ref(q.CONTROL/'controller.json'))
    if (b.get('purpose')!='round2_fixed_patchtst_enc2_boundary_v1' or b.get('scope')!=q.s.ID or b.get('owner')!=controller['owner'] or not q.same(b['owner'])
        or b.get('commit')!=q.closure() or enc2.get('commit')!=b['commit'] or b.get('original_main_ref')!=bound(SOURCE_REF)['refs']['main']
        or b.get('main_index_ref',{}).get('path')!=str(main_index_path()) or b.get('adoption_policy_ref')!=contract()['fixed_encoder_policy_ref']
        or b.get('adoption_id')!=digest(dict(policy_ref=contract()['fixed_encoder_policy_ref'],original_main_ref=b['original_main_ref'],enc2_boundary_ref=b['enc2_boundary_ref']))
        or b.get('effective_counts')!=dict(MS=203,M=168,total=371)or b.get('replaced_cells')!=24 or b.get('retained_cells')!=347
        or b.get('chosen_PatchTST_M_encoder')!=2 or b.get('metric_based_reselection')is not False or b.get('technical_complete')is not True or b.get('result_review')!='pending'):
        raise ValueError('fixed encoder2 boundary/config/producer/selection contract')
    # Hash-only integrity, no result audit, checkpoint or numeric replay per task.
    if ref(main_index_path())!=b['main_index_ref']:raise ValueError('sealed fixed371 index bytes changed')
    return b


def continuation_actions():return {MAIN_ADOPTION_STATE:seal_fixed_main}
def completion_fields(receipts):return dict(round2_fixed_enc2_boundary=receipts[MAIN_ADOPTION_STATE],chosen_PatchTST_M_encoder=2)


def validate_complete(complete):
    if any(complete.get(k)!=v for k,v in resource_binding().items()):raise ValueError('complete exclusive resource contract')
    value=complete.get('round2_fixed_enc2_boundary');b=bound(value)
    if value['path']!=str(main_boundary_path())or b['main_index_ref']['path']!=str(main_index_path())or b['original_boundary_ref']!=complete['round2_boundary']or b['enc2_boundary_ref']!=complete['PATCH_ENC2_boundary']or complete.get('chosen_PatchTST_M_encoder')!=2:raise ValueError('complete must include original and fixed-encoder2 main boundaries')
    main=validate_fixed_main_index(bound(b['main_index_ref']))
    if any(b.get(key)!=main.get(key)for key in ('adoption_policy_ref','adoption_id','original_main_ref','enc2_boundary_ref','effective_counts','replaced_cells','retained_cells','metric_based_reselection')) or b.get('technical_complete')is not True:raise ValueError('fixed main complete adoption metadata')
    return main


def depth_results():
    from utils.ch3_type1_summary import observed
    from utils import ch3_type1_chain as q
    original={ (r['dataset'],r['H']):r for r in original_second_round()['cells'] if r['task']=='M' and r['model']=='PatchTST'}
    rows=[];means=[];cs=q.configs()
    for depth in (1,2,3):
        for dataset in M_DATASETS:
            values=[]
            for h in (96,192,336,720):
                if depth==3:value=dict(original[(dataset,h)],status='complete')
                else:
                    c=cs['PATCH_ENC'+str(depth)];t=next(t for t in c['tasks'] if (t['dataset'],t['h'])==(dataset,h))
                    value=observed(RESULT/('PATCH_ENC'+str(depth))/'formal-PatchTST'/t['id']/'result.json',t['id'],c['baseline_unified']['id'],digest(profile(c,t)))
                rows.append(dict(encoder=depth,dataset=dataset,H=h,result=value));values.append(value)
            means.append(dict(encoder=depth,dataset=dataset,mse=sum(v['mse'] for v in values)/4 if all(v['status']=='complete' for v in values) else None,mae=sum(v['mae'] for v in values)/4 if all(v['status']=='complete' for v in values) else None))
    return dict(rows=rows,dataset_four_H_means=means,main_table_selection='pre_results_fixed_encoder_2',adoption_policy_ref=contract()['fixed_encoder_policy_ref'],metric_based_reselection=False,enc1_retained_not_main=True,enc3_original_sources_retained=True,no_cross_dataset_raw_metric_mean=True,result_review='pending',**resource_binding())


def evidence():
    v=bound(REUSE_REF)
    if v['source_ref']!=SOURCE_REF or v['old_config_ref']!=contract()['old_config_refs']['URBAN_SUBSET']:raise ValueError('specific saved Urban serial producer')
    return v


def accepted_urban_readonly_refs(report):
    """Only the registered successful H3 wave, never its failed H12 sibling."""
    expected_scope='m6-baseline-type1-followup-v3-recovery1-URBAN_SUBSET-probe'
    if report.get('scope')!=expected_scope or report.get('probe_recovery_ref')!=REUSE_REF:
        raise PermissionError('exact registered Urban stage/source required')
    saved=evidence();old=bound(saved['old_config_ref']);readonly={}
    run='AMD-UrbanEV-F4-type1-followup-v3-recovery1-f1-h3-s2024'
    if set(saved['accepted'])!={run}:raise ValueError('only the registered Urban H3 serial')
    row=saved['accepted'][run];producer=row['producer']
    process=OLD_RESULT/'probe/URBAN_SUBSET/AMD-cross-fold-f1-f2-H3-H12/serial/wave-0/process.json'
    permit=bound(ref(OLD_RESULT/'queue/URBAN_SUBSET/probe-permit.json'))
    if (row['process']['path']!=str(process) or row['task_ids']!=[run] or producer!=permit
        or producer.get('commit')!=PRODUCER or producer.get('successor_scope')!=expected_scope
        or producer.get('protocol_sha')!=digest(old) or producer.get('authorized_task_ids')!=[t['id']for t in old['tasks']]
        or row['self_comparison'].get('passed')is not True):raise ValueError('exact accepted task/producer/process binding')
    def record(name):
        p=process.parent.parent/run/name
        value=row['artifacts'].get(str(p))
        if not value or value['path']!=str(p):raise ValueError('registered original success record absent')
        return bound(value)
    cfg=record('config.json');runtime=record('runtime.json');trajectory=record('trajectory.json')
    task=next(t for t in old['tasks']if t['id']==run)
    if (cfg.get('task')!=run or cfg.get('unified_stage')!='URBAN_SUBSET' or cfg.get('purpose')!='ch3_probe'
        or cfg.get('protocol_sha')!=digest(old) or cfg.get('approval')!=producer
        or runtime.get('task')!=run or runtime.get('error')is not None or trajectory.get('id')!=run
        or trajectory.get('profile_sha')!=digest(profile(old,task)) or trajectory.get('finite')is not True
        or trajectory.get('steps')!=6 or record('budget.json').get('counts')!=dict(adam=6,backward=6,forward=8)):
        raise ValueError('original Urban success/schema/profile evidence changed')
    from utils.ch3_native_execution import wave_passed
    if not wave_passed(bound(row['process'])):raise ValueError('failed Urban resource wave cannot be retained')
    for key,entry in report.get('evidence',{}).items():
        if entry.get('process',{}).get('path')==str(process) and (entry.get('process')!=row['process'] or entry.get('task_ids')!=[run] or key!='AMD-cross-fold-f1-f6-H3-H12/serial/0'):
            raise ValueError('historical process attached to wrong task/group/wave')
    readonly.update(row['artifacts']);readonly[str(process)]=row['process']
    return readonly


def retained_refs(report):
    if report.get('probe_recovery_ref') not in (None,REUSE_REF):raise PermissionError('unregistered retained probe evidence')
    if report.get('scope','').endswith('URBAN_SUBSET-probe'):return accepted_urban_readonly_refs(report)
    if report.get('single_serial_source'):
        if not report.get('scope','').endswith('PATCH_ENC1-probe'):raise PermissionError('single serial evidence only belongs to PATCH_ENC1')
        if report['single_serial_source']['path']!=str(SERIAL_RESULT/'probe/PATCH_ENC1/serial-check.json'):raise PermissionError('only the precise single serial readonly source')
        saved=bound(report['single_serial_source'])
        return dict(saved['artifacts'],**{item['path']:item for item in [saved['approval']]+[v['process']for v in saved['evidence'].values()]})
    return {}


def retained_payload(c,point):
    if c['baseline_unified']['stage']=='PATCH_ENC1':
        refs={p:v for row in bound(R5_SOURCE_REF)['accepted'].values()for p,v in row['artifacts'].items()}
        return all(point[k]in refs and ref(point[k])==refs[point[k]]for k in ('schema_file','data_file','meta_file')if k in point)
    if c['baseline_unified']['stage']=='PATCH_ENC1':
        saved=SERIAL_RESULT/'probe/PATCH_ENC1/serial-check.json'
        if not saved.exists():return False
        body=bound(ref(saved))
        if body['protocol_sha']!=digest(c) or body['task_id']!=SERIAL_TASK or body['serial_check_passed']is not True:return False
        refs=body['artifacts']
    elif c['baseline_unified']['stage']=='URBAN_SUBSET':refs={p:r for v in evidence()['accepted'].values() for p,r in v['artifacts'].items()}
    else:return False
    for k in ('schema_file','data_file','meta_file'):
        if k in point and (point[k] not in refs or ref(point[k])!=refs[point[k]]):return False
    return True


def load_seed(c,a):
    validate_permit_link(c,a,True)
    if c['baseline_unified']['stage']=='PATCH_ENC1':
        old=verify_r5_source();actual=old['historical_actual']
        from utils import ch3_type1_tasks as s
        return dict(budget=dict(caps=s.probe_budget(c)['caps'],reserved=dict(actual),actual=dict(actual),historical_actual=dict(actual),new_actual=dict(adam=0,backward=0,forward=0),diagnostic_actual=dict(actual),refund=False),evidence={},decisions={},artifacts={},numeric_reference_ref=R5_SOURCE_REF)
    v=evidence();artifacts={};entries={}
    for run,row in v['accepted'].items():
        t=next(t for t in c['tasks'] if t['id']==run)
        if profile(c,t)!=profile(bound(v['old_config_ref']),t) or row['self_comparison']['passed']is not True:raise ValueError('unchanged actual retained serial profile/self evidence')
        from utils.ch3_native_execution import wave_passed
        if not wave_passed(bound(row['process'])):raise ValueError('retained resource failure cannot be admitted')
        for p,r in row['artifacts'].items():
            if ref(p)!=r:raise ValueError('saved serial artifact changed')
        g=next(g for g in groups(c) if run in g['representatives']);key=g['id']+'/serial/'+str(g['representatives'].index(run));entries[key]=dict(process=row['process'],task_ids=[run]);artifacts.update(row['artifacts'])
    from utils import ch3_type1_tasks as s
    actual=v['historical_actual']
    return dict(budget=dict(caps=s.probe_budget(c)['caps'],reserved=dict(actual),actual=dict(actual),historical_actual=dict(actual),new_actual=dict(adam=0,backward=0,forward=0),diagnostic_actual=v['failed_actual'],refund=False),evidence=entries,decisions={},artifacts=artifacts)


def retained_wave(report,key):
    if not report.get('probe_recovery_ref'):return None
    entry=report.get('evidence',{}).get(key)
    if not entry:return None
    if report.get('single_serial_source'):
        if report['single_serial_source']['path']!=str(SERIAL_RESULT/'probe/PATCH_ENC1/serial-check.json'):raise PermissionError('only the precise single serial readonly source')
        saved=bound(report['single_serial_source'])
        if key in saved['evidence'] and entry==saved['evidence'][key]:
            return dict(process=entry['process'],task_ids=entry['task_ids'],artifacts=saved['artifacts'],producer=bound(saved['approval']),self_comparison=saved['serial_self_check'])
    for row in evidence()['accepted'].values():
        if entry['process']==row['process'] and entry['task_ids']==row['task_ids']:return row
    return None


def location(report,key,default):
    row=retained_wave(report,key)
    return Path(row['process']['path']).parent if row else Path(default)


def producer(report,key,current):
    row=retained_wave(report,key)
    return row['producer'] if row else current


def self_review(report,path):
    return any(str(path) in row['artifacts'] and row['self_comparison']['passed']is True for row in evidence()['accepted'].values()) if report.get('probe_recovery_ref')==REUSE_REF else False


def numeric_reference(c,run):
    if c['baseline_unified']['stage']!='PATCH_ENC1':return None
    for row in bound(R5_SOURCE_REF)['accepted'].values():
        if row['task_ids']==[run]:return row['trajectory_ref']
    return None


def verify_r5_source():
    value=bound(R5_SOURCE_REF)
    if value.get('producer_commit')!='6f525489cff125248cbc0b256e9744acf67590da' or value.get('historical_actual')!=dict(adam=42,backward=42,forward=56) or value.get('timing_comparable')is not False:raise ValueError('exact r5 adoption/cost/timing boundary')
    for item in value['protected_refs'].values():
        if ref(item['path'])!=item:raise ValueError('r5 failure/success source changed')
    failure=bound(value['failed_q4_ref'])
    if failure.get('failure_kind')!='observation' or failure.get('resource_admission')is not False or failure.get('returncodes')!=[-15]*4:raise ValueError('r5 q4 remains negative')
    from utils.ch3_type1_upstream import assert_owned_exited
    owners=[]
    for row in value['accepted'].values():
        if row['producer']['commit']!=value['producer_commit'] or row['producer']['resource_contract_ref']!=LEGACY_RESOURCE_CONTRACT_REF:raise ValueError('r5 original producer/policy retained')
        process=bound(row['process'])
        if process.get('returncodes')!=[0] or process.get('resource_admission')is not True:raise ValueError('successful original serial only')
        owners+=process['worker_lifecycles']
    assert_owned_exited(owners)
    return value


def group_review(report,group):return None
def extra_actual(report):
    if report.get('probe_recovery_ref')==REUSE_REF:
        if report.get('scope','').endswith('PATCH_ENC1-probe'):return bound(R5_SOURCE_REF)['historical_actual']
        if report.get('scope','').endswith('URBAN_SUBSET-probe'):return evidence()['failed_actual']
    return dict(adam=0,backward=0,forward=0)
