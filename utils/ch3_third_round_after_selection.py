"""Independent four-H B lifecycle; withdrawn B results never authorize new runs."""
import contextlib,copy,hashlib,hmac,json,os,sys
from pathlib import Path
from utils.ch3_contract import ROOT,digest,profile
from utils.ch3_native_recovery_records import bound,ref,sha,exclusive
from utils import ch3_patchtst_depth_urban6_recovery as parent

SCIENCE_PACKAGE=parent.EVENT_PACKAGE/'patchtst-width-urban-audit-v1/separate-entrypoints-v1/dm4-selection-third-round-v1'
STARTUP_PACKAGE=SCIENCE_PACKAGE/'startup-timeout-recovery-v1'
LEGACY_PACKAGE=STARTUP_PACKAGE/'ast-proof-repair-v1'
PREVIOUS_PACKAGE=LEGACY_PACKAGE/'urban6-four-H-third-v1'
PACKAGE=PREVIOUS_PACKAGE/'reviewed-concurrency-reuse-v1'
PLAN_PACKAGE=PACKAGE/'timexer-epf-four-plus-one-v1/stage-plans-v1'
SUPERSEDED_STARTUP_PROOF_REF={'path':str(STARTUP_PACKAGE/'producer-delta-proof.json'),'sha256':'9bf2b2cedb36fbe6a37ba6c4742b69364311403180170594b1b682e84723d5b9'}
AST_PROOF_ALGORITHM='full-field-ast-json-v1'
STARTUP_RECOVERY_REF={'path':str(PACKAGE/'startup-recovery.contract.v4.json'),'sha256':'9f30997bace0e310fb696a357b00009601aae3a95e08fc6592eff01b11cb90f6'}
CONTRACT_REF={'path':str(PACKAGE/'third-round-reviewed-concurrency.contract.v3.json'),'sha256':'cda6b1a1ebd2e1eb4056899b3c01f36bb06caa73bc1c057f7de585bfd92b1722'}
REUSE_REF=CONTRACT_REF
from utils.ch3_patchtst_width_reproduction import SOURCE_REF
LEGACY_SCOPE='m6-third-round-selected-patchtst-dm4-v1'
ID='m6-third-round-dm4-urban6-four-H-reviewed-concurrency-v1'
ATTEMPT='THIRD-PatchTST-dm4-Urban6-four-H-reviewed-q-r1'
SESSION='ch3-m6-third-fourH-reviewed-q-r1'
RESULT=parent.RESULT.with_name('baseline-unified-v3-ms-seal-m128-recovery1-third-selected-dm4-urban6-four-H-reviewed-q-r1')
LOG=PACKAGE/'B-reviewed-concurrency-launcher.log'
CONCURRENCY_POLICY='reviewed_concurrency_reuse_without_mandatory_probe_v1'
ENTRY=ROOT/'m6_third_round_after_selection_entry.py'
WRAPPER=ROOT/'scripts/ch3/start_third_round_after_selection.sh'
STAGES=('URBAN_SUBSET','EPF_ALL','M_ALL')
CURRENT_STAGE_RUNS={'URBAN_SUBSET':168,'EPF_ALL':35,'M_ALL':168}
APPROVED_AST_CHANGES={'utils/ch3_type1_chain.py':{'dynamic','validate_permit','create_permit','seal_runtime','validate_runtime','validate_boundary_light','seal_boundary','run'},
    'utils/ch3_third_round_after_selection.py':{'contract','completion_record','validate_complete','validate_permit_link'}}
THIRD_ROUND_ONLY=True
INDEPENDENT_SCIENCE_QUEUE=False
COMPLETED_PREFIX=False
COMPLETED_STAGES=SEED_STAGES=()
ACTIVE=False
_R6={}


def contract():
    from utils.ch3_type1_tasks import EPF_GROUPING_POLICY
    v=bound(CONTRACT_REF)
    if (v.get('purpose')!='third_round_user_selected_dm4_urban6_four_H_v1' or v.get('selected_d_model')!=4
        or v.get('stage_order')!=list(STAGES) or v.get('stage_runs')!=CURRENT_STAGE_RUNS
        or v.get('remaining_formal_runs')!=371 or v.get('max_run_epochs')!=5670
        or v.get('urban_horizons')!=[3,6,9,12] or v.get('urban_folds')!=list(range(1,7))
        or v.get('new_queue_completed_from_old_B')!=0 or v.get('adopt_old_urban_probe')is not False
        or v.get('Weather_d_model')!=128
        or v.get('Weather_changed')is not False or v.get('no_depth_redispatch')is not True
        or v.get('no_A_dispatch')is not True or v.get('execution_permitted')is not False
        or v.get('epf_grouping_policy')!=EPF_GROUPING_POLICY
        or v.get('concurrency_policy')!=CONCURRENCY_POLICY or v.get('automatic_probe')is not False
        or v.get('probe_admission_claim')is not False
        or v.get('r6_source_ref')!=SOURCE_REF
        or any(v.get(k)!=x for k,x in resource_binding().items())):
        raise PermissionError('exact user-selected dm4 third-round contract required')
    return v


def resource_binding():return parent.resource_binding()
def config_refs():return contract()['config_refs']


def code_binding():
    return {name:sha(ROOT/name)for name in('utils/ch3_reviewed_concurrency.py','utils/ch3_type1_execution.py','utils/ch3_third_round_after_selection.py','utils/ch3_type1_tasks.py')}


def selected_main(value=None):
    spec=contract();v=bound(spec['selected_main_ref']) if value is None else value
    old=bound(bound(spec['r6_source_ref'])['refs']['round2-amendment/queue/round2-main-results.patchtst-enc2-v1.json'])
    audit=bound(spec['width_source_audit_ref']);selection=bound(spec['selection_ref'])
    if selection.get('selected_d_model')!=4 or selection.get('dm8_adopted')is not False:raise ValueError('explicit uniform dm4 selection only')
    rows={(r['task']['dataset'],r['task']['h']):r for r in audit['stages']['PATCH_DM4']['rows']}
    if len(v.get('cells',[]))!=371 or v.get('effective_counts')!={'MS':203,'M':168,'total':371} or v.get('switched_cells')!=20 or v.get('retained_cells')!=351 or v.get('selection_ref')!=spec['selection_ref']:raise ValueError('exact 371 cells, 20 replacements and 351 preserved')
    changed=0;seen=set()
    for before,after in zip(old['cells'],v['cells']):
        if before['cell_id']!=after['cell_id'] or after['cell_id']in seen:raise ValueError('one-to-one cell slots; no duplicate')
        seen.add(after['cell_id'])
        target=before['model']=='PatchTST'and before['task']=='M'and before['dataset']!='Weather'
        if not target:
            if after!=before:raise ValueError('all 351 retained cells must be field-identical, including Weather4')
            continue
        source=rows[(before['dataset'],before['H'])];result=bound(source['refs']['result.json']);manifest=bound(source['refs']['manifest.json'])
        if (after.get('result_ref')!=source['refs']['result.json'] or after.get('manifest_ref')!=source['refs']['manifest.json']
            or after.get('runtime_ref')!=source['refs']['runtime.json'] or after.get('execution_commit')!=audit['producer_commit']
            or after.get('profile_sha')!=digest(source['profile']) or after.get('protocol_sha')!=result['protocol_sha']
            or after.get('run_id')!=source['task']['id'] or after.get('previous_run_id')!=bound(before['manifest_ref'])['task']['id']
            or after.get('supersedes')!=before['result_ref'] or after.get('d_model')!=4 or after.get('encoder')!=2
            or after.get('selection_ref')!=spec['selection_ref'] or (after.get('mse'),after.get('mae'))!=(result['mse'],result['mae'])
            or manifest['profile']!=source['profile'] or manifest['task']!=source['task']):raise ValueError('exact selected dm4 producer/profile/result source; no mixed width')
        changed+=1
    if changed!=20 or v!=bound(spec['selected_main_ref']):raise ValueError('review-bound selected index bytes/fields')
    return v


def validate_revision(c):
    spec=contract();stage=c['baseline_unified']['stage']
    if stage not in STAGES or c!=bound(spec['config_refs'][stage]):raise ValueError('exact reviewed third-round effective configuration')
    if stage=='URBAN_SUBSET':return validate_urban_four_H(c)
    if stage!='M_ALL':return c
    old=bound(spec['parent_M_ALL_ref']);expected=copy.deepcopy(old);data=bound(c['baseline_unified']['data_ref']);olddata=bound(old['baseline_unified']['data_ref']);changed=0
    for i,t in enumerate(old['tasks']):
        if t['model']!='PatchTST'or t['dataset']=='Weather':continue
        nt=dict(t,group=t['group']+'-dm4-selected-v1',id=t['id']+'-dm4-selected-v1',profile=t['profile']+'-dm4-selected-v1',variant='encoder-2-d_model-4-user-selected')
        p=profile(old,t)
        if [p['structure'][k]for k in('d_model','n_heads','e_layers','d_ff')]!=[16,4,2,128]:raise ValueError('true nonWeather parent')
        p['structure']['d_model']=4;expected['tasks'][i]=nt;del expected['resolved_profiles'][t['id']];expected['resolved_profiles'][nt['id']]=p
        expected['baseline_unified']['parent_refs'].pop(t['id']);expected['baseline_unified']['parent_refs'][nt['id']]=dict(config_ref=spec['parent_M_ALL_ref'],task_id=t['id'],profile_sha=digest(profile(old,t)))
        if any(data[k][t['dataset']][nt['id']]!=olddata[k][t['dataset']][t['id']]for k in('metadata','data_bindings'))or data['mapping'][nt['id']]!=dict(task_id=t['id'],data_ref=old['baseline_unified']['data_ref']):raise ValueError('same data/scaler/window identity')
        if p['training']['scheduler']['name']!='type1_horizon_scaled_v1':raise ValueError('third-round scheduler must remain type1')
        changed+=1
    aliases={x['id']:y['id']for x,y in zip(old['tasks'],expected['tasks'])if x['id']!=y['id']}
    for g in expected['groups']:
        if g['model']=='PatchTST'and g['dataset']!='Weather':
            g['id']+='-dm4-selected-v1'
            for key in('representatives','task_ids'):g[key]=[aliases[t]for t in g[key]]
            g['waves']=[[aliases[t]for t in wave]for wave in g['waves']]
    expected['baseline_unified'].update(data_ref=c['baseline_unified']['data_ref'],direct_parent_ref=spec['parent_M_ALL_ref'],selected_width_ref=spec['selection_ref'])
    if changed!=20 or c!=expected or len(c['tasks'])!=168:raise ValueError('only20 width/identity changes, all other148 profiles unchanged')
    for t in old['tasks']:
        if t['model']=='PatchTST'and t['dataset']=='Weather':
            if t not in c['tasks']or profile(c,t)!=profile(old,t)or [profile(c,t)['structure'][k]for k in('d_model','n_heads','e_layers','d_ff')]!=[128,16,2,256]:raise ValueError('entire original Weather128 profile/task must remain')
    return c


def urban_task(prior):
    from utils import ch3_type1_tasks as s
    t=s.new_task(prior);t['id']+='-urban6-four-H-v1';t['group']+='-urban6-four-H-v1'
    t['profile']=t['id']+'-profile'
    return t


def validate_urban_four_H(c):
    """Reconstruct from exact frozen per-fold/H sources, never test data."""
    from utils import ch3_type1_tasks as s
    from utils.ch3_contract import step_arithmetic
    spec=contract();old=bound(spec['urban_parent_ref']);frozen=bound(spec['frozen_urban_source_ref'])
    if spec['frozen_urban_source_ref']!=s.PARENT_REF:raise ValueError('exact frozen four-H source')
    source=bound(frozen['baseline_unified']['data_ref']);data=bound(c['baseline_unified']['data_ref'])
    expected=copy.deepcopy(old);expected.update(tasks=[],resolved_profiles={},groups=[])
    expected['baseline_unified']['parent_refs']={}
    metadata=dict(purpose='urban_six_fold_four_H_train_validation_metadata_v1',source_refs=[s.PARENT_REF,spec['urban_parent_ref']],
        source_states={str(Path(old['datasets']['UrbanEV']['path'])/name):source['source_states'][str(Path(old['datasets']['UrbanEV']['path'])/name)]for name in('volume.csv','e_price.csv','s_price.csv','weather_central.csv')},test_observations_accessed=False,metadata={'UrbanEV':{}},data_bindings={'UrbanEV':{}},mapping={})
    for model in s.MODELS:
        bank=[]
        for fold in range(1,7):
            for h in (3,6,9,12):
                prior=next(t for t in frozen['tasks']if t['dataset']=='UrbanEV'and(t['model'],t['fold'],t['h'])==(model,fold,h))
                t=urban_task(prior);p=s.inherited_profile(prior)
                if (p['T'],p['pred_len'],p['C'],p['training']['batch'],p['training']['epochs'],p['training']['patience'])!=(12,1,11,128,20,5):raise ValueError('fixed single-offset Urban training')
                if h in(3,12):
                    before=next(x for x in old['tasks']if(x['model'],x['fold'],x['h'])==(model,fold,h))
                    if p!=profile(old,before):raise ValueError('all84 existing scientific profiles unchanged')
                expected['tasks'].append(t);expected['resolved_profiles'][t['id']]=p;bank.append(t['id'])
                expected['baseline_unified']['parent_refs'][t['id']]=dict(config_ref=s.PARENT_REF,task_id=prior['id'],profile_sha=digest(profile(frozen,prior)))
                m=source['metadata']['UrbanEV'][prior['id']]
                metadata['metadata']['UrbanEV'][t['id']]=m;metadata['data_bindings']['UrbanEV'][t['id']]=digest(m)
                metadata['mapping'][t['id']]=dict(task_id=prior['id'],data_ref=frozen['baseline_unified']['data_ref'])
                a=step_arithmetic(c,t)
                if m['window_counts']!=dict(train=a['train_windows'],validation=a['validation_windows'])or p['training']['scheduler']['steps_per_epoch']!=a['train_batches']:raise ValueError('real fold/H window and scheduler counts')
        expected['groups'].append(dict(id=t['group'],dataset='UrbanEV',model=model,input_variant='F4',q=4,representatives=bank,task_ids=bank,
            waves=[bank[i:i+4]for i in range(0,len(bank),4)],equivalence='every fold/H independently represented; no inherited partial-H admission'))
    for rule in expected['baseline_unified']['numeric_policies'].values():rule['horizons']=[3,6,9,12]
    expected['baseline_unified'].update(data_ref=c['baseline_unified']['data_ref'],four_H_parent_ref=spec['urban_parent_ref'])
    if c!=expected or data!=metadata or len({t['profile']for t in c['tasks']})!=168:raise ValueError('exact168 unique profiles and frozen data mappings; no other scientific changes')
    return c


def urban_probe_groups(c):
    from utils import ch3_type1_tasks as s
    return [dict(id=model+'-cross-fold-f1-f6-H3-H6-H9-H12',model=model,
        representatives=[t['id']for t in c['tasks']if t['model']==model],planned_q=4,
        coverage={t['id']:[t['id']]for t in c['tasks']if t['model']==model},
        identities={t['id']:digest(profile(c,t))for t in c['tasks']if t['model']==model},
        equivalence='all24 fold/H representatives independently measured; no old B/r6 admission')for model in s.MODELS]


def urban_probe_budget(c):
    from utils import ch3_type1_tasks as s
    counts={t['id']:s.worker_counts(c,t)for t in c['tasks']}
    return dict(nominal={k:2*sum(v[k]for v in counts.values())for k in('adam','backward','forward')},
        caps={k:3*sum(v[k]for v in counts.values())for k in('adam','backward','forward')},
        nominal_workers=336,max_workers=504,
        fallback='resource-only q4 -> once q2 -> measured serial q1; q2 -> measured serial q1; no refund')


def verify_stopped_B():
    """Read-only revocation source; no checkpoint reads or new runtime credit."""
    v=bound(contract()['withdrawn_B_ref']);inventory=bound(v['retained_artifact_inventory_ref'])
    if (v.get('old_attempt')!='THIRD-PatchTST-enc2-dm4-selected-startup-r2'
        or v.get('old_formal_eligibility')is not False or v.get('old_probe_eligibility_for_new_queue')is not False
        or v.get('may_resume')is not False or v.get('new_queue_completed_from_old_B')!=0):raise ValueError('withdrawn B cannot be adopted or resumed')
    for r in [v['authorization_ref'],v['log_ref'],v['launch_ref'],*v['refs'].values()]:
        if Path(r['path']).is_symlink()or sha(r['path'])!=r['sha256']:raise ValueError('exact withdrawn B stop source bytes')
    for row in inventory['files']:
        p=Path(row['path']);st=p.stat()
        if p.is_symlink()or(st.st_size,st.st_mtime_ns)!=(row['size'],row['mtime_ns']):raise ValueError('withdrawn B artifact stat changed')
        if row['sha256']is not None and sha(p)!=row['sha256']:raise ValueError('withdrawn B artifact bytes changed')
    from utils import ch3_type1_chain as q
    for identity in v['identities']:
        if identity.get('start_ticks')is None:
            if Path('/proc',str(identity['pid'])).exists():raise PermissionError('withdrawn worker PID currently occupied; cannot assert exit identity')
        elif q.same(identity):raise PermissionError('withdrawn B owner still active')
    import subprocess
    if subprocess.run(['tmux','has-session','-t',v['old_session']],capture_output=True).returncode==0:raise PermissionError('withdrawn B session active')
    if not Path(v['refs']['STOP']['path']).exists():raise PermissionError('persistent old B STOP required')
    return v



def startup_contract():
    v=bound(STARTUP_RECOVERY_REF)
    if (v.get('purpose')!='B_pre_controller_timeout_recovery_v1' or v.get('science_scope')!=ID
        or v.get('science_contract_ref')!=CONTRACT_REF or v.get('execution_attempt')!=ATTEMPT
        or v.get('session')!=SESSION or v.get('result_root')!=str(RESULT)
        or v.get('stage_runs')!=CURRENT_STAGE_RUNS or v.get('remaining_formal_runs')!=371
        or v.get('max_run_epochs')!=contract()['max_run_epochs']
        or v.get('scientific_change')is not False or v.get('execution_permitted')is not False
        or v.get('query_policy')!=dict(timeout_seconds=10,max_attempts=3,retry_delay_seconds=2,
            retry_exception='subprocess.TimeoutExpired',maximum_query_and_wait_seconds=34)):
        raise PermissionError('exact B technical startup contract, no scientific change')
    return v


def startup_query_policy():
    from utils import ch3_type1_chain as q
    if q.PROBE_RECOVERY is not sys.modules[__name__] or not ACTIVE:
        raise PermissionError('registered B startup retry policy only')
    p=startup_contract()['query_policy']
    return dict(max_attempts=p['max_attempts'],retry_delay_seconds=p['retry_delay_seconds'])


def historical_startup_failure():
    v=bound(startup_contract()['prior_failure_ref'])
    if (v.get('state')!='PRE_CONTROLLER_START_FAILED' or v.get('scope')!=LEGACY_SCOPE
        or v.get('execution_attempt')!='THIRD-PatchTST-enc2-dm4-selected-r1'
        or v.get('production_commit')!='8de112a2f1ff2117e94c22e5119d86ccaf0bdec7'
        or v.get('formal_completed')!=0 or v.get('new_validation_test')!=0):
        raise ValueError('exact zero-compute historical startup failure')
    for key,h in [('authorization_ref','be6da901ba58f24a73f2656a0467b849c63c49cdfc050ba671c1e32193ede802'),
        ('log_ref','3b8f1adc66153453662ab14a32cde82f0c193bea6f98ff384f04c4445ccc8772'),
        ('launch_ref','121396fb9aee1bb500568c571c87b018a3362a3762f7c9b9cd8df26604f5db8a')]:
        value=v[key]
        if value['sha256']!=h or Path(value['path']).is_symlink()or sha(value['path'])!=h:
            raise ValueError('historical B startup evidence changed: '+key)
    launch=bound(v['launch_ref']);a=bound(v['authorization_ref'])
    if launch['approval']!=v['authorization_ref']or a['closure_commit']!=v['production_commit']:
        raise ValueError('old exact launch/approval/source')
    from utils import ch3_type1_chain as q
    if q.same(launch['wrapper']):raise PermissionError('old startup owner remains active')
    import subprocess
    if subprocess.run(['tmux','has-session','-t',v['session']],capture_output=True).returncode==0:
        raise PermissionError('old startup session remains active')
    if Path(v['log_ref']['path']+'.claimed.json').exists():raise PermissionError('unexpected old startup claimed')
    return v


def assert_startup_token_usable():
    if (PACKAGE/'startup-failure.json').exists():
        raise PermissionError('failed B startup token is invalidated; no repeat arm')


def before_startup_query(a,launch=False):
    from utils import ch3_type1_chain as q
    q.validate_start(a);q.stop_check();assert_startup_token_usable()
    if (PACKAGE/'STOP').exists():raise InterruptedError('persistent startup STOP')
    verify_source();cs=q.validate_amd_configs(q.configs())
    from ch3_runner import environment_binding
    from utils.ch3_m6 import source_states
    if bound(q.ENVIRONMENT_REF)['environment']!=environment_binding():raise ValueError('fixed startup environment changed')
    for c in cs.values():
        if source_states(c)!=bound(c['baseline_unified']['data_ref'])['source_states']:raise ValueError('startup data source changed')
        import subprocess
        for src in c['sources'].values():
            if (subprocess.check_output(['git','-C',src['repository'],'rev-parse','HEAD'],text=True).strip()!=src['commit']
                or any(sha(p)!=h for p,h in src['files'].items())):raise ValueError('locked author source changed')
    if RESULT.exists()or RESULT.is_symlink():raise FileExistsError('exclusive third-round result already exists')
    if not launch:
        for p in(LOG,Path(str(LOG)+'.launch.json'),Path(str(LOG)+'.claimed.json')):
            if p.exists()or p.is_symlink():raise FileExistsError('retained new startup log/launch; no repeat')
        import subprocess
        if subprocess.run(['tmux','has-session','-t',SESSION],capture_output=True).returncode==0:
            raise FileExistsError('retained new B startup session')


def record_startup_failure(start_ref,exc,launch):
    """After a verified owned launch only; retain failure and invalidate its token."""
    from utils import ch3_type1_chain as q,ch3_event_resources as event
    from datetime import datetime,timezone
    event.invalidate_startup()
    if RESULT.exists():raise PermissionError('startup failure cannot replace a controller failure')
    a=bound(start_ref);q.validate_start(a)
    if launch['approval']!=start_ref or launch['scope']!=ID:raise PermissionError('exact current owned launch')
    audit=event.query_audit()
    body=dict(state='PRE_CONTROLLER_START_FAILED',scope=ID,execution_attempt=ATTEMPT,
        failure_time_UTC=datetime.now(timezone.utc).isoformat(),error_type=(audit or{}).get('error_type',type(exc).__name__),
        controller_error_type=type(exc).__name__,error=str(exc),authorization_ref=start_ref,
        log_capture=dict(path=str(LOG),bytes=LOG.stat().st_size,prefix_sha256=sha(LOG)),
        launch_ref=ref(str(LOG)+'.launch.json'),startup_recovery_ref=STARTUP_RECOVERY_REF,
        query_audit=audit,formal_completed=0,new_validation_test=0,launch_invalidated=True,
        resource_fallback=False,prior_failure_ref=startup_contract()['prior_failure_ref'])
    return exclusive(PACKAGE/'startup-failure.json',body)


def startup_status():
    history=historical_startup_failure();r=dict(previous_startup_failures=[history],
        execution_attempt=ATTEMPT,permission_exists=(PACKAGE/'start-review.json').exists(),pre_controller_start_failed=False)
    path=PACKAGE/'startup-failure.json'
    if path.exists():
        failure=bound(ref(path))
        if (failure.get('execution_attempt')!=ATTEMPT or failure.get('scope')!=ID
            or failure.get('startup_recovery_ref')!=STARTUP_RECOVERY_REF or failure.get('launch_invalidated')is not True):
            raise ValueError('current B startup failure identity')
        for key in('authorization_ref','launch_ref'):bound(failure[key])
        capture=failure['log_capture'];raw=Path(capture['path']).read_bytes()
        if hashlib.sha256(raw[:capture['bytes']]).hexdigest()!=capture['prefix_sha256']:raise ValueError('startup failure log prefix changed')
        r.update(state='PRE_CONTROLLER_START_FAILED',pre_controller_start_failed=True,startup_failure_ref=ref(path),startup_failure=failure,current_log_ref=ref(LOG))
    return r


def code_binding():
    return {p:sha(ROOT/p)for p in('utils/ch3_third_round_after_selection.py','m6_third_round_after_selection_entry.py','scripts/ch3/start_third_round_after_selection.sh','utils/ch3_type1_chain.py','utils/ch3_probe_schema_recovery.py','m6_type1_followup_entry.py','utils/ch3_type1_summary.py','utils/ch3_event_resources.py')}


def protected_ast_fingerprint(node):
    """All semantic fields, including empty/None; stable on Python 3.11/3.13.

    Older parsers omit the type_params field entirely. Its absent form means
    an empty list, never an exemption for nonempty type parameters. Location
    attributes remain excluded, as in the original AST proof.
    """
    import ast
    def encode(value):
        if isinstance(value,ast.AST):
            fields=dict(ast.iter_fields(value))
            if isinstance(value,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                fields.setdefault('type_params',[])
            return [type(value).__name__,[[key,encode(item)]for key,item in sorted(fields.items())]]
        if isinstance(value,list):return ['list',[encode(item)for item in value]]
        if value is None or value is Ellipsis or isinstance(value,(str,bytes,bool,int,float,complex)):
            return [type(value).__name__,repr(value)]
        raise TypeError('unsupported semantic AST field: '+type(value).__name__)
    payload=json.dumps(encode(node),ensure_ascii=False,separators=(',',':'))
    return hashlib.sha256(payload.encode()).hexdigest()


def verify_production_inheritance():
    """Original66 plus exact reviewed execution deltas; mathematics/guard unchanged."""
    import ast,subprocess
    proof=bound(ref(PACKAGE/'producer-delta-proof.v11.json'))
    previous_ref=startup_contract()['previous_producer_proof_ref'];previous=bound(previous_ref)
    registry=lambda value:{name:set(symbols)for name,symbols in value['protected_AST'].items()}
    if (proof.get('purpose')!='B_reviewed_concurrency_exact_delta_v1'
        or proof.get('AST_algorithm')!=AST_PROOF_ALGORITHM or registry(proof)!=registry(previous)
        or sum(map(len,proof['protected_AST'].values()))!=66
        or proof.get('parent_commit')!='0a0049a8fbe3505c7f3ca719b8c392df071a0e2c'
        or proof.get('previous_proof_ref')!=previous_ref or proof.get('parent_AST')!=previous['protected_AST']
        or proof.get('model_math_changed')is not False or proof.get('scientific_change')is not False
        or set(proof['protected_exact'])!=set(previous['protected_exact'])
        or proof.get('approved_AST_changes')!={k:sorted(v)for k,v in APPROVED_AST_CHANGES.items()}):
        raise ValueError('exact reviewed concurrency delta with original66 protection')
    changed={name:{symbol for symbol,h in symbols.items()if h!=previous['protected_AST'][name][symbol]}
        for name,symbols in proof['protected_AST'].items()}
    changed={name:symbols for name,symbols in changed.items()if symbols}
    if changed!=APPROVED_AST_CHANGES:raise ValueError('only explicit reviewed scheduling AST deltas allowed')
    exact_changes={k for k,h in proof['protected_exact'].items()if h!=previous['protected_exact'][k]}
    if exact_changes!={'ch3_runner.py','utils/ch3_type1_tasks.py','utils/ch3_type1_summary.py'}:
        raise ValueError('original math, adapter, guard and restricted tools remain byte-identical')
    expected_files=set(previous['after_code'])|set(previous['protected_exact'])|{'utils/ch3_type1_execution.py','utils/ch3_reviewed_concurrency.py'}
    if set(proof['after_code'])|set(proof['protected_exact'])!=expected_files:
        raise ValueError('exact original and added execution source coverage')
    for name,h in {**proof['protected_exact'],**proof['after_code']}.items():
        if sha(ROOT/name)!=h:raise ValueError('exact reviewed source changed: '+name)
    for name,symbols in proof['protected_AST'].items():
        now=ast.parse((ROOT/name).read_text());parent_tree=ast.parse(subprocess.check_output(['git','show',proof['parent_commit']+':'+name],cwd=ROOT,text=True))
        actual={n.name:protected_ast_fingerprint(n)for n in now.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        prior={n.name:protected_ast_fingerprint(n)for n in parent_tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        if any(actual.get(k)!=h or prior.get(k)!=proof['parent_AST'][name][k]for k,h in symbols.items()):raise ValueError('protected mathematics/runtime function changed: '+name)
    # The three changed formerly exact files receive full function-level delta
    # protection, so author computations outside the declared control edits stay fixed.
    for name,allowed in {'ch3_runner.py':{'code_binding','formal_identity'},'utils/ch3_type1_tasks.py':{'formal_waves','plan','probe_groups','_probe_groups','historical_epf_probe_groups_v1'},'utils/ch3_type1_summary.py':{'result_index'}}.items():
        before=ast.parse(subprocess.check_output(['git','show',proof['parent_commit']+':'+name],cwd=ROOT,text=True));after=ast.parse((ROOT/name).read_text())
        table=lambda tree:{n.name:protected_ast_fingerprint(n)for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        x,y=table(before),table(after)
        if {k for k in set(x)|set(y)if x.get(k)!=y.get(k)}!=allowed:raise ValueError('only approved control functions changed: '+name)
    from tools.restricted_regression.run_restricted import verify_bundle
    if verify_bundle()!=proof['bundle_sha']or proof['bundle_sha']!=previous['bundle_sha']:raise ValueError('original restricted bundle changed')
    return proof


def verify_stopped_four_H():
    v=bound(contract()['stopped_four_H_ref']);inventory=bound(v['retained_artifact_inventory_ref'])
    if (v.get('old_attempt')!='THIRD-PatchTST-enc2-dm4-Urban6-four-H-r1'
        or v.get('old_formal_eligibility')is not False or v.get('old_probe_eligibility_for_new_queue')is not False
        or v.get('may_resume')is not False or v.get('new_queue_completed_from_old_attempt')!=0
        or v.get('new_queue_admission_credit_from_old_attempt')!=0):raise PermissionError('stopped four-H lifecycle never grants new credit')
    for row in inventory['files']:
        p=Path(row['path']);st=p.stat()
        if p.is_symlink()or(st.st_size,st.st_mtime_ns)!=(row['size'],row['mtime_ns']):raise ValueError('stopped four-H artifact identity changed')
        if row['sha256']is not None and sha(p)!=row['sha256']:raise ValueError('stopped four-H artifact bytes changed')
    from utils import ch3_type1_chain as q
    if any(q.same(i)for i in v['identities']):raise PermissionError('stopped four-H owner remains active')
    import subprocess
    if subprocess.run(['tmux','has-session','-t',v['old_session']],capture_output=True).returncode==0:raise PermissionError('stopped four-H session remains active')
    if not Path(v['STOP_ref']['path']).exists()or sha(v['STOP_ref']['path'])!=v['STOP_ref']['sha256']:raise PermissionError('persistent stopped four-H STOP required')
    return v


def verify_concurrency_sources():
    from utils import ch3_reviewed_concurrency as concurrency,ch3_type1_chain as q
    plan=bound(contract()['concurrency_plan_ref']);history=bound(plan['history_ref'])
    stopped_root=Path(bound(contract()['stopped_four_H_ref'])['STOP_ref']['path']).parents[2]
    for c in q.configs().values():concurrency.stage_plan(c)
    for value in history['sources'].values():
        report=bound(value['report_ref']);decision=report['decisions'][value['group']]
        if decision.get('status')!='Passed'or decision.get('concurrency')!=value['recorded_q']:raise ValueError('historical measured source only')
        for r in(value['waves']+value['worker_config_refs']+value['formal_record_refs']):
            if Path(r['path']).resolve().is_relative_to(stopped_root):raise PermissionError('stopped four-H evidence cannot grant new concurrency credit')
            v=bound(r)
            if r in value['waves']and(v.get('failure')is not None or any(v['returncodes'])):raise ValueError('failed historical wave cannot be presented as success')
            if r in value['formal_record_refs']and v.get('technical_complete')is not True:raise ValueError('successful historical formal evidence required')
    return plan


def verify_source():
    historical_startup_failure()
    verify_stopped_B()
    verify_stopped_four_H();verify_concurrency_sources()
    spec=contract();source=bound(SOURCE_REF);audit=bound(spec['width_source_audit_ref']);complete=bound(spec['width_complete_ref'])
    if audit.get('technical_complete')is not True or audit.get('producer_commit')!='7d9fec8c5f66e9f38dbf396bf8d1136276b8e78e'or audit.get('new_test_calls')!=0 or complete.get('technical_complete')is not True or complete.get('total_runs')!=40:raise ValueError('real independent width40 completion and read-only audit')
    for k in('PATCH_ENC1','PATCH_ENC2','probe/URBAN_SUBSET/complete.json','queue/controller/failure.json'):
        v=bound(source['refs'][k])
        if k.startswith('PATCH_')and(v.get('technical_complete')is not True or len(v['task_ids'])!=24):raise ValueError('retained48 original completion')
    if bound(source['refs']['queue/controller/failure.json'])['error']!="ValueError('probe artifact path/symlink outside exact scope')":raise ValueError('old r6 failure must remain negative')
    from utils.ch3_type1_upstream import assert_owned_exited
    from utils.ch3_patchtst_width_reproduction import RESULT as A_RESULT,SESSION as A_SESSION
    owners=list(source['owned_exited'])+[bound(ref(A_RESULT/'queue/controller/controller.json'))['owner']]
    assert_owned_exited(owners)
    import subprocess
    for session in(parent.SESSION,A_SESSION):
        if subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:raise ValueError('source lifecycle still active')
    selected_main();bound(spec['selected_boundary_ref']);verify_production_inheritance()
    from utils import ch3_type1_chain as q,ch3_type1_tasks as s
    for stage,c in q.configs().items():
        if bound(ref(PLAN_PACKAGE/(stage.lower()+'-plan.json')))!=s.plan(c):raise ValueError('current stage plan projection must match reviewed concurrency')
    return spec


def anchors():
    v=contract()
    return dict(execution_attempt=ATTEMPT,startup_recovery_ref=STARTUP_RECOVERY_REF,contract_ref=CONTRACT_REF,source_ref=SOURCE_REF,selection_ref=v['selection_ref'],selected_main_ref=v['selected_main_ref'],selected_boundary_ref=v['selected_boundary_ref'],withdrawn_B_ref=v['withdrawn_B_ref'],old_B_completed_credit=0,stopped_four_H_ref=v['stopped_four_H_ref'],stopped_four_H_completed_credit=0,concurrency_policy=CONCURRENCY_POLICY,epf_grouping_policy=v['epf_grouping_policy'],concurrency_plan_ref=v['concurrency_plan_ref'],automatic_probe=False,probe_admission_claim=False,urban_horizons=[3,6,9,12],producer_delta_ref=ref(PACKAGE/'producer-delta-proof.v11.json'),candidate_config_refs=config_refs(),completed_old_new_formal=196,completed_depth_formal=48,completed_independent_width_formal=40,expected_remaining_formal=371,approved_third_formal=371,Weather_d_model=128,selected_nonWeather_d_model=4,**resource_binding())


def status():
    # Read-only state: no GPU query, source numerical replay or old checkpoint scan.
    verify_source()
    return dict(state='FOUR_H_B_WAIT_REVIEW_CLOSURE_AND_PERMISSION',anchors=anchors(),selected_nonWeather_d_model=4,remaining_formal_runs=371,READY_FOR_GPU_EXECUTION=False,result_review='pending',result_exists=RESULT.exists(),log_exists=LOG.exists(),permission_exists=(PACKAGE/'start-review.json').exists(),previous_startup_failures=[historical_startup_failure()],withdrawn_B=verify_stopped_B())


def import_ms():
    from utils import ch3_type1_chain as q
    verify_source()
    body=dict(state='SELECTED_DM4_371_AND_COMPLETED_SOURCES_ADOPTED',READY_FOR_GPU_EXECUTION=True,anchors=anchors(),boundaries=dict(selected_round2=contract()['selected_boundary_ref']),handoff_scope=ID,successor_owner=q.owner(),adoption_commit=q.closure(),result_review='pending')
    secret=os.environ.get(q.SECRET)
    if not secret:raise PermissionError('new B owned adoption lifecycle required')
    body['mac']=hmac.new(secret.encode(),digest(body).encode(),hashlib.sha256).hexdigest()
    return exclusive(q.CONTROL/'upstream-technical-boundary.json',body)


def validate_permit_link(c,a,probe):
    if probe:raise PermissionError('reviewed formal queue has no automatic probe authorization')
    if a.get('execution_attempt')!=ATTEMPT or a.get('probe_recovery_ref')!=REUSE_REF or c!=bound(config_refs()[c['baseline_unified']['stage']]):raise PermissionError('exact selected B lifecycle/config/permit')
    if a.get('base287_boundary_ref')is not None or a.get('round2_boundary_ref')is not None:raise PermissionError('historical MACs cannot authorize B')


def monitor_binding(configs):
    from utils import ch3_type1_chain as q
    expected=resource_binding()
    if not ACTIVE or q.PROBE_RECOVERY is not sys.modules[__name__]:raise PermissionError('registered B resource context')
    permits=[]
    for cfg in configs:
        a=cfg.get('approval')if cfg.get('purpose')=='ch3_probe'else bound(cfg['formal_permit_ref'])
        if cfg.get('type1_scope')!=ID or cfg.get('probe_schema_recovery_ref')!=REUSE_REF or any(cfg.get(k)!=v or a.get(k)!=v for k,v in expected.items()):raise PermissionError('bound B worker resource mode/permit')
        c=bound(config_refs()[cfg['unified_stage']])
        if cfg['protocol_sha']!=digest(c)or cfg['task']not in a['authorized_task_ids']:raise PermissionError('exact B worker protocol/task')
        q.validate_permit(c,a,cfg['purpose']=='ch3_probe',worker=True);permits.append(a)
    from utils.ch3_event_resources import validate_startup
    for a in permits:validate_startup(a['startup_hardware_ref'],live=True)
    if any(cfg.get('startup_hardware_ref')!=permits[0]['startup_hardware_ref']for cfg in configs):raise PermissionError('one verified startup grant')
    return expected


def worker_metadata(c):
    v=contract();values=[CONTRACT_REF,SOURCE_REF,v['selection_ref'],v['selected_main_ref'],v['selected_boundary_ref'],v['width_source_audit_ref'],v['parent_M_ALL_ref'],v['urban_parent_ref'],v['frozen_urban_source_ref'],resource_binding()['resource_contract_ref'],ref(PACKAGE/'producer-delta-proof.v11.json'),v['concurrency_plan_ref']]
    values.extend([STARTUP_RECOVERY_REF,startup_contract()['prior_failure_ref'],startup_contract()['previous_producer_proof_ref']])
    values.append(bound(v['parent_M_ALL_ref'])['baseline_unified']['data_ref'])
    for r in config_refs().values():values.extend([r,bound(r)['baseline_unified']['data_ref']])
    return values


@contextlib.contextmanager
def historical_urban_context():
    """Read saved original producer evidence under its original schema; no launch."""
    from utils import ch3_type1_tasks as s,ch3_type1_chain as q
    saved={k:getattr(s,k)for k in _R6};recovery=q.PROBE_RECOVERY
    try:
        for k,v in _R6.items():setattr(s,k,v)
        q.PROBE_RECOVERY=parent
        yield
    finally:
        for k,v in saved.items():setattr(s,k,v)
        q.PROBE_RECOVERY=recovery


def retained_refs(report):
    if report.get('probe_recovery_ref')not in(None,REUSE_REF):raise PermissionError('foreign readonly registry')
    if report.get('scope','').endswith('URBAN_SUBSET-probe')and report['scope']!=ID+'-URBAN_SUBSET-probe':raise PermissionError('exact four-H B stage only')
    # All168 representatives are freshly admitted in this lifecycle. Neither
    # stopped B nor the partial-H r6 registry grants external artifact access.
    return {}


def adopt_urban_probe(receipts):
    raise PermissionError('four-H requires fresh complete Urban admission; stopped B/r6 probe cannot authorize it')


def validate_saved_probe(c,report):
    if report.get('adoption_only')or report.get('adopted_source_ref'):raise PermissionError('no historical probe completion adoption in new four-H lifecycle')
    return False


def location(report,key,default):return Path(default)
def producer(report,key,current):return current
def retained_payload(c,point):return False
def self_review(report,path):return False
def group_review(report,group):return None
def continuation_actions():
    from utils import ch3_type1_chain as q
    return {'VERIFY_ADOPT_SELECTED_DM4_SOURCES':lambda _:import_ms()}


def completion_counts():
    return dict(total_runs=615,third_round_runs=371,adopted_new_formal_runs=244,executed_new_formal_runs=371,independent_width_formal_runs=40,withdrawn_B_completed_credit=0)


def completion_record(receipts):
    from utils import ch3_type1_chain as q
    v=contract()
    return dict(scope=ID,technical_complete=True,result_review='pending',stage_boundaries={s:receipts[q.STAGE_STATES[s][3]]for s in STAGES},selected_main_ref=contract()['selected_main_ref'],selection_ref=contract()['selection_ref'],adopted_original_new_formal_runs=196,adopted_depth_formal_runs=48,completed_independent_width_formal_runs=40,executed_third_formal_runs=371,total_third_formal_runs=371,withdrawn_B_ref=contract()['withdrawn_B_ref'],withdrawn_B_completed_credit=0,stopped_four_H_ref=contract()['stopped_four_H_ref'],stopped_four_H_completed_credit=0,concurrency_policy=CONCURRENCY_POLICY,epf_grouping_policy=v['epf_grouping_policy'],concurrency_plan_ref=contract()['concurrency_plan_ref'],probe_admission_claim=False,**resource_binding())


def validate_complete(v):
    from utils import ch3_type1_chain as q
    if v.get('scope')!=ID or v.get('technical_complete')is not True or v.get('result_review')!='pending' or v.get('executed_third_formal_runs')!=371 or v.get('total_third_formal_runs')!=371 or v.get('withdrawn_B_completed_credit')!=0 or v.get('withdrawn_B_ref')!=contract()['withdrawn_B_ref'] or v.get('selected_main_ref')!=contract()['selected_main_ref'] or v.get('selection_ref')!=contract()['selection_ref']or set(v.get('stage_boundaries',{}))!=set(STAGES):raise ValueError('exact four-H B technical complete; withdrawn B excluded; result review separate')
    if (v.get('concurrency_policy')!=CONCURRENCY_POLICY or v.get('epf_grouping_policy')!=contract()['epf_grouping_policy']or v.get('concurrency_plan_ref')!=contract()['concurrency_plan_ref']or v.get('probe_admission_claim')is not False or v.get('stopped_four_H_ref')!=contract()['stopped_four_H_ref']or v.get('stopped_four_H_completed_credit')!=0):raise PermissionError('current reviewed queue only; no stopped four-H/probe credit')
    for stage,r in v['stage_boundaries'].items():q.validate_boundary_light(r,stage)
    return v


def activate():
    global ACTIVE,_R6
    if ACTIVE:return
    parent.activate()
    from utils import ch3_type1_tasks as s,ch3_type1_chain as q,ch3_round2_amendment as amend
    names=('ID','PROTOCOL','STAGES','RESULT','PACKAGE','file','package','context','plan','probe_groups','probe_budget','validate','expected_tasks','parent_ref','selected')
    _R6={k:getattr(s,k)for k in names};original_context=s.context
    s.ID=ID;s.STAGES=STAGES;s.RESULT=RESULT;s.PACKAGE=PACKAGE
    s.file=lambda stage:Path(config_refs()[stage]['path']);s.package=lambda stage:SCIENCE_PACKAGE
    # Keep the historical source generator's parent; the new direct parent is
    # independently bound in validate_revision/worker_metadata.
    s.parent_ref=_R6['parent_ref']
    s.validate=validate_revision;s.expected_tasks=lambda stage:bound(config_refs()[stage])['tasks']
    old_selected,old_groups,old_budget=s.selected,s.probe_groups,s.probe_budget
    s.selected=lambda stage:[t for t in s.parent()['tasks']if t['dataset']=='UrbanEV'and t['fold']in range(1,7)and t['h']in(3,6,9,12)]if stage=='URBAN_SUBSET'else old_selected(stage)
    s.probe_groups=lambda c:urban_probe_groups(c)if c['baseline_unified']['stage']=='URBAN_SUBSET'else old_groups(c)
    s.probe_budget=lambda c:dict(nominal=dict(adam=0,backward=0,forward=0),caps=dict(adam=0,backward=0,forward=0),nominal_workers=0,max_workers=0,automatic_probe=False,manual_diagnostics='only on explicit user request under an applicable independent diagnostic authorization')
    s.package=lambda stage:PLAN_PACKAGE
    s.context=lambda c:dict(original_context(c),fixture=PACKAGE/c['baseline_unified']['stage']/'fixtures')
    # Original protocol/profile identities remain in unchanged Urban/EPF configs.
    q.STAGE_STATES={stage:tuple(stage+suffix for suffix in('_PROBE','_AUTO_AUDIT','_FORMAL','_BOUNDARY'))for stage in STAGES}
    q.STATES=('VERIFY_ADOPT_SELECTED_DM4_SOURCES','FOLLOWUP_PROTOCOL_PREFLIGHT',*(state for stage in STAGES for state in q.STAGE_STATES[stage][2:]),'COMPLETE')
    q.CONTROL=RESULT/'queue/controller';q.LOG=LOG;q.SESSION=SESSION;q.ENTRY=ENTRY;q.WRAPPER=WRAPPER;q.PROBE_RECOVERY=sys.modules[__name__]
    amend.RESULT=RESULT/'round2-amendment';amend.summary=selected_main
    ACTIVE=True


def cli(argv=None):
    activate()
    from m6_type1_followup_entry import cli as shared_cli
    if argv is None:return shared_cli()
    previous=sys.argv
    try:
        sys.argv=[str(ENTRY),*argv];return shared_cli()
    finally:sys.argv=previous
