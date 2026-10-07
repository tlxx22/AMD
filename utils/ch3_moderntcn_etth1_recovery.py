"""One reviewed-policy candidate and exact retained M_BASE prefix; no generic retry."""
import ast,copy,hashlib,hmac,json,os,subprocess,sys
from pathlib import Path
from utils.ch3_contract import ROOT,digest
from utils.ch3_native_recovery_records import bound,ref,sha,exclusive
from utils.ch3_moderntcn_etth1_diagnostic import PACKAGE,BASE,CONFIG as COLLECTION_FILE,CONFIG_SHA

SOURCE_REF=dict(path=str(PACKAGE/'numeric-source-anchors.json'),sha256='634ee1b02c10313395fc44fd0e1b350faab299245069f21713875d138faf65a4')
POLICY_REF=dict(path=str(PACKAGE/'policy-revision.json'),sha256='02574234af8f826c2949366166d6f063c7ad0d7687a859df478a3fdf03367d00')
POLICY_PACKAGE=PACKAGE/'baseline-numeric-admission-v1'
ALL_M_POLICY_REF=dict(path=str(POLICY_PACKAGE/'policy-extension.json'),sha256='5c62a148ba9e7c76d3d159fcf5a30718f536f6e2552c2ccac6d252a2ac33bf90')
CONFIG_REF=dict(path=str(ROOT/'configs/ch3_round2_m_batch128_etth1_numeric_r1.json'),sha256='a1b763e88fdcdffe9ed82d819f55f98d68774b77d8befa1abdf58fbf14284d0e')
CONFIG_REFS=dict(M_BASE=CONFIG_REF,
    M_AMEND=dict(path=str(ROOT/'configs/ch3_round2_m_amend1_recovery1_moderntcn2e4_r1.json'),sha256='e00288b5c188008c9304940cebd559952d983ce6c5c07a35eb233b78f7330e15'),
    M_ALL=dict(path=str(ROOT/'configs/ch3_type1_m_all_v3_recovery1_moderntcn2e4_r1.json'),sha256='44955f95f4b0a65f20a2c8a0575f51e68c192cc27d62a70368a0e34ea99e5333'),
    URBAN_SUBSET=dict(path=str(ROOT/'configs/ch3_type1_urban_subset_v3_recovery1_numeric_admission_v1.json'),sha256='6a695caad7aefa0c4b760ab734ccf5eeb4e98d3c49b1aabfccb93a42a92826e1'),
    EPF_ALL=dict(path=str(ROOT/'configs/ch3_type1_epf_all_v3_recovery1_numeric_admission_v1.json'),sha256='d4e76d9d327cb1469dfff0e9e6545120fa1305076583521084a7d8903a16b69e'),
)
PRIOR_REUSE_REF=dict(path=str(PACKAGE/'numeric-reuse-evidence.json'),sha256='e5398240898fc54cc9ddd68e57a7e25d4790b8bf3abd21b2a2fed95feeed8a3a')
INCLUSION_REF=dict(path=str(POLICY_PACKAGE/'policy-inclusion.json'),sha256='b9ec9c93ef845787e81314d49c79e87ae6d41827765bc837cb837befdacf800d')
REUSE_REF=dict(path=str(POLICY_PACKAGE/'numeric-reuse-evidence.json'),sha256='fe9f16b516cf4a94b69551aae782f168051f3f17acb57233b20e1e2aec3208ca')
LEGACY_POLICY_REFS=(
    dict(path=str(PACKAGE/'moderntcn-all-m-policy-v1/policy-extension.json'),sha256='17f4ca6c24ef6abff43259490e24b296dbd9f89b5f6b011f90ef119d8aa0c179'),
    dict(path=str(PACKAGE/'moderntcn-all-m-policy-v2/policy-extension.json'),sha256='805069ff11623c2f1158fcebbe62d32bbc47a6e795c2a8dfe89e37cba45165d1'))
OLD_RESULT=ROOT.parent/'amd-execution-evidence/m6/m6-formal-launch-dhozikhu/baseline-unified-v3-ms-seal-m128-recovery1-probe-schema-r1'
RESULT=OLD_RESULT.with_name('baseline-unified-v3-ms-seal-m128-recovery1-moderntcn-etth1-numeric-r2')
ATTEMPT='M_BASE-ModernTCN-ETTh1-numeric-r2'
ENTRY=ROOT/'m6_moderntcn_etth1_recovery_entry.py'
WRAPPER=ROOT/'scripts/ch3/start_moderntcn_etth1_recovery.sh'
SESSION='ch3-m6-m128-moderntcn-etth1-numeric-r2'
REPAIR_PACKAGE=POLICY_PACKAGE/'import-path-repair-v1'
LOG=REPAIR_PACKAGE/'followup-launcher.log'
PARENT_TECHNICAL_FAILURE_REF=dict(path=str(REPAIR_PACKAGE/'r1-technical-failure-anchors.json'),sha256='3f22526dbcb544e6bc47caa5fd9de8296a4cf1f8b6d325c1f17962389aa2b989')
ACTIVE=False

def validate_revision(c):
    """One shared materialized default table; historical revisions stay explicit."""
    from utils.ch3_amend_ett_identity_recovery import weather_base
    c=weather_base(c)
    from utils.ch3_contract import BASELINE_NUMERIC_ADMISSION_ID,BASELINE_NUMERIC_DEFAULTS,materialize_baseline_numeric_admission,validate_baseline_numeric_registry
    b=c['baseline_unified'];stage=b.get('stage');value=b.get('numeric_revision_ref')
    if b.get('numeric_admission_version')is not None:
        if stage not in CONFIG_REFS or value!=ALL_M_POLICY_REF:raise ValueError('only the registered five-stage routine numeric revision')
        v=bound(ALL_M_POLICY_REF)
        if v['purpose']!='user_authorized_baseline_numeric_admission_v1' or v['admission_version']!=BASELINE_NUMERIC_ADMISSION_ID or v['default_table']!=BASELINE_NUMERIC_DEFAULTS or v['user_authorized']is not True:
            raise ValueError('exact user-approved routine numeric default table')
        old=bound(v['stages'][stage]['old_config_ref']);want=materialize_baseline_numeric_admission(old);want['baseline_unified']['numeric_revision_ref']=value
        if want['baseline_unified']['numeric_policies']!=v['stages'][stage]['evaluation_policies']or c!=want:raise ValueError('only numeric policy/version/reference changes; science and identity unchanged')
        return validate_baseline_numeric_registry(c)
    # These frozen data interpretations do not authorize old permits at a new HEAD.
    if value in LEGACY_POLICY_REFS:
        v=bound(value)
        if stage not in v['stages']:raise ValueError('historical revision stage')
        old=bound(v['stages'][stage]['old_config_ref']);want=copy.deepcopy(old)
        want['baseline_unified']['numeric_policies'].update(v['stages'][stage]['evaluation_policies'])
    elif value==POLICY_REF and stage=='M_BASE':
        v=bound(POLICY_REF);old=bound(v['old_config_ref']);want=copy.deepcopy(old)
        want['baseline_unified']['numeric_policies']['ModernTCN-ETTh1']=v['evaluation_policy']
    else:raise ValueError('unregistered numeric revision; no exact fallback')
    want['baseline_unified']['numeric_revision_ref']=value
    if c!=want:raise ValueError('historical numeric revision bytes/contract mismatch')
    return want['baseline_unified']['numeric_policies']

def activate():
    global ACTIVE
    if ACTIVE:return
    from utils import ch3_type1_tasks as s,ch3_type1_chain as q,ch3_ms_seal_recovery as ms,ch3_round2_amendment as amendment
    ACTIVE=True;s.RESULT=RESULT;ms.RESULT=RESULT;ms.M_FILE=Path(CONFIG_REF['path']);amendment.RESULT=RESULT/'round2-amendment'
    original_package=s.package;original_file=s.file
    s.package=lambda stage:POLICY_PACKAGE if stage in CONFIG_REFS else original_package(stage)
    s.file=lambda stage:Path(CONFIG_REFS[stage]['path'])if stage in CONFIG_REFS else original_file(stage)
    original_context=s.context
    def context(c):return dict(original_context(c),fixture=PACKAGE/c['baseline_unified']['stage']/'fixtures')
    s.context=context
    q.CONTROL=RESULT/'queue/controller';q.LOG=LOG;q.SESSION=SESSION;q.ENTRY=ENTRY;q.WRAPPER=WRAPPER;q.PROBE_RECOVERY=sys.modules[__name__];q.QUEUE_LOCK=PACKAGE/'queue.lock'

def anchors():
    return dict(recovery='specific_ModernTCN_ETTh1_M_numeric_failure',source_anchors_ref=SOURCE_REF,probe_reuse_ref=REUSE_REF,
        policy_revision_ref=POLICY_REF,all_m_policy_ref=ALL_M_POLICY_REF,candidate_config_refs=CONFIG_REFS,original_training_commit='df6a16403e10d51097db8c88829909c533d15652',
        retained_producer_commit=BASE,expected_runs=dict(MS=203,M=84,total=287),expected_remaining_formal=427,
        execution_attempt=ATTEMPT,parent_technical_failure_ref=PARENT_TECHNICAL_FAILURE_REF)

def evidence():
    v=bound(REUSE_REF);policy=bound(POLICY_REF);confirmation=bound(v['confirmation_ref'])
    previous=bound(PRIOR_REUSE_REF);inclusion=bound(INCLUSION_REF)
    if inclusion['collection_adoption_ref']!=PRIOR_REUSE_REF or inclusion['source_anchors_ref']!=SOURCE_REF or inclusion['evaluation_config_ref']!=CONFIG_REF or inclusion['evaluation_policy_ref']!=ALL_M_POLICY_REF:raise ValueError('exact old/new policy inclusion references')
    expected=adopt_numeric_defaults(previous,bound(CONFIG_REF),inclusion)
    if v!=expected:raise ValueError('exact prior measurements plus metadata-proven default policy inclusion')
    if v['purpose']!='M_BASE_ModernTCN_ETTh1_scoped_numeric_adoption_v1'or v['source_anchors_ref']!=SOURCE_REF or v['policy_ref']!=POLICY_REF or v['config_ref']!=CONFIG_REF or v['producer_commit']!=BASE:
        raise ValueError('exact registered numeric failure/reuse identity')
    if v['confirmation_ref']!=policy['confirmation_ref']or confirmation['policy_revision_conditions_met']is not True or confirmation['whole_M_probe_admission']is not False:
        raise ValueError('short confirmation is not whole-stage admission')
    if (v['retained_groups'],v['unaffected_groups'],v['retained_probe_trajectories'],v['independent_confirmation_trajectories'])!=(16,15,128,8):
        raise ValueError('exact prefix and all original/confirmation measurements required')
    return v

def compile_policy_inclusion(previous,old,current):
    """One bounded JSON/schema audit; never read tensor bytes or run a model."""
    from utils.ch3_contract import profile,task_by_id,numeric_probe_policy,validate_baseline_full_schema
    from utils import ch3_type1_tasks as s
    from ch3_runner import _probe_validation_metrics
    if old['tasks']!=current['tasks']or old['resolved_profiles']!=current['resolved_profiles']:raise ValueError('retained scientific identity changed')
    diagnostic=bound(POLICY_REF);groups={g['id']:g for g in s.probe_groups(old)};records={};schema_cache={}
    for group,decision in previous['decisions'].items():
        g=groups[group]
        if decision['status']!='Passed'or decision['coverage']!=g['coverage']or len(decision['numerical_comparisons'])!=len(g['representatives']):raise ValueError('registered completed group coverage')
        rows=[]
        for index,run in enumerate(g['representatives']):
            t=task_by_id(old,run);collection=numeric_probe_policy(old,t);evaluation=numeric_probe_policy(current,task_by_id(current,run));comparison=decision['numerical_comparisons'][index]
            special=t['model']=='ModernTCN'and t['dataset']=='ETTh1'
            basis=diagnostic['evaluation_policy']if special else collection
            if not comparison['passed']or basis is not None and (basis['kind']!='full_float_state' or comparison.get('exact_residual_state')is not True):raise ValueError('old comparison lacks full-state/exact proof')
            if basis is None:
                if comparison.get('mode')!='exact'or comparison.get('bitwise_equal')is not True:raise ValueError('old exact evidence incomplete')
                limits=dict(state_atol=0,metric_atol=0,loss_atol=0,loss_rtol=0)
            else:limits=dict(state_atol=basis['state_atol'],metric_atol=basis['metric_atol'],loss_atol=basis.get('loss_atol',basis['metric_atol']),loss_rtol=basis.get('loss_rtol',0))
            if any(limits[k]>evaluation[k]for k in limits):raise ValueError('old policy is not contained by the new policy')
            candidates={}
            for key,e in previous['evidence'].items():
                if key.startswith(group+'/')and run in e['task_ids']and ('/serial/'in key or key.startswith(group+'/q'+str(decision['concurrency'])+'/')):
                    candidates['serial'if '/serial/'in key else'parallel']=Path(e['process']['path']).parent.parent/run/'trajectory.json'
            if set(candidates)!={'serial','parallel'}:raise ValueError('registered actual wave/trajectory mapping')
            refs={phase:previous['artifacts'][str(path)]for phase,path in candidates.items()};tr={phase:bound(value)for phase,value in refs.items()};a,b=tr['serial'],tr['parallel']
            identity=('id','profile_sha','initial','initial_rng','batch_ids','validation_tail','steps','final_rng','scheduler_trace','time_mark')
            if any(a.get(k)!=b.get(k)for k in identity)or a['id']!=run or a['profile_sha']!=digest(profile(current,t))or a['steps']!=6 or len(a['batch_ids'])!=6:raise ValueError('retained exact identity/scheduler/RNG/batch')
            for value in tr.values():
                if value.get('finite')is not True or len(value.get('M_full_state_trace',[]))!=6 or len(value['trajectory'])!=6:raise ValueError('retained finite/full six-step coverage')
                _probe_validation_metrics(old,t,value['validation'],value['validation_tail'])
                if collection is not None and (value.get('numeric_policy_sha')!=digest(collection) or value.get('full_numeric_trace')!=value['M_full_state_trace']):raise ValueError('retained collection policy/generic payload identity')
            state_refs=[];entries=0
            for step,(left,right)in enumerate(zip(a['M_full_state_trace'],b['M_full_state_trace']),1):
                if left['step']!=step or right['step']!=step or left['schema_sha']!=right['schema_sha']:raise ValueError('retained state step/schema exact identity')
                for point in (left,right):
                    sr=previous['artifacts'][point['schema_file']];dr=previous['artifacts'][point['data_file']]
                    if sr['sha256']!=point['schema_sha']or dr['sha256']!=point['data_sha']:raise ValueError('retained schema/tensor digest references')
                    if sr['path']not in schema_cache:schema_cache[sr['path']]=bound(sr)
                    schema=schema_cache[sr['path']];expected_id=collection['id']if collection else t['group']+'-exact-full-state'
                    validate_baseline_full_schema(schema,dict(id=expected_id))
                    if point['bytes']!=schema['total_bytes']:raise ValueError('retained full payload byte coverage')
                if basis is None and left['data_sha']!=right['data_sha']:raise ValueError('old exact complete payloads are not equal')
                state_refs.append(dict(step=step,serial=left,parallel=right));entries+=sum(e['numel']for e in schema['entries']if e['mode']=='bounded')
            rows.append(dict(task_id=run,collection_policy=collection,basis_policy=basis,evaluation_policy=evaluation,collection_comparison=comparison,
                relation='complete_exact_payload_subset'if basis is None else'full_float_bound_subset',old_limits=limits,new_limits={k:evaluation[k]for k in limits},
                trajectory_refs=refs,state_refs=state_refs,complete_floating_elements=entries,identity_verified=True,finite_proof='bound producer capture + saved gate, no new tensor replay',new_numeric_replays=0))
        records[group]=rows
    if len(records)!=16 or sum(len(v)for v in records.values())!=64:raise ValueError('exact retained sixteen groups/sixty-four representatives')
    return dict(purpose='baseline_numeric_policy_inclusion_v1',collection_adoption_ref=PRIOR_REUSE_REF,source_anchors_ref=SOURCE_REF,
        evaluation_config_ref=CONFIG_REF,evaluation_policy_ref=ALL_M_POLICY_REF,records=records,retained_groups=16,unaffected_groups=15,
        retained_probe_trajectories=128,independent_confirmation_ref=previous['confirmation_ref'],new_numeric_replays=0,new_compute=0,whole_M_admission=False)

def adopt_numeric_defaults(previous,c,inclusion):
    """Attach evaluation identity to proven old verdicts; retain their collection."""
    from utils.ch3_contract import task_by_id,numeric_probe_policy
    if set(inclusion['records'])!=set(previous['decisions'])or (inclusion['retained_groups'],inclusion['retained_probe_trajectories'])!=(16,128):raise ValueError('exact default inclusion scope')
    result=copy.deepcopy(previous)
    for group,decision in result['decisions'].items():
        rows=inclusion['records'][group]
        if len(rows)!=len(decision['numerical_comparisons']):raise ValueError('default inclusion complete comparisons')
        values=[]
        for old,row in zip(decision['numerical_comparisons'],rows):
            rule=numeric_probe_policy(c,task_by_id(c,row['task_id']))
            if old!=row['collection_comparison']or rule!=row['evaluation_policy']or row['identity_verified']is not True or row['new_numeric_replays']!=0 or row['complete_floating_elements']<=0 or any(row['old_limits'][k]>row['new_limits'][k]for k in row['old_limits']):raise ValueError('default inclusion verdict/policy/coverage mismatch')
            collection=row['collection_policy'];value=copy.deepcopy(old)
            value.update(mode='bounded_numeric',policy_id=rule['id'],policy_sha=digest(rule),state_atol=rule['state_atol'],metric_atol=rule['metric_atol'],loss_atol=rule['loss_atol'],loss_rtol=rule['loss_rtol'],rtol=0,equal_nan=False,
                collection_policy_id=collection['id']if collection else None,collection_policy_sha=digest(collection)if collection else None,
                collection_comparison=old,evaluation_policy_ref=ALL_M_POLICY_REF,policy_inclusion_ref=INCLUSION_REF,adoption=row['relation'],exact_residual_state=True)
            values.append(value)
        decision['numerical_comparisons']=values
    result.update(config_ref=CONFIG_REF,protocol_sha=digest(c),parent_adoption_ref=PRIOR_REUSE_REF,all_m_policy_ref=ALL_M_POLICY_REF,candidate_config_refs=CONFIG_REFS,policy_inclusion_ref=INCLUSION_REF)
    return result

def verify_production_inheritance(permit):
    """Exact frozen math plus explicitly reviewable guarded/identity plumbing delta."""
    allowed={'ch3_runner.py':{'code_binding','probe_worker','FrozenConvReplay','_compare_full_numeric_files'},
        'utils/ch3_contract.py':{'numeric_probe_policy','baseline_numeric_admission_policy','materialize_baseline_numeric_admission','validate_baseline_numeric_registry','validate_baseline_full_schema'},
        'tools/restricted_regression/m5_formal_entry.py':{'execution_profiles','validate_config','bootstrap','run_configs','worker'},
        'utils/ch3_native_execution.py':{'compare','run_probe','validate_probe_completion'},
        'utils/ch3_native_recovery_records.py':{'manifest_projection','scan_manifest'},
        'utils/ch3_probe_schema_recovery.py':{'current_recovery','activate_worker','retained_payload'},
        'utils/ch3_ms_seal_recovery.py':{'validate_m'},
        'utils/ch3_type1_chain.py':{'create_permit'},
        'utils/ch3_type1_tasks.py':{'validate'},
        'utils/ch3_round2_amendment.py':{'validate'},
        'utils/ch3_type1_execution.py':{'read_config','metadata_files'},
        'tests/test_m6_ms_seal_recovery.py':{'Dispatch','ProcessIdentity'},
        'tests/test_m6_probe_schema_recovery.py':{'trajectory'}}
    changes={}
    for name,h in permit['code'].items():
        p=ROOT/name
        if sha(p)==h:continue
        if name=='tools/restricted_regression/bundle.sha256':
            from tools.restricted_regression.run_restricted import verify_bundle
            verify_bundle();continue
        if name not in allowed:raise ValueError('unreviewed producer delta: '+name)
        old=subprocess.check_output(['git','-C',str(ROOT),'show',permit['commit']+':'+name],text=True)
        def functions(text):return {n.name:ast.dump(n,include_attributes=False)for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        a,b=functions(old),functions(p.read_text());delta={k for k in set(a)|set(b)if a.get(k)!=b.get(k)}
        if not delta<=allowed[name]:raise ValueError('computation-producing AST changed: '+name+':'+str(delta))
        changes[name]=sorted(delta)
    return changes

def verify_registered_source():
    source=bound(SOURCE_REF);refs=source['refs'];records={k:bound(r)for k,r in refs.items()}
    if source['producer_commit']!=BASE or records['authorization']['closure_commit']!=BASE or records['controller']['authorization']!=refs['authorization']:
        raise ValueError('registered failed execution/authorization')
    if records['failure']['error']!="RuntimeError('owned child technical failure: M_BASE-probe')"or records['probe_failure']['error']!="ValueError('numeric gate failure; no resource fallback')":
        raise ValueError('only this precisely registered numeric failure is recoverable')
    if records['producer_permit']['commit']!=BASE or records['producer_permit']['protocol_sha']!=digest(records['config']):raise ValueError('retained producer protocol')
    ms,verification=records['MS_boundary'],records['source_verification']
    if ms['technical_complete']is not True or ms['training_commit']!='df6a16403e10d51097db8c88829909c533d15652'or ms['source_verification_ref']!=refs['source_verification']or ms['task_ids']!=verification['task_ids']or len(set(ms['task_ids']))!=203 or ms['receipts']!=verification['receipts']:
        raise ValueError('original MS203 seal/provenance must remain exact')
    if records['upstream_boundary']['boundaries']['MS']!=refs['MS_boundary']:raise ValueError('previous lifecycle MS reference')
    from utils.ch3_type1_upstream import assert_owned_exited
    assert_owned_exited(records['exit_verification']['instances'])
    assert_owned_exited(bound(evidence()['confirmation_ref'])['worker_instances'])
    for path in (OLD_RESULT/'queue/controller/STOP',OLD_RESULT/'probe/M_BASE/STOP'):
        if path.exists():raise ValueError('STOP is not the registered numeric failure')
    return records

def status():
    records=verify_registered_source()
    try:verify_production_inheritance(records['producer_permit'])
    except (ImportError,RuntimeError) as exc:
        raise ValueError('parent-controller producer/bundle verification failed: '+str(exc)[:300]) from exc
    return dict(state='MS203_SEAL_AND_SCOPED_NUMERIC_PREFIX_REUSE_AWAITING_START',READY_FOR_GPU_EXECUTION=False,anchors=anchors(),result_review='pending')

def import_ms():
    from utils import ch3_type1_chain as q
    q.stop_check();records=verify_registered_source();verify_production_inheritance(records['producer_permit'])
    secret=os.environ.get(q.SECRET)
    if not secret:raise PermissionError('controlled lifecycle required')
    ms_ref=bound(SOURCE_REF)['refs']['MS_boundary']
    v=dict(state='ORIGINAL_MS203_VERIFIED_SEALED_AND_RELEASED',READY_FOR_GPU_EXECUTION=True,anchors=anchors(),boundaries=dict(MS=ms_ref),
        owned_exited=records['upstream_boundary']['owned_exited'],handoff_scope=q.s.ID,successor_owner=q.owner(),result_review='pending',
        reused_seal_ref=ms_ref,reuse_review_ref=REUSE_REF,adoption_commit=q.closure(),new_MS_training_calls=0,new_MS_test_calls=0)
    v['mac']=hmac.new(secret.encode(),digest(v).encode(),hashlib.sha256).hexdigest()
    return exclusive(q.CONTROL/'upstream-technical-boundary.json',v)

def validate_permit_link(c,a,probe):
    if a.get('execution_attempt')!=ATTEMPT or a.get('probe_recovery_ref')!=REUSE_REF:raise PermissionError('exact numeric attempt/policy adoption binding')
    if c['baseline_unified']['stage']in CONFIG_REFS:validate_revision(c)

def retained_refs(report):
    if not report.get('probe_recovery_ref'):return {}
    if not ACTIVE or report['probe_recovery_ref']!=REUSE_REF:raise PermissionError('unregistered numeric adoption root')
    v=evidence();r=dict(v['artifacts'])
    for p in list(v['producer_refs'].values())+[e['process']for e in v['evidence'].values()]:r[p['path']]=p
    return r

def retained_payload(c,point):
    if not ACTIVE or c['baseline_unified']['stage']!='M_BASE':return False
    allowed=retained_refs(dict(probe_recovery_ref=REUSE_REF))
    for k,h in (('schema_file','schema_sha'),('data_file','data_sha')):
        p=Path(point[k]);r=dict(path=str(p),sha256=point[h])
        if p.is_symlink()or allowed.get(str(p))!=r:return False
    return True

def load_seed(c,a):
    validate_permit_link(c,a,True);v=evidence();records=verify_registered_source();validate_revision(c)
    verify_production_inheritance(records['producer_permit'])
    from utils import ch3_type1_chain as q
    now=q.dynamic(c)
    for k in ('environment','hardware','source_states'):
        if now[k]!=records['producer_permit'][k]:raise ValueError('retained measurement condition changed: '+k)
    # One adoption checksum scan; no new numeric replay or MS checkpoint scan.
    for p,r in retained_refs(dict(probe_recovery_ref=REUSE_REF)).items():
        if Path(p).is_symlink()or ref(p)!=r:raise ValueError('retained measurement changed: '+p)
    budget=dict(caps=a['caps'],reserved=copy.deepcopy(v['historical_actual']),actual=copy.deepcopy(v['historical_actual']),
        historical_actual=copy.deepcopy(v['historical_actual']),new_actual=dict(adam=0,backward=0,forward=0),
        diagnostic_actual=copy.deepcopy(v['diagnostic_actual']),refund=False)
    return dict(budget=budget,decisions=copy.deepcopy(v['decisions']),evidence=copy.deepcopy(v['evidence']),artifacts=copy.deepcopy(v['artifacts']))

def location(report,key,default):
    if report.get('probe_recovery_ref')and key in evidence()['evidence']:
        e=evidence()['evidence'][key]
        if report['evidence'].get(key)!=e:raise ValueError('retained wave changed')
        return Path(e['process']['path']).parent
    return Path(default)

def producer(report,key,current):
    if report.get('probe_recovery_ref')and key in evidence()['producer_refs']:return bound(evidence()['producer_refs'][key])
    return current

def self_review(report,path):
    if not report.get('probe_recovery_ref'):return False
    v=evidence();return str(path)in v['trajectory_reviews']and report['artifacts'].get(str(path))==v['artifacts'][str(path)]

def group_review(report,g):
    if not report.get('probe_recovery_ref'):return None
    v=evidence()
    if g['id']in v['decisions']:
        if report['decisions'][g['id']]!=v['decisions'][g['id']]:raise ValueError('retained group or evaluation policy changed')
        return v['decisions'][g['id']]['numerical_comparisons']
    return None

def extra_actual(report):
    if report.get('probe_recovery_ref')and report['probe_recovery_ref']!=REUSE_REF:raise PermissionError('unregistered diagnostic cost reference')
    return evidence()['diagnostic_actual']if report.get('probe_recovery_ref')else dict(adam=0,backward=0,forward=0)
