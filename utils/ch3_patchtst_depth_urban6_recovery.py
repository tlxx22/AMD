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
OBSERVATION_PACKAGE = REVISION_PACKAGE / 'patch-enc1-observation-repair-v1'
OBSERVATION_REF = dict(path=str(OBSERVATION_PACKAGE / 'observation-source.json'), sha256='99ee6307fe9888988447ef52ece1fe74eab913aeb746aa865db63928a084dab1')
OLD_RESULT = prefix.RESULT
RESULT = OLD_RESULT.with_name('baseline-unified-v3-ms-seal-m128-recovery1-patchtst-depth-urban6-r2')
ATTEMPT = 'PATCHTST-depth-Urban6-exit-r2'
SESSION = 'ch3-m6-patchtst-depth-urban6-r2'
LOG = OBSERVATION_PACKAGE / 'followup-launcher.log'
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
    s.package = lambda stage:OBSERVATION_PACKAGE if stage=='PATCH_ENC1' else REVISION_PACKAGE if stage=='M_ALL' else PACKAGE if stage in spec['new_config_refs'] else packages[stage]
    s.validate = lambda c:validate_revision(c) if c['baseline_unified']['stage'] in spec['new_config_refs'] else old_validate(c)
    s.selected = lambda stage:[t for t in s.parent()['tasks'] if t['dataset']=='UrbanEV' and t['h'] in (3,12)] if stage=='URBAN_SUBSET' else old_selected(stage)
    s.probe_groups = lambda c:groups(c) if c['baseline_unified']['stage'] in ('PATCH_ENC1','PATCH_ENC2','URBAN_SUBSET') else old_groups(c)
    def budget(c):
        v=old_budget(c)
        if c['baseline_unified']['stage']=='PATCH_ENC1':
            actual=bound(OBSERVATION_REF)['retained_actual']
            for k in actual:v['nominal'][k]+=actual[k]
            v['nominal_workers']+=1
            v.update(retained_historical_actual=actual,new_nominal_workers=48,observation_failure_cost_retained=True)
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
        return dict(v,models=('PatchTST',) if c['baseline_unified']['stage'].startswith('PATCH_ENC') else s.MODELS,fixture=OBSERVATION_PACKAGE/c['baseline_unified']['stage']/'fixtures')
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
    return {str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ENTRY,WRAPPER,ROOT/'tests/test_m6_patchtst_depth_urban6.py',ROOT/'tests/test_m6_patch_enc1_exit_observation.py')}


def delta_ref():return ref(OBSERVATION_PACKAGE/'producer-delta-proof.json')


def worker_metadata(c):
    values=[CONTRACT_REF,contract()['fixed_encoder_policy_ref'],OBSERVATION_REF]
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
    if proof.get('purpose')!='exit_observer_depth_urban6_exact_delta_v1' or proof['producer_commit']!=PRODUCER or proof['contract_ref']!=CONTRACT_REF or proof['new_code']!=code_binding():raise ValueError('reviewed explicit continuation delta')
    for name,row in proof['changes'].items():
        old=subprocess.check_output(['git','-C',str(ROOT),'show',PRODUCER+':'+name])
        if hashlib.sha256(old).hexdigest()!=row['before_sha256'] or sha(ROOT/name)!=row['after_sha256']:raise ValueError('exact producer delta bytes: '+name)
        if name.endswith('.py'):
            def functions(raw):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
            before,after=functions(old),functions((ROOT/name).read_bytes())
            changed=sorted(k for k in set(before)|set(after) if before.get(k)!=after.get(k))
            if changed!=row['changed_symbols']:raise ValueError('exact reviewed function delta')
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


def verify_observation_source():
    """Specific r1 observation failure; never an arbitrary ignore-failure path."""
    value=bound(OBSERVATION_REF)
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
    return dict(recovery='specific_PATCH_ENC1_observation_after_completed_round2_371',execution_attempt=ATTEMPT,parent_technical_failure_ref=OBSERVATION_REF,prior_Urban_failure_ref=SOURCE_REF,completed_round2_ref=SOURCE_REF,completed_prefix_ref=PREFIX_REF,probe_reuse_ref=REUSE_REF,contract_ref=CONTRACT_REF,fixed_encoder_policy_ref=contract()['fixed_encoder_policy_ref'],producer_delta_ref=delta_ref(),candidate_config_refs=config_refs(),completed_new_formal=196,expected_remaining_formal=335,approved_total_new_formal=531,third_round_runs=287,depth_supplement_runs=48,automatic_main_table_replacement=True,main_table_replacement_cells=24,chosen_M_encoder=2,metric_based_reselection=False)


def status():
    verify_source_light()
    try:verify_production_inheritance()
    except (ImportError,RuntimeError)as exc:raise ValueError('completed371 producer/bundle verification failed: '+str(exc)[:300])from exc
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
        source_rule='user-frozen encoder2 before results; all24 PatchTST M switched together; all other347 sources retained',history_test_seen=True)


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
        replaced_cells=24,retained_cells=347,chosen_PatchTST_M_encoder=2,metric_based_reselection=False,technical_complete=True,result_review='pending')
    value=exclusive(main_boundary_path(),sign(body));validate_main_boundary(value);return value


def validate_main_boundary(value):
    """Compact sealed link for workers; the 371-cell adoption audit runs once."""
    from utils import ch3_type1_chain as q
    b=bound(value);content={k:v for k,v in b.items()if k!='mac'};secret=os.environ.get(q.SECRET)
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
    return dict(rows=rows,dataset_four_H_means=means,main_table_selection='pre_results_fixed_encoder_2',adoption_policy_ref=contract()['fixed_encoder_policy_ref'],metric_based_reselection=False,enc1_retained_not_main=True,enc3_original_sources_retained=True,no_cross_dataset_raw_metric_mean=True,result_review='pending')


def evidence():
    v=bound(REUSE_REF)
    if v['source_ref']!=SOURCE_REF or v['old_config_ref']!=contract()['old_config_refs']['URBAN_SUBSET']:raise ValueError('specific saved Urban serial producer')
    return v


def retained_refs(report):
    if report.get('probe_recovery_ref') not in (None,REUSE_REF):raise PermissionError('unregistered retained probe evidence')
    if report.get('scope','').endswith('URBAN_SUBSET-probe'):return {p:r for v in evidence()['accepted'].values() for p,r in v['artifacts'].items()}
    return {}


def retained_payload(c,point):
    if c['baseline_unified']['stage']!='URBAN_SUBSET':return False
    refs={p:r for v in evidence()['accepted'].values() for p,r in v['artifacts'].items()}
    for k in ('schema_file','data_file','meta_file'):
        if k in point and (point[k] not in refs or ref(point[k])!=refs[point[k]]):return False
    return True


def load_seed(c,a):
    validate_permit_link(c,a,True)
    if c['baseline_unified']['stage']=='PATCH_ENC1':
        value=verify_observation_source()
        for item in value['artifact_refs'].values():
            if ref(item['path'])!=item:raise ValueError('failed six-step artifact changed; never reclassify resource failure')
        from utils import ch3_type1_tasks as s
        actual=value['retained_actual']
        return dict(budget=dict(caps=s.probe_budget(c)['caps'],reserved=dict(actual),actual=dict(actual),historical_actual=dict(actual),new_actual=dict(adam=0,backward=0,forward=0),diagnostic_actual=dict(actual),diagnostic_failure_ref=OBSERVATION_REF,refund=False),evidence={},decisions={},artifacts={})
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


def group_review(report,group):return None
def extra_actual(report):
    if report.get('probe_recovery_ref')==REUSE_REF:
        if report.get('scope','').endswith('PATCH_ENC1-probe'):return bound(OBSERVATION_REF)['retained_actual']
        if report.get('scope','').endswith('URBAN_SUBSET-probe'):return evidence()['failed_actual']
    return dict(adam=0,backward=0,forward=0)
