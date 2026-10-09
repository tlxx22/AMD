"""Two authorized width candidates; stop after forty runs for user selection."""
import copy,hashlib,hmac,json,os,sys
from pathlib import Path
from utils.ch3_contract import ROOT,digest,profile,step_arithmetic,numeric_probe_policy
from utils.ch3_native_recovery_records import bound,ref,sha,exclusive
from utils import ch3_patchtst_depth_urban6_recovery as parent

PACKAGE=parent.EVENT_PACKAGE/'patchtst-width-urban-audit-v1'
SOURCE_REF={'path': '/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/baseline-unified-v3-ms-seal-m128-recovery1/moderntcn-etth1-numeric-diagnosis-v1/baseline-numeric-admission-v1/patchtst-depth-urban6-repair-v1/patchtst-enc2-adoption-v1/patch-enc1-identity-settle-repair-v1/whole-card-monitor-v1/worker-guard-binding-fix-v1/runtime-no-telemetry-v1/patchtst-width-urban-audit-v1/r6-source.json', 'sha256': 'e2619fdd9734d911ce3d499e355fefaf0c97ac0b465fc0fe17e96e1e6a8cc67f'}
REUSE_REF={'path': '/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/baseline-unified-v3-ms-seal-m128-recovery1/moderntcn-etth1-numeric-diagnosis-v1/baseline-numeric-admission-v1/patchtst-depth-urban6-repair-v1/patchtst-enc2-adoption-v1/patch-enc1-identity-settle-repair-v1/whole-card-monitor-v1/worker-guard-binding-fix-v1/runtime-no-telemetry-v1/patchtst-width-urban-audit-v1/contract.json', 'sha256': '9beae9b8a38a0136302da1def6b639a7d4bf5889e8764bdbe049afc5fd987d54'}
CONTRACT_REF=REUSE_REF
ATTEMPT='PatchTST-enc2-width4-8-reproduction-r1'
SESSION='ch3-m6-patchtst-width4-8-r1'
RESULT=parent.RESULT.with_name('baseline-unified-v3-ms-seal-m128-recovery1-patchtst-enc2-width4-8-r1')
LOG=PACKAGE/'followup-launcher.log'
ENTRY=ROOT/'m6_patchtst_width_reproduction_entry.py'
WRAPPER=ROOT/'scripts/ch3/start_patchtst_width_reproduction.sh'
INDEPENDENT_SCIENCE_QUEUE=True
COMPLETED_PREFIX=False
COMPLETED_STAGES=SEED_STAGES=()
DATASETS=('ETTh1','ETTh2','ETTm1','ETTm2','Exchange')
HORIZONS=(96,192,336,720)
STAGES=('PATCH_DM4','PATCH_DM8')
ID='m6-patchtst-enc2-width-reproduction-v1'
PROTOCOL='patchtst-enc2-width-reproduction-v1'
ACTIVE=False


def contract():
    v=bound(CONTRACT_REF)
    if (v.get('purpose')!='PatchTST_enc2_width40_user_authorized_v1' or v.get('source_ref')!=SOURCE_REF
        or v.get('stage_order')!=list(STAGES) or v.get('datasets')!=list(DATASETS) or v.get('horizons')!=list(HORIZONS)
        or v.get('candidates')!=[4,8]or v.get('formal_runs')!=40 or v.get('max_run_epochs')!=400
        or v.get('Weather_retraining')is not False or v.get('Weather_d_model')!=128
        or v.get('automatic_adoption')is not False or v.get('automatic_third_round')is not False
        or v.get('resource_contract_ref')!=parent.RESOURCE_CONTRACT_REF):raise ValueError('fixed width40 contract and Weather preservation')
    return v


def config_refs():return contract()['config_refs']


def derive_task(prior,width):
    group=f'PatchTST-{prior["dataset"]}-M-round2-enc2-dm{width}-v1'
    return dict(prior,group=group,id=f'{group}-f1-h{prior["h"]}-s2024',profile=f'{group}-h{prior["h"]}',variant=f'encoder-2-d_model-{width}')


def validate_revision(c):
    spec=contract();stage=c['baseline_unified']['stage']
    if stage not in STAGES or c!=bound(spec['config_refs'][stage]):raise ValueError('exact width-specific immutable configuration')
    width=int(stage.removeprefix('PATCH_DM'));old=bound(spec['parent_ref']);data=bound(c['baseline_unified']['data_ref']);olddata=bound(old['baseline_unified']['data_ref'])
    prior=[t for t in old['tasks']if t['dataset']!='Weather']
    if c['tasks']!=[derive_task(t,width)for t in prior]or len(c['tasks'])!=20 or set(c['datasets'])!=set(DATASETS):raise ValueError('exact five-domain four-H independent task set')
    if c['type1_followup']!=ID or c['baseline_unified']['id']!=PROTOCOL or c['baseline_unified']['scope']!=ID or c['baseline_unified']['direct_parent_ref']!=spec['parent_ref']:raise ValueError('independent science identity')
    if c['sources']!=old['sources']or c['datasets']!={d:old['datasets'][d]for d in DATASETS}:raise ValueError('source/data contract unchanged')
    for t,before_task in zip(c['tasks'],prior):
        before=profile(old,before_task);expected=copy.deepcopy(before)
        if [before['structure'][k]for k in('d_model','n_heads','e_layers','d_ff')]!=[16,4,2,128]or(before['training']['epochs'],before['training']['patience'])!=(10,None):raise ValueError('true per-domain enc2 parent')
        expected['structure']['d_model']=width
        if profile(c,t)!=expected or c['baseline_unified']['parent_refs'][t['id']]!=dict(config_ref=spec['parent_ref'],task_id=before_task['id'],profile_sha=digest(before)):raise ValueError('d_model is the only scientific delta')
        if numeric_probe_policy(c,t)!=numeric_probe_policy(old,before_task):raise ValueError('unchanged M numeric policy')
        m=olddata['metadata'][t['dataset']][before_task['id']]
        if data['metadata'][t['dataset']][t['id']]!=m or data['data_bindings'][t['dataset']][t['id']]!=digest(m)or data['mapping'][t['id']]!=dict(task_id=before_task['id'],data_ref=old['baseline_unified']['data_ref']):raise ValueError('exact source metadata/scaler/window mapping')
        a=step_arithmetic(c,t)
        if expected['training']['scheduler']['steps_per_epoch']!=a['train_batches']or m['window_counts']!=dict(train=a['train_windows'],validation=a['validation_windows']):raise ValueError('fixed original scheduler/window arithmetic')
    for value in(spec['parent_ref'],spec['third_parent_ref']):
        weather=bound(value)
        for t in weather['tasks']:
            if t['model']=='PatchTST'and t['dataset']=='Weather':
                p=profile(weather,t)
                if [p['structure'][k]for k in('d_model','n_heads','e_layers','d_ff')]!=[128,16,2,256]or(p['training']['epochs'],p['training']['patience'])!=(20,10):raise ValueError('Weather128 original full parent must remain')
    return c


def groups(c):
    stage=c['baseline_unified']['stage'];rows=[]
    for d in DATASETS:
        ts=[t for t in c['tasks']if t['dataset']==d];ids=[t['id']for t in ts]
        rows.append(dict(id='PatchTST-'+stage+'-'+d+'-four-H',model='PatchTST',representatives=ids,planned_q=4,coverage={i:[i]for i in ids},identities={t['id']:digest(profile(c,t))for t in ts},equivalence='new width; own independent serial and same-mode bounded-q makespan, no inherited concurrency'))
    return rows


def resource_binding():return parent.resource_binding()


def verify_source():
    v=bound(SOURCE_REF)
    if v['producer_commit']!='bd4b63dfcce78f38cc233e74f3c50271b020d8de' or v['third_formal']!=0 or v['third_round_paused']is not True:raise ValueError('exact paused r6 source')
    records={k:bound(x)for k,x in v['refs'].items()}
    if records['queue/controller/failure.json']['error']!="ValueError('probe artifact path/symlink outside exact scope')":raise ValueError('registered historical audit failure unchanged')
    from utils.ch3_type1_upstream import assert_owned_exited
    assert_owned_exited(v['owned_exited'])
    import subprocess
    if subprocess.run(['tmux','has-session','-t',parent.SESSION],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:raise ValueError('old r6 session still active')
    for stage in('PATCH_ENC1','PATCH_ENC2'):
        b=records[stage];receipt=records[stage+'-receipt']
        if b['commit']!=v['producer_commit']or b['technical_complete']is not True or len(b['task_ids'])!=24 or receipt['task_ids']!=b['task_ids']or receipt['technical_complete']is not True:raise ValueError('retained48 actual source boundary/receipt')
    if records['probe/URBAN_SUBSET/complete.json']['execution_complete']is not True:raise ValueError('saved Urban compute completion required; audit is separate')
    from tools.restricted_regression.run_restricted import verify_bundle
    verify_bundle();verify_production_inheritance()
    return v


def verify_production_inheritance():
    proof=bound(ref(PACKAGE/'separate-entrypoints-v1/producer-delta-proof.review.json'))
    if proof['parent_commit']!='bd4b63dfcce78f38cc233e74f3c50271b020d8de' or proof['model_math_changed']is not False:raise ValueError('exact reviewed plumbing proof')
    for name,h in proof['protected_exact'].items():
        if sha(ROOT/name)!=h:raise ValueError('frozen computation/guard changed: '+name)
    for name,h in proof['after_code'].items():
        if sha(ROOT/name)!=h:raise ValueError('reviewed source delta changed: '+name)
    return proof


def anchors():
    return dict(execution_attempt=ATTEMPT,contract_ref=CONTRACT_REF,source_ref=SOURCE_REF,producer_delta_ref=ref(PACKAGE/'separate-entrypoints-v1/producer-delta-proof.review.json'),candidate_config_refs=config_refs(),completed_old_new_formal=196,completed_depth_formal=48,expected_remaining_formal=40,approved_additional_formal=40,max_additional_run_epochs=400,old_plan_total=531,third_round_paused=True,automatic_adoption=False,**resource_binding())


def status():
    verify_source()
    return dict(state='WIDTH40_PREPARED_THIRD_FORMAL_PAUSED',READY_FOR_GPU_EXECUTION=False,anchors=anchors(),result_review='pending')


def import_ms():
    """Seal source provenance in this lifecycle; adopt no old scientific result."""
    from utils import ch3_type1_chain as q
    source=verify_source();body=dict(state='PAUSED_R6_SOURCE_VERIFIED',READY_FOR_GPU_EXECUTION=True,anchors=anchors(),boundaries={},handoff_scope=ID,successor_owner=q.owner(),source_ref=SOURCE_REF,adoption_commit=q.closure(),result_review='pending')
    secret=os.environ.get(q.SECRET)
    if not secret:raise PermissionError('controlled source handoff absent')
    body['mac']=hmac.new(secret.encode(),digest(body).encode(),hashlib.sha256).hexdigest()
    return exclusive(q.CONTROL/'upstream-technical-boundary.json',body)


def validate_permit_link(c,a,probe):
    if a.get('execution_attempt')!=ATTEMPT or a.get('probe_recovery_ref')!=REUSE_REF or c!=bound(config_refs()[c['baseline_unified']['stage']]):raise PermissionError('width variant/attempt/contract exact binding')
    if a.get('base287_boundary_ref')is not None or a.get('round2_boundary_ref')is not None:raise PermissionError('new40 cannot reuse old MAC lifecycle')


def monitor_binding(configs):
    from utils import ch3_type1_chain as q
    expected=resource_binding();permits=[]
    if not ACTIVE or q.PROBE_RECOVERY is not sys.modules[__name__]:raise PermissionError('registered width resource context')
    for cfg in configs:
        a=cfg.get('approval')if cfg.get('purpose')=='ch3_probe'else bound(cfg['formal_permit_ref'])
        if cfg.get('type1_scope')!=ID or cfg.get('probe_schema_recovery_ref')!=REUSE_REF or any(cfg.get(k)!=v or a.get(k)!=v for k,v in expected.items()):raise PermissionError('exact width worker/permit resource mode')
        permits.append(a)
    if any(a!=permits[0]or cfg['unified_stage']!=configs[0]['unified_stage']or cfg['purpose']!=configs[0]['purpose']for cfg,a in zip(configs,permits)):raise PermissionError('one exact width wave')
    q.validate_permit(q.configs()[configs[0]['unified_stage']],permits[0],configs[0]['purpose']=='ch3_probe')
    from utils.ch3_event_resources import validate_startup
    validate_startup(permits[0]['startup_hardware_ref'],permits[0],live=True)
    if any(cfg.get('startup_hardware_ref')!=permits[0]['startup_hardware_ref']for cfg in configs):raise PermissionError('exact startup hardware grant')
    return expected


def worker_metadata(c):
    spec=contract();values=[CONTRACT_REF,SOURCE_REF,spec['parent_ref'],spec['third_parent_ref'],resource_binding()['resource_contract_ref'],ref(PACKAGE/'separate-entrypoints-v1/producer-delta-proof.review.json')]
    for value in config_refs().values():
        v=bound(value);values.extend([value,v['baseline_unified']['data_ref']])
    values.append(bound(spec['parent_ref'])['baseline_unified']['data_ref'])
    return values


def code_binding():
    return {name:sha(ROOT/name)for name in('utils/ch3_patchtst_width_reproduction.py','m6_patchtst_width_reproduction_entry.py','scripts/ch3/start_patchtst_width_reproduction.sh')}


def retained_refs(report):
    if report.get('probe_recovery_ref')not in(None,REUSE_REF):raise PermissionError('foreign width readonly source')
    return {}


def retained_payload(c,point):return False

def location(report,key,default):return Path(default)

def producer(report,key,current):return current

def self_review(report,path):return False

def group_review(report,group):return None

def extra_actual(report):return dict(adam=0,backward=0,forward=0)

def continuation_actions():
    from utils import ch3_type1_chain as q
    return {'VERIFY_PAUSED_R6_SOURCE':lambda receipts:q.wait_upstream()}


def completion_counts():return dict(total_runs=40,third_round_runs=0,adopted_new_formal_runs=0,executed_new_formal_runs=40)


def completion_record(receipts):
    from utils import ch3_type1_chain as q
    return dict(scope=ID,technical_complete=True,result_review='pending',awaiting_user_width_selection=True,automatic_adoption=False,third_round_paused=True,source_ref=SOURCE_REF,boundaries={stage:receipts[q.STAGE_STATES[stage][3]]for stage in STAGES},**completion_counts(),**resource_binding())


def validate_complete(v):
    from utils import ch3_type1_chain as q
    if (v.get('scope')!=ID or v.get('technical_complete')is not True or v.get('result_review')!='pending' or v.get('awaiting_user_width_selection')is not True or v.get('automatic_adoption')is not False or v.get('third_round_paused')is not True or v.get('source_ref')!=SOURCE_REF
        or any(v.get(k)!=x for k,x in dict(completion_counts(),**resource_binding()).items())or set(v.get('boundaries',{}))!=set(STAGES)):raise ValueError('width40 complete does not authorize adoption or third round')
    for stage,value in v['boundaries'].items():q.validate_boundary_light(value,stage)
    return v


def width_results():
    from utils import ch3_type1_chain as q
    from utils.ch3_native_tasks import result_path
    complete=validate_complete(bound(ref(q.CONTROL/'complete.json')));cs=q.configs();reference=bound(contract()['paper_ref'])['PatchTST'];rows={}
    for stage,c in cs.items():
        boundary=bound(complete['boundaries'][stage]);receipt=bound(boundary['receipts']['PatchTST']);bank=[]
        for t in c['tasks']:
            value=receipt['artifacts'][t['id']];out=result_path(c,t)
            if value['result.json']['path']!=str(out/'result.json'):raise ValueError('exact variant result path')
            result=bound(value['result.json']);manifest=bound(value['manifest.json']);p=profile(c,t)
            if result['id']!=t['id']or manifest['task']!=t or manifest['profile']!=p or manifest['identity']['profile_sha']!=digest(p)or manifest['identity']['commit']!=boundary['commit']or result['commit']!=boundary['commit']or result['protocol_sha']!=digest(c)or result.get('final_test',{}).get('calls')!=1:raise ValueError('actual run/validation-selected result provenance')
            bank.append(dict(task=t,metrics={k:result[k]for k in('mse','mae')},result_ref=value['result.json'],manifest_ref=value['manifest.json'],producer_commit=manifest['identity']['commit'],seed=t['seed'],profile_sha=digest(p),actual_counts=bound(value['budget.json'])['counts']))
        domains={}
        for d in DATASETS:
            ds=[x for x in bank if x['task']['dataset']==d];avg={k:sum(x['metrics'][k]for x in ds)/4 for k in('mse','mae')};target=reference.get(d)
            domains[d]=dict(rows=ds,four_H_mean=avg,paper=target,delta=None if target is None else{k:avg[k]-target[k]for k in avg})
        rows[stage]=domains
    closer={}
    for d,target in reference.items():
        by={k:('tie'if abs(rows[STAGES[0]][d]['delta'][k])==abs(rows[STAGES[1]][d]['delta'][k])else 4 if abs(rows[STAGES[0]][d]['delta'][k])<abs(rows[STAGES[1]][d]['delta'][k])else 8)for k in('mse','mae')}
        closer[d]=dict(by_metric=by,metric_conflict=by['mse']!=by['mae'])
    return dict(purpose='PatchTST_width40_comparison_without_selection_v1',candidates=rows,closer=closer,selected_width=None,Weather_retrained=False,third_round_paused=True,result_review='pending',paper_ref=contract()['paper_ref'])


def activate():
    global ACTIVE
    if ACTIVE:return
    parent.activate()
    from utils import ch3_type1_tasks as s,ch3_type1_chain as q,ch3_round2_amendment as amend
    old_context,old_plan=s.context,s.plan
    s.ID=ID;s.PROTOCOL=PROTOCOL;s.STAGES=STAGES;s.RESULT=RESULT;s.PACKAGE=PACKAGE;amend.RESULT=RESULT/'round2-amendment'
    s.file=lambda stage:Path(config_refs()[stage]['path'])
    s.parent_ref=lambda stage:contract()['parent_ref']
    s.package=lambda stage:PACKAGE
    s.validate=validate_revision;s.probe_groups=groups
    s.expected_tasks=lambda stage:bound(config_refs()[stage])['tasks']
    # Drop the parent attempt's historical probe reservation patch.
    def budget(c):
        nominal={k:sum(2*s.worker_counts(c,t)[k]for t in c['tasks'])for k in('adam','backward','forward')}
        caps={k:sum(3*s.worker_counts(c,t)[k]for t in c['tasks'])for k in nominal}
        return dict(nominal=nominal,caps=caps,nominal_workers=40,max_workers=60,fallback='resource-only q4 -> once q2 -> measured serial q1; no refund')
    s.probe_budget=budget
    def context(c):return dict(old_context(c),models=('PatchTST',),fixture=PACKAGE/c['baseline_unified']['stage']/'fixtures')
    s.context=context
    def plan(c):
        v=old_plan(c);v['models']=['PatchTST'];v['formal_waves']={'PatchTST':v['formal_waves']['PatchTST']};return v
    s.plan=plan
    q.STAGE_STATES={stage:tuple(stage+suffix for suffix in('_PROBE','_AUTO_AUDIT','_FORMAL','_BOUNDARY'))for stage in STAGES}
    q.STATES=('VERIFY_PAUSED_R6_SOURCE','FOLLOWUP_PROTOCOL_PREFLIGHT',*(state for stage in STAGES for state in q.STAGE_STATES[stage]),'COMPLETE')
    q.CONTROL=RESULT/'queue/controller';q.LOG=LOG;q.SESSION=SESSION;q.ENTRY=ENTRY;q.WRAPPER=WRAPPER;q.PROBE_RECOVERY=sys.modules[__name__]
    ACTIVE=True
