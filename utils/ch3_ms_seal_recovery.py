"""One specific failed MS seal, read-only import and fresh M128 continuation."""
import copy,json,subprocess
from pathlib import Path
from utils.ch3_contract import ROOT,digest,step_arithmetic,BestState
from utils.ch3_native_recovery_records import bound,ref,sha

NAME='baseline-unified-v3-ms-seal-m128-recovery1'
PACKAGE=ROOT.parent/'amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1'/NAME
RESULT=ROOT.parent/'amd-execution-evidence/m6/m6-formal-launch-dhozikhu'/NAME
M_PROTOCOL='baseline-unified96-onecycle001-v3-mbatch128-recovery1'
M_FILE=ROOT/'configs/ch3_round2_m_batch128_recovery1.json'
M_PARENT=dict(path=str(ROOT/'configs/ch3_baseline_m_u96_oc01_v3.json'),sha256='f627260225b971d13d1e881d8123a0426c63304e64d8e8afa56fa0ab73a3cdb2')
SOURCE_REF=dict(path=str(PACKAGE/'ms203-source-verification.json'),sha256='7f77d387f4ee19a96fa69220e73842664a51a68803d14a83987804e1b5662e32')
IDENTITY_REF=dict(path=str(PACKAGE/'pid-identity-repair-v1/owned-identity-reconciliation.json'),sha256='b88eaf82752a84947bfef94834e9143889735db83ac9899dee40159ef13f21be')
ERROR='TypeError("dict() got multiple values for keyword argument \'protocol_sha\'")'
FILES={'manifest.json','result.json','history.jsonl','budget.json','runtime.json','best.pt','last.pt'}

def m_parent():return bound(M_PARENT)
def m_tasks():
    rows=[]
    for old in m_parent()['tasks']:
        t=copy.deepcopy(old);group=t['model']+'-'+t['dataset']+'-M-u96-oc01-v3-mbatch128-r1'
        t.update(id=group+'-f1-h'+str(t['h'])+'-s2024',group=group,profile=group+'-h'+str(t['h']))
        rows.append(t)
    return rows
def old_task(t):return next(x for x in m_parent()['tasks']if all(x[k]==t[k]for k in ('model','dataset','h','fold','seed')))
def m_profile(t):
    old=m_parent();prior=old_task(t);p=copy.deepcopy(old['resolved_profiles'][prior['id']])
    tr=p['training'];tr.update(batch=128,eval_batch=128)
    windows=old['datasets'][t['dataset']]['endpoints'][0]-p['T']-p['pred_len']+1
    tr['scheduler']['steps_per_epoch']=windows//128 if tr['train_drop_last']else (windows+127)//128
    return p
def validate_m(c):
    from utils import ch3_type1_tasks as s
    p=m_parent();b=c['baseline_unified']
    if c.get('type1_followup')!=s.ID or b['stage']!='M_BASE' or b['id']!=M_PROTOCOL or c['tasks']!=m_tasks():raise ValueError('exact fresh 84-task M128 continuation')
    if b['direct_parent_ref']!=M_PARENT or b['recipe_ref']!=__import__('utils.ch3_round2_amendment',fromlist=['RECIPE_REF']).RECIPE_REF:raise ValueError('M128 frozen parent/OneCycle recipe')
    if any(c[k]!=p[k]for k in ('sources','datasets','urban_folds','urban_input_variants'))or b['numeric_policies']!=p['baseline_unified']['numeric_policies']:raise ValueError('M128 data/source/policy changed')
    data=bound(b['data_ref'])
    for t in c['tasks']:
        prior=old_task(t);resolved=m_profile(t);expected=dict(task_id=prior['id'],profile_sha=digest(p['resolved_profiles'][prior['id']]),config_ref=M_PARENT)
        if c['resolved_profiles'][t['id']]!=resolved or b['parent_refs'][t['id']]!=expected:raise ValueError('M128 only batch and derived scheduler change')
        arithmetic=step_arithmetic(c,t);m=data['metadata'][t['dataset']][t['id']]
        if m['window_counts']!={'train':arithmetic['train_windows'],'validation':arithmetic['validation_windows']}or data['data_bindings'][t['dataset']][t['id']]!=digest(m):raise ValueError('M128 actual metadata/window binding')
    return c

def identity_evidence():
    """Small review-bound reconciliation, separate from the untouched MS proof."""
    expected=bound(SOURCE_REF);evidence=bound(IDENTITY_REF)
    if evidence.get('purpose')!='ms203_owned_identity_reconciliation_v1' or evidence.get('source_ref')!=SOURCE_REF or evidence.get('historical_refs')!=expected['owned_exited']:raise ValueError('review-bound process reconciliation/source mismatch')
    follower=bound(expected['follower_controller_ref'])
    if evidence.get('follower_ref')!=expected['follower_controller_ref'] or evidence['upstream']['historical_refs']+[follower['owner']]!=expected['owned_exited']:raise ValueError('historical process sample projection changed')
    want=evidence['upstream']['instances']+[dict(**follower['owner'],scope=follower['scope'],wave='controller',source=expected['follower_controller_ref'])]
    if evidence.get('instances')!=want or any(v.get('start_ticks')is None for v in want):raise ValueError('unresolved or changed process instance reconciliation')
    return evidence

def snapshot(checksums=True):
    """No deserialization, numerical replay, model, optimizer or test execution."""
    from utils import ch3_type1_upstream as u
    start,controller=u.records();control=u.OLD_RESULT/'queue/controller'
    failure_ref=ref(control/'failure.json');failure=bound(failure_ref)
    if failure!={'automatic_retry':False,'error':ERROR,'result_review':'pending','scope':u.OLD_SCOPE}:raise ValueError('only the registered protocol_sha MS-seal failure is recoverable')
    progress_ref=ref(control/'progress.json')
    if bound(progress_ref).get('state')!='SEAL_MS_BOUNDARY' or (control/'STOP').exists() or (control/'complete.json').exists():raise ValueError('exact unsealed failed MS state')
    for args in (('rev-parse','HEAD'),('rev-parse','@{u}')):
        if subprocess.check_output(['git',*args],cwd=u.OLD_WORK,text=True).strip()!=u.BASE:raise ValueError('old training source commit changed')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=u.OLD_WORK,text=True).strip():raise ValueError('old training worktree changed')
    for relative in ('M','probe/M','queue/M'):
        if (u.OLD_RESULT/relative).exists():raise ValueError('M already attempted; zero-M recovery assumption invalid')
    # The waiting successor must also remain a retained failure without computation.
    follower=u.OLD_RESULT.with_name('baseline-type1-followup-v3')
    fc=follower/'queue/controller';fcontrol_ref=ref(fc/'controller.json');fcontrol=bound(fcontrol_ref);ffailure_ref=ref(fc/'failure.json')
    if fcontrol.get('scope')!='m6-baseline-type1-followup-v3' or bound(ffailure_ref).get('error')!="RuntimeError('upstream STOP/failure; successor cannot run')" or bound(ref(fc/'progress.json')).get('state')!='WAIT_V3_COMPLETE_AND_RELEASED':raise ValueError('fixed failed waiting successor identity')
    if any((follower/x).exists()for x in ('probe','URBAN_SUBSET','EPF_ALL','M_ALL')):raise ValueError('successor computation already exists')
    from utils.ch3_round2_amendment import HISTORICAL_RESULT
    if HISTORICAL_RESULT.exists():raise ValueError('old amendment computation already exists')
    ownership=u.owned_evidence();evidence=identity_evidence()
    owners=ownership['historical_refs']+[fcontrol['owner']]
    if ownership!=evidence['upstream'] or owners!=evidence['historical_refs']:raise ValueError('registered ownership sampling/reference changed')
    u.assert_owned_exited(evidence['instances'])
    c=bound(start['config_refs']['MS']);ids=[t['id']for t in c['tasks']]
    from utils.ch3_baseline_unified_tasks import MODELS
    if len(ids)!=203 or len(set(ids))!=203 or {t['model']for t in c['tasks']}!=set(MODELS):raise ValueError('exact seven-model 203-task MS required')
    permit_ref=ref(u.OLD_RESULT/'queue/MS/formal-permit.json');permit=bound(permit_ref);code=bound(u.CODE_REF)['code'];env=bound(u.ENV_REF)
    if permit.get('start_authorization_ref')!=u.START_REF or permit.get('unified_scope')!=u.OLD_SCOPE or permit.get('commit')!=u.BASE or permit.get('protocol_sha')!=digest(c) or permit.get('authorized_task_ids')!=ids or permit.get('code')!=code or permit.get('data_binding_ref')!=c['baseline_unified']['data_ref']:raise ValueError('original MS formal permit ancestry')
    for filename,expected in code.items():
        import hashlib
        if hashlib.sha256(subprocess.check_output(['git','show',u.BASE+':'+filename],cwd=ROOT)).hexdigest()!=expected:raise ValueError('original source byte binding')
    summary=bound(permit['summary_ref']);report=bound(summary['complete_ref']);probe=bound(report['approval'])
    if summary.get('technical_admission')is not True or summary.get('owner')!=u.OWNER or summary.get('protocol_sha')!=digest(c) or probe.get('start_authorization_ref')!=u.START_REF or report.get('execution_complete')is not True or report.get('commit')!=u.BASE:raise ValueError('original MS probe/admission lineage; no replay')
    runtime_ref=ref(u.OLD_RESULT/'queue/MS/runtime-admission.json');runtime=bound(runtime_ref)
    if runtime.get('owner')!=u.OWNER or runtime.get('permit_ref')!=permit_ref or runtime.get('protocol_sha')!=digest(c) or runtime.get('integrity_scan_passed')is not True:raise ValueError('original MS runtime admission')
    data=bound(permit['data_binding_ref']);receipts={};actual=dict(adam=0,backward=0,forward=0)
    seen=[];artifact_refs={}
    for model in MODELS:
        receipt=ref(u.OLD_RESULT/'queue/MS'/('group-'+model)/'complete.json');group=bound(receipt);want=[t['id']for t in c['tasks']if t['model']==model]
        if group.get('technical_complete')is not True or group.get('result_review')!='pending' or group.get('model')!=model or group.get('task_ids')!=want or set(group.get('artifacts',{}))!=set(want):raise ValueError('exact original model-group receipt')
        receipts[model]=receipt
        for t in (t for t in c['tasks']if t['model']==model):
            refs=group['artifacts'][t['id']];root=u.OLD_RESULT/'MS'/('formal-'+model)/t['id'];p=c['resolved_profiles'][t['id']]
            if set(refs)!=FILES:raise ValueError('all seven standard artifacts required')
            for name,v in refs.items():
                file=Path(v['path'])
                if file!=root/name or file.is_symlink() or not file.is_file():raise ValueError('exact original artifact path')
                if checksums and sha(file)!=v['sha256']:raise ValueError('original artifact checksum changed: '+str(file))
                artifact_refs[str(file)]=v
            m=bound(refs['manifest.json']);r=bound(refs['result.json']);budget=bound(refs['budget.json']);rt=bound(refs['runtime.json'])
            if m['task']!=t or m['profile']!=p or m['identity'].get('commit')!=u.BASE or r.get('id')!=t['id'] or r.get('commit')!=u.BASE or r.get('scientific_protocol')!=u.OLD_PROTOCOL or r.get('protocol_sha')!=digest(c) or r.get('profile_sha')!=digest(p) or r.get('data_sha')!=data['data_bindings'][t['dataset']][t['id']]:raise ValueError('old MS scientific identity')
            final=r.get('final_test',{})
            if final.get('calls')!=1 or final.get('selected')!='best.pt' or final.get('sha256')!=refs['best.pt']['sha256'] or final.get('epoch')!=r['best_epoch'] or rt.get('error')is not None or rt.get('task')!=t['id']:raise ValueError('known test-once/best/runtime facts required')
            history=[json.loads(line)for line in (root/'history.jsonl').read_text().splitlines()];best=BestState(p['training']['patience']);arithmetic=step_arithmetic(c,t)
            for e,row in enumerate(history,1):
                if row['epoch']!=e or row['steps']!=e*arithmetic['train_batches'] or best.stopped:raise ValueError('contiguous original history/scheduler/early stop')
                best.update(row['validation']['mse'],e)
                if row['best_epoch']!=best.epoch:raise ValueError('validation-selected best history')
            if best.epoch!=r['best_epoch'] or budget['counts']['adam']!=r['scheduler_updates'] or not history or history[-1]['steps']!=r['scheduler_updates'] or (len(history)!=p['training']['epochs'] and not best.stopped):raise ValueError('original history/best/optimizer accounting')
            for k in actual:actual[k]+=budget['counts'][k]
            seen.append(t['id'])
    actual_paths={p.parent.name for p in (u.OLD_RESULT/'MS').glob('formal-*/*/result.json')}
    if set(seen)!=set(ids) or len(seen)!=203 or actual_paths!=set(ids):raise ValueError('MS missing/duplicate/extra result')
    return dict(purpose='specific_ms203_seal_failure_source_verification_v1',scope=u.OLD_SCOPE,training_commit=u.BASE,training_protocol=u.OLD_PROTOCOL,training_protocol_sha=digest(c),config_ref=start['config_refs']['MS'],start_ref=u.START_REF,controller_ref=u.CONTROLLER_REF,failure_ref=failure_ref,progress_ref=progress_ref,formal_permit_ref=permit_ref,runtime_ref=runtime_ref,probe_summary_ref=permit['summary_ref'],follower_controller_ref=fcontrol_ref,follower_failure_ref=ffailure_ref,receipts=receipts,artifact_refs=artifact_refs,task_ids=ids,owned_exited=owners,actual_formal_consumed=actual,original_probe_actual=report['budget']['actual'],code=code,environment=env['environment'],hardware=env['hardware'],source_states=permit['source_states'],checkpoints_deserialized=0,numeric_replays=0,new_test_calls=0,budget_refund=False,result_review='pending',scientific_failure=False)

def anchors():
    from utils import ch3_type1_upstream as u
    return dict(recovery='specific_protocol_sha_seal_failure',source_verification_ref=SOURCE_REF,ownership_identity_ref=IDENTITY_REF,original_start_ref=u.START_REF,original_controller_ref=u.CONTROLLER_REF,training_commit=u.BASE,expected_runs=dict(MS=203,M=84,total=287),expected_remaining_formal=427)
def verify_source():
    expected=bound(SOURCE_REF);actual=snapshot()
    if actual!=expected:raise ValueError('review-bound MS source verification changed')
    return actual
def status(full=False):
    if full:
        v=verify_source();return dict(state='RECOVERABLE_MS203_VERIFIED_AND_RELEASED',READY_FOR_GPU_EXECUTION=True,source_verification_ref=SOURCE_REF,anchors=anchors(),owned_exited=v['owned_exited'],result_review='pending')
    expected=bound(SOURCE_REF)
    for key in ('failure_ref','progress_ref','controller_ref','follower_failure_ref','follower_controller_ref'):bound(expected[key])
    from utils import ch3_type1_upstream as u
    u.assert_owned_exited(identity_evidence()['instances'])
    return dict(state='MS_SEAL_RECOVERY_AWAITING_FULL_START_CHECK',READY_FOR_GPU_EXECUTION=False,anchors=anchors(),result_review='pending')
