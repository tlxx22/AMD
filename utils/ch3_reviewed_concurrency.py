"""Reviewed fixed waves for the independent M6 queue; no probe admission claim."""
import hashlib,hmac,json,os
from utils.ch3_contract import digest,profile
from utils.ch3_native_recovery_records import bound,exclusive

POLICY='reviewed_concurrency_reuse_without_mandatory_probe_v1'


def enabled():
    from utils import ch3_type1_chain as q
    return getattr(q.PROBE_RECOVERY,'CONCURRENCY_POLICY',None)==POLICY


def recovery():
    from utils import ch3_type1_chain as q
    if not enabled() or not getattr(q.PROBE_RECOVERY,'ACTIVE',False):
        raise PermissionError('registered reviewed-concurrency lifecycle required')
    b=q.PROBE_RECOVERY
    if b.contract().get('concurrency_policy')!=POLICY:
        raise PermissionError('reviewed concurrency must be bound by the science contract')
    return b


def shape(c,t):
    p=profile(c,t)
    return dict(model=t['model'],task=t['task'],T=p['T'],pred_len=p['pred_len'],C=p['C'],
        batch=p['training']['batch'],eval_batch=p['training']['eval_batch'],
        structure=p['structure'],dtype='float32',precision='original-float32-no-AMP',
        input_variant=t.get('input_variant'),features=p.get('features'),threads=4)


def stage_plan(c):
    from utils import ch3_type1_tasks as s
    b=recovery();r=b.contract()['concurrency_plan_ref'];v=bound(r)
    stage=c['baseline_unified']['stage']
    if (v.get('purpose')!='m6_reviewed_fixed_concurrency_plan_v1' or v.get('policy')!=POLICY
        or v.get('scope')!=b.ID or v.get('execution_attempt')!=b.ATTEMPT
        or v.get('science_config_refs')!=b.config_refs() or v.get('resource_binding')!=b.resource_binding()
        or v.get('epf_grouping_policy')!=s.EPF_GROUPING_POLICY
        or v.get('automatic_probe')is not False or v.get('probe_admission_claim')is not False
        or v.get('stopped_attempt_credit')!=0 or v.get('result_review')!='pending'
        or set(v.get('stages',{}))!=set(b.STAGES)):
        raise PermissionError('exact fixed concurrency plan/lifecycle/science/resource binding')
    row=v['stages'][stage]
    validate_stage(c,row)
    return row


def validate_stage(c,row):
    from utils import ch3_type1_tasks as s
    stage=c['baseline_unified']['stage'];groups=s.probe_groups(c)
    if (row.get('stage')!=stage or row.get('protocol_sha')!=digest(c)
        or row.get('numeric_policy_sha')!=digest(c['baseline_unified']['numeric_policies'])
        or row.get('profile_shas')!={t['id']:digest(profile(c,t))for t in c['tasks']}
        or len(row.get('groups',[]))!=len(groups)
        or(stage=='EPF_ALL'and row.get('grouping_policy')!=s.EPF_GROUPING_POLICY)):
        raise ValueError('reviewed plan exact protocol/policy/profile/group coverage')
    seen=[]
    tasks={t['id']:t for t in c['tasks']}
    for actual,g in zip(row['groups'],groups):
        ids=g['representatives'];width=actual.get('q')
        if (actual.get('id')!=g['id'] or actual.get('model')!=g['model']
            or(stage=='EPF_ALL'and any(actual.get(k)!=g[k]for k in('grouping_policy','resource_review_required','resource_risks')))
            or actual.get('task_ids')!=ids or type(width)is not int or width not in(1,2,4)
            or width>g['planned_q'] or actual.get('shapes')!={t:shape(c,tasks[t])for t in ids}
            or actual.get('evidence_class')not in('historical_measured','similar_shape_inference','unmeasured_estimate')
            or not actual.get('rationale') or not isinstance(actual.get('limitations'),list)
            or not isinstance(actual.get('history_refs'),list)
            or any(k in actual for k in('status','Passed','Measured','numerical_passed','resource_admission'))):
            raise ValueError('fixed q, exact shapes and truthful evidence classification required')
        seen.extend(ids)
    if len(seen)!=len(set(seen))or set(seen)!=set(tasks):raise ValueError('each formal task exactly once')
    return row


def binding(c):
    b=recovery();stage_plan(c)
    return dict(concurrency_policy=POLICY,concurrency_plan_ref=b.contract()['concurrency_plan_ref'],
        mandatory_probe=False,probe_admission_claim=False)


def validate_permit(c,a):
    if any(a.get(k)!=v for k,v in binding(c).items()):raise PermissionError('formal permit fixed concurrency binding')
    if any(k in a for k in('summary_ref','manifest_ref','technical_admission')):
        raise PermissionError('reviewed-plan permit cannot masquerade as probe admission')
    return stage_plan(c)


def formal_waves(c,row,model):
    # Equality to the contract-bound projection prevents arbitrary caller dictionaries.
    if row!=stage_plan(c):raise PermissionError('only the review-bound stage plan selects formal q')
    return [g['task_ids'][i:i+g['q']]for g in row['groups']if g['model']==model
        for i in range(0,len(g['task_ids']),g['q'])]


def seal_runtime(c,permit_ref):
    from utils import ch3_type1_chain as q,ch3_type1_tasks as s
    a=q.validate_permit(c,bound(permit_ref));validate_permit(c,a);q.stop_check()
    body=dict(purpose='m6_reviewed_concurrency_runtime_v1',scope=s.ID,stage=s.context(c)['stage'],
        owner=q.owner(),permit_ref=permit_ref,protocol_sha=digest(c),commit=a['commit'],code=a['code'],
        upstream_boundary_ref=a['upstream_boundary_ref'],science_protocol=c['baseline_unified']['id'],
        result_review='pending',**binding(c),**q.PROBE_RECOVERY.resource_binding(),
        startup_hardware_ref=a['startup_hardware_ref'])
    secret=os.environ.get(q.SECRET)
    if not secret:raise PermissionError('current controlled runtime secret required')
    body['mac']=hmac.new(secret.encode(),digest(body).encode(),hashlib.sha256).hexdigest()
    return exclusive(s.context(c)['control']/'runtime-admission.json',body)


def validate_runtime(c,v,permit_ref):
    from utils import ch3_type1_chain as q,ch3_type1_tasks as s
    a=bound(permit_ref);validate_permit(c,a)
    if (v.get('purpose')!='m6_reviewed_concurrency_runtime_v1'
        or v.get('scope')!=s.ID or v.get('stage')!=s.context(c)['stage']
        or v.get('permit_ref')!=permit_ref or v.get('protocol_sha')!=digest(c)
        or v.get('owner')!=json.loads((q.CONTROL/'controller.json').read_text())['owner']
        or not q.same(v.get('owner')) or v.get('science_protocol')!=c['baseline_unified']['id']
        or v.get('commit')!=a['commit'] or v.get('code')!=a['code']
        or v.get('startup_hardware_ref')!=a.get('startup_hardware_ref')
        or v.get('upstream_boundary_ref')!=a['upstream_boundary_ref']
        or any(v.get(k)!=x for k,x in binding(c).items())
        or any(k in v for k in('summary_ref','manifest_ref','integrity_scan_passed','full_scans'))):
        raise PermissionError('current owner/runtime/permit/plan required; no invented probe audit')
    return v
