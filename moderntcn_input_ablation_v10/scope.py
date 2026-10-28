"""Only Urban fold6 F1/F2/F3 UV may execute; F4 is a text result reference."""
import copy
import hashlib
import json
from pathlib import Path
from .spec import CANDIDATE_ID, REVISION
from utils.feature_schema import get_feature_schema
SCOPE='urban_fold6_uv_input_ablation_v10_v1'
REFERENCE_SHA='18116e926996b6f27272d9e162bb654b612cab4991146a48cfe0e455b00e99fc'
HORIZONS=(3,6,9,12)
VARIANTS=('F1','F2','F3')

def reference():
    raw=Path(__file__).with_name('scope_reference.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=REFERENCE_SHA:raise ValueError('input scope reference changed')
    return json.loads(raw)

def expected_tasks():
    from .training_config import digest, arithmetic
    parents={e['original_task']['h']:e for e in reference()['F4_reuse']}
    out=[]
    for variant in VARIANTS:
        schema=get_feature_schema(variant);names=list(schema.feature_names)
        for h in HORIZONS:
            entry=parents[h];t=copy.deepcopy(entry['original_task']);p=t['structural_source_profile']
            p.update(input_variant=variant,features=names,C=len(names),aux_idx=list(range(1,len(names))),target_idx=0)
            t.update(id=f'{CANDIDATE_ID}|input-v1|UrbanEV|{variant}|f6|h{h}|UV|s2024',candidate_id=CANDIDATE_ID,model=CANDIDATE_ID,structural_revision=copy.deepcopy(REVISION),task_scope=SCOPE,input_variant=variant,goal=3,profile_sha=digest(p),schema_fingerprint=schema.fingerprint,
                     derivation_changed_fields=['input_variant','features','C','aux_idx','profile_sha','schema_fingerprint','task_scope','goal','candidate/revision'],
                     structural_parent=dict(original_task_id=entry['original_task_id'],producer_commit=entry['producer_commit'],profile_sha=entry['original_task']['profile_sha'],note='F4 native hyperparameters/data parent, not an F1/F2/F3 B result'))
            t['arithmetic']=arithmetic(t['data_config'],p,6);out.append(t)
    return out

def validate_task_scope(t):
    p=t['structural_source_profile'];variant=t.get('input_variant')
    if (t.get('task_scope'),t.get('dataset'),t.get('fold'),t.get('arm'),t.get('task'),t.get('use_u'),t.get('use_v'))!=(SCOPE,'UrbanEV',6,'UV','MS',True,True):raise ValueError('only fold6 Urban UV input scope')
    if variant not in VARIANTS or t.get('h') not in HORIZONS:raise ValueError('input variant/H outside scope')
    schema=get_feature_schema(variant);names=list(schema.feature_names)
    if (p['C'],p['target_idx'],p['aux_idx'],p['features'],p['pred_len'],p['T'])!=(len(names),0,list(range(1,len(names))),names,1,12):raise ValueError('ordered schema/target/L/P changed')
    if t.get('schema_fingerprint')!=schema.fingerprint:raise ValueError('schema fingerprint changed')
    if t['id']!=f'{CANDIDATE_ID}|input-v1|UrbanEV|{variant}|f6|h{t["h"]}|UV|s2024':raise ValueError('input task id changed')
    return t

def expected_waves():
    ids=[t['id'] for t in expected_tasks()]
    return [ids[i:i+4] for i in (0,4,8)]

def validate_scope(c):
    r=reference()
    if c.get('task_scope')!=SCOPE or c.get('scope_reference_sha256')!=REFERENCE_SHA:raise ValueError('input scope binding')
    if c['tasks']!=expected_tasks():raise ValueError('exact ordered twelve fresh input tasks required')
    for key in ('F4_reuse','parent_config_ref','P1_equivalence_ref','completed_module_goals_ref','global_lock'):
        if c.get(key)!=r[key]:raise ValueError('input reference changed: '+key)
    if c.get('waves')!=expected_waves() or (c.get('max_run_epochs'),c.get('max_optimizer_steps'))!=(240,1782120):raise ValueError('three waves/input budget')
    if (sum(t['arithmetic']['max_epochs'] for t in c['tasks']),sum(t['arithmetic']['max_optimizer_steps'] for t in c['tasks']))!=(240,1782120):raise ValueError('independent input arithmetic')
    if any(c.get(k)!='cfg_adapted_v1' for k in ('B_ref','B_cfg','backbone_variant')):raise ValueError('baseline changed')
    return c

def waves(c,ids):
    ordered=[t['id'] for t in c['tasks']]
    if not ids or len(ids)!=len(set(ids)) or ids!=[i for i in ordered if i in ids]:raise ValueError('ordered unique task subset required')
    if c['purpose']=='synthetic':return [ids[i:i+4] for i in range(0,len(ids),4)]
    validate_scope(c)
    return [[i for i in w if i in ids] for w in expected_waves() if any(i in ids for i in w)]

def verified_f4_records(c):
    from .training_config import load_ref,digest
    from moderntcn_dev_v10.scope import verified_p1_records
    from moderntcn_dev_v10.worker import source_binding as original_binding
    r=reference()
    if c.get('F4_reuse')!=r['F4_reuse'] or c.get('P1_equivalence_ref')!=r['P1_equivalence_ref']:raise ValueError('four approved F4 sources required')
    parent=load_ref(r['parent_config_ref']);proof=load_ref(r['P1_equivalence_ref'])
    if proof.get('status')!='Passed' or proof.get('candidate_id')!=CANDIDATE_ID or proof.get('structural_revision')!=REVISION or proof.get('source_binding')!=original_binding():raise ValueError('original P1 equivalence changed')
    records={t['h']:(t,m) for t,m in verified_p1_records(parent) if t['arm']=='UV'}
    if set(records)!=set(HORIZONS):raise ValueError('F4 coverage')
    seen=set()
    for entry in r['F4_reuse']:
        old=entry['original_task'];h=old['h']
        if h in seen or (old['dataset'],old['input_variant'],old['fold'],old['arm'])!=('UrbanEV','F4',6,'UV'):raise ValueError('F4 source identity')
        seen.add(h)
        if entry['producer_commit']!='385f53b458f1140c143c2c1edc558b2ceb9e24b7':raise ValueError('F4 producer changed')
        o={k:load_ref(v) for k,v in entry['references'].items()}
        if digest(o['data-metadata.json'])!=o['manifest.json']['identity']['data_sha']:raise ValueError('F4 data/scaler binding')
        for fresh in [t for t in c['tasks'] if t['h']==h]:
            for key in ('data_config','data_stat','baseline_training','baseline_prefix_metadata','baseline_record_prefix'):
                if fresh[key]!=old[key]:raise ValueError('input/F4 shared data or training changed')
    return [records[h] for h in HORIZONS]

def validate_launch_scope(c,ids,a):
    validate_scope(c);waves(c,ids)
    if a.get('task_scope')!=SCOPE:raise ValueError('input scope authorization required')
    if a.get('resume_review_approved') is True:
        if a.get('resume_task_ids')!=ids or not a.get('previous_runtime_ref'):raise ValueError('explicit owned input resume required')
    elif ids!=[t['id'] for t in c['tasks']]:raise ValueError('fresh input launch requires twelve tasks')
    if (a.get('max_run_epochs'),a.get('max_optimizer_updates'))!=(240,1782120):raise ValueError('input launch budget')
    if a.get('task_profile_bindings')!={t['id']:t['profile_sha'] for t in c['tasks']}:raise ValueError('input profile bindings')
    for key in ('F4_reuse','P1_equivalence_ref','completed_module_goals_ref'):
        if a.get(key)!=c[key]:raise ValueError('input launch reference binding')
    from .training_config import load_ref
    from .worker import source_binding
    proof=load_ref(a['input_scope_evidence_ref'])
    if proof.get('status')!='Passed' or proof.get('task_scope')!=SCOPE or proof.get('source_binding')!=source_binding():raise ValueError('reviewed input scope implementation required')
    verified_f4_records(c);load_ref(c['completed_module_goals_ref'])

def derive():
    from .training_config import validate
    from .worker import source_binding
    from .training_state import SCHEMA
    from .gate_policy import policy
    r=reference()
    return validate(dict(schema='moderntcn-uv-input-ablation-training-v10',candidate_id=CANDIDATE_ID,structural_revision=copy.deepcopy(REVISION),purpose='development',training_protocol='round2_complete_onecycle_v3',execution_authorized=False,checkpoint_schema=SCHEMA,gate_policy=policy(),tasks=expected_tasks(),source_binding=source_binding(),output_namespace='moderntcn-uv-v10-input-ablation-runs',preparation_head='ef38340a6fbe7a98cdc73270e4da121cf17ee70f',backbone_variant='cfg_adapted_v1',B_ref='cfg_adapted_v1',B_cfg='cfg_adapted_v1',max_run_epochs=240,max_optimizer_steps=1782120,task_scope=SCOPE,scope_reference_sha256=REFERENCE_SHA,waves=expected_waves(),**r))
