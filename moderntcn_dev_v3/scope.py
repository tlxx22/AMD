"""Exact v3 fresh V/UV scope plus explicitly attributed v1 U-only references."""
import copy,json,hashlib
from pathlib import Path
from .spec import CANDIDATE_ID,REVISION
SCOPE='development_fold6_epf_target_state_v3'
REFERENCE_SHA='0c085ee08195212cc389f27a0a6508c25d0ec42568bf39b75dabe3022ed9b052'
MARKETS=('PJM','NP','BE','FR','DE')
def reference():
    raw=Path(__file__).with_name('scope_reference.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=REFERENCE_SHA:raise ValueError('frozen v3 scope reference changed')
    return json.loads(raw)
def expected_tasks(arms=('V','UV')):
    out=[]
    for arm in arms:
        for p in reference()['prototypes']:
            t=copy.deepcopy(p);parts=t['id'].split('|');parts[0]=CANDIDATE_ID;parts[1]='v3';parts[-2]=arm
            t.update(id='|'.join(parts),model=CANDIDATE_ID,candidate_id=CANDIDATE_ID,structural_revision=copy.deepcopy(REVISION),arm=arm,use_u=arm in ('U','UV'),use_v=arm in ('V','UV'))
            # Latest Urban fold6 module goal; parent metadata was an extension.
            t['goal']=1 if t['dataset']=='UrbanEV' else 2
            out.append(t)
    return out
def expected_waves():
    ids=[t['id'] for t in expected_tasks()]
    return [ids[start+a:start+b] for start in (0,9) for a,b in ((0,4),(4,8),(8,9))]
def future_input_scope():
    from utils.feature_schema import get_feature_schema
    return [{'dataset':'UrbanEV','fold':6,'h':h,'input_variant':v,'arm':'UV','task':'MS','C':len(get_feature_schema(v).feature_names),'features':list(get_feature_schema(v).feature_names),'target_idx':0,'aux_idx':list(range(1,len(get_feature_schema(v).feature_names))),'status':'scope_only_not_executable','candidate_id':CANDIDATE_ID,'structural_revision':copy.deepcopy(REVISION)} for v in ('F1','F2','F3') for h in (3,6,9,12)]
def validate_scope(c):
    r=reference()
    if c.get('task_scope')!=SCOPE or c.get('scope_reference_sha256')!=REFERENCE_SHA:raise ValueError('v3 explicit scope')
    if c['tasks']!=expected_tasks():raise ValueError('exact ordered 18 fresh V/UV tasks required')
    if c.get('U_reuse')!=r['U_reuse'] or c.get('existing_B')!=r['existing_B'] or c.get('parent_config_ref')!=r['parent_config_ref']:raise ValueError('exact nine U/B references')
    if c.get('waves')!=expected_waves() or c.get('global_lock')!=r['global_lock']:raise ValueError('fixed six waves/global mutex')
    if (c.get('max_run_epochs'),c.get('max_optimizer_steps'))!=(360,1245080):raise ValueError('fresh-only budget')
    if (sum(t['arithmetic']['max_epochs'] for t in c['tasks']),sum(t['arithmetic']['max_optimizer_steps'] for t in c['tasks']))!=(360,1245080):raise ValueError('independent task arithmetic')
    if any(c.get(k)!='cfg_adapted_v1' for k in ('B_ref','B_cfg','backbone_variant')):raise ValueError('adapted baseline only')
    if c.get('future_input_scope')!=future_input_scope():raise ValueError('fold6 input scope must be retained without execution')
    reuse={f'f6-H{t["h"]}':t['id'] for t in c['tasks'] if t['dataset']=='UrbanEV' and t['arm']=='UV'}
    if c.get('F4_reuse')!=reuse:raise ValueError('v3 F4 identity only')
    return c
def waves(c,ids):
    if c['purpose']=='synthetic':
        ordered=[t['id'] for t in c['tasks']]
        if not ids or len(ids)!=len(set(ids)) or not set(ids)<=set(ordered):raise ValueError('synthetic task set')
        return [ids[i:i+4] for i in range(0,len(ids),4)]
    validate_scope(c);ordered=[t['id'] for t in c['tasks']]
    if not ids or len(set(ids))!=len(ids) or ids!=[i for i in ordered if i in ids]:raise ValueError('exact ordered fresh task subset')
    return [[i for i in w if i in ids] for w in expected_waves() if any(i in ids for i in w)]
def verified_u_records(c):
    """Read sealed TEXT metadata, never a legacy checkpoint or a new test."""
    from .training_config import load_ref,digest
    r=reference();runtime=load_ref(r['original_runtime_ref']);parent=load_ref(r['parent_config_ref'])
    result=[]
    for target,old,entry in zip(expected_tasks(('U',)),r['prototypes'],c['U_reuse']):
        objects={k:load_ref(ref) for k,ref in entry['references'].items()}
        m=objects['manifest.json'];metrics=objects['metrics.json'];identity=m['identity']
        if entry['task_id']!=old['id'] or identity['task']!=old or metrics['task_id']!=old['id']:raise ValueError('U reuse exact original task')
        if identity['candidate_id']!='moderntcn-uv-dev-v1' or identity['source']!=entry['source_binding'] or identity['source']!=runtime['source_binding']:raise ValueError('U reuse original source')
        if identity['producer_commit']!=entry['producer_commit'] or identity['config_sha']!=entry['config_sha'] or digest(parent)!=entry['config_sha']:raise ValueError('U reuse producer/config')
        if m['status']!='completed' or metrics['test_calls']!=1 or m['identity_sha']!=entry['identity_sha'] or digest(identity)!=entry['identity_sha'] or objects['seal.json']['identity_sha']!=entry['identity_sha']:raise ValueError('U reuse seal/test identity')
        spec=objects['spec.json']
        if spec!=identity['structural_spec'] or spec['use_u'] is not True or spec['use_v'] is not False or identity['profile']!=target['structural_source_profile']:raise ValueError('U-only equivalent science required')
        if old['data_config']!=target['data_config'] or old['data_stat']!=target['data_stat'] or old['baseline_training']!=target['baseline_training']:raise ValueError('U data/training mismatch')
        result.append((target,dict(metrics,task_id=target['id'],original_task_id=old['id'],provenance='v1 equivalent U-only reuse; not a v3 fresh result')))
    return result
def validate_launch_scope(c,ids,a):
    waves(c,ids)
    if a.get('task_scope')!=SCOPE:raise ValueError('v3 scope authorization required')
    if a.get('resume_review_approved') is True:
        if a.get('resume_task_ids')!=ids or not a.get('previous_runtime_ref'):raise ValueError('explicit owned resume required')
    elif ids!=[t['id'] for t in c['tasks']]:raise ValueError('fresh v3 launch requires exact 18 tasks')
    if a.get('max_run_epochs')!=360 or a.get('max_optimizer_updates')!=1245080:raise ValueError('launch fresh budget mismatch')
    profiles={t['id']:t['profile_sha'] for t in c['tasks']}
    if a.get('task_profile_bindings')!=profiles:raise ValueError('launch exact task/profile bindings')
    if a.get('U_reuse')!=c['U_reuse']:raise ValueError('launch reuse provenance required')
    from .training_config import load_ref
    proof=load_ref(a['U_equivalence_evidence_ref'])
    if proof.get('status')!='Passed' or proof.get('candidate_id')!=CANDIDATE_ID or proof.get('structural_revision')!=REVISION:raise ValueError('reviewed U-only equivalence evidence required')
    from .worker import source_binding
    if proof.get('source_binding')!=source_binding():raise ValueError('equivalence evidence source changed')
    verified_u_records(c)
def derive():
    from .training_config import validate,ROOT
    from .worker import source_binding
    from .training_state import SCHEMA
    from .gate_policy import policy
    r=reference();tasks=expected_tasks()
    return validate(dict(schema='moderntcn-uv-round2-training-v3',candidate_id=CANDIDATE_ID,structural_revision=REVISION,purpose='development',training_protocol='round2_complete_onecycle_v3',execution_authorized=False,checkpoint_schema=SCHEMA,gate_policy=policy(),tasks=tasks,existing_B=r['existing_B'],U_reuse=r['U_reuse'],F4_reuse={f'f6-H{t["h"]}':t['id'] for t in tasks if t['dataset']=='UrbanEV' and t['arm']=='UV'},source_binding=source_binding(),output_namespace='moderntcn-uv-v3-development-runs',preparation_head='170c27e685686ed1932849090b65c9c79de36e8c',backbone_variant='cfg_adapted_v1',B_ref='cfg_adapted_v1',B_cfg='cfg_adapted_v1',global_lock=r['global_lock'],max_run_epochs=360,max_optimizer_steps=1245080,task_scope=SCOPE,scope_reference_sha256=REFERENCE_SHA,parent_config_ref=r['parent_config_ref'],waves=expected_waves(),future_input_scope=future_input_scope()))
