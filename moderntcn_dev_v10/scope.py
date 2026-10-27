"""Exact v10 fresh U/UV scope plus explicitly attributed v3 V-only references."""
import copy,json,hashlib
from pathlib import Path
from .spec import CANDIDATE_ID,REVISION
SCOPE='development_fold6_epf_horizon_coupled_spline_v10'
REFERENCE_SHA='b63c3a7d013d1b65e00d42cb189cd584e6d87c472d4c05e4e40b5bd9535be0af'
MARKETS=('PJM','NP','BE','FR','DE')
def reference():
    raw=Path(__file__).with_name('scope_reference.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=REFERENCE_SHA:raise ValueError('frozen v10 scope reference changed')
    return json.loads(raw)
def expected_tasks(arms=('U','UV'), *, include_reused=False):
    out=[]
    for arm in arms:
        for p in reference()['prototypes']:
            t=copy.deepcopy(p);parts=t['id'].split('|');parts[0]=CANDIDATE_ID;parts[1]='v10';parts[-2]=arm
            t.update(id='|'.join(parts),model=CANDIDATE_ID,candidate_id=CANDIDATE_ID,structural_revision=copy.deepcopy(REVISION),arm=arm,use_u=arm in ('U','UV'),use_v=arm in ('V','UV'))
            # Latest Urban fold6 module goal; parent metadata was an extension.
            t['goal']=1 if t['dataset']=='UrbanEV' else 2
            out.append(t)
    return out if include_reused or arms==('V',) else [t for t in out if t['dataset']!='UrbanEV']
def expected_waves():
    ids=[t['id'] for t in expected_tasks()]
    return [ids[start+a:start+b] for start in (0,5) for a,b in ((0,4),(4,5))]
def future_input_scope():
    from utils.feature_schema import get_feature_schema
    return [{'dataset':'UrbanEV','fold':6,'h':h,'input_variant':v,'arm':'UV','task':'MS','C':len(get_feature_schema(v).feature_names),'features':list(get_feature_schema(v).feature_names),'target_idx':0,'aux_idx':list(range(1,len(get_feature_schema(v).feature_names))),'status':'scope_only_not_executable','candidate_id':CANDIDATE_ID,'structural_revision':copy.deepcopy(REVISION)} for v in ('F1','F2','F3') for h in (3,6,9,12)]
def validate_scope(c):
    r=reference()
    if c.get('task_scope')!=SCOPE or c.get('scope_reference_sha256')!=REFERENCE_SHA:raise ValueError('v10 explicit scope')
    if c['tasks']!=expected_tasks():raise ValueError('exact ordered 10 fresh U/UV tasks required')
    if c.get('V_reuse')!=r['V_reuse'] or c.get('existing_B')!=r['existing_B'] or c.get('parent_config_ref')!=r['parent_config_ref']:raise ValueError('exact nine V/B references')
    if c.get('P1_v9_reuse')!=r['P1_v9_reuse']:raise ValueError('exact eight v9 P1 references')
    if c.get('waves')!=expected_waves() or c.get('global_lock')!=r['global_lock']:raise ValueError('fixed four waves/global mutex')
    if (c.get('max_run_epochs'),c.get('max_optimizer_steps'))!=(200,57000):raise ValueError('fresh-only budget')
    if (sum(t['arithmetic']['max_epochs'] for t in c['tasks']),sum(t['arithmetic']['max_optimizer_steps'] for t in c['tasks']))!=(200,57000):raise ValueError('independent task arithmetic')
    if any(c.get(k)!='cfg_adapted_v1' for k in ('B_ref','B_cfg','backbone_variant')):raise ValueError('adapted baseline only')
    if c.get('future_input_scope')!=future_input_scope():raise ValueError('fold6 input scope must be retained without execution')
    reuse={f'f6-H{r["original_task"]["h"]}':dict(original_task_id=r['original_task_id'],producer='v9',status='equivalent_reference_only; future input ablation contract must authorize') for r in reference()['P1_v9_reuse'] if r['original_task']['arm']=='UV'}
    if c.get('F4_reuse')!=reuse:raise ValueError('v10 F4 identity only')
    return c
def waves(c,ids):
    if c['purpose']=='synthetic':
        ordered=[t['id'] for t in c['tasks']]
        if not ids or len(ids)!=len(set(ids)) or not set(ids)<=set(ordered):raise ValueError('synthetic task set')
        return [ids[i:i+4] for i in range(0,len(ids),4)]
    validate_scope(c);ordered=[t['id'] for t in c['tasks']]
    if not ids or len(set(ids))!=len(ids) or ids!=[i for i in ordered if i in ids]:raise ValueError('exact ordered fresh task subset')
    return [[i for i in w if i in ids] for w in expected_waves() if any(i in ids for i in w)]
def verified_v_records(c):
    """Read sealed TEXT metadata, never a legacy checkpoint or a new test."""
    from .training_config import load_ref,digest
    r=reference();runtime=load_ref(r['original_runtime_ref']);parent=load_ref(r['parent_config_ref'])
    result=[]
    for target,old,entry in zip(expected_tasks(('V',)),r['prototypes'],c['V_reuse']):
        objects={k:load_ref(ref) for k,ref in entry['references'].items()}
        m=objects['manifest.json'];metrics=objects['metrics.json'];identity=m['identity']
        if objects['config.json']!=parent:raise ValueError('V reuse original saved configuration')
        if entry['task_id']!=old['id'] or identity['task']!=old or metrics['task_id']!=old['id']:raise ValueError('V reuse exact original task')
        if identity['candidate_id']!='moderntcn-uv-dev-v3' or identity['structural_spec']['structural_revision']!=r['original_structural_revision'] or identity['source']!=entry['source_binding'] or identity['source']!=runtime['source_binding']:raise ValueError('V reuse original source')
        if identity['producer_commit']!=entry['producer_commit'] or identity['config_sha']!=entry['config_sha'] or digest(parent)!=entry['config_sha']:raise ValueError('V reuse producer/config')
        if m['status']!='completed' or metrics['test_calls']!=1 or m['identity_sha']!=entry['identity_sha'] or digest(identity)!=entry['identity_sha'] or objects['seal.json']['identity_sha']!=entry['identity_sha']:raise ValueError('V reuse seal/test identity')
        spec=objects['spec.json']
        if spec!=identity['structural_spec'] or spec['use_u'] is not False or spec['use_v'] is not True or identity['profile']!=target['structural_source_profile']:raise ValueError('V-only equivalent science required')
        if old['data_config']!=target['data_config'] or old['data_stat']!=target['data_stat'] or old['baseline_training']!=target['baseline_training']:raise ValueError('U data/training mismatch')
        result.append((target,dict(metrics,task_id=target['id'],original_task_id=old['id'],provenance='v3 equivalent V-only reuse; not a v10 fresh result')))
    return result
def validate_launch_scope(c,ids,a):
    waves(c,ids)
    if a.get('task_scope')!=SCOPE:raise ValueError('v10 scope authorization required')
    if a.get('resume_review_approved') is True:
        if a.get('resume_task_ids')!=ids or not a.get('previous_runtime_ref'):raise ValueError('explicit owned resume required')
    elif ids!=[t['id'] for t in c['tasks']]:raise ValueError('fresh v10 launch requires exact 10 tasks')
    if a.get('max_run_epochs')!=200 or a.get('max_optimizer_updates')!=57000:raise ValueError('launch fresh budget mismatch')
    profiles={t['id']:t['profile_sha'] for t in c['tasks']}
    if a.get('task_profile_bindings')!=profiles:raise ValueError('launch exact task/profile bindings')
    if a.get('V_reuse')!=c['V_reuse']:raise ValueError('launch reuse provenance required')
    from .training_config import load_ref
    proof=load_ref(a['V_equivalence_evidence_ref'])
    if proof.get('status')!='Passed' or proof.get('candidate_id')!=CANDIDATE_ID or proof.get('structural_revision')!=REVISION:raise ValueError('reviewed V-only equivalence evidence required')
    from .worker import source_binding
    if proof.get('source_binding')!=source_binding():raise ValueError('equivalence evidence source changed')
    if a.get('P1_v9_reuse')!=c['P1_v9_reuse']:raise ValueError('launch P1 provenance required')
    p1=load_ref(a['P1_v9_equivalence_evidence_ref'])
    if p1.get('status')!='Passed' or p1.get('candidate_id')!=CANDIDATE_ID or p1.get('structural_revision')!=REVISION or p1.get('source_binding')!=source_binding():raise ValueError('P1 equivalence/source required')
    verified_v_records(c);verified_p1_records(c)
def derive():
    from .training_config import validate,ROOT
    from .worker import source_binding
    from .training_state import SCHEMA
    from .gate_policy import policy
    r=reference();tasks=expected_tasks()
    return validate(dict(schema='moderntcn-uv-round2-training-v10',candidate_id=CANDIDATE_ID,structural_revision=REVISION,purpose='development',training_protocol='round2_complete_onecycle_v3',execution_authorized=False,checkpoint_schema=SCHEMA,gate_policy=policy(),tasks=tasks,existing_B=r['existing_B'],V_reuse=r['V_reuse'],P1_v9_reuse=r['P1_v9_reuse'],F4_reuse={f'f6-H{q["original_task"]["h"]}':dict(original_task_id=q['original_task_id'],producer='v9',status='equivalent_reference_only; future input ablation contract must authorize') for q in r['P1_v9_reuse'] if q['original_task']['arm']=='UV'},source_binding=source_binding(),output_namespace='moderntcn-uv-v10-development-runs',preparation_head='385f53b458f1140c143c2c1edc558b2ceb9e24b7',backbone_variant='cfg_adapted_v1',B_ref='cfg_adapted_v1',B_cfg='cfg_adapted_v1',global_lock=r['global_lock'],max_run_epochs=200,max_optimizer_steps=57000,task_scope=SCOPE,scope_reference_sha256=REFERENCE_SHA,parent_config_ref=r['parent_config_ref'],waves=expected_waves(),future_input_scope=future_input_scope()))


def verified_p1_records(c):
    """Eight immutable v9 text results; no checkpoint migration or new test."""
    from .training_config import load_ref, digest
    r=reference()
    if c.get('P1_v9_reuse')!=r['P1_v9_reuse']:
        raise ValueError('P1 reference list changed')
    runtime=load_ref(r['v9_runtime_ref']); parent=load_ref(r['v9_config_ref'])
    targets=[t for t in expected_tasks(include_reused=True) if t['dataset']=='UrbanEV']
    by_cell={(t['dataset'],t['h'],t['fold'],t['arm']):t for t in targets}
    seen=set(); result=[]
    for entry in r['P1_v9_reuse']:
        cell=entry['consumer_cell']; key=tuple(cell[k] for k in ('dataset','h','fold','arm'))
        if key not in by_cell or key in seen:raise ValueError('P1 duplicate/foreign logical cell')
        seen.add(key); target=by_cell[key]
        o={k:load_ref(ref) for k,ref in entry['references'].items()}
        m=o['manifest.json']; identity=m['identity']; spec=o['spec.json']; metrics=o['metrics.json']; old=entry['original_task']
        if identity['task']!=old or old['id']!=entry['original_task_id'] or metrics['task_id']!=old['id']:
            raise ValueError('P1 original task identity')
        if identity['candidate_id']!='moderntcn-uv-dev-v9' or identity['producer_commit']!=entry['producer_commit'] or identity['structural_spec']['structural_revision']!=entry['producer_revision']:
            raise ValueError('P1 original producer')
        if o['config.json']!=parent or digest(parent)!=entry['config_sha'] or identity['config_sha']!=entry['config_sha']:
            raise ValueError('P1 saved configuration')
        if identity['source']!=entry['source_binding'] or identity['source']!=runtime['source_binding']:
            raise ValueError('P1 original source')
        if spec!=identity['structural_spec'] or spec['pred_len']!=1 or not spec['use_u'] or spec['use_v']!=target['use_v']:
            raise ValueError('P1 output/arm branch mismatch')
        for key in ('dataset','h','fold','arm','data_config','data_stat','baseline_training','structural_source_profile'):
            if old[key]!=target[key]:raise ValueError('P1 scientific profile mismatch: '+key)
        if identity['profile']!=target['structural_source_profile'] or m['status']!='completed' or metrics['test_calls']!=1:
            raise ValueError('P1 science/completion/test declaration')
        if digest(identity)!=entry['identity_sha'] or m['identity_sha']!=entry['identity_sha'] or o['seal.json']['identity_sha']!=entry['identity_sha']:
            raise ValueError('P1 sealed identity')
        result.append((target,dict(metrics,task_id=target['id'],original_task_id=old['id'],provenance='v9 exact P1 U/UV reuse; not v10 fresh/test')))
    if len(result)!=8:raise ValueError('eight P1 results required')
    return result
