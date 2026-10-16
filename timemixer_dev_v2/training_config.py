"""Resolved second-round contracts, separate from non-executing StructuralSpec."""
import copy
import hashlib
import json
from pathlib import Path
from .spec import CANDIDATE_ID, REVISION
from utils.ch3_contract import digest
from .gate_policy import policy, validate_config_policy

ROOT = Path(__file__).resolve().parents[1]
REVIEW_ROOT = ROOT.parent / 'amd-execution-evidence/m6/m6-original-round2-round3-result-review-v1-20261015'
REVIEW_SHAS = {'round2-raw-result-review.json':'b69ebfce815d029bb2819f4ea33fa7c60f13ca74ccbe3f432dce3e3937bd4783',
              'round3-raw-result-review.json':'cbee1e91b61b63d0bd3a0dfa890c1ca28b3987e4d7d66b0a5a02b3ada21563dd',
              'task-audit.v2.json':'1dbc648f6e9a9243f06937578622561d0fb92ad3fb1e35a8766472be22bfd74f'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_ref(ref):
    if sha(ref['path']) != ref['sha256']: raise ValueError('bound file SHA mismatch')
    return json.loads(Path(ref['path']).read_text())

def windows(data, profile, fold):
    ds = profile['dataset']; T = profile['T']; P = profile['pred_len']
    if ds == 'UrbanEV':
        a,b,c = data['urban_folds'][fold-1]
        return {k:(end-start-T-profile['H']+1)*data.get('synthetic_nodes',275)
                for k,start,end in [('train',0,a),('validation',a,b),('test',b,c)]}
    a,b,c = data['datasets'][ds]['endpoints']
    return dict(train=a-T-P+1,validation=b-a-P+1,test=c-b-P+1)

def arithmetic(data, p, fold):
    n = windows(data,p,fold); tr = p['training']
    if min(n.values()) <= 0: raise ValueError('empty declared split')
    batches = n['train']//tr['batch']
    if batches < 1 or tr['scheduler']['steps_per_epoch'] != batches: raise ValueError('OneCycle/window arithmetic mismatch')
    return dict(windows=n,train_batches=batches,eval_batches={k:(n[k]+tr['eval_batch']-1)//tr['eval_batch'] for k in ('validation','test')},
                max_epochs=tr['epochs'],max_optimizer_steps=batches*tr['epochs'])

def validate_task(task, purpose):
    from utils.ch3_onecycle import validate_config
    from .spec import StructuralSpec
    p=task['structural_source_profile'];tr=p['training']
    if (p['dataset'],p['input_variant'],p['H'])!=(task['dataset'],task['input_variant'],task['h']) or task['seed']!=2024 or task['task']!=p.get('task','MS'):raise ValueError('task/profile identity mismatch')
    if task['model'] != CANDIDATE_ID or p['model'] != 'TimeMixer': raise ValueError('candidate/source identities must be distinct')
    if task.get('candidate_id') != CANDIDATE_ID or task.get('structural_revision') != REVISION: raise ValueError('v2 task/revision identity required')
    if task['arm'] not in ('B','U','V','UV') or (purpose=='development' and task['arm']=='B'): raise ValueError('no real B retraining')
    if (task['use_u'],task['use_v']) != (task['arm'] in ('U','UV'),task['arm'] in ('V','UV')): raise ValueError('arm/switch mismatch')
    StructuralSpec.from_profile(p,use_u=task['use_u'],use_v=task['use_v'],task=task['task'],fold=task['fold'])
    validate_config(tr['scheduler'])
    if (tr['seed'],tr['optimizer'],tr['dtype'],tr['accumulation']) != (2024,'Adam','float32',1): raise ValueError('frozen optimization')
    if tr['from_scratch'] is not True or tr['search']!=0:raise ValueError('fresh/no-search contract required')
    if tr['betas'] != [0.9,0.999] or (tr['eps'],tr['weight_decay'],tr['lr']) != (1e-8,1e-7,0.01): raise ValueError('frozen Adam')
    if not tr['train_shuffle'] or not tr['train_drop_last'] or tr['eval_drop_last'] or tr['workers'] != 0: raise ValueError('unsupported inherited loader options')
    if task['arithmetic'] != arithmetic(task['data_config'],p,task['fold']): raise ValueError('arithmetic changed')
    if task['profile_sha'] != digest(p): raise ValueError('profile identity changed')
    if purpose=='development':
        if tr != task['baseline_training']: raise ValueError('training differs from second-round B')
        if tr['batch']!=128 or tr['eval_batch']!=128 or tr['threads']!=4: raise ValueError('round-two batch/thread mismatch')
        E,pat=(20,5) if task['dataset'] in ('UrbanEV','PJM','NP','BE','FR','DE') else ((20,10) if task['dataset']=='Weather' else (10,None))
        if (tr['epochs'],tr['patience']) != (E,pat): raise ValueError('round-two cap/patience mismatch')
        if p['T'] != (12 if task['dataset']=='UrbanEV' else 96): raise ValueError('round-two L mismatch')
    return task

def validate(config):
    validate_config_policy(config)
    if config.get('candidate_id') != CANDIDATE_ID or config.get('structural_revision') != REVISION: raise ValueError('v2 configuration/revision identity required')
    if config['schema']!='timemixer-uv-round2-training-v1' or config['training_protocol']!=('round2_complete_onecycle_v3' if config['purpose']=='development' else 'synthetic_round2_pipeline_v1'): raise ValueError('training schema/protocol mismatch')
    if config['purpose'] not in ('development','synthetic'): raise ValueError('purpose required')
    if config['execution_authorized'] is not False:raise ValueError('scientific configuration is not a launch authorization')
    ids=[t['id'] for t in config['tasks']]
    if len(ids)!=len(set(ids)): raise ValueError('duplicate task')
    for t in config['tasks']:validate_task(t,config['purpose'])
    if config['purpose']=='development':
        from .training_state import SCHEMA
        if config.get('checkpoint_schema')!=SCHEMA:raise ValueError('checkpoint technical schema binding')
        from collections import Counter
        if len(ids)!=77 or Counter(t['goal'] for t in config['tasks'])!={1:24,3:24,2:29}:raise ValueError('exact 77 scope required')
        if sum(t['arithmetic']['max_epochs'] for t in config['tasks'])!=1340:raise ValueError('1340 epoch cap required')
        if len({t['baseline_cell_id'] for t in config['tasks'] if t['goal']!=3})!=37:raise ValueError('37 B bindings')
        import itertools
        expected={(1,'UrbanEV',f,h,'F4',a) for f,h,a in itertools.product((1,4),(3,6,9,12),('U','V','UV'))}
        expected|={(3,'UrbanEV',f,h,v,'UV') for f,h,v in itertools.product((1,4),(3,6,9,12),('F1','F2','F3'))}
        expected|={(2,d,1,h,'M','UV') for d,h in itertools.product(('ETTh1','ETTh2','ETTm1','ETTm2','Weather','Exchange'),(96,192,336,720))}
        expected|={(2,d,1,24,'MS','UV') for d in ('PJM','NP','BE','FR','DE')}
        if {(t['goal'],t['dataset'],t['fold'],t['h'],t['input_variant'],t['arm']) for t in config['tasks']}!=expected:raise ValueError('development grid expanded/changed')
        for t in config['tasks']:
            expected_id='|'.join((CANDIDATE_ID,t['dataset'],t['input_variant'],'f'+str(t['fold']),'h'+str(t['h']),t['arm'],'s2024'))
            if t['id']!=expected_id:raise ValueError('exact scientific task ID required')
        if set(config['F4_reuse'].values())!={t['id'] for t in config['tasks'] if t['goal']==1 and t['arm']=='UV'}:raise ValueError('F4 reuse identity mismatch')
        expected_reuse={'f'+str(t['fold'])+'-H'+str(t['h']):t['id'] for t in config['tasks'] if t['goal']==1 and t['arm']=='UV'}
        if config['F4_reuse']!=expected_reuse:raise ValueError('exact same-version fold/H F4 reuse required')
        if set(config['existing_B'])!={t['baseline_pair_cell_id'] for t in config['tasks'] if t['goal']!=3} or any(t['baseline_pair_cell_id'] is not None for t in config['tasks'] if t['goal']==3):raise ValueError('B pairing/input ablation roles')
    return config

def derive():
    """Copy the pinned reviewed v1 recipe; change only version/technical identity and gate policy."""
    source = ROOT/'configs/timemixer_uv_round2_resolved_v1.json'
    expected = '4839ccdaf2c2e67edcf2bfda86cf6786c96f2e7a64b17599bd7d3879aaf618a5'
    if sha(source) != expected: raise ValueError('reviewed v1 resolved snapshot changed')
    config = json.loads(source.read_text())
    old_id = config['candidate_id']
    mapping = {t['id']:t['id'].replace(old_id, CANDIDATE_ID, 1) for t in config['tasks']}
    for task in config['tasks']:
        task.update(id=mapping[task['id']], model=CANDIDATE_ID, candidate_id=CANDIDATE_ID,
                    structural_revision=copy.deepcopy(REVISION))
    config['F4_reuse'] = {k.replace(old_id, CANDIDATE_ID):mapping[v] for k,v in config['F4_reuse'].items()}
    config.update(candidate_id=CANDIDATE_ID, structural_revision=copy.deepcopy(REVISION), gate_policy=policy(),
                  derivation_source=dict(path=str(source), sha256=expected, commit='437b43b5ace68329ceb5a18cf75952baaa9103c4'),
                  output_namespace='timemixer-uv-v2-development-runs')
    from .worker import source_binding
    from .training_state import SCHEMA
    config['checkpoint_schema']=SCHEMA
    config['source_binding']=source_binding()
    return validate(config)
