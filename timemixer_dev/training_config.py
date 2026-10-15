"""Resolved second-round contracts, separate from non-executing StructuralSpec."""
import copy
import hashlib
import json
from pathlib import Path
from .spec import CANDIDATE_ID
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
        if set(config['existing_B'])!={t['baseline_pair_cell_id'] for t in config['tasks'] if t['goal']!=3} or any(t['baseline_pair_cell_id'] is not None for t in config['tasks'] if t['goal']==3):raise ValueError('B pairing/input ablation roles')
    return config

def derive():
    """Metadata only: no model construction, real CSV or checkpoint reads."""
    from utils.feature_schema import get_feature_schema
    plan=json.loads((ROOT/'configs/timemixer_uv_development_plan_v1.json').read_text())
    bank=json.loads((ROOT/plan['baseline_sources']).read_text())
    if sha(ROOT/plan['baseline_sources'])!=plan['baseline_sources_sha256']:raise ValueError('reviewed bank changed')
    audits=load_ref(dict(path=str(REVIEW_ROOT/'task-audit.v2.json'),sha256=REVIEW_SHAS['task-audit.v2.json']))
    for name,h in REVIEW_SHAS.items():load_ref(dict(path=str(REVIEW_ROOT/name),sha256=h))
    bank_by={r['cell_id']:r for r in bank['required_cells']}; review_by={(r['dataset'],r['fold'],r['H']):r for r in audits if r['round']==2 and r['model']=='TimeMixer'}
    rows=[];paired={};cache={}
    for t in plan['tasks']:
        b=bank_by[t['structural_source_cell_id']];s=b['second_round'];a=review_by[(b['dataset'],b['fold'],b['H'])]
        for k,v in [('producer',s['training_commit']),('profile_sha',s['profile_sha256']),('protocol_sha',s['protocol_sha256']),('data_sha',s['data_sha']),('mse',s['mse']),('mae',s['mae'])]:
            if a[k]!=v:raise ValueError('37-cell raw review mismatch: '+k)
        if a['artifact_refs']['result.json'] != s['result_ref'] or a['artifact_refs']['manifest.json']!=s['manifest_ref']:raise ValueError('original local source changed')
        if s['config_ref']['path'] not in cache:cache[s['config_ref']['path']]=load_ref(s['config_ref'])
        c=cache[s['config_ref']['path']]
        source=copy.deepcopy(c['resolved_profiles'][a['run_id']]);p=copy.deepcopy(source)
        if digest(source)!=s['profile_sha256']:raise ValueError('source profile changed')
        if t['dataset']=='UrbanEV':
            names=list(get_feature_schema(t['input_variant']).feature_names)
            p.update(input_variant=t['input_variant'],features=names,C=len(names),target_idx=0,aux_idx=list(range(1,len(names))))
        changed=sorted(k for k in p if p[k]!=source[k])
        if not set(changed)<= {'input_variant','features','C','target_idx','aux_idx'}:raise ValueError('non-schema source change')
        data={'datasets':{t['dataset']:copy.deepcopy(c['datasets'][t['dataset']])}}
        if t['dataset']=='UrbanEV':data['urban_folds']=copy.deepcopy(c['urban_folds'])
        metadata=json.loads(Path(s['manifest_ref']['path']).read_text())['metadata']
        files=[Path(data['datasets'][t['dataset']]['path'])/f for f in ('volume.csv','e_price.csv','s_price.csv','weather_central.csv')] if t['dataset']=='UrbanEV' else [Path(data['datasets'][t['dataset']]['path'])]
        stat_refs={str(f):{k:getattr(f.stat(),k) for k in ('st_size','st_mtime_ns','st_ino','st_dev')} for f in files}
        rows.append(dict(id=t['id'],model=CANDIDATE_ID,candidate_id=CANDIDATE_ID,goal=t['goal'],arm=t['arm'],use_u=t['use_u'],use_v=t['use_v'],dataset=t['dataset'],task=t['task'],fold=t['fold'],h=t['H'],seed=2024,input_variant=t['input_variant'],
                         structural_source_profile=p,profile_sha=digest(p),baseline_training=source['training'],baseline_cell_id=b['cell_id'],baseline_pair_cell_id=None if t['goal']==3 else b['cell_id'],baseline_source=s['config_ref'],baseline_profile_sha=s['profile_sha256'],derivation_changed_fields=changed,
                         data_config=data,data_stat=stat_refs,baseline_prefix_metadata=metadata.get('prefixes',{}),
                         baseline_record_prefix={k:metadata[k] for k in ('parsed_records','record_prefix_sha256','fingerprint_semantics') if k in metadata},arithmetic=arithmetic(data,p,t['fold'])))
        paired[b['cell_id']]=dict(round=2,task_id=a['run_id'],mse=a['mse'],mae=a['mae'],metric_space='train-standardized',result_ref=s['result_ref'],profile_sha=s['profile_sha256'],review_scope='accepted source/execution/raw metrics/sealed integrity; no fresh tensor/optimizer/RNG load audit',review_accepted_by_user=True)
    config=dict(schema='timemixer-uv-round2-training-v1',candidate_id=CANDIDATE_ID,purpose='development',training_protocol='round2_complete_onecycle_v3',gate_policy=policy(),execution_authorized=False,
                preparation_head='491ce9848575f744125def9950e994a9bc5a65e8',review_refs={n:dict(path=str(REVIEW_ROOT/n),sha256=h) for n,h in REVIEW_SHAS.items()},tasks=rows,existing_B=paired,
                F4_reuse=plan['goal3']['F4_same_identity_reuse'],max_run_epochs=1340,max_optimizer_steps=sum(t['arithmetic']['max_optimizer_steps'] for t in rows),
                global_lock='/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-formal-launch-dhozikhu/project-gpu0.lock',q_max=4,
                resource_policy='reviewed_q4_no_mandatory_probe',batch_authorization='explicit task ID subset in separately reviewed post-closure launch binding; this file grants none')
    from .worker import source_binding
    from .training_state import SCHEMA
    config['checkpoint_schema']=SCHEMA
    config['source_binding']=source_binding()
    return validate(config)
