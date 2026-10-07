"""Fixed round-two amendment and source selection; no model/test execution."""
import copy
from pathlib import Path
from functools import lru_cache
from utils.ch3_contract import ROOT, digest, step_arithmetic
from utils.ch3_native_recovery_records import bound, ref
from utils import ch3_type1_ett as ett

PROTOCOL = 'baseline-unified96-onecycle001-v3-amend1-m128-recovery1'
ID = 'm6-baseline-unified96-oc01-v3-amend1-m128-recovery1'
STAGE = 'M_AMEND'
DATASETS = ('ETTh2', 'ETTm1', 'ETTm2', 'Weather')
MODELS = ('AMD','DLinear','PatchTST','iTransformer','TimeMixer','ModernTCN','TimeXer')
from utils import ch3_ms_seal_recovery as recovery
PACKAGE = recovery.PACKAGE/'amendment'
RESULT = recovery.RESULT/'round2-amendment'
HISTORICAL_RESULT = ROOT.parent/'amd-execution-evidence/m6/m6-formal-launch-dhozikhu/baseline-unified96-onecycle001-v3-amend1'
PARENT_REF = dict(path=str(recovery.M_FILE),sha256='7f4720a723e31e3f5b291466c5b483021d480e0152c4317871f8d8097b4b14a5')
RECIPE_REF = dict(path=str(recovery.PACKAGE.parent/'baseline-unified96-onecycle001-v1/onecycle-author-recipe.json'),sha256='e2e1cb27c93d2e865ea68c1e39858363ebc216a5c91df0ec773862bbc0a9185f')

@lru_cache(maxsize=1)
def parent(): return bound(PARENT_REF)

def selected():
    rows = parent()['tasks']; result = []
    for model in MODELS:
        for dataset in DATASETS:
            for h in (96,192,336,720):
                old = next(t for t in rows if t['model']==model and t['dataset']==('Weather' if dataset=='Weather' else 'ETTh1') and t['h']==h)
                t = copy.deepcopy(old); t['dataset'] = dataset
                result.append(t)
    return result

def task(t):
    t = copy.deepcopy(t); group = t['model']+'-'+t['dataset']+'-M-oc01-v3-amend1-m128-recovery1'
    t.update(id=group+'-f1-h'+str(t['h'])+'-s2024',group=group,profile=group+'-h'+str(t['h']))
    return t

def tasks(): return [task(t) for t in selected()]

def source_task(t):
    return next(x for x in parent()['tasks'] if x['model']==t['model'] and x['dataset']==('Weather' if t['dataset']=='Weather' else 'ETTh1') and x['h']==t['h'])

def inherited(t,c=None):
    p = copy.deepcopy(parent()['resolved_profiles'][source_task(t)['id']])
    if t['dataset']=='Weather':
        p['training']['epochs']=20; p['training']['scheduler']['epochs']=20
    else:
        p['dataset']=t['dataset']; evidence=ett.sources()['datasets'][t['dataset']]
        if 'freq' in p['structure']: p['structure']['freq']=evidence['timeF_freq']
        if p.get('time_mark'):
            from utils.ch3_time_marks import policy
            p['time_mark']=policy(p)
        p['training'].update(batch=128,eval_batch=128,epochs=10,patience=None)
        windows=ett.dataset(t['dataset'])['endpoints'][0]-p['T']-p['pred_len']+1
        p['training']['scheduler']['steps_per_epoch']=windows//128
    if c is not None:
        from utils.ch3_amend_ett_identity_recovery import weather_profile
        p=weather_profile(c,t,p)
    return p

def dataset(name):
    return ett.dataset(name) if name in ett.NEW_DATASETS else copy.deepcopy(parent()['datasets'][name])

def policy(model,name):
    return ett.numeric_policy(model,name) if name in ett.NEW_DATASETS else copy.deepcopy(parent()['baseline_unified']['numeric_policies'][model+'-'+name])

def validate(c):
    from utils import ch3_type1_tasks as s
    if c.get('type1_followup')!=s.ID or c['baseline_unified']['id']!=PROTOCOL or c['baseline_unified']['stage']!=STAGE or c['tasks']!=tasks(): raise ValueError('exact 112 fresh amendment tasks')
    if c['baseline_unified']['direct_parent_ref']!=PARENT_REF or c['baseline_unified']['recipe_ref']!=RECIPE_REF: raise ValueError('fixed original v3/OneCycle ancestry')
    bound(RECIPE_REF)
    if c['sources']!=parent()['sources'] or c['urban_folds']!=parent()['urban_folds'] or c['urban_input_variants']!=parent()['urban_input_variants']: raise ValueError('amendment source/split identity')
    if set(c['datasets'])!=set(DATASETS) or any(c['datasets'][d]!=dataset(d) for d in DATASETS): raise ValueError('independent ETT or frozen Weather data identity')
    expected_refs=dict(source=ett.SOURCE_REF,paper=ett.PAPER_REF,onecycle=RECIPE_REF)
    if c['baseline_unified'].get('extension_refs')!=expected_refs: raise ValueError('amendment source/recipe bindings')
    expected_policies={m+'-'+d:policy(m,d) for m in MODELS for d in DATASETS}
    if c['baseline_unified'].get('numeric_revision_ref'):
        from utils.ch3_moderntcn_etth1_recovery import validate_revision
        expected_policies=validate_revision(c)
    if c['baseline_unified']['numeric_policies']!=expected_policies: raise ValueError('M-only inherited numerical policies')
    data=bound(c['baseline_unified']['data_ref'])
    from utils.ch3_time_marks import metadata
    for t in c['tasks']:
        p=inherited(t,c); old=source_task(t)
        expected=dict(task_id=old['id'],profile_sha=digest(parent()['resolved_profiles'][old['id']]),config_ref=PARENT_REF)
        if c['resolved_profiles'][t['id']]!=p or c['baseline_unified']['parent_refs'][t['id']]!=expected: raise ValueError('only authorized effective profile changes')
        a=step_arithmetic(c,t); m=data['metadata'][t['dataset']][t['id']]
        if any(m.get(k)!=v for k,v in metadata(p).items()) or m['window_counts']!={'train':a['train_windows'],'validation':a['validation_windows']} or data['data_bindings'][t['dataset']][t['id']]!=digest(m): raise ValueError('real ETT/Weather metadata, marks and windows')
    return c

def key(t): return '|'.join(str(t[k]) for k in ('task','model','dataset','fold','h','seed'))

def cell_tasks():
    from utils.ch3_type1_upstream import records
    start,_=records(); ms=bound(start['config_refs']['MS']); m=parent()
    return ms['tasks']+m['tasks']+[t for t in tasks() if t['dataset']!='Weather']

def expected_cells():return [key(t) for t in cell_tasks()]

def validate_index(v):
    cells=v.get('cells',[]); ids=[x['cell_id'] for x in cells];expected_tasks={key(t):t for t in cell_tasks()}
    if len(ids)!=371 or len(set(ids))!=371 or set(ids)!=set(expected_tasks) or v.get('effective_counts')!={'MS':203,'M':168,'total':371} or v.get('planned_formal_executions')!=399: raise ValueError('371 effective cells are distinct from 399 planned runs')
    for row in cells:
        t=expected_tasks[row['cell_id']]
        if any(row.get(k)!=t['h' if k=='H' else k] for k in ('task','model','dataset','fold','H','seed')):raise ValueError('exact cell/source identity; dataset labels cannot redefine selection')
        expected='amend1' if row['dataset'] in DATASETS else 'original_v3' if row['task']=='MS' else 'base_m128'
        if row['origin']!=expected: raise ValueError('fixed all-Weather20 selection; no metric-based choice')
        if row['dataset']=='Weather' and not row.get('supersedes'): raise ValueError('old Weather10 remains referenced as superseded')
        if row['origin']=='amend1' and row['scientific_protocol']!=PROTOCOL: raise ValueError('amendment execution identity cannot masquerade as original')
        if row['origin']=='base_m128' and row['scientific_protocol']!=recovery.M_PROTOCOL:raise ValueError('ETTh1/Exchange must use new M128 source')
    return v

def artifact_rows(boundary_ref):
    b=bound(boundary_ref); result={}
    for receipt in b['receipts'].values():
        group=bound(receipt)
        for task_id,files in group['artifacts'].items():
            if task_id in result: raise ValueError('duplicate sealed task')
            result[task_id]=files
    if set(result)!=set(b['task_ids']): raise ValueError('sealed artifact index incomplete')
    return result

def build_index(old_boundaries,amendment_ref,c):
    from utils.ch3_type1_upstream import records,OLD_RESULT,OLD_PROTOCOL,BASE
    start,_=records(); old={'MS':bound(start['config_refs']['MS']),'M':parent()}
    sealed_m=bound(old_boundaries['M'])
    if sealed_m.get('protocol_sha') and sealed_m['protocol_sha']!=digest(old['M']):
        from utils.ch3_type1_chain import configs
        actual=configs()['M_BASE']
        if sealed_m['protocol_sha']!=digest(actual) or any(actual[k]!=old['M'][k] for k in ('tasks','resolved_profiles','datasets','sources')):raise ValueError('exact retained M producer config, unchanged science')
        old['M']=actual
    banks={stage:artifact_rows(old_boundaries[stage]) for stage in old}; new=artifact_rows(amendment_ref)
    if set(new)!={t['id'] for t in c['tasks']}: raise ValueError('all 112 amendment results must be sealed before selection')
    cells=[]
    def add(config,t,files,origin,supersedes=None):
        r=bound(files['result.json']);m=bound(files['manifest.json']);p=config['resolved_profiles'][t['id']]
        science=config['baseline_unified']['id']
        sealed=bound(old_boundaries['MS' if origin=='original_v3' else 'M']) if origin!='amend1' else bound(amendment_ref)
        source_root=recovery.RESULT/'M_BASE' if origin=='base_m128' else OLD_RESULT/'MS' if origin=='original_v3' else RESULT/STAGE
        if sealed.get('adopted_source_ref') and origin=='base_m128':
            producer=bound(sealed['adopted_source_ref'])
            if any(sealed.get(k)!=v for k,v in producer.items()):raise ValueError('adopted M boundary changed original producer identity')
            source_root=Path(sealed['adopted_source_ref']['path']).parents[2]/'M_BASE'
        root=source_root/('formal-'+t['model'])/t['id']
        if files['result.json']['path']!=str(root/'result.json') or files['manifest.json']['path']!=str(root/'manifest.json') or m['task']!=t or m['profile']!=p or r['id']!=t['id'] or r['profile_sha']!=digest(p) or r['protocol_sha']!=digest(config) or r['scientific_protocol']!=science or r['commit']!=m['identity']['commit'] or r.get('final_test',{}).get('calls')!=1: raise ValueError('exact source/config/execution/test-once provenance')
        if origin=='original_v3' and r['commit']!=BASE: raise ValueError('original execution commit must remain original')
        if origin!='original_v3' and r['commit']!=sealed['commit']:raise ValueError('actual stage producer commit; adoption cannot relabel training')
        cells.append(dict(cell_id=key(t),task=t['task'],model=t['model'],dataset=t['dataset'],fold=t['fold'],H=t['h'],seed=t['seed'],origin=origin,scientific_protocol=science,execution_commit=r['commit'],profile_sha=r['profile_sha'],protocol_sha=r['protocol_sha'],result_ref=files['result.json'],manifest_ref=files['manifest.json'],runtime_ref=files['runtime.json'],mse=r['mse'],mae=r['mae'],supersedes=supersedes,std='N/A'))
    for stage,config in old.items():
        for t in config['tasks']:
            if t['dataset']=='Weather': continue
            add(config,t,banks[stage][t['id']],'original_v3' if stage=='MS' else 'base_m128')
    for t in c['tasks']:
        prior=source_task(t);supersedes=banks['M'][prior['id']]['result.json'] if t['dataset']=='Weather' else None
        add(c,t,new[t['id']],'amend1',supersedes)
    return validate_index(dict(purpose='round2_revised_main_results_v1',revision=PROTOCOL,effective_counts=dict(MS=203,M=168,total=371),planned_formal_executions=399,original_runs=287,amendment_runs=112,Weather_replacement_runs=28,cells=cells,technical_complete=True,result_review='pending',source_rule='all Weather20 replaces Weather10 irrespective of metrics',history_test_seen=True))

def summary():
    path=RESULT/'queue/round2-main-results.json'
    if not path.exists(): return dict(revision='Pending',effective_expected=371,planned_formal_executions=399,Weather20='Pending; no Weather10 value relabeled as 20',result_review='pending')
    boundary=bound(ref(RESULT/'queue/round2-boundary.json'))
    if boundary.get('technical_complete') is not True: raise ValueError('round2 main result boundary incomplete')
    return validate_index(bound(boundary['main_index_ref']))
