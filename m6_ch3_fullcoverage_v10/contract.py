"""Fixed 54 logical cells, 39 missing-sample evaluations and one offline merge."""
from pathlib import Path
from m6_ch3_supplement_v10.contract import ROOT,read,sha,write_new,load_ref
from .analysis import MODELS,MARKETS,HS,digest,index_sha,urban_plan

SCOPE='m6_ch3_fullcoverage_reporting_v10_v1'
PURPOSE='posthoc_frozen_test_data_collection'
AUTH_SCHEMA='m6-ch3-fullcoverage-launch-v1'
ATTEMPT='all275-4h-epf5-sixmodels-first-r1'
BUDGET=dict(model_jobs=39,offline_jobs=1,waves=40,new_test_accesses=39,batches=10973,samples=1257300,elements=4866000,training=0,backward=0,optimizer=0,logical_cells=54,whole_references=15,partial_references=5,region_metric_cells=6600)


def jobs(s):
    out=[]
    for d in MARKETS:
        for m in MODELS:
            cell=f'P-{d}-{m}'
            if cell in s['inherited']:continue
            p=s['rows'][cell]['profile'];bs=p['training']['eval_batch'];n=10460
            out.append(dict(id=cell,phase='P',cell=cell,dataset=d,model=m,H=24,P=24,split='test',purpose=PURPOSE,subset='full_test',sample_set_sha256=digest(dict(dataset=d,indices_sha256=index_sha(range(n)),source_data=s['rows'][cell]['data_ref'])),expected=dict(samples=n,elements=n*24,batches=(n+bs-1)//bs,eval_batch=bs)))
    for h in HS:
        for m in MODELS:
            cell=f'C-H{h}-{m}';partial=cell in s['inherited'];key=f'H{h}-'+('remaining265' if partial else 'all275');ref=s['plans'][key];plan=load_ref(ref);n=len(plan['dataset_indices']);bs=s['rows'][cell]['profile']['training']['eval_batch']
            out.append(dict(id=cell,phase='C',cell=cell,dataset='UrbanEV',model=m,H=h,P=1,split='test',purpose=PURPOSE,subset=plan['subset'],plan_ref=ref,sample_set_sha256=plan['sample_set_sha256'],expected=dict(samples=n,elements=n,batches=(n+bs-1)//bs,eval_batch=bs)))
    return out+[dict(id='O',phase='O',purpose='offline_full_data_merge_no_selection')]


def validate(c):
    if c.get('scope')!=SCOPE or c.get('purpose')!=PURPOSE or c.get('execution_authorized') is not False or c.get('base_candidate_id')!='moderntcn-uv-dev-v10' or c.get('attempt')!=ATTEMPT:raise PermissionError('fixed explicit read-only collection scope')
    s=load_ref(c['sources_ref']);expected={f'P-{d}-{m}' for d in MARKETS for m in MODELS}|{f'C-H{h}-{m}' for h in HS for m in MODELS}
    reuse={f'P-{d}-{m}' for d in ('NP','BE','DE') for m in MODELS if m!='PatchTST'}|{f'C-H6-{m}' for m in MODELS if m!='PatchTST'}
    if set(s['rows'])!=expected or set(s['inherited'])!=reuse:raise ValueError('exact54 cells /15 whole /5 partial')
    if c['jobs']!=jobs(s) or len(c['jobs'])!=40 or c['waves']!=[[j['id']] for j in c['jobs']] or c['q_max']!=1:raise ValueError('exact ordered40 serial waves')
    if c['budget']!=BUDGET or c['numeric_policy']!=dict(atol=2e-6,rtol=2e-5,identity='exact'):raise ValueError('fixed budget/policy')
    if c['phase_concurrency']!={'P':1,'C':1,'O':1} or c['resource_plan']['waves']!=c['waves'] or c['resource_plan']['phase_concurrency']!=c['phase_concurrency'] or c['resource_plan']['q_max']!=1 or c['resource_plan']['mandatory_probe'] is not False:raise ValueError('serial inference/no probes')
    if c['display_policy']!=dict(auto_select_regions=False,final_region_nodes=None,final_EPF_models=None,region_interface='explicit H/node_ids/ordered models only',EPF_interface='explicit market/ordered models only'):raise ValueError('selection is deferred')
    if [sum(j['expected'][k] for j in c['jobs'][:-1]) for k in ('batches','samples','elements')]!=[10973,1257300,4866000]:raise ValueError('arithmetic')
    for cell,r in s['rows'].items():
        p,t=r['profile'],r['task'];urban=cell.startswith('C-');d='UrbanEV' if urban else cell.split('-')[1];h=int(cell.split('-')[1][1:]) if urban else 24;m=cell.split('-')[-1]
        if p['dataset']!=d or t['dataset']!=d or p['H']!=h or t['h']!=h or t['task']!='MS' or t['seed']!=2024 or t['fold']!=(6 if urban else 1):raise ValueError('cell dataset/H/task/seed/fold')
        if (p['T'],p['pred_len'],p['C'],p['target_idx'])!=((12,1,11,0) if urban else (96,24,3,2)):raise ValueError('original input/target shape')
        if Path(r['weights_ref']['path']).name!='best.pt' or r.get('kind')=='sensitivity':raise ValueError('only default validation-best source')
        if m=='UV':
            if t.get('arm')!='UV' or r['identity'].get('candidate_id')!=('moderntcn-uv-dev-v9' if urban else 'moderntcn-uv-dev-v10'):raise ValueError('original approved default UV producer')
        elif r['kind']!='baseline' or t['model']!=m:raise ValueError('baseline identity')
        if m=='PatchTST' and p['structure']['e_layers']!=(2 if urban else 3):raise ValueError('current MS PatchTST depth')
        if urban and (p['input_variant']!='F4' or len(r['data']['node_order'])!=275 or r['test_arithmetic']['endpoints']!=[3475,3909,4344]):raise ValueError('UrbanEV schema/split')
    for key,ref in s['plans'].items():
        h=int(key.split('-')[0][1:]);plan=load_ref(ref)
        if plan!=urban_plan(s['rows'][f'C-H{h}-UV'],key.endswith('remaining265')):raise ValueError('metadata-derived sample plan changed')
    for j in c['jobs'][:-1]:
        bs=16 if j['phase']=='P' and j['model']=='TimeXer' else 128
        if j['expected']['eval_batch']!=bs:raise ValueError('native evaluated batch changed')
    return c


def verify_sources(s):
    # Reuse only the previously accepted typed dependencies, including sealed
    # protocol/resolved profile/active data_ref/raw stat. Historical bytes stay bound.
    from m6_ch3_scope_amendment_v10.contract import verify_source_refs
    verify_source_refs(dict(s,centers={},prototypes={},inherited={}))
    for x in s['inherited'].values():
        for k in ('receipt_ref','result_ref','runtime_ref','prediction_ref'):
            if sha(x[k]['path'])!=x[k]['sha256']:raise ValueError('inherited '+k)
        receipt=load_ref(x['receipt_ref']);result=load_ref(x['result_ref'])
        if receipt['result']!=result or receipt['runtime_sha256']!=x['runtime_ref']['sha256']:raise ValueError('inherited receipt/result/runtime')
        folder=Path(x['runtime_ref']['path']).parent
        for p,h in receipt['files'].items():
            if not (folder/p).resolve().is_relative_to(folder.resolve()) or sha(folder/p)!=h:raise ValueError('inherited payload changed')
        for k in ('identity','weights_ref','profile','data'):
            if x['source'][k]!=s['rows'][x['cell']][k]:raise ValueError('reuse producer mismatch')


def permission(config_path,auth):
    if auth.get('schema')!=AUTH_SCHEMA or auth.get('scope')!=SCOPE or auth.get('purpose')!=PURPOSE or auth.get('review_approved') is not True or not auth.get('commit'):raise PermissionError('new explicit AUTH required')
    c=validate(read(config_path))
    if auth.get('binding')!=c or auth.get('config_sha256')!=sha(config_path) or auth.get('resume_review_approved') is not False or auth.get('resume_task_ids')!=[] or auth.get('previous_runtime_ref') is not None:raise PermissionError('full binding/no automatic resume')
    for p,h in c['source_binding'].items():
        if sha(ROOT/p)!=h:raise PermissionError('source changed')
    for ref in c['resource_plan']['evidence_refs']:
        if sha(ref['path'])!=ref['sha256']:raise PermissionError('resource evidence changed')
    return c


def capability(c,j,row):
    if j not in c['jobs'] or j['phase'] not in ('P','C') or row!=load_ref(c['sources_ref'])['rows'][j['cell']]:raise PermissionError('no training or unbound source')
    return dict(scope=SCOPE,purpose=PURPOSE,job=j['id'],evaluation_key=digest(dict(scope=SCOPE,job=j,identity=row['identity'],weights=row['weights_ref'],profile=row['profile'],data=row['data'])),mode='authorized_frozen_test')


def require(c,j,row,cap):
    if cap!=capability(c,j,row):raise PermissionError('claim-bound frozen test only')
