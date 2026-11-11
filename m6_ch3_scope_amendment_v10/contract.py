"""Explicit amendment grant; preparation never reads raw data or weights."""
import copy
import math
from pathlib import Path
from m6_ch3_supplement_v10.contract import ROOT, read, sha, load_ref, write_new
from m6_ch3_test_reporting_v10.contract import digest

SCOPE = 'm6_ch3_scope_amendment_v10_v1'
AUTH_SCHEMA = 'm6-ch3-amendment-launch-v1'
ATTEMPT = 'urbanev4h-np-amd-timexer-first-r1'
PAIRS = ((3, 4), (7, 4), (5, 2), (5, 6))
MODELS = ('ModernTCN', 'TimeMixer', 'TimeXer', 'AMD', 'UV')
CELLS = ('UrbanEV-H3', 'UrbanEV-H9', 'UrbanEV-H12', 'NP-H24')
STEPS = (7435, 7422, 7416, 285)


def jobs_and_waves(sources):
    jobs = []
    for key, row in sources['rows'].items():
        phase, dataset, model = key.split('-')
        n = 1680 if phase == 'C' else row['test_arithmetic']['full_test_samples']
        p = row['profile']; bs = p['training']['eval_batch']
        jobs.append(dict(id=key, phase=phase, dataset=dataset, H=p['H'], model=model,
                         source_key=key, split='test', purpose='posthoc_baseline_extension',
                         expected=dict(samples=n, elements=n*p['pred_len'], batches=(n+bs-1)//bs, eval_batch=bs)))
    order = [f'P-{d}-{m}' for d in ('NP','BE','DE') for m in ('AMD','TimeXer')]+['C-UrbanEV-AMD','C-UrbanEV-TimeXer']
    jobs.sort(key=lambda j:order.index(j['id']))
    waves = [[j['id']] for j in jobs]
    trains = []
    for cell in CELLS:
        dataset, h = cell.split('-H'); wave = []
        for G, R in PAIRS:
            variant = f'{SCOPE}|{dataset}|F4|f6|H{h}|P1|G{G}|R{R}|s2024' if dataset=='UrbanEV' else f'{SCOPE}|NP|MS|f1|H24|P24|G{G}|R{R}|s2024'
            j = dict(id=f'TR-{cell}-G{G}-R{R}', phase='T', dataset=dataset, H=int(h), model='UV',
                     G=G, R=R, prototype=cell, analysis_variant_id=variant, split='train_validation', purpose='validation_only_sensitivity')
            trains.append(j); wave.append(j['id'])
        waves.append(wave)
    jobs += trains
    for train in trains:
        p = sources['prototypes'][train['prototype']]['profile']; a = sources['prototypes'][train['prototype']]['test_arithmetic']
        j = dict(train, id=train['id'].replace('TR-','ST-',1), phase='S', producer_job=train['id'],
                 split='test', purpose='first_test_of_sealed_validation_best',
                 expected=dict(samples=a['full_test_samples'], elements=a['elements'], batches=a['batches'], eval_batch=p['training']['eval_batch']))
        jobs.append(j); waves.append([j['id']])
    jobs.append(dict(id='O', phase='O', purpose='cache_only_publication')); waves.append(['O'])
    return jobs, waves


def derived_task(config, job):
    source = load_ref(config['sources_ref'])['prototypes'][job['prototype']]
    task = copy.deepcopy(source['task'])
    task.update(id=job['analysis_variant_id'], candidate_id='moderntcn-uv-dev-v10', model='moderntcn-uv-dev-v10',
                structural_revision=config['base_structural_revision'], analysis_variant_id=job['analysis_variant_id'])
    return task, source['profile'], source['data']


def validate(c):
    if c.get('scope')!=SCOPE or c.get('execution_authorized') is not False or c.get('attempt')!=ATTEMPT:
        raise PermissionError('only the explicit non-authorizing amendment config')
    s = load_ref(c['sources_ref']); jobs, waves = jobs_and_waves(s)
    if c['jobs']!=jobs or c['waves']!=waves or len(jobs)!=41 or len(waves)!=29:
        raise ValueError('fixed 41 jobs / 29 waves and task order')
    if c['phase_concurrency']!={'T':4,'S':1,'P':1,'C':1,'O':1} or c['q_max']!=4:
        raise ValueError('fixed phase concurrency')
    if c['budget']!=dict(train_jobs=16,max_epochs=320,max_updates=1804640,test_jobs=24,test_batches=17218,test_samples=1480760,test_elements=3886560,offline_jobs=1,sensitivity_cells=25,sensitivity_references=9):
        raise ValueError('fixed scientific budget')
    evals = [j for j in jobs if j['phase'] in ('S','P','C')]
    if [sum(j['expected'][k] for j in evals) for k in ('batches','samples','elements')] != [17218,1480760,3886560]:
        raise ValueError('test arithmetic')
    if c['numeric_policy']!={'atol':2e-6,'rtol':2e-5,'identity':'exact'}:raise ValueError('numeric policy')
    for cell,steps in zip(CELLS,STEPS):
        src=s['prototypes'][cell];t=src['task'];p=src['profile'];d,h=cell.split('-H');urban=d=='UrbanEV'
        if t['dataset']!=d or p['dataset']!=d or t['h']!=int(h) or p['H']!=int(h) or t['fold']!=(6 if urban else 1):raise ValueError('H/dataset/fold source mismatch')
        if p!=t['structural_source_profile'] or p['training']!=src['source']['profile']['training']:raise ValueError('unaltered source profile required')
        if (p['T'],p['pred_len'],p['C'],p['target_idx'])!=((12,1,11,0) if urban else (96,24,3,2)):raise ValueError('shape/schema')
        tr=p['training'];a=t['arithmetic'];sched=tr['scheduler']
        if a['train_batches']!=steps or sched['steps_per_epoch']!=steps or tr['epochs']!=20 or tr['patience']!=5 or tr['seed']!=2024 or tr['batch']!=128 or tr['eval_batch']!=128:raise ValueError('training arithmetic/recipe')
        if (tr['lr'],tr['betas'],tr['eps'],tr['weight_decay'],tr['dtype'])!=(.01,[.9,.999],1e-8,1e-7,'float32'):raise ValueError('frozen Adam/precision')
        if tr['scheduler']['pct_start']!=.2 or not tr['train_drop_last']:raise ValueError('OneCycle/drop_last')
        ends=src['test_arithmetic']['endpoints'];n=(ends[2]-ends[1]-12-int(h)+1)*275 if urban else ends[2]-ends[1]-24+1
        if n!=src['test_arithmetic']['full_test_samples']:raise ValueError('test window arithmetic')
        if urban and (ends!=[3475,3909,4344] or len(src['data']['node_order'])!=275):raise ValueError('UrbanEV split/nodes')
    for j in jobs[:8]:
        row=s['rows'][j['source_key']];p=row['profile'];t=row['task']
        bs=128 if j['model']=='AMD' or j['phase']=='C' else (16 if j['dataset']=='BE' else 4)
        if t['model']!=j['model'] or t['dataset']!=j['dataset'] or t['task']!='MS' or p['training']['eval_batch']!=bs or p['H']!=j['H']:raise ValueError('latest MS baseline identity/batch')
    if set(s['centers'])!={f'UrbanEV-H{h}' for h in (3,6,9,12)}|{'NP-H24'}:raise ValueError('five unique centers')
    if len(s['inherited'])!=16 or any(s['inherited'][f'ST-UrbanEV-G{G}-R{R}']['source']['profile']['H']!=6 for G,R in PAIRS):raise ValueError('H6 references cannot impersonate other H')
    plan=load_ref(s['case_plan_ref']);selection=load_ref(s['display_selection_ref'])
    if plan['nodes']!=['208','330','333','570','969','1083','1113','1120','1162','1173'] or selection['selected_nodes']!=['1162','1083','1120','969']:raise ValueError('locked ten/four nodes')
    if plan['target_times'][0]!='2023-02-13 00:00:00' or plan['target_times'][-1]!='2023-02-19 23:00:00':raise ValueError('locked week')
    return c


def verify_source_refs(s):
    """Verify typed execution artifacts, not every historical pointer inside metadata.

    The sealed baseline protocol retains an older native_time_mark artifact SHA.
    Its entire byte identity stays bound. The actual unified data_ref, resolved
    profile and loader source-state are checked separately; history is not edited.
    """
    def bound(ref,weight=False):
        if weight:
            if not Path(ref['path']).is_file():raise ValueError('missing weight file')
        elif sha(ref['path'])!=ref['sha256']:raise ValueError('source ref changed: '+ref['path'])
    def artifacts(x):
        for key,ref in x.items():
            if (key.endswith('_ref') or key=='main_index') and isinstance(ref,dict) and 'path' in ref:
                bound(ref,key=='weights_ref')
    def row(r):
        artifacts(r)
        if Path(r['weights_ref']['path']).stat().st_size!=r['weights_stat']['bytes']:raise ValueError('weight stat changed')
        if r['kind']=='baseline':
            protocol=load_ref(r['protocol_ref'])
            if protocol!=r['data_protocol']:raise ValueError('sealed protocol contents differ')
            from utils.ch3_contract import profile
            if profile(protocol,r['task'])!=r['profile']:raise ValueError('resolved profile differs')
            bound(protocol['baseline_unified']['data_ref'])
        else:protocol=r['task']['data_config']
        from utils.ch3_data import verify_source_state
        verify_source_state(protocol['datasets'][r['task']['dataset']])
    artifacts(s)
    for r in [*s['rows'].values(),*s['centers'].values()]:row(r)
    for x in s['prototypes'].values():row(x['source'])
    for x in s['inherited'].values():artifacts(x);row(x['source'])


def permission(config_path, auth):
    if auth.get('schema')!=AUTH_SCHEMA or auth.get('scope')!=SCOPE or auth.get('review_approved') is not True or not auth.get('commit'):raise PermissionError('new amendment AUTH required')
    c=validate(read(config_path))
    if auth.get('config_sha256')!=sha(config_path) or auth.get('resume_review_approved') is not False or auth.get('resume_task_ids')!=[] or auth.get('previous_runtime_ref') is not None:raise PermissionError('no checkpoint resume or unbound config')
    if auth.get('binding')!=c:raise PermissionError('full scope/config binding')
    for name,s in c['source_binding'].items():
        if sha(ROOT/name)!=s:raise PermissionError('source changed: '+name)
    for r in c['resource_plan']['evidence_refs']:
        if sha(r['path'])!=r['sha256']:raise PermissionError('resource evidence changed')
    return c


def test_key(c, job, row):
    if job not in c['jobs'] or job['phase'] not in ('S','P','C'):raise PermissionError('no test grant for this job')
    return digest(dict(scope=SCOPE,job=job,weights=row['weights_ref'],identity=row['identity'],data=row['data'],profile=row['profile']))


def require_test(c, job, row, cap):
    expected=dict(scope=SCOPE,job=job['id'],key=test_key(c,job,row),purpose=job['purpose'],mode='authorized_frozen_test')
    if cap!=expected:raise PermissionError('only owned explicit frozen test capability')
