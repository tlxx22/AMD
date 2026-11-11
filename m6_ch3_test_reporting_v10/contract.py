"""Exact twenty-job test grant. No dataset or model is imported here."""
import hashlib
import json
from pathlib import Path
from m6_ch3_supplement_v10.contract import ROOT, sha, read, write_new, load_ref

SCOPE = 'm6_ch3_test_reporting_v10_v1'
PURPOSE = 'posthoc_frozen_test_reporting'
AUTH_SCHEMA = 'm6-ch3-test-reporting-launch-v1'
PAIRS = ((3, 4), (7, 4), (5, 2), (5, 6))
MODELS = ('ModernTCN', 'TimeMixer', 'UV')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def skeleton():
    jobs = []
    for dataset in ('UrbanEV', 'PJM'):
        for G, R in PAIRS:
            jobs.append(dict(id=f'ST-{dataset}-G{G}-R{R}', phase='S', dataset=dataset, model='UV', G=G, R=R,
                             split='test', purpose='first_test_of_validation_only_sensitivity', sample_scope='full_test'))
    for dataset in ('NP', 'BE', 'DE'):
        for model in MODELS:
            jobs.append(dict(id=f'PT-{dataset}-{model}', phase='P', dataset=dataset, model=model,
                             split='test', purpose='posthoc_output_position_reporting', sample_scope='full_test'))
    for model in MODELS:
        jobs.append(dict(id=f'CT-UrbanEV-{model}', phase='C', dataset='UrbanEV', model=model,
                         split='test', purpose='posthoc_ten_node_case_reporting', sample_scope='ten_nodes_fixed_test_week'))
    return jobs


def selected(config, job):
    if job not in config['jobs']:
        raise PermissionError('job outside exact reporting grant')
    return load_ref(config['sources_ref'])['rows'][job['source_key']]


def evaluation_key(config, job):
    row = selected(config, job)
    return digest(dict(scope=SCOPE, purpose=job['purpose'], split='test', job=job,
                       weights=row['weights_ref'], identity=row['identity'], profile=row['profile'],
                       data=row['data'], preselection=config['preselection_ref'] if job['phase']=='C' else None))


def validate(config):
    if config.get('scope') != SCOPE or config.get('purpose') != PURPOSE or config.get('execution_authorized') is not False:
        raise PermissionError('explicit non-authorizing test reporting config required')
    if config.get('base_candidate_id') != 'moderntcn-uv-dev-v10' or config.get('attempt') != 'test-reporting-v10-first-r1':
        raise ValueError('fixed candidate/attempt')
    expected = skeleton()
    if len(config['jobs']) != 20 or len({j['id'] for j in config['jobs']}) != 20:
        raise ValueError('twenty unique evaluations required')
    for job, fixed in zip(config['jobs'], expected):
        if {k: job.get(k) for k in fixed} != fixed or set(job) != set(fixed)|{'source_key','expected'} or job['source_key'] != job['id']:
            raise ValueError('fixed ordered test job contract')
    if config['waves'] != [[j['id']] for j in config['jobs']]+[['O']] or config['q_max'] != 1:
        raise ValueError('serial twenty evaluations then offline O')
    if config['budget'] != dict(model_jobs=20, offline_jobs=1, sensitivity_first_test=8, repeated_posthoc_test=12,
                               center_references=2, case_presentations=5040, training=0, backward=0, optimizer=0):
        raise ValueError('fixed added test access budget')
    if config['numeric_policy'] != dict(atol=2e-6, rtol=2e-5, identity='exact'):
        raise ValueError('fixed numerical policy')
    ledger = load_ref(config['sources_ref'])
    if set(ledger['rows']) != {j['id'] for j in config['jobs']} or set(ledger['centers']) != {'UrbanEV','PJM'}:
        raise ValueError('source/center coverage')
    for job in config['jobs']:
        row = ledger['rows'][job['id']]; p = row['profile']; t = row['task']
        if t['dataset'] != job['dataset'] or p['dataset'] != job['dataset'] or p['H'] != (6 if job['dataset']=='UrbanEV' else 24):
            raise ValueError('source task/profile mismatch')
        if p['pred_len'] != (1 if job['dataset']=='UrbanEV' else 24) or p['target_idx'] != (0 if job['dataset']=='UrbanEV' else 2):
            raise ValueError('source target/P mismatch')
        urban=job['dataset']=='UrbanEV'
        features=['volume','e_price','s_price','Ta','P','h','hour_sin','hour_cos','weekday_sin','weekday_cos','is_weekend'] if urban else row['data_protocol']['datasets'][job['dataset']]['features']
        if p['T']!=(12 if urban else 96) or p['C']!=(11 if urban else 3) or p['features']!=features or p['aux_idx']!=[i for i in range(p['C']) if i!=p['target_idx']] or t['task']!='MS' or t['seed']!=2024:
            raise ValueError('frozen input schema/MS/seed')
        if urban and (t['fold']!=6 or p['input_variant']!='F4' or t['input_variant']!='F4' or len(row['data']['node_order'])!=275):
            raise ValueError('only UrbanEV fold6 F4 allowed')
        if job['phase']=='S':
            a=row['identity']['analysis']
            if row['kind']!='sensitivity' or (a['G'],a['R'])!=(job['G'],job['R']) or row['identity']['purpose']!='validation_only_sensitivity':
                raise ValueError('wrong sensitivity producer')
        elif (job['model']=='UV' and row['kind']=='baseline') or (job['model']!='UV' and (row['kind']!='baseline' or t['model']!=job['model'])):
            raise ValueError('wrong frozen producer arm/model')
        a=row['test_arithmetic'];n=1680 if job['phase']=='C' else a['full_test_samples'];bs=p['training']['eval_batch']
        if job['expected']!=dict(samples=n,elements=n*p['pred_len'],batches=(n+bs-1)//bs,eval_batch=bs):
            raise ValueError('original loader arithmetic')
    pre=load_ref(config['preselection_ref'])
    if len(pre['eligible'])!=137 or len(set(pre['draw_indices']))!=10 or not set(pre['draw_indices'])<=set(pre['eligible']) or pre['seed']!=2024 or pre['selected_plots']!=4:
        raise ValueError('fixed train-only random preselection')
    if any(Path(r['weights_ref']['path']).name!='best.pt' for r in ledger['rows'].values()):
        raise ValueError('only original validation-best allowed')
    return config


def permission(config_path, auth):
    if auth.get('schema')!=AUTH_SCHEMA or auth.get('scope')!=SCOPE or auth.get('purpose')!=PURPOSE or auth.get('review_approved') is not True or not auth.get('commit'):
        raise PermissionError('new explicit test reporting authorization required')
    c=validate(read(config_path))
    if auth.get('config_sha256')!=sha(config_path) or auth.get('resume_review_approved') is not False or auth.get('resume_task_ids')!=[] or auth.get('previous_runtime_ref') is not None:
        raise PermissionError('config binding; no checkpoint resume')
    for k in ('jobs','waves','budget','environment_ref','sources_ref','preselection_ref','source_binding','gpu_uuid','global_lock','resource_plan'):
        if auth.get(k)!=c[k]:raise PermissionError('authorization differs: '+k)
    for name,s in c['source_binding'].items():
        if sha(ROOT/name)!=s:raise PermissionError('source bytes changed: '+name)
    for k in ('environment_ref','sources_ref','preselection_ref','compatibility_ref','old_completion_ref'):
        load_ref(c[k])
    return c


def require_evaluation(config, job, capability):
    if capability != dict(scope=SCOPE, purpose=PURPOSE, job=job['id'], evaluation_key=evaluation_key(config,job), mode='authorized_frozen_test'):
        raise PermissionError('test observations require new runtime job capability')
    return selected(config,job)
