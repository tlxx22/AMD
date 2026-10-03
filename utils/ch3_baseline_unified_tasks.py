"""Source-bound 203 MS + 84 M fresh profiles; no observation/model imports."""
import copy
import json
import math
from functools import lru_cache
from pathlib import Path
from utils.ch3_contract import ROOT, digest
from utils.ch3_onecycle import configuration
from utils.ch3_native_recovery_records import bound, sha

BASE='22b6447c4e62078d903446690225504f26e7a3ed'
PROTOCOL='baseline-unified96-onecycle001-v2'
ID='m6-baseline-unified96-oc01-v2'
MODELS=('AMD','DLinear','PatchTST','iTransformer','TimeMixer','ModernTCN','TimeXer')
MS_DOMAINS=('UrbanEV','PJM','NP','BE','FR','DE')
M_DOMAINS=('ETTh1','Weather','Exchange')
PACKAGE=ROOT.parent/'amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1'/PROTOCOL
RESULT=ROOT.parent/'amd-execution-evidence/m6/m6-formal-launch-dhozikhu'/PROTOCOL
PARENT_PACKAGE=PACKAGE.with_name('baseline-unified96-onecycle001-v1')
CATALOG=dict(path=str(PARENT_PACKAGE/'parent-profile-catalog.json'),sha256='1f295507e158ba5b3101027820c62655ef6ae0ea0dab5552940a8d3615eb35b6')
AUTHOR_RECIPE=dict(path=str(PARENT_PACKAGE/'onecycle-author-recipe.json'),sha256='e2e1cb27c93d2e865ea68c1e39858363ebc216a5c91df0ec773862bbc0a9185f')
RETIREMENT=dict(path=str(PARENT_PACKAGE/'baseline-unified-protocol-retirement.json'),sha256='64869e3735c49334dfd2e127691c1c2c958670c0e3d5f3d3b59f5e696aa26604')
MODERNTCN_EPF_POLICY=dict(
    id='baseline-unified-v2-moderntcn-epf-fullfloat-atol5e-4',kind='full_float_state',
    state_atol=5e-4,loss_atol=1e-6,metric_atol=1e-6,loss_rtol=0.0,rtol=0.0,equal_nan=False,
    capture='preallocated raw sidecar v1',
    floating_state='all model parameters/buffers, gradients, Adam exp_avg and exp_avg_sq',
    exact_state='initialization/RNG/batch identity/order, optimizer step/nonfloating state, param groups, tensor shape/dtype, task/profile/source/data; scheduler configuration/step/LR/beta1',
    initialization_rng_batches='exact',nonfloating_and_step='exact')


@lru_cache(maxsize=1)
def catalog():
    return bound(CATALOG)


@lru_cache(maxsize=1)
def parents():
    return {k:bound(row['ref']) for k,row in catalog()['parents'].items()}


def file(stage):
    if stage not in ('MS','M'):raise ValueError('exact unified stage')
    return ROOT/'configs'/('ch3_baseline_'+stage.lower()+'_u96_oc01_v2.json')


def parent_rows(stage):
    return [r for r in catalog()['rows'] if r['stage']==stage]


def task(row):
    old=row['parent_task'];p=row['parent_profile'];stage=row['stage']
    group=old['model']+'-'+old['dataset']+'-'+('F4' if old['dataset']=='UrbanEV' else stage)+'-u96-oc01-v2'
    value={k:copy.deepcopy(old[k]) for k in ('model','dataset','fold','h','seed')}
    value.update(id=group+'-f'+str(old['fold'])+'-h'+str(old['h'])+'-s2024',
                 input_variant=p['input_variant'],group=group,profile=group+'-h'+str(old['h']),
                 task=stage,metric_scope='all_channels' if stage=='M' else 'target_only')
    return value


def expected_tasks(stage):
    return [task(r) for r in parent_rows(stage)]


def resolved_parent(row):
    p=copy.deepcopy(row['parent_profile']);t=row['parent_task']
    p['T']=12 if t['dataset']=='UrbanEV' else 96
    old=parents()[row['parent_config']];d=old['datasets'][t['dataset']]
    if t['dataset']=='UrbanEV':
        a,z,n=old['urban_folds'][t['fold']-1]
        train=(a-p['T']-t['h']+1)*275
    else:train=d['endpoints'][0]-p['T']-p['pred_len']+1
    p['training']['lr']=0.01
    p['training']['scheduler']=configuration(p['training']['epochs'],train//p['training']['batch'])
    return p


def profile(c,t):
    if t not in c['tasks']:raise ValueError('foreign unified task')
    return copy.deepcopy(c['resolved_profiles'][t['id']])


def numeric_policy(c,t):
    if t not in c['tasks']:raise ValueError('foreign numeric scope')
    return copy.deepcopy(c['baseline_unified']['numeric_policies'][t['model']+'-'+t['dataset']])


def expected_numeric_policy(row):
    t=row['parent_task']
    if row['stage']=='MS' and t['model']=='ModernTCN' and t['dataset'] in ('PJM','NP','BE','FR','DE'):
        if row['numeric_policy'] is not None:raise ValueError('v2 requires the original exact ModernTCN EPF policy')
        return copy.deepcopy(MODERNTCN_EPF_POLICY)
    return copy.deepcopy(row['numeric_policy'])


def validate(c):
    stage=c['baseline_unified']['stage'];old=parents();rows=parent_rows(stage)
    if c['baseline_unified']['id']!=PROTOCOL or c['tasks']!=expected_tasks(stage):raise ValueError('exact unified fresh task registry')
    if c['baseline_unified']['catalog_ref']!=CATALOG or c['baseline_unified']['recipe_ref']!=AUTHOR_RECIPE:raise ValueError('source-bound parent/recipe')
    if len(c['tasks'])!=(203 if stage=='MS' else 84) or len({t['id'] for t in c['tasks']})!=len(c['tasks']):raise ValueError('unique unified task count')
    if set(c['datasets'])!=set(MS_DOMAINS if stage=='MS' else M_DOMAINS) or any(t['model']not in MODELS for t in c['tasks']):raise ValueError('J/N/S/ECL excluded')
    policies={r['parent_task']['model']+'-'+r['parent_task']['dataset']:expected_numeric_policy(r) for r in rows}
    if c['baseline_unified']['numeric_policies']!=policies:raise ValueError('exact user-authorized v2 numeric registry required')
    for k,r in catalog()['parents'].items():
        if sha(r['ref']['path'])!=r['ref']['sha256'] or digest(old[k])!=r['protocol_sha']:raise ValueError('parent source changed')
    source=old['production'] if stage=='MS' else old['m']
    if c['sources']!={m:source['sources'][m] for m in MODELS if m in source['sources']}:raise ValueError('author source unchanged')
    for name,d in c['datasets'].items():
        expected=copy.deepcopy(source['datasets'][name]);expected['T']=12 if name=='UrbanEV' else 96
        if d!=expected:raise ValueError('only dataset T may change')
    for row,t in zip(rows,c['tasks']):
        p=profile(c,t);expected=resolved_parent(row)
        if p!=expected or digest(row['parent_profile'])!=row['parent_profile_sha']:raise ValueError('profile changes outside T/LR/scheduler whitelist')
        if numeric_policy(c,t)!=expected_numeric_policy(row):raise ValueError('per-scope numerical policy outside the exact v2 revision')
        if p['training']['optimizer']!='Adam' or p['training']['seed']!=2024 or not p['training']['from_scratch']:raise ValueError('Adam/seed/fresh contract')
    return c


def compute_identity(c,t,compatibility=False):
    p=profile(c,t);train=copy.deepcopy(p['training'])
    train.pop('patience',None)
    if compatibility:
        train.pop('epochs',None)
        for k in ('epochs','steps_per_epoch'):train['scheduler'].pop(k,None)
    return dict(model=t['model'],T=p['T'],pred_len=p['pred_len'],C=p['C'],structure=p['structure'],
                training=train,time_mark=p.get('time_mark'),task=t['task'],metric_scope=t['metric_scope'])


def probe_groups(c):
    result=[];stage=c['baseline_unified']['stage']
    for m in MODELS:
        selected=[t for t in c['tasks'] if t['model']==m]
        if stage=='M':
            banks=[([t for t in selected if t['dataset']==d],m+'-'+d+'-M') for d in M_DOMAINS]
        else:
            banks=[([t for t in selected if t['dataset']=='UrbanEV' and t['fold']==1],m+'-UrbanEV-F4')]
            epf={}
            for t in selected:
                if t['dataset']!='UrbanEV':epf.setdefault(digest(compute_identity(c,t,True)),[]).append(t)
            banks += [(ts,m+'-EPF-compatible-'+str(n)) for n,ts in enumerate(epf.values())]
        for ts,gid in banks:
            reps=[t['id'] for t in ts];q=2 if len(reps)==2 else 4 if len(reps)>1 else 1
            coverage={t['id']:[x['id'] for x in selected if x['dataset']=='UrbanEV' and x['h']==t['h']] if t['dataset']=='UrbanEV' else [t['id']] for t in ts}
            result.append(dict(id=gid,model=m,representatives=reps,planned_q=q,coverage=coverage,
                               identities={t['id']:digest(compute_identity(c,t)) for t in ts},
                               equivalence='same resource shapes; each representative compares its own exact scheduler profile; UrbanEV fold1 covers network shapes, per-fold schedule arithmetic stays task-bound'))
    return result


def worker_counts(c,t):
    endpoint=c['baseline_unified']['stage']=='MS' and t['model']=='TimeMixer' and t['dataset']=='UrbanEV'
    return dict(adam=6,backward=6,forward=10 if endpoint else 8)


def probe_budget(c):
    nominal=dict(adam=0,backward=0,forward=0);caps=dict(nominal);workers=maximum=0
    for g in probe_groups(c):
        factors=1 if g['planned_q']==1 else 2
        extra=1 if g['planned_q']==4 else 0
        workers+=factors*len(g['representatives']);maximum+=(factors+extra)*len(g['representatives'])
        for r in g['representatives']:
            counts=worker_counts(c,next(t for t in c['tasks'] if t['id']==r))
            for k in counts:nominal[k]+=counts[k]*factors;caps[k]+=counts[k]*(factors+extra)
    return dict(nominal=nominal,caps=caps,nominal_workers=workers,max_workers=maximum,
                fallback='only resource: q4 -> once q2 -> already passed serial q1; q2 -> serial q1; no refund')


def formal_budget(c):
    from utils.ch3_contract import step_arithmetic
    total=dict(runs=len(c['tasks']),run_epochs=0,adam=0,backward=0,forward=0);rows={}
    for t in c['tasks']:
        p=profile(c,t);a=step_arithmetic(c,t);epochs=p['training']['epochs'];v=p['training']['eval_batch']
        counts=dict(run_epochs=epochs,adam=a['max_optimizer_steps'],backward=a['max_optimizer_steps'],
                    forward=epochs*(a['train_batches']+math.ceil(a['validation_windows']/v))+math.ceil(a['test_windows_arithmetic_only']/v))
        rows[t['id']]=dict(arithmetic=a,**counts)
        for k in counts:total[k]+=counts[k]
    return dict(total=total,tasks=rows,proposal_only=True,early_stopping_retained=True)


def context(c):
    stage=c['baseline_unified']['stage'];b=formal_budget(c)['total']
    return dict(stage=stage,probe_scope=ID+'-'+stage+'-probe',formal_scope=ID+'-'+stage+'-formal',
                probe_root=RESULT/'probe'/stage,result_root=RESULT/stage,control=RESULT/'queue'/stage,
                protocol_file=file(stage),fixture=PACKAGE/'fixtures',models=MODELS,
                caps=probe_budget(c)['caps'],runs=b['runs'],epochs=b['run_epochs'],optimizer_steps=b['adam'])


def decision_for(c,report,t):
    for g in probe_groups(c):
        if any(t['id']in ids for ids in g['coverage'].values()):return report['decisions'][g['id']]
    raise ValueError('exact measured group coverage required')


def formal_waves(c,report,model):
    result=[]
    for g in probe_groups(c):
        if g['model']!=model:continue
        d=report['decisions'][g['id']];q=d['concurrency']
        if d['status']!='Passed' or q not in (1,2,4) or q>g['planned_q']:raise ValueError('unmeasured concurrency')
        if any('-UrbanEV-'in r for r in g['representatives']):
            banks=[[t['id'] for t in c['tasks'] if t['model']==model and t['dataset']=='UrbanEV' and t['fold']==fold] for fold in range(1,7)]
        else:banks=[g['representatives']]
        for ids in banks:result += [ids[i:i+q] for i in range(0,len(ids),q)]
    return result


def plan(c):
    nominal=dict(decisions={g['id']:dict(status='Passed',concurrency=g['planned_q']) for g in probe_groups(c)})
    return dict(protocol=PROTOCOL,scope=ID,stage=c['baseline_unified']['stage'],models=list(MODELS),
                task_ids=[t['id'] for t in c['tasks']],profile_shas={t['id']:digest(profile(c,t)) for t in c['tasks']},
                groups=probe_groups(c),formal_waves={m:formal_waves(c,nominal,m) for m in MODELS},
                formal_budget=formal_budget(c),probe_budget=probe_budget(c),science_review='pending',
                proposed_concurrency=True,additional_search=0,from_scratch=True)


def result_index(configs):
    """Only new task/stage/protocol identities; no legacy/J ranking or test replay."""
    rows=[]
    for stage,c in configs.items():
        for t in c['tasks']:
            path=context(c)['result_root']/('formal-'+t['model'])/t['id']/'result.json'
            row=dict(task=stage,dataset=t['dataset'],model=t['model'],H=t['h'],fold=t['fold'],seed=2024,
                     task_id=t['id'],path=str(path),status='missing')
            if path.exists():
                if path.is_symlink() or path.resolve()!=path:raise ValueError('result path alias rejected')
                r=json.loads(path.read_text())
                if r.get('scientific_protocol')!=PROTOCOL or r.get('id')!=t['id'] or r.get('protocol_sha')!=digest(c) or r.get('profile_sha')!=digest(profile(c,t)) or r.get('task')!=stage:raise ValueError('foreign result/J/legacy cannot enter unified summary')
                row.update(status='complete',mse=r['mse'],mae=r['mae'],result_sha=sha(path))
                if stage=='M':row.update(channels=r['channels'],MS_target_diagnostic=r['MS_target_diagnostic'])
            rows.append(row)
    return dict(scientific_protocol=PROTOCOL,result_review='pending',rows=rows,
                grouping='separate task/dataset/H; never average absolute errors across datasets or mix M/MS ranks',
                completed=sum(r['status']=='complete' for r in rows),missing=sum(r['status']=='missing' for r in rows))
