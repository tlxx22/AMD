"""Exact 28 UrbanEV + 35 EPF tasks, pinned directly to the reviewed v3 profiles."""
import copy,json,math
from pathlib import Path
from functools import lru_cache
from utils.ch3_contract import ROOT,digest,step_arithmetic
from utils.ch3_native_recovery_records import bound,sha
from utils.ch3_type1 import configuration
BASE='df6a16403e10d51097db8c88829909c533d15652'
PROTOCOL='baseline-type1-followup-v1'
ID='m6-baseline-type1-followup-v1'
MODELS=('AMD','DLinear','PatchTST','iTransformer','TimeMixer','ModernTCN','TimeXer')
STAGES=('URBAN_SUBSET','EPF_ALL')
PACKAGE=ROOT.parent/'amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1'/PROTOCOL
RESULT=ROOT.parent/'amd-execution-evidence/m6/m6-formal-launch-dhozikhu'/PROTOCOL
PARENT_REF=dict(path=str(ROOT/'configs/ch3_baseline_ms_u96_oc01_v3.json'),sha256='6bdfc55357f8fad0d102efec83d453ef77418cc567de40158ff3ed974124aa09')
AUTHOR_RECIPE=dict(path=str(PACKAGE/'type1-author-recipe.json'),sha256='fc1081b5acaef98181bc4edd0d5bcbab7ca7620e69eff4a1b97850833cf3da71')
@lru_cache(maxsize=1)
def parent():return bound(PARENT_REF)
def file(stage):
    if stage not in STAGES:raise ValueError('exact type1 ring')
    return ROOT/'configs'/('ch3_type1_'+stage.lower()+'.json')
def selected(stage):
    return [t for t in parent()['tasks'] if (t['dataset']=='UrbanEV' and t['fold']in (1,6) and t['h']in (3,12)) if stage=='URBAN_SUBSET'] if stage=='URBAN_SUBSET' else [t for t in parent()['tasks'] if t['dataset']!='UrbanEV']
def new_task(t):
    value=copy.deepcopy(t);group=t['model']+'-'+t['dataset']+'-'+t['input_variant']+'-type1-followup-v1'
    value.update(id=group+'-f'+str(t['fold'])+'-h'+str(t['h'])+'-s2024',group=group,profile=group+'-h'+str(t['h']))
    return value
def expected_tasks(stage):return [new_task(t) for t in selected(stage)]
def inherited_profile(t):
    p=copy.deepcopy(parent()['resolved_profiles'][t['id']]);p['T']=12 if t['dataset']=='UrbanEV' else 168
    train=p['training'];train['lr']=1e-4
    d=parent()['datasets'][t['dataset']]
    windows=(parent()['urban_folds'][t['fold']-1][0]-p['T']-t['h']+1)*275 if t['dataset']=='UrbanEV' else d['endpoints'][0]-p['T']-p['pred_len']+1
    train['scheduler']=configuration(train['epochs'],windows//train['batch'])
    return p
def profile(c,t):
    if t not in c['tasks']:raise ValueError('foreign type1 task')
    return copy.deepcopy(c['resolved_profiles'][t['id']])
def numeric_policy(c,t):
    if t not in c['tasks']:raise ValueError('foreign type1 policy')
    return copy.deepcopy(c['baseline_unified']['numeric_policies'][t['model']+'-'+t['dataset']])
def validate(c):
    stage=c['baseline_unified']['stage'];old=parent()
    if stage not in STAGES or c.get('type1_followup')!=ID or c['baseline_unified']['id']!=PROTOCOL or c['tasks']!=expected_tasks(stage):raise ValueError('exact type1 ring/task namespace')
    if c['baseline_unified']['direct_parent_ref']!=PARENT_REF or c['baseline_unified']['recipe_ref']!=AUTHOR_RECIPE:raise ValueError('fixed reviewed parent/recipe')
    bound(AUTHOR_RECIPE)
    expected_policies={t['model']+'-'+t['dataset']:old['baseline_unified']['numeric_policies'][t['model']+'-'+t['dataset']] for t in selected(stage)}
    if c['baseline_unified']['numeric_policies']!=expected_policies or c['sources']!=old['sources'] or c['urban_folds']!=old['urban_folds'] or c['urban_input_variants']!=old['urban_input_variants']:raise ValueError('source/policy/features/split changed')
    domains={'UrbanEV'} if stage=='URBAN_SUBSET' else {'PJM','NP','BE','FR','DE'}
    if set(c['datasets'])!=domains or len(c['tasks'])!=(28 if stage=='URBAN_SUBSET' else 35) or len({t['id']for t in c['tasks']})!=len(c['tasks']):raise ValueError('63 unique runs; J/N/S/ECL/M excluded')
    for name,d in c['datasets'].items():
        expected=copy.deepcopy(old['datasets'][name]);expected['T']=12 if name=='UrbanEV' else 168
        if d!=expected:raise ValueError('dataset only T changes')
    for prior,t in zip(selected(stage),c['tasks']):
        if profile(c,t)!=inherited_profile(prior) or c['baseline_unified']['parent_refs'][t['id']]!=dict(task_id=prior['id'],profile_sha=digest(old['resolved_profiles'][prior['id']]),config_ref=PARENT_REF):raise ValueError('only T/lr/scheduler changes from v3')
    data=bound(c['baseline_unified']['data_ref'])
    from utils.ch3_time_marks import metadata as mark_metadata
    for t in c['tasks']:
        m=data['metadata'][t['dataset']][t['id']];p=profile(c,t);a=step_arithmetic(c,t)
        if any(m.get(k)!=v for k,v in mark_metadata(p).items()) or m['window_counts']!={'train':a['train_windows'],'validation':a['validation_windows']} or data['data_bindings'][t['dataset']][t['id']]!=digest(m):raise ValueError('exact T-dependent historical mark/window metadata required')
    return c
def probe_groups(c):
    stage=c['baseline_unified']['stage'];result=[]
    for model in MODELS:
        ts=[t for t in c['tasks']if t['model']==model]
        banks=[(ts,4,'cross-fold-f1-f6-H3-H12')] if stage=='URBAN_SUBSET' else [(ts,4,'EPF-4-plus-1')] if model!='TimeXer' else [([t for t in ts if t['dataset']in ('PJM','BE','FR')],4,'EPF-batch16'),([t for t in ts if t['dataset']in ('NP','DE')],2,'EPF-batch4')]
        for rows,q,label in banks:
            result.append(dict(id=model+'-'+label,model=model,representatives=[t['id']for t in rows],planned_q=q,coverage={t['id']:[t['id']]for t in rows},identities={t['id']:digest(profile(c,t))for t in rows},equivalence='own-profile independent six-step serial; fixed actual waves; type1 epoch behavior proved separately with no-model fixtures'))
    return result
def worker_counts(c,t):return dict(adam=6,backward=6,forward=10 if t['model']=='TimeMixer'and t['dataset']=='UrbanEV'else 8)
def probe_budget(c):
    nominal=dict(adam=0,backward=0,forward=0);caps=dict(nominal);workers=maximum=0
    for g in probe_groups(c):
        workers+=2*len(g['representatives']);factor=3 if g['planned_q']==4 else 2;maximum+=factor*len(g['representatives'])
        for r in g['representatives']:
            counts=worker_counts(c,next(t for t in c['tasks']if t['id']==r))
            for k in counts:nominal[k]+=2*counts[k];caps[k]+=factor*counts[k]
    return dict(nominal=nominal,caps=caps,nominal_workers=workers,max_workers=maximum,fallback='resource-only q4 -> once q2 -> measured serial q1; q2 -> measured serial q1; no refund')
def formal_budget(c):
    total=dict(runs=len(c['tasks']),run_epochs=0,adam=0,backward=0,forward=0);rows={}
    for t in c['tasks']:
        p=profile(c,t);a=step_arithmetic(c,t);epochs=p['training']['epochs'];v=p['training']['eval_batch']
        counts=dict(run_epochs=epochs,adam=a['max_optimizer_steps'],backward=a['max_optimizer_steps'],forward=epochs*(a['train_batches']+math.ceil(a['validation_windows']/v))+math.ceil(a['test_windows_arithmetic_only']/v))
        rows[t['id']]=dict(arithmetic=a,**counts)
        for k in counts:total[k]+=counts[k]
    return dict(total=total,tasks=rows,early_stopping_retained=True,refund=False)
def context(c):
    stage=c['baseline_unified']['stage'];b=formal_budget(c)['total']
    return dict(stage=stage,probe_scope=ID+'-'+stage+'-probe',formal_scope=ID+'-'+stage+'-formal',probe_root=RESULT/'probe'/stage,result_root=RESULT/stage,control=RESULT/'queue'/stage,protocol_file=file(stage),fixture=PACKAGE/'fixtures',models=MODELS,caps=probe_budget(c)['caps'],runs=b['runs'],epochs=b['run_epochs'],optimizer_steps=b['adam'])
def decision_for(c,report,t):
    return report['decisions'][next(g['id']for g in probe_groups(c)if t['id']in g['representatives'])]
def formal_waves(c,report,model):
    waves=[]
    for g in probe_groups(c):
        if g['model']!=model:continue
        d=report['decisions'][g['id']];q=d['concurrency']
        if d['status']!='Passed'or q not in (1,2,4)or q>g['planned_q']:raise ValueError('unmeasured concurrency')
        waves.extend(g['representatives'][i:i+q]for i in range(0,len(g['representatives']),q))
    return waves
def plan(c):
    nominal=dict(decisions={g['id']:dict(status='Passed',concurrency=g['planned_q'])for g in probe_groups(c)})
    return dict(protocol=PROTOCOL,scope=ID,stage=c['baseline_unified']['stage'],models=list(MODELS),task_ids=[t['id']for t in c['tasks']],profile_shas={t['id']:digest(profile(c,t))for t in c['tasks']},groups=probe_groups(c),formal_waves={m:formal_waves(c,nominal,m)for m in MODELS},formal_budget=formal_budget(c),probe_budget=probe_budget(c),science_review='pending',proposed_concurrency=True,additional_search=0,from_scratch=True)
