"""112 round-two amendments followed by 231 fixed third-round tasks."""
import copy,json,math
from pathlib import Path
from functools import lru_cache
from utils.ch3_contract import ROOT,digest,step_arithmetic
from utils.ch3_native_recovery_records import bound,sha
from utils.ch3_type1_scaled import configuration
BASE='5bd62dc93d611b3271467a46fd16335b622e0af1'
PROTOCOL='baseline-type1-followup-v3-recovery1'
ID='m6-baseline-type1-followup-v3-recovery1'
MODELS=('AMD','DLinear','PatchTST','iTransformer','TimeMixer','ModernTCN','TimeXer')
EPF_GROUPING_POLICY='epf_all_baselines_4_plus_1_v1'
EPF_MARKET_ORDER=('PJM','NP','BE','FR','DE')
EPF_REVIEWED_MAIN_SHAPES_SHA={'AMD': ('fd87e7cf0093f270184f1e2616e772abc9c40cd407604c051346fb257427a5cc',), 'DLinear': ('ad79f0972427a5d23da8a11a05cc02584e3659a529a168c57b18d9401f72fc9a',), 'ModernTCN': ('4af40750ea2183827a5b2adba2061128dfce621ece4d31cd7fce9c5ef1cd7ede',), 'PatchTST': ('023b13b2564fca429636491069c3f995206b2efb26a0ea0eed0ba631d2ac1f64',), 'TimeMixer': ('d4136fc86153bfec282d075810583b3f24918664a87dd6eb8204cb1f37844dac',), 'TimeXer': ('81c765aded8c4ce1d01276e9cd5b52af058f024b6f1d14386e0b1ddc229c2c32',), 'iTransformer': ('0b355416edd72514bbf5ec25539aa084d3eb2c188fdf02aee915b47e0500912b',)}
HISTORICAL_EPF_SPLIT_PROTOCOL_SHA='9a63ea039e571f4f8a3195798d54226f44b22284e95e43e26d27169576a27365'
STAGES=('M_BASE','M_AMEND','URBAN_SUBSET','EPF_ALL','M_ALL')
from utils.ch3_ms_seal_recovery import PACKAGE,RESULT
PARENT_REF=dict(path=str(ROOT/'configs/ch3_baseline_ms_u96_oc01_v3.json'),sha256='6bdfc55357f8fad0d102efec83d453ef77418cc567de40158ff3ed974124aa09')
M_PARENT_REF=dict(path=str(ROOT/'configs/ch3_baseline_m_u96_oc01_v3.json'),sha256='f627260225b971d13d1e881d8123a0426c63304e64d8e8afa56fa0ab73a3cdb2')
AUTHOR_RECIPE=dict(path=str(PACKAGE.parent/'baseline-type1-followup-v2/type1-author-recipe.json'),sha256='fc1081b5acaef98181bc4edd0d5bcbab7ca7620e69eff4a1b97850833cf3da71')
@lru_cache(maxsize=2)
def parent(stage='URBAN_SUBSET'):return bound(parent_ref(stage))
def parent_ref(stage):
    if stage=='M_BASE':
        from utils.ch3_ms_seal_recovery import M_PARENT
        return M_PARENT
    if stage=='M_AMEND':
        from utils.ch3_round2_amendment import PARENT_REF as value
        return value
    return M_PARENT_REF if stage=='M_ALL' else PARENT_REF
def package(stage):
    if stage=='M_AMEND':
        from utils.ch3_round2_amendment import PACKAGE as value
        return value
    return PACKAGE
def file(stage):
    if stage not in STAGES:raise ValueError('exact type1 ring')
    if stage=='M_BASE':
        from utils.ch3_ms_seal_recovery import M_FILE
        return M_FILE
    return ROOT/'configs/ch3_round2_m_amend1_recovery1.json' if stage=='M_AMEND' else ROOT/'configs'/('ch3_type1_'+stage.lower()+'_v3_recovery1.json')
def selected(stage):
    if stage not in STAGES:raise ValueError('exact ring')
    if stage=='M_BASE':
        from utils.ch3_ms_seal_recovery import m_parent
        return m_parent()['tasks']
    if stage=='M_AMEND':
        from utils.ch3_round2_amendment import selected as handler
        return handler()
    rows=parent(stage)['tasks']
    if stage=='M_ALL':
        from utils.ch3_type1_ett import NEW_DATASETS
        result=[]
        for model in MODELS:
            bank=[t for t in rows if t['model']==model];result.extend(bank)
            for name in NEW_DATASETS:
                for t in bank:
                    if t['dataset']!='ETTh1':continue
                    value=copy.deepcopy(t);value['dataset']=name
                    for k in ('id','profile','group'):value[k]=value[k].replace('ETTh1',name)
                    result.append(value)
        return result
    return [t for t in rows if t['dataset']=='UrbanEV' and t['fold']in (1,2) and t['h']in (3,12)] if stage=='URBAN_SUBSET' else [t for t in rows if t['dataset']!='UrbanEV']
def new_task(t):
    value=copy.deepcopy(t);group=t['model']+'-'+t['dataset']+'-'+t['input_variant']+'-type1-followup-v3-recovery1'
    value.update(id=group+'-f'+str(t['fold'])+'-h'+str(t['h'])+'-s2024',group=group,profile=group+'-h'+str(t['h']))
    return value
def expected_tasks(stage):
    if stage=='M_BASE':
        from utils.ch3_ms_seal_recovery import m_tasks
        return m_tasks()
    if stage=='M_AMEND':
        from utils.ch3_round2_amendment import tasks
        return tasks()
    return [new_task(t) for t in selected(stage)]
def inherited_profile(t,c=None):
    from utils import ch3_type1_ett as ett
    if t['dataset']in ett.NEW_DATASETS:return ett.profile(t)
    stage='M_ALL' if t['task']=='M' else 'URBAN_SUBSET' if t['dataset']=='UrbanEV' else 'EPF_ALL';old=parent(stage)
    p=copy.deepcopy(old['resolved_profiles'][t['id']]);p['T']=12 if stage=='URBAN_SUBSET' else 96 if stage=='M_ALL' else 168
    train=p['training'];train.update(lr=1e-4,batch=32 if stage=='EPF_ALL' else 128,eval_batch=32 if stage=='EPF_ALL' else 128,epochs=20 if stage=='URBAN_SUBSET' or t['dataset']=='Weather' else 10,patience=5 if stage=='URBAN_SUBSET' else 3 if stage=='EPF_ALL' else None)
    d=old['datasets'][t['dataset']]
    windows=(old['urban_folds'][t['fold']-1][0]-p['T']-t['h']+1)*275 if stage=='URBAN_SUBSET' else d['endpoints'][0]-p['T']-p['pred_len']+1
    train['scheduler']=configuration(train['epochs'],windows//train['batch'])
    if c is not None:
        from utils.ch3_amend_ett_identity_recovery import weather_profile
        p=weather_profile(c,t,p)
    return p
def profile(c,t):
    if t not in c['tasks']:raise ValueError('foreign type1 task')
    return copy.deepcopy(c['resolved_profiles'][t['id']])
def numeric_policy(c,t):
    if t not in c['tasks']:raise ValueError('foreign type1 policy')
    return copy.deepcopy(c['baseline_unified']['numeric_policies'][t['model']+'-'+t['dataset']])
def validate(c):
    from utils import ch3_type1_ett as ett
    if c['baseline_unified']['stage']=='M_BASE':
        from utils.ch3_ms_seal_recovery import validate_m
        return validate_m(c)
    if c['baseline_unified']['stage']=='M_AMEND':
        from utils.ch3_round2_amendment import validate as handler
        return handler(c)
    stage=c['baseline_unified']['stage'];old=parent(stage)
    if stage not in STAGES or c.get('type1_followup')!=ID or c['baseline_unified']['id']!=PROTOCOL or c['tasks']!=expected_tasks(stage):raise ValueError('exact type1 ring/task namespace')
    if c['baseline_unified']['direct_parent_ref']!=parent_ref(stage) or c['baseline_unified']['recipe_ref']!=AUTHOR_RECIPE:raise ValueError('fixed reviewed parent/recipe')
    bound(AUTHOR_RECIPE)
    expected_policies={t['model']+'-'+t['dataset']:(ett.numeric_policy(t['model'],t['dataset'])if t['dataset']in ett.NEW_DATASETS else old['baseline_unified']['numeric_policies'][t['model']+'-'+t['dataset']]) for t in selected(stage)}
    if c['baseline_unified'].get('numeric_revision_ref'):
        from utils.ch3_moderntcn_etth1_recovery import validate_revision
        expected_policies=validate_revision(c)
    if c['baseline_unified']['numeric_policies']!=expected_policies or c['sources']!=old['sources'] or c['urban_folds']!=old['urban_folds'] or c['urban_input_variants']!=old['urban_input_variants']:raise ValueError('source/policy/features/split changed')
    domains={'UrbanEV'} if stage=='URBAN_SUBSET' else set(ett.M_DATASETS) if stage=='M_ALL' else {'PJM','NP','BE','FR','DE'}
    if set(c['datasets'])!=domains or len(c['tasks'])!={'URBAN_SUBSET':28,'EPF_ALL':35,'M_ALL':168}[stage] or len({t['id']for t in c['tasks']})!=len(c['tasks']):raise ValueError('231 unique runs; J/N/S/ECL excluded')
    if stage=='M_ALL'and c['baseline_unified'].get('extension_refs')!=dict(template=ett.TEMPLATE_REF,source=ett.SOURCE_REF,paper=ett.PAPER_REF):raise ValueError('new ETT template/source/paper binding')
    for name,d in c['datasets'].items():
        expected=ett.dataset(name)if name in ett.NEW_DATASETS else copy.deepcopy(old['datasets'][name]);expected['T']=12 if name=='UrbanEV' else 96 if stage=='M_ALL' else 168
        if d!=expected:raise ValueError('dataset only T changes')
    for prior,t in zip(selected(stage),c['tasks']):
        expected_ref=ett.parent_ref(prior)if prior['dataset']in ett.NEW_DATASETS else dict(task_id=prior['id'],profile_sha=digest(old['resolved_profiles'][prior['id']]),config_ref=parent_ref(stage))
        if profile(c,t)!=inherited_profile(prior,c) or c['baseline_unified']['parent_refs'][t['id']]!=expected_ref:raise ValueError('only approved parent profile changes or fixed ETTh1 ETT migration')
    data=bound(c['baseline_unified']['data_ref'])
    from utils.ch3_time_marks import metadata as mark_metadata
    for t in c['tasks']:
        m=data['metadata'][t['dataset']][t['id']];p=profile(c,t);a=step_arithmetic(c,t)
        if any(m.get(k)!=v for k,v in mark_metadata(p).items()) or m['window_counts']!={'train':a['train_windows'],'validation':a['validation_windows']} or data['data_bindings'][t['dataset']][t['id']]!=digest(m):raise ValueError('exact T-dependent historical mark/window metadata required')
    return c
def probe_groups(c):
    """Default for all newly created EPF work: every baseline uses4+1."""
    return _probe_groups(c,False)


def historical_epf_probe_groups_v1(c):
    """Read the exact frozen type1 split; never a new-task default."""
    if c['baseline_unified']['stage']!='EPF_ALL'or digest(c)!=HISTORICAL_EPF_SPLIT_PROTOCOL_SHA:
        raise PermissionError('exact historical EPF protocol required for3+2 projection')
    return _probe_groups(c,True)


def _probe_groups(c,historical_epf_split):
    stage=c['baseline_unified']['stage'];result=[]
    for model in MODELS:
        ts=[t for t in c['tasks']if t['model']==model]
        if stage=='EPF_ALL'and not historical_epf_split:
            if len(ts)!=5 or{t['dataset']for t in ts}!=set(EPF_MARKET_ORDER):raise ValueError('exact five EPF markets required')
            ts=[next(t for t in ts if t['dataset']==market)for market in EPF_MARKET_ORDER]
        from utils.ch3_type1_ett import M_DATASETS
        if stage=='M_BASE':M_DATASETS=('ETTh1','Weather','Exchange')
        if stage=='M_AMEND':
            from utils.ch3_round2_amendment import DATASETS as M_DATASETS
        banks=[(ts,4,'cross-fold-f1-f2-H3-H12')] if stage=='URBAN_SUBSET' else [([t for t in ts if t['dataset']==d],4,'M-'+d+'-four-H')for d in M_DATASETS if d in c['datasets']] if stage in ('M_BASE','M_ALL','M_AMEND') else [(ts,4,'EPF-4-plus-1')] if model!='TimeXer'or not historical_epf_split else [([t for t in ts if t['dataset']in ('PJM','BE','FR')],4,'EPF-structure-PJM-BE-FR-batch32'),([t for t in ts if t['dataset']in ('NP','DE')],2,'EPF-structure-NP-DE-batch32')]
        for rows,q,label in banks:
            result.append(dict(id=model+'-'+label,model=model,representatives=[t['id']for t in rows],planned_q=q,coverage={t['id']:[t['id']]for t in rows},identities={t['id']:digest(profile(c,t))for t in rows},equivalence='own-profile independent six-step serial; fixed actual waves; type1 epoch behavior proved separately with no-model fixtures'))
            if stage=='EPF_ALL'and not historical_epf_split:
                profiles=[profile(c,t)for t in rows]
                shapes=[dict(T=p['T'],pred_len=p['pred_len'],C=p['C'],batch=p['training']['batch'],eval_batch=p['training']['eval_batch'],structure=p['structure'])for p in profiles]
                risks=[]
                if len({digest(p)for p in shapes})!=1:risks.append('market batch or main model shape differs; resource review required')
                if any(digest(p)not in EPF_REVIEWED_MAIN_SHAPES_SHA[model]for p in shapes):
                    risks.append('main batch or model shape differs from reviewed EPF configuration; resource review required')
                result[-1].update(grouping_policy=EPF_GROUPING_POLICY,resource_review_required=bool(risks),resource_risks=risks,
                    equivalence='default4+1 scheduling; numerical equivalence is not claimed by grouping')
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
    result=RESULT;identity=ID
    if stage=='M_AMEND':
        from utils.ch3_round2_amendment import RESULT as result,ID as identity
    return dict(stage=stage,probe_scope=identity+'-'+stage+'-probe',formal_scope=identity+'-'+stage+'-formal',probe_root=result/'probe'/stage,result_root=result/stage,control=result/'queue'/stage,protocol_file=file(stage),fixture=package(stage)/'fixtures',package=package(stage),models=MODELS,caps=probe_budget(c)['caps'],runs=b['runs'],epochs=b['run_epochs'],optimizer_steps=b['adam'])
def decision_for(c,report,t):
    return report['decisions'][next(g['id']for g in probe_groups(c)if t['id']in g['representatives'])]
def formal_waves(c,report,model):
    from utils import ch3_reviewed_concurrency as concurrency
    if concurrency.enabled():return concurrency.formal_waves(c,report,model)
    waves=[]
    for g in probe_groups(c):
        if g['model']!=model:continue
        d=report['decisions'][g['id']];q=d['concurrency']
        if d['status']!='Passed'or q not in (1,2,4)or q>g['planned_q']:raise ValueError('unmeasured concurrency')
        waves.extend(g['representatives'][i:i+q]for i in range(0,len(g['representatives']),q))
    return waves
def plan(c):
    from utils import ch3_reviewed_concurrency as concurrency
    if concurrency.enabled():
        row=concurrency.stage_plan(c)
        return dict(protocol=c['baseline_unified']['id'],scope=ID,stage=c['baseline_unified']['stage'],
            models=list(MODELS),task_ids=[t['id']for t in c['tasks']],profile_shas=row['profile_shas'],
            concurrency_policy=concurrency.POLICY,concurrency_plan_ref=concurrency.recovery().contract()['concurrency_plan_ref'],
            groups=row['groups'],formal_waves={m:formal_waves(c,row,m)for m in MODELS},
            formal_budget=formal_budget(c),probe_budget=probe_budget(c),science_review='pending',
            proposed_concurrency=True,automatic_probe=False,probe_admission_claim=False,additional_search=0,from_scratch=True)
    nominal=dict(decisions={g['id']:dict(status='Passed',concurrency=g['planned_q'])for g in probe_groups(c)})
    return dict(protocol=c['baseline_unified']['id'],scope=ID,stage=c['baseline_unified']['stage'],models=list(MODELS),task_ids=[t['id']for t in c['tasks']],profile_shas={t['id']:digest(profile(c,t))for t in c['tasks']},groups=probe_groups(c),formal_waves={m:formal_waves(c,nominal,m)for m in MODELS},formal_budget=formal_budget(c),probe_budget=probe_budget(c),science_review='pending',proposed_concurrency=True,additional_search=0,from_scratch=True)
