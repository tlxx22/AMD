"""Formal task/configuration contract; no torch or dataset import."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE_FILE = ROOT / 'configs/ch3_formal_profiles.json'
CONTRACT = 'ch3-target-ms-formal-v2'
MODELS = ('AMD','J','DLinear','PatchTST','iTransformer','TimeMixer','ModernTCN','TimeXer','N','S')
BASELINES = frozenset(MODELS)-{'AMD','J','N','S'}
TRAINING_OVERRIDES = frozenset(('batch','eval_batch','lr'))


# Fixed six-step numerical-equivalence policies. They never change formal training math.
NAMED_NUMERIC_POLICY = {
    'id':'timemixer-exchange-tokenconv-atol1e-7-v1','kind':'named_tensor',
    'model':'TimeMixer','dataset':'Exchange','horizons':[96,192,336,720],
    'parameter':'enc_embedding.value_embedding.tokenConv.weight',
    'shape':[16,1,3],'dtype':'torch.float32',
    'atol':1e-7,'metric_atol':1e-7,'rtol':0.0,'equal_nan':False,
    'tensor_fields':['parameter','gradient','exp_avg','exp_avg_sq'],
    'initial_identity':'exact','other_state':'exact',
    'metrics':'loss and normalized validation errors atol1e-7',
}
FULL_STATE_NUMERIC_POLICIES = [
    {
        'id':'timemixer-etth1-fullfloat-atol1e-4-v1','kind':'full_float_state',
        'model':'TimeMixer','dataset':'ETTh1','horizons':[96,192,336,720],
        'state_atol':1e-4,'metric_atol':1e-6,'rtol':0.0,'equal_nan':False,
        'floating_state':'all model parameters/buffers, gradients and Adam moments',
        'exact_state':'initial identity/RNG/batch, optimizer step, nonfloating state and param-group structure',
        'capture':'preallocated raw sidecar v1',
    },
    {
        'id':'timemixer-ecl-fullfloat-atol1e-4-v1','kind':'full_float_state',
        'model':'TimeMixer','dataset':'ECL','horizons':[96,192,336,720],
        'state_atol':1e-4,'metric_atol':1e-6,'rtol':0.0,'equal_nan':False,
        'floating_state':'all model parameters/buffers, gradients and Adam moments',
        'exact_state':'initial identity/RNG/batch, optimizer step, nonfloating state and param-group structure',
        'capture':'preallocated raw sidecar v1',
    },
    {
        'id':'moderntcn-etth1-fullfloat-atol1e-4-v1','kind':'full_float_state',
        'model':'ModernTCN','dataset':'ETTh1','horizons':[96,192,336,720],
        'state_atol':1e-4,'metric_atol':1e-6,'rtol':0.0,'equal_nan':False,
        'floating_state':'all model parameters/buffers, gradients and Adam moments',
        'exact_state':'initial identity/RNG/batch, optimizer step, nonfloating state and param-group structure',
        'capture':'preallocated raw sidecar v1',
    },
]
# Conditional extensions: production startup stays blocked until kernel and
# paired numeric evidence is accepted. The older named Exchange policy is historical.
KERNEL_NUMERIC_POLICIES = [
    dict(FULL_STATE_NUMERIC_POLICIES[0], id='timemixer-exchange-fullfloat-atol1e-4-kernel-v1', model='TimeMixer', dataset='Exchange'),
    dict(FULL_STATE_NUMERIC_POLICIES[2], id='moderntcn-weather-fullfloat-atol1e-4-kernel-v1', model='ModernTCN', dataset='Weather'),
    dict(FULL_STATE_NUMERIC_POLICIES[2], id='moderntcn-ecl-fullfloat-atol1e-3-loss-scaled-kernel-v1', model='ModernTCN', dataset='ECL', state_atol=1e-3, loss_atol=1e-6, loss_rtol=1e-5),
    dict(FULL_STATE_NUMERIC_POLICIES[0], id='timemixer-weather-fullfloat-atol1e-4-loss-scaled-kernel-v1', model='TimeMixer', dataset='Weather', loss_atol=1e-6, loss_rtol=1e-5),
]
NUMERIC_PROBE_POLICIES = FULL_STATE_NUMERIC_POLICIES + KERNEL_NUMERIC_POLICIES

RSS_PROBE_POLICY = dict(id='rss-material-growth-platform-v1', short_window=4,
    warmup_updates=6, long_total_updates=24, min_cumulative_bytes=32*1024**2,
    min_average_bytes_per_step=1024**2, plateau_window=8,
    plateau_range_bytes=8*1024**2, plateau_abs_slope_bytes_per_step=256*1024)


# Opt-in, materialized routine admission; legacy/equivalence contracts stay frozen.
BASELINE_NUMERIC_ADMISSION_ID = 'baseline_numeric_admission_v1'
BASELINE_NUMERIC_DEFAULTS = dict(
    M=dict(state_atol=2e-4, metric_atol=1e-6, loss_atol=1e-6, loss_rtol=0),
    UrbanEV_MS=dict(state_atol=1e-4, metric_atol=1e-6, loss_atol=1e-6, loss_rtol=0),
    EPF_MS=dict(state_atol=1e-4, metric_atol=1e-6, loss_atol=1e-6, loss_rtol=0),
    exceptions={'M/TimeMixer/Weather':dict(loss_rtol=1e-5),
                'EPF_MS/ModernTCN':dict(state_atol=5e-4)})

def baseline_numeric_admission_policy(c, task):
    """Resolve only registered baseline M/UrbanEV-MS/EPF-MS task identities."""
    model,name,kind=task.get('model'),task.get('dataset'),task.get('task')
    if model not in tuple(m for m in MODELS if m not in ('J','N','S')) or task not in c['tasks']:
        raise ValueError('routine numeric admission: unregistered baseline/task')
    p=c['resolved_profiles'][task['id']];d=c['datasets'][name];C=p['C']
    if type(C)is not int or C<=0 or p['features']!=d['features'] or len(p['features'])!=C:
        raise ValueError('routine numeric admission: channel/feature identity')
    if kind=='M' and name in ('ETTh1','ETTh2','ETTm1','ETTm2','Weather','Exchange'):
        scope='M';allowed=(96,192,336,720)
        if (p.get('task')!='M' or task.get('input_variant')!='M' or task.get('metric_scope')!='all_channels'
                or p.get('metric_scope')!='all_channels' or p.get('supervised_channels')!=list(range(C))
                or p.get('output_order')!=d['features'] or p['pred_len']!=task['h']):
            raise ValueError('routine numeric admission: exact M supervision/output identity')
    elif kind=='MS' and name=='UrbanEV':
        scope='UrbanEV_MS';allowed=(3,6,9,12)
        if p['pred_len']!=1 or task.get('input_variant')!='F4':raise ValueError('routine numeric admission: UrbanEV label identity')
    elif kind=='MS' and name in ('PJM','NP','BE','FR','DE'):
        scope='EPF_MS';allowed=(24,)
        if p['pred_len']!=24 or task.get('input_variant')!='MS':raise ValueError('routine numeric admission: EPF horizon identity')
    else:raise ValueError('routine numeric admission: undefined task/data category; no exact fallback')
    if kind=='MS' and (p.get('task','MS')!='MS' or task.get('metric_scope')!='target_only'
            or p.get('metric_scope','target_only')!='target_only'):
        raise ValueError('routine numeric admission: MS/M identity conflict')
    horizons=sorted({t['h']for t in c['tasks']if (t['model'],t['dataset'],t['task'])==(model,name,kind)})
    if any(type(h)is not int or h not in allowed or h not in d['horizons']for h in horizons):
        raise ValueError('routine numeric admission: unregistered horizon')
    fields=dict(BASELINE_NUMERIC_DEFAULTS[scope])
    for key in (scope+'/'+model,scope+'/'+model+'/'+name):fields.update(BASELINE_NUMERIC_DEFAULTS['exceptions'].get(key,{}))
    return dict(fields,id=BASELINE_NUMERIC_ADMISSION_ID+'/'+scope+'/'+model+'/'+name,
        admission_version=BASELINE_NUMERIC_ADMISSION_ID,kind='full_float_state',task=kind,
        model=model,dataset=name,horizons=horizons,rtol=0,equal_nan=False,
        floating_state='all model parameters/buffers, gradients and Adam moments',
        exact_state='initial identity/RNG/batch, optimizer step, nonfloating state and param-group structure',
        capture='preallocated raw sidecar v1')

def materialize_baseline_numeric_admission(c):
    """Explicit new config generation, never a mutation of historical input."""
    import copy
    value=copy.deepcopy(c);policies={}
    for task in value['tasks']:
        key=task['model']+'-'+task['dataset'];rule=baseline_numeric_admission_policy(value,task)
        if key in policies and policies[key]!=rule:raise ValueError('routine numeric admission: overlapping task modes')
        policies[key]=rule
    value['baseline_unified'].update(numeric_admission_version=BASELINE_NUMERIC_ADMISSION_ID,numeric_policies=policies)
    return value

def validate_baseline_numeric_registry(c):
    b=c['baseline_unified']
    if b.get('numeric_admission_version')!=BASELINE_NUMERIC_ADMISSION_ID:raise ValueError('unsupported routine numeric admission version')
    expected=materialize_baseline_numeric_admission(c)['baseline_unified']['numeric_policies']
    if b.get('numeric_policies')!=expected:raise ValueError('routine numeric admission: explicit complete registry required; no None/exact fallback')
    return expected

def validate_baseline_full_schema(schema, rule):
    """Complete captured state structure, separate from floating tolerances."""
    import math
    if set(schema)!={'policy_id','entries','optimizer_groups','optimizer_non_tensor_state','total_bytes'} or schema['policy_id']!=rule['id']:
        raise ValueError('routine full state schema/policy identity')
    sizes={'torch.float16':2,'torch.float32':4,'torch.float64':8,'torch.int8':1,'torch.uint8':1,'torch.int16':2,'torch.int32':4,'torch.int64':8,'torch.bool':1}
    paths={};offset=0;parameters=set();gradients=set();states={}
    if not isinstance(schema['entries'],list)or not isinstance(schema['optimizer_groups'],list)or not isinstance(schema['optimizer_non_tensor_state'],dict):raise ValueError('routine full state schema types')
    for e in schema['entries']:
        if set(e)!={'path','dtype','shape','mode','offset','nbytes','numel'}or e['path']in paths:raise ValueError('routine full state entry coverage/schema')
        if e['dtype']not in sizes or not isinstance(e['shape'],list)or any(type(x)is not int or x<0 for x in e['shape']):raise ValueError('routine full state shape/dtype')
        if any(type(e[x])is not int for x in ('offset','nbytes','numel'))or e['offset']!=offset or e['numel']!=math.prod(e['shape'])or e['nbytes']!=e['numel']*sizes[e['dtype']]:raise ValueError('routine full state contiguous shape/byte coverage')
        floating=e['dtype'].startswith('torch.float');name=e['path']
        if name.startswith('model/parameter/'):parameters.add(name[len('model/parameter/'):])
        elif name.startswith('model/buffer/'):pass
        elif name.startswith('gradient/'):gradients.add(name[len('gradient/'):])
        elif name.startswith('optimizer/'):
            parameter,field=name[len('optimizer/'):].rsplit('/',1)
            if field not in ('step','exp_avg','exp_avg_sq','max_exp_avg_sq'):raise ValueError('routine full state Adam field')
            states.setdefault(parameter,set()).add(field)
        else:raise ValueError('routine full state unknown path')
        mode='exact'if name.startswith('optimizer/')and name.endswith('/step')or not floating else 'bounded'
        if e['mode']!=mode:raise ValueError('routine full state floating/nonfloating/optimizer-step mode')
        paths[name]=e;offset+=e['nbytes']
    optimizer_names=[]
    for group in schema['optimizer_groups']:
        if not isinstance(group,dict)or not isinstance(group.get('params'),list):raise ValueError('routine full state parameter groups')
        optimizer_names.extend(group['params'])
    if not parameters or len(set(optimizer_names))!=len(optimizer_names)or set(optimizer_names)!=parameters or not gradients<=parameters or not set(states)<=parameters:
        raise ValueError('routine full state parameter/gradient/Adam coverage')
    if any(not {'step','exp_avg','exp_avg_sq'}<=fields for fields in states.values())or gradients!=set(states):raise ValueError('routine full state gradients/Adam moments missing')
    if type(schema['total_bytes'])is not int or schema['total_bytes']!=offset:raise ValueError('routine full state total byte coverage')
    return schema


def numeric_probe_policy(c, task):
    if c.get('baseline_unified',{}).get('numeric_admission_version')is not None:
        b=c['baseline_unified']
        if b['numeric_admission_version']!=BASELINE_NUMERIC_ADMISSION_ID:raise ValueError('unsupported routine numeric admission version')
        rule=baseline_numeric_admission_policy(c,task)
        if b['numeric_policies'].get(task['model']+'-'+task['dataset'])!=rule:raise ValueError('routine numeric admission: missing/foreign explicit policy')
        return rule
    if c.get('type1_followup'):
        from utils.ch3_type1_tasks import numeric_policy as type1_handler
        return type1_handler(c,task)
    if 'baseline_unified' in c:
        from utils.ch3_baseline_unified_tasks import numeric_policy
        return numeric_policy(c,task)
    if 'native_replacement' in c:
        from utils.ch3_native_tasks import numeric_policy
        return numeric_policy(c,task)
    if 'm_experiment' in c:
        from utils.ch3_m_tasks import numeric_policy
        return numeric_policy(c,task)
    policies=c['execution']['probe'].get('numeric_equivalence')
    if policies is None:return None
    if policies!=NUMERIC_PROBE_POLICIES:raise ValueError('unauthorized numeric equivalence policy set')
    matches=[p for p in policies if (task['model'],task['dataset'])==(p['model'],p['dataset'])]
    if len(matches)>1:raise ValueError('overlapping numeric equivalence policies')
    if not matches:return None
    policy=matches[0]
    if task['h'] not in policy['horizons']:raise ValueError('numeric equivalence horizon outside scope')
    return policy

def baseline_training(c, task):
    layer=c.get('baseline_training_overrides',{})
    for model,domains in layer.items():
        if model not in BASELINES:raise ValueError('external training override forbidden for AMD family')
        for domain,bank in domains.items():
            if domain not in c['datasets']:raise ValueError('override dataset unknown')
            for horizon,entry in bank.items():
                if horizon!='*' and horizon not in [str(h) for h in c['datasets'][domain]['horizons']]:
                    raise ValueError('override horizon unknown')
                if set(entry)!={'values','source','pending'} or set(entry['values'])-TRAINING_OVERRIDES:
                    raise ValueError('only batch/eval_batch/lr overrides authorized')
                if not entry['source'] or set(entry['pending'])-TRAINING_OVERRIDES:
                    raise ValueError('source/decision binding missing')
                for k,v in entry['values'].items():
                    import math
                    if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:
                        raise ValueError('invalid training override')
                    if k!='lr' and not isinstance(v,int):raise ValueError('batch must be integer')
                if ('batch' in entry['values'])!=('eval_batch' in entry['values']):
                    raise ValueError('explicit train/eval batch pair required')
    bank=layer.get(task['model'],{}).get(task['dataset'],{})
    return bank.get(str(task['h']),bank.get('*',dict(values={},source=[],pending={})))


def training_blockers(c, task):
    return [task['model']+'/'+task['dataset']+'/'+str(task['h'])+': '+k+' '+str(v)
            for k,v in baseline_training(c,task)['pending'].items()]



def followup_limits(c, approval=None):
    """Check exact coverage and a pooled hard allowance, before creating output."""
    f=c['execution'].get('followup')
    if not f:return c['execution']['probe']['first_adam_max']
    inherited=f['inherit_groups'];retest=f['retest_groups']
    all_ids={g['id'] for g in c['groups']}
    if (len(set(inherited))!=len(inherited) or len(set(retest))!=len(retest)
            or set(inherited)&set(retest) or set(inherited+retest)!=all_ids):
        raise ValueError('follow-up exact partition mismatch')
    groups=[g for g in c['groups'] if g['id'] in retest]
    q=sum(g['q'] for g in groups)
    for k in ('remaining_first_adam','approved_extra_adam'):
        if type(f.get(k)) is not int or f[k]<0:raise ValueError('invalid allowance')
    maximum=f['remaining_first_adam']+f['approved_extra_adam']
    if f.get('scheduling_budget_policy')=='fixed-order-capped-resource-fallback-v1':
        base=sum(6*g['q']*(1 if g['q']==1 else 2) for g in groups)
        if (f['Q']!=q or f['core_adam_max']!=base or base>maximum
                or f['planned_adam']!=maximum or f['planned_backward']!=maximum
                or f['planned_forward']!=maximum//6*8 or f['planned_validation']!=maximum//6*2):
            raise ValueError('capped follow-up arithmetic mismatch')
    else:
        factor=sum(g['q']*(1 if g['q']==1 else 3) for g in groups)
        expected=dict(Q=q,planned_adam=6*factor,planned_forward=8*factor,
                      planned_backward=6*factor,planned_validation=2*factor)
        if any(f.get(k)!=v for k,v in expected.items()) or f['planned_adam']>maximum:
            raise ValueError('follow-up exceeds approved original plus extra budget')
    if approval is not None and (approval.get('followup_sha')!=digest(f)
            or approval.get('approved_extra_adam')!=f['approved_extra_adam']):
        raise ValueError('exact follow-up scope/extra approval missing')
    return maximum

def step_arithmetic(c, task):
    p=profile(c,task);d=c['datasets'][task['dataset']];b=p['training']['batch'];v=p['training']['eval_batch']
    if task['dataset']=='UrbanEV':
        a,z,n=c['urban_folds'][task['fold']-1]
        counts=[(length-p['T']-task['h']+1)*275 for length in (a,z-a,n-z)]
    else:
        a,z,n=d['endpoints']
        counts=[a-p['T']-p['pred_len']+1,z-a-p['pred_len']+1,n-z-p['pred_len']+1]
    if min(counts)<=0:raise ValueError('empty task windows')
    return dict(train_windows=counts[0],train_batches=counts[0]//b,train_dropped=counts[0]%b,
                max_optimizer_steps=counts[0]//b*p['training']['epochs'],
                validation_windows=counts[1],validation_full_batches=counts[1]//v,validation_tail=counts[1]%v,
                test_windows_arithmetic_only=counts[2],test_full_batches=counts[2]//v,test_tail=counts[2]%v)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def read_profiles(path=PROFILE_FILE):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def generate_tasks(c):
    if c.get('type1_followup'):
        from utils.ch3_type1_tasks import expected_tasks as type1_handler
        return type1_handler(c['baseline_unified']['stage'])
    if 'baseline_unified' in c:
        from utils.ch3_baseline_unified_tasks import expected_tasks
        return expected_tasks(c['baseline_unified']['stage'])
    if 'native_replacement' in c:
        from utils.ch3_native_tasks import tasks
        return tasks()
    if "m_experiment" in c:
        from utils.ch3_m_tasks import tasks
        return tasks()
    if 'extension' in c:
        from utils.ch3_extension import expected_tasks
        return expected_tasks(c)
    tasks = []
    for model in MODELS:
        domains = ['UrbanEV'] if model in ('N','S') else (
            ['ETTh1','Weather','ECL','Exchange'] if model == 'TimeMixer' else list(c['datasets']))
        for dataset in domains:
            d = c['datasets'][dataset]
            input_variants = (['F1','F2','F3','F4'] if model in ('AMD','J') else ['F4']) if dataset=='UrbanEV' else ['MS']
            for input_variant in input_variants:
                for fold in range(1,7) if dataset=='UrbanEV' else [1]:
                    for h in d['horizons']:
                        run = f'{model}-{dataset}-{input_variant}-f{fold}-h{h}-s2024'
                        group = f'{model}-{dataset}-{input_variant}'
                        tasks.append(dict(id=run, model=model, dataset=dataset, input_variant=input_variant,
                                          fold=fold, h=h, group=group, profile=group+f'-h{h}', seed=2024))
    return tasks


def generate_groups(tasks):
    groups = {}
    for task in tasks:
        group = groups.setdefault(task['group'], dict(id=task['group'], model=task['model'],
                                  dataset=task['dataset'], input_variant=task['input_variant'], task_ids=[], representatives=[]))
        group['task_ids'].append(task['id'])
        if task['fold']==1: group['representatives'].append(task['id'])
    for g in groups.values():
        g['q'] = len(g['representatives'])
        g['equivalence'] = ('Same T/C/output/batch/structure across folds; each H represented; '
                            'fold-specific CPU data/labels/RSS checked separately' if g['dataset']=='UrbanEV'
                            else 'Every distinct H represented; no H720 domination assumption')
        g['waves'] = [g['task_ids'][i:i+g['q']] for i in range(0,len(g['task_ids']),g['q'])]
    return list(groups.values())


def task_by_id(c, run_id):
    matches = [t for t in c['tasks'] if t['id']==run_id]
    if len(matches)!=1: raise ValueError('unknown or duplicate run ID')
    return matches[0]


def profile(c, task):
    if c.get('type1_followup'):
        from utils.ch3_type1_tasks import profile as type1_handler
        return type1_handler(c,task)
    if 'baseline_unified' in c:
        from utils.ch3_baseline_unified_tasks import profile as unified_profile
        return unified_profile(c,task)
    if 'native_replacement' in c:
        from utils.ch3_native_tasks import resolved
        return resolved(c,task)
    if "m_experiment" in c:
        from utils.ch3_m_tasks import resolved
        return resolved(c,task)
    d = c['datasets'][task['dataset']]
    input_variant = task['input_variant']
    names = c['urban_input_variants'][input_variant] if task['dataset']=='UrbanEV' else (
        [d['target']] if input_variant=='TargetOnly' else d['features'])
    target = names.index(d['target'])
    h = task['h']
    if task['model'] in ('J','S') and (len(names)<2 or input_variant=='F0'):
        raise ValueError('J/S require nonempty ordered aux; F0 cannot retain S2 identity')
    base = dict(contract=CONTRACT, model=task['model'], dataset=task['dataset'], input_variant=input_variant,
                T=d['T'], H=h, pred_len=1 if task['dataset']=='UrbanEV' else h,
                features=names, C=len(names), target_idx=target,
                aux_idx=[i for i in range(len(names)) if i!=target],
                training=dict(c['training_common'], **d['training']))
    base['training'].update(baseline_training(c,task)['values'])
    if task['model'] in ('AMD','J','N','S'):
        base['structure'] = dict(c['amd'], patch=d['amd_patch'], layernorm=task['dataset']!='ECL',
                                 kernel_small=3 if task['dataset']=='UrbanEV' else 5,
                                 kernel_large=7 if task['dataset']=='UrbanEV' else 31)
    else:
        bank=c['structures'][task['model']][task['dataset']]
        base['structure']=bank[str(h) if str(h) in bank else '*']['values']
    return base


def validate_manifest(c):
    if c.get('type1_followup'):
        from utils.ch3_type1_tasks import validate as type1_handler
        return type1_handler(c)
    if 'baseline_unified' in c:
        from utils.ch3_baseline_unified_tasks import validate
        return validate(c)
    if 'native_replacement' in c:
        from utils.ch3_native_tasks import validate
        return validate(c)
    if "m_experiment" in c:
        from utils.ch3_m_tasks import validate
        return validate(c)
    if c['contract']!=CONTRACT: raise ValueError('wrong formal contract')
    if c['execution']['probe'].get('rss_policy')!=RSS_PROBE_POLICY:raise ValueError('unauthorized RSS rule')
    if c['execution']['probe'].get('numeric_equivalence') is not None:
        for policy in NUMERIC_PROBE_POLICIES:
            numeric_probe_policy(c,dict(model=policy['model'],dataset=policy['dataset'],h=policy['horizons'][0]))
    if 'extension' in c:
        from utils.ch3_extension import validate_extension
        return validate_extension(c)
    expected=generate_tasks(c)
    if c['tasks']!=expected or c['groups']!=generate_groups(expected):
        raise ValueError('task/group manifest mismatch')
    if len(expected)!=495 or len({t['id'] for t in expected})!=495: raise ValueError('task count/duplicates')
    if len(c['groups'])!=54 or sum(g['q'] for g in c['groups'])!=195: raise ValueError('probe Q budget')
    if sum(profile(c,t)['training']['epochs'] for t in expected)!=5340: raise ValueError('epoch budget')
    if any(g['q'] not in (1,2,3,4) for g in c['groups']): raise ValueError('worker group size')
    for t in expected:
        p=profile(c,t)
        if p['C']==1 and t['model']!='AMD': raise ValueError('target-only J forbidden')
    return c


def validate_amd_declaration(declaration, *, input_shape, pred_len, patch, layernorm,
                             target_idx, aux_idx, norm, task_mode, s2, thls):
    if isinstance(declaration,dict) and declaration.get('task')=='M':
        p=declaration
        if (p['model']!='AMD' or p['dataset']not in ('ETTh1','Weather','Exchange') or
            tuple(input_shape)!=(p['T'],p['C'])or pred_len!=p['pred_len']or patch!=p['structure']['patch']or
            layernorm!=p['structure']['layernorm']or target_idx!=p['target_idx']or tuple(aux_idx) or
            not norm or task_mode!='parallel_multivariate'or s2 or thls):raise ValueError('M AMD native all-channel identity')
        return
    if not isinstance(declaration,dict) or declaration.get('contract')!=CONTRACT:
        raise ValueError('explicit new formal declaration required')
    p=declaration
    if p['model'] in ('J','S') and (not p['aux_idx'] or p.get('input_variant')=='F0'):
        raise ValueError('J/S require nonempty ordered aux; never disable S2 implicitly')
    if p['model'] not in ('AMD','N','S','J') or p['dataset'] not in ('UrbanEV','PJM','ETTh1','Weather','ECL','Exchange','NP','BE','FR','DE'):
        raise ValueError('formal domain/arm')
    expected_patch={'NP':24,'BE':24,'FR':24,'DE':24,'UrbanEV':12,'PJM':24,'ETTh1':16,'Weather':16,'ECL':16,'Exchange':4}
    if (tuple(input_shape)!=(p['T'],p['C']) or pred_len!=p['pred_len']
            or patch!=expected_patch[p['dataset']] or layernorm!=(p['dataset']!='ECL')
            or target_idx!=p['target_idx'] or tuple(aux_idx)!=tuple(p['aux_idx'])
            or not norm or task_mode!='target_exogenous'
            or s2!=(p['model'] in ('S','J')) or thls!=(p['model'] in ('N','J'))):
        raise ValueError('formal AMD shape/normalization/module identity mismatch')
    if p['model'] in ('N','S') and (p['dataset'],p['input_variant'])!=('UrbanEV','F4'):
        raise ValueError('N/S formal scope is UrbanEV F4 only')


class BestState:
    def __init__(self, patience=None):
        self.best=None; self.epoch=None; self.bad=0; self.patience=patience
    def update(self, mse, epoch):
        import math
        if not math.isfinite(mse): raise ValueError('nonfinite validation')
        improved=self.best is None or mse<self.best
        if improved: self.best,self.epoch,self.bad=mse,epoch,0
        else: self.bad+=1
        return improved
    @property
    def stopped(self): return self.patience is not None and self.bad>=self.patience


def summarize(c, rows):
    seen=set();panels={}
    for row in rows:
        t=task_by_id(c,row['id'])
        if row['id'] in seen: raise ValueError('duplicate result')
        seen.add(row['id'])
        if (row['purpose']!='ch3_formal' or row['protocol_sha']!=digest(c)
                or row.get('input_variant')!=t['input_variant']): raise ValueError('foreign result identity')
        panels.setdefault(t['group'],[]).append((t,row))
    result={}
    for group,items in panels.items():
        required=next(g['task_ids'] for g in c['groups'] if g['id']==group)
        if {t['id'] for t,_ in items}!=set(required): raise ValueError('incomplete panel')
        per_h={}
        for t,row in items:per_h.setdefault(t['h'],[]).append(row)
        result[group]={metric:sum(sum(v[metric] for v in rows)/len(rows) for rows in per_h.values())/len(per_h)
                       for metric in ('mse','mae')}
        result[group].update(input_variant=items[0][0]['input_variant'],seed=2024,std=None,stability='Not evaluated',metric_space='train-standardized')
    return result
