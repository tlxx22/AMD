"""Formal task/configuration contract; no torch or dataset import."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE_FILE = ROOT / 'configs/ch3_formal_profiles.json'
CONTRACT = 'ch3-target-ms-formal-v2'
MODELS = ('AMD','J','DLinear','PatchTST','iTransformer','TimeMixer','ModernTCN','TimeXer','N','S')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def read_profiles(path=PROFILE_FILE):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def generate_tasks(c):
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
    if task['model'] in ('AMD','J','N','S'):
        base['structure'] = dict(c['amd'], patch=d['amd_patch'], layernorm=task['dataset']!='ECL',
                                 kernel_small=3 if task['dataset']=='UrbanEV' else 5,
                                 kernel_large=7 if task['dataset']=='UrbanEV' else 31)
    else:
        bank=c['structures'][task['model']][task['dataset']]
        base['structure']=bank[str(h) if str(h) in bank else '*']['values']
    return base


def validate_manifest(c):
    if c['contract']!=CONTRACT: raise ValueError('wrong formal contract')
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
    if not isinstance(declaration,dict) or declaration.get('contract')!=CONTRACT:
        raise ValueError('explicit new formal declaration required')
    p=declaration
    if p['model'] in ('J','S') and (not p['aux_idx'] or p.get('input_variant')=='F0'):
        raise ValueError('J/S require nonempty ordered aux; never disable S2 implicitly')
    if p['model'] not in ('AMD','N','S','J') or p['dataset'] not in ('UrbanEV','PJM','ETTh1','Weather','ECL','Exchange'):
        raise ValueError('formal domain/arm')
    expected_patch={'UrbanEV':12,'PJM':24,'ETTh1':16,'Weather':16,'ECL':16,'Exchange':4}
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
