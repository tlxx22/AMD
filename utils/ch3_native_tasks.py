"""Exact native-time-mark replacement and successor contexts; no model imports."""
import copy
import json
import subprocess
from functools import lru_cache
from pathlib import Path
from utils.ch3_contract import ROOT, digest, profile, step_arithmetic, generate_groups
from utils import ch3_m_tasks as m
from utils.ch3_time_marks import MODE, MODELS, policy

BASE = '42cee1d453aa56e5dd6fb725817cc8d0b97077a6'
ID = 'm6-native-tmark-chain-v4'
PACKAGE = m.PACKAGE / 'native-time-mark-chain-v4'
EVIDENCE = ROOT.parent / 'amd-execution-evidence/m6/m6-formal-launch-dhozikhu'
REPLACEMENT = EVIDENCE / 'revisions/native-time-mark-v4'
M_RESULT = EVIDENCE / 'm-tasks/m-baselines-native-time-mark-v4'
PROFILE_FILE = ROOT / 'configs/ch3_native_time_mark_profiles.json'
DOMAINS = ('UrbanEV', 'PJM', 'NP', 'BE', 'FR', 'DE')

# Replacement only. M's independently frozen numerical policies are untouched.
REPLACEMENT_NUMERIC_POLICY = dict(
    id='native-tmark-replacement-fullfloat-atol1e-4-v4', kind='full_float_state',
    state_atol=1e-4, loss_atol=1e-6, metric_atol=1e-6,
    loss_rtol=0.0, rtol=0.0, equal_nan=False,
    floating_state='all model parameters/buffers, gradients, Adam exp_avg and exp_avg_sq',
    exact_state='initialization/RNG/batch identity/order, optimizer step/nonfloating state, param groups, tensor shape/dtype, task/profile/source/data',
    nonfloating_and_step='exact',
    initialization_rng_batches='exact', capture='preallocated raw sidecar v1')


def replacement_numeric_rule():
    return copy.deepcopy(REPLACEMENT_NUMERIC_POLICY)


@lru_cache(maxsize=1)
def parent_config():
    return json.loads(subprocess.check_output(['git', '-C', str(ROOT), 'show', m.BASE + ':configs/ch3_formal_profiles.json']))


@lru_cache(maxsize=1)
def tasks():
    bank = parent_config()['tasks']
    result = []
    for model in MODELS:
        # Retain each model's actual frozen registry order, including the
        # TimeMixer catchup ordering; do not invent a common domain sort.
        selected = [t for t in bank if t['model'] == model and t['dataset'] in DOMAINS
                    and (t['dataset'] != 'UrbanEV' or t['input_variant'] == 'F4')]
        for t in selected:
            value = copy.deepcopy(t)
            value.update(id=t['id'] + '-native-tmark-v1', parent_run_id=t['id'], revision='native-time-mark-v1')
            result.append(value)
    if len(result) != 87:
        raise ValueError('exact 87 replacement tasks')
    return result


def resolved(c, t):
    if t not in tasks():
        raise ValueError('foreign native replacement task')
    parent = c['native_replacement']['parents'][t['id']]
    p = copy.deepcopy(parent['profile'])
    p.update(time_mark=policy(p), parent_run_id=t['parent_run_id'],
             parent_profile_sha=parent['sha256'], revision='native-time-mark-v1')
    return p


def numeric_policy(c, t):
    if t not in c['tasks']:
        raise ValueError('foreign native numerical scope')
    rule = c['native_replacement']['numeric_policies'][t['model'] + '-' + t['dataset']]['rule']
    if rule != replacement_numeric_rule():
        raise ValueError('exact preregistered replacement v4 numerical policy')
    return rule


def validate(c):
    old = parent_config()
    policies = c['native_replacement']['numeric_policies']
    if set(policies) != {model + '-' + domain for model in MODELS for domain in DOMAINS}:
        raise ValueError('exact 18 replacement numerical scopes')
    if any(row.get('rule') != replacement_numeric_rule() for row in policies.values()):
        raise ValueError('all replacement scopes require the fixed v4 bounded rule')
    if c['tasks'] != tasks() or c['groups'] != generate_groups(tasks()):
        raise ValueError('replacement exact tasks/groups')
    if c['native_replacement']['parent_protocol_sha'] != digest(old):
        raise ValueError('replacement parent protocol')
    if c['datasets'] != {d: old['datasets'][d] for d in DOMAINS} or c['urban_folds'] != old['urban_folds']:
        raise ValueError('replacement data/window changed')
    if c['sources'] != {k: old['sources'][k] for k in MODELS}:
        raise ValueError('replacement author source changed')
    for t in tasks():
        pt = next(x for x in old['tasks'] if x['id'] == t['parent_run_id'])
        expected = profile(old, pt)
        if c['native_replacement']['parents'][t['id']] != dict(profile=expected, sha256=digest(expected)):
            raise ValueError('replacement parent profile changed')
        p = resolved(c, t)
        reduced = {k: v for k, v in p.items() if k not in ('time_mark', 'parent_run_id', 'parent_profile_sha', 'revision')}
        if reduced != expected or p['structure'].get('use_future_temporal_feature', 0) != 0:
            raise ValueError('replacement changed beyond historical native mark')
    if sum(resolved(c, t)['training']['epochs'] for t in tasks()) != 1020 or sum(step_arithmetic(c, t)['max_optimizer_steps'] for t in tasks()) != 3687530:
        raise ValueError('replacement exact recomputed budget')
    return c


def context(c):
    from utils.ch3_native_recovery import active_context
    recovery = active_context(c)
    if recovery is not None:
        return recovery
    replacement = 'native_replacement' in c
    stage = 'tmark' if replacement else 'm'
    return dict(stage=stage, probe_scope=ID + '-' + stage + '-probe', formal_scope=ID + '-' + stage + '-formal',
                probe_root=PACKAGE / (stage + '-probe-execution-v1'),
                result_root=REPLACEMENT if replacement else M_RESULT,
                control=(REPLACEMENT if replacement else M_RESULT) / 'queues' / ID,
                models=MODELS if replacement else m.MODELS,
                protocol_file=PROFILE_FILE if replacement else ROOT / 'configs/ch3_formal_profiles.json',
                caps=dict(adam=438, forward=608, backward=438) if replacement else m.CAPS,
                runs=87 if replacement else 84, epochs=1020 if replacement else 840,
                optimizer_steps=3687530 if replacement else 294790)


def result_path(c, t):
    path = context(c)['result_root'] / ('formal-' + t['model']) / t['id']
    if path.is_symlink() or path.resolve() != path:
        raise ValueError('native successor exact result namespace')
    return path


def computational_identity(c, t):
    p = profile(c, t)
    # Label offset/fold remains task-bound. UrbanEV's network output is one
    # step for all four offsets: it is not four different computational shapes.
    train = {k: v for k, v in p['training'].items() if k not in ('epochs', 'patience')}
    return dict(model=t['model'], T=p['T'], prediction_H=p['pred_len'], C=p['C'],
                training=train, structure=p['structure'], mark_shape=[p['T'], 4],
                mark_mode=p['time_mark'], output_shape=[p['pred_len'], 1],
                numeric_rule=numeric_policy(c, t))


def probe_groups(c):
    if 'native_replacement' not in c:
        return [dict(g, representatives=g['task_ids'], planned_q=4) for g in c['groups']]
    result = []
    for model in MODELS:
        urban = [t for t in c['tasks'] if t['model']==model and t['dataset']=='UrbanEV']
        reps = [t['id'] for t in urban if t['fold']==1]
        result.append(dict(id=model + '-UrbanEV-native', model=model, representatives=reps, planned_q=4,
                           identities={r:digest(computational_identity(c,next(t for t in urban if t['id']==r))) for r in reps},
                           coverage={r:[t['id'] for t in urban if t['h']==next(x for x in urban if x['id']==r)['h']] for r in reps},
                           equivalence='fold1 actual four-H serial/parallel; other folds retain exact label/profile/data bindings and identical computational shape'))
        bank = {}
        for t in c['tasks']:
            if t['model']==model and t['dataset']!='UrbanEV':
                bank.setdefault(digest(computational_identity(c,t)),[]).append(t)
        for n,(key,rows) in enumerate(bank.items()):
            if len(rows)<2:raise ValueError('EPF singleton has no new parallel combination')
            # Test the actual concurrent set. The fifth compatible market is
            # a fixed singleton wave, covered by the same input/compute identity.
            selected=rows[:4];reps=[t['id'] for t in selected]
            result.append(dict(id=model+'-EPF-combination-'+str(n),model=model,
                               representatives=reps,planned_q=4 if len(reps)>2 else 2,
                               identities={r:key for r in reps},
                               coverage={r:[r] for r in reps[:-1]}|{reps[-1]:[t['id'] for t in rows[len(reps)-1:]]},
                               equivalence='actual cross-market concurrent workers; same frozen T/H/C/batch/optimizer/structure/mark/numeric rule; remaining singleton inherits only this exact compute identity'))
    return result


def formal_waves(c,report,model):
    """Fixed registered combinations, clipped only by a measured resource fallback."""
    from utils.ch3_m_execution import wave_ids
    if 'native_replacement' not in c:
        return [wave for g in c['groups'] if g['model']==model
                for wave in wave_ids(g['task_ids'],report['decisions'][g['id']]['concurrency'])]
    waves=[]
    positions={t['id']:i for i,t in enumerate(c['tasks'])}
    for g in sorted(probe_groups(c),key=lambda g:min(positions[r] for r in g['representatives'])):
        if g['model']!=model:continue
        q=report['decisions'][g['id']]['concurrency']
        if type(q)is not int or q not in (1,2,4) or q>g['planned_q']:raise ValueError('unmeasured formal concurrency')
        reps=g['representatives']
        if task_by_run(c,reps[0])['dataset']=='UrbanEV':
            for fold in range(1,7):
                ids=[t['id'] for t in c['tasks'] if t['model']==model and t['dataset']=='UrbanEV' and t['fold']==fold]
                waves+=wave_ids(ids,q)
        else:
            covered={r for ids in g['coverage'].values() for r in ids}
            ids=[t['id'] for t in c['tasks'] if t['id'] in covered]
            waves+=wave_ids(ids,q)
    flat=[r for wave in waves for r in wave]
    expected=[t['id'] for t in c['tasks'] if t['model']==model]
    if len(flat)!=len(set(flat)) or set(flat)!=set(expected):raise ValueError('exact formal combination coverage')
    return waves


def task_by_run(c,run):
    return next(t for t in c['tasks'] if t['id']==run)


def attempt_widths(g):
    return (4,2) if g['planned_q']==4 else (2,) if g['planned_q']==2 else ()


def plan(c):
    ctx = context(c)
    groups = probe_groups(c)
    representatives = sum(len(g['representatives']) for g in groups)
    nominal = sum(len(g['representatives']) * (2 if g['planned_q'] > 1 else 1) for g in groups)
    maximum = nominal + sum(len(g['representatives']) for g in groups if g['planned_q'] == 4)
    identities={v for g in groups for v in g.get('identities',{}).values()}
    if 'native_replacement' in c and (len(identities),representatives, nominal, maximum) != (7,25,50,73):
        raise ValueError('native distinct computational coverage drift')
    costs={r:worker_counts(c,task_by_run(c,r)) for g in groups for r in g['representatives']}
    nominal_cost={k:sum(costs[r][k]*2 for g in groups for r in g['representatives']) for k in ('adam','forward','backward')}
    maximum_cost={k:nominal_cost[k]+sum(costs[r][k] for g in groups if g['planned_q']==4 for r in g['representatives']) for k in nominal_cost}
    if any(maximum_cost[k]>ctx['caps'][k] for k in maximum_cost) or ('native_replacement' in c and maximum_cost!=ctx['caps']):raise ValueError('probe caps must cover exact maximum worker dispatch arithmetic')
    planned_report=dict(decisions={g['id']:dict(concurrency=g['planned_q']) for g in groups})
    return dict(id=ctx.get('id',ID), stage=ctx['stage'], protocol_sha=digest(c),
                task_ids=[t['id'] for t in c['tasks']], profile_shas={t['id']:digest(profile(c,t)) for t in c['tasks']},
                order=list(ctx['models']), run_budget=dict(runs=ctx['runs'],run_epochs=ctx['epochs']),
                max_optimizer_steps=sum(step_arithmetic(c,t)['max_optimizer_steps'] for t in c['tasks']),
                groups=groups, computational_identities=len(identities) if identities else representatives,
                representative_tasks=representatives,nominal_workers=nominal, max_workers=maximum,
                nominal_cost=nominal_cost,caps=ctx['caps'], per_worker=costs,
                planned_formal_waves={model:formal_waves(c,planned_report,model) for model in ctx['models']},
                fallback='resource-only q4 -> once q2 -> legal measured serial q1; no other retries',
                status='Prepared; actual-byte review and new clean closure required')


def worker_counts(c,t):
    endpoint = 'native_replacement' in c and t['model']=='TimeMixer' and t['dataset']=='UrbanEV'
    return dict(adam=6,forward=10 if endpoint else 8,backward=6)


def effective_results(c):
    return dict(policy='fixed replacement independent of measured performance',
                legacy_protocol='legacy/common-input; x_mark=None; retained untouched',
                replacement={t['parent_run_id']:dict(run_id=t['id'],result_path=str(result_path(c,t)),
                                                   revision='native-time-mark-v1') for t in c['tasks']},
                all_other_cells='original formal sources; no MS standard-domain replacement')
