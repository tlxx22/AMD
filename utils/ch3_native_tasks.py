"""Exact native-time-mark replacement and successor contexts; no model imports."""
import copy
import json
import subprocess
from functools import lru_cache
from pathlib import Path
from utils.ch3_contract import ROOT, digest, profile, step_arithmetic, generate_groups
from utils import ch3_m_tasks as m
from utils.ch3_time_marks import MODE, MODELS, policy

BASE = 'be347d8f1990e20893883fd1203e21ab21897c78'
ID = 'm6-native-tmark-chain-v1'
PACKAGE = m.PACKAGE / 'native-time-mark-chain-v1'
EVIDENCE = ROOT.parent / 'amd-execution-evidence/m6/m6-formal-launch-dhozikhu'
REPLACEMENT = EVIDENCE / 'revisions/native-time-mark-v1'
M_RESULT = EVIDENCE / 'm-tasks/m-baselines-native-time-mark-v1'
PROFILE_FILE = ROOT / 'configs/ch3_native_time_mark_profiles.json'
DOMAINS = ('UrbanEV', 'PJM', 'NP', 'BE', 'FR', 'DE')


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
    return c['native_replacement']['numeric_policies'][t['model'] + '-' + t['dataset']]['rule']


def validate(c):
    old = parent_config()
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
    replacement = 'native_replacement' in c
    stage = 'tmark' if replacement else 'm'
    return dict(stage=stage, probe_scope=ID + '-' + stage + '-probe', formal_scope=ID + '-' + stage + '-formal',
                probe_root=PACKAGE / (stage + '-probe-execution-v1'),
                result_root=REPLACEMENT if replacement else M_RESULT,
                control=(REPLACEMENT if replacement else M_RESULT) / 'queues' / ID,
                models=MODELS if replacement else m.MODELS,
                protocol_file=PROFILE_FILE if replacement else ROOT / 'configs/ch3_formal_profiles.json',
                caps=dict(adam=240, forward=392, backward=240) if replacement else m.CAPS,
                runs=87 if replacement else 84, epochs=1020 if replacement else 840,
                optimizer_steps=3687530 if replacement else 294790)


def result_path(c, t):
    path = context(c)['result_root'] / ('formal-' + t['model']) / t['id']
    if path.is_symlink() or path.resolve() != path:
        raise ValueError('native successor exact result namespace')
    return path


def computational_identity(c, t):
    p = profile(c, t)
    # Raw label H is kept separate even UrbanEV's network output has pred_len=1.
    train = {k: v for k, v in p['training'].items() if k not in ('epochs', 'patience')}
    return dict(model=t['model'], T=p['T'], label_H=t['h'], prediction_H=p['pred_len'], C=p['C'],
                training=train, structure=p['structure'], mark_shape=[p['T'], 4],
                mark_mode=p['time_mark'], output_shape=[p['pred_len'], 1],
                numeric_rule=numeric_policy(c, t))


def probe_groups(c):
    if 'native_replacement' not in c:
        return [dict(g, representatives=g['task_ids'], planned_q=4) for g in c['groups']]
    bank = {}
    for t in c['tasks']:
        key = digest(computational_identity(c, t))
        bank.setdefault(key, []).append(t)
    result = []
    for model in MODELS:
        urban = [rows for rows in bank.values() if rows[0]['model'] == model and rows[0]['dataset'] == 'UrbanEV']
        reps = [rows[0]['id'] for rows in urban]
        result.append(dict(id=model + '-UrbanEV-native', model=model, representatives=reps, planned_q=4,
                           identities={rows[0]['id']:digest(computational_identity(c, rows[0])) for rows in urban},
                           coverage={rows[0]['id']:[t['id'] for t in rows] for rows in urban},
                           equivalence='same network shape/H/batch/structure/mark; fold statistics and counts remain loader-bound; full eval batch bounds tail resource shape'))
        for rows in bank.values():
            if rows[0]['model'] != model or rows[0]['dataset'] == 'UrbanEV':
                continue
            representative = rows[0]
            result.append(dict(id=model + '-EPF-native-' + str(len(result)), model=model,
                               representatives=[representative['id']], planned_q=1,
                               identities={representative['id']:digest(computational_identity(c, representative))},
                               coverage={representative['id']:[t['id'] for t in rows]},
                               equivalence='same T/H/C/batch/optimizer/structure/hourly native interface; dataset names/ordered business names differ only'))
    return result


def plan(c):
    ctx = context(c)
    groups = probe_groups(c)
    representatives = sum(len(g['representatives']) for g in groups)
    nominal = sum(len(g['representatives']) * (2 if g['planned_q'] > 1 else 1) for g in groups)
    maximum = nominal + sum(len(g['representatives']) for g in groups if g['planned_q'] > 1)
    if 'native_replacement' in c and (representatives, nominal, maximum) != (16, 28, 40):
        raise ValueError('native distinct computational coverage drift')
    return dict(id=ID, stage=ctx['stage'], protocol_sha=digest(c),
                task_ids=[t['id'] for t in c['tasks']], profile_shas={t['id']:digest(profile(c,t)) for t in c['tasks']},
                order=list(ctx['models']), run_budget=dict(runs=ctx['runs'],run_epochs=ctx['epochs']),
                max_optimizer_steps=sum(step_arithmetic(c,t)['max_optimizer_steps'] for t in c['tasks']),
                groups=groups, nominal_workers=nominal, max_workers=maximum,
                caps=ctx['caps'], per_worker={r:worker_counts(c,next(t for t in c['tasks'] if t['id']==r)) for g in groups for r in g['representatives']},
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
