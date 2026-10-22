"""Exact, separately registered fold6/EPF three-arm task scope; no model changes."""
import copy
import hashlib
import json
import math
from pathlib import Path

SCOPE = 'development_extension_f6_epf_3arm_v1'
REFERENCE_SHA = 'ba670266429aa28281c6b63c5d12e23b3cdb587e0a1f8c1b20be7f2311739a0a'
MARKETS = ('PJM', 'NP', 'BE', 'FR', 'DE')
ARMS = ('U', 'V', 'UV')


def reference():
    path = Path(__file__).with_name('extension_reference.json')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA:
        raise ValueError('frozen extension scientific reference changed')
    return json.loads(raw)


def expected_tasks():
    tasks = []
    for arm in ARMS:
        for prototype in reference()['prototypes']:
            t = copy.deepcopy(prototype)
            t.update(arm=arm, use_u=arm in ('U', 'UV'), use_v=arm in ('V', 'UV'))
            parts = t['id'].split('|'); parts[-2] = arm; t['id'] = '|'.join(parts)
            tasks.append(t)
    return tasks


def expected_waves():
    ids = [t['id'] for t in expected_tasks()]
    return [ids[start+lo:start+hi] for start in (0, 9, 18)
            for lo, hi in ((0, 4), (4, 8), (8, 9))]


def validate_extension(config):
    ref = reference()
    if config.get('task_scope') != SCOPE or config.get('extension_reference_sha256') != REFERENCE_SHA:
        raise ValueError('explicit frozen extension scope/reference required')
    if config['tasks'] != expected_tasks():
        raise ValueError('exact ordered 27 extension scientific tasks required')
    if config.get('existing_B') != ref['existing_B'] or config.get('F4_reuse') != {}:
        raise ValueError('exact nine B cells; fold6 is not goal3 F4 reuse')
    if config.get('parent_config_ref') != ref['parent_config_ref']:
        raise ValueError('original 53-task source identity changed')
    if config.get('extension_waves') != expected_waves():
        raise ValueError('exact nine U then V then UV waves required')
    epochs = sum(t['arithmetic']['max_epochs'] for t in config['tasks'])
    updates = sum(t['arithmetic']['max_optimizer_steps'] for t in config['tasks'])
    urban = sum(t['arithmetic']['max_optimizer_steps'] for t in config['tasks'] if t['dataset']=='UrbanEV')
    if (epochs, updates, urban, updates-urban) != (540, 1867620, 1782120, 85500):
        raise ValueError('fixed extension budgets changed')
    if (config.get('max_run_epochs'), config.get('max_optimizer_steps')) != (epochs, updates):
        raise ValueError('extension aggregate budget mismatch')
    if any(config.get(k) != 'cfg_adapted_v1' for k in ('B_ref', 'B_cfg', 'backbone_variant')):
        raise ValueError('adapted baseline identity required')
    return config


def waves(config, ids):
    validate_extension(config)
    ordered = [t['id'] for t in config['tasks']]
    if not ids or len(set(ids)) != len(ids) or ids != [i for i in ordered if i in ids]:
        raise ValueError('extension selection must be unique and retain frozen order')
    # Approved recovery subsets retain original wave boundaries; fresh launches
    # are separately required to contain the complete 27-task list.
    return [[i for i in wave if i in ids] for wave in expected_waves() if any(i in ids for i in wave)]


def validate_launch_scope(config, ids, authorization):
    if config.get('task_scope') != SCOPE:
        return
    waves(config, ids)
    if authorization.get('resume_review_approved') is True:
        if authorization.get('resume_task_ids') != ids or not authorization.get('previous_runtime_ref'):
            raise ValueError('explicit extension recovery identity required')
    elif ids != [t['id'] for t in config['tasks']]:
        raise ValueError('fresh extension launch requires all 27 tasks')
    if authorization.get('task_scope') != SCOPE:
        raise ValueError('launch scope cannot be borrowed from original batch')


def derive():
    from .training_config import ROOT, validate
    from .worker import source_binding
    ref = reference(); path = ROOT/ref['parent_config_ref']['path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != ref['parent_config_ref']['sha256']:
        raise ValueError('original config bytes changed')
    config = copy.deepcopy(json.loads(path.read_text()))
    config.update(task_scope=SCOPE, extension_reference_sha256=REFERENCE_SHA,
                  parent_config_ref=ref['parent_config_ref'], tasks=expected_tasks(),
                  existing_B=ref['existing_B'], F4_reuse={}, extension_waves=expected_waves(),
                  max_run_epochs=540, max_optimizer_steps=1867620,
                  source_binding=source_binding(), preparation_head='5ff5ea17732891556270789aeb504e2ab356ecfb')
    return validate(config)


def aggregate_extension(config, rows):
    """Saved-metric presentation only; goal1/3 scopes are deliberately absent."""
    tasks = config['tasks']; metrics = ('mse', 'mae')
    def cell(ds, h):
        selected = [t for t in tasks if t['dataset']==ds and t['h']==h]
        result = {'B': config['existing_B'][selected[0]['baseline_pair_cell_id']]}
        for arm in ARMS:
            task = next(t for t in selected if t['arm']==arm)
            result[arm] = rows.get(task['id'])
        return result
    urban = {str(h): cell('UrbanEV', h) for h in (3, 6, 9, 12)}
    urban_mean = {arm: ({m: math.fsum(v[arm][m] for v in urban.values())/4 for m in metrics}
                       if all(v[arm] is not None for v in urban.values()) else None)
                  for arm in ('B', *ARMS)}
    epf = {ds: cell(ds, 24) for ds in MARKETS}
    coverage = all(v['UV'] is not None for v in epf.values())
    goal2 = dict(coverage_complete=coverage, denominator=10, required_winning_cells=7,
                 numerical_status='Not evaluated', numerical_conditions_met=None,
                 result_review='Pending', metrics={})
    if coverage:
        for m in metrics:
            winners = [d for d, v in epf.items() if v['UV'][m] < v['B'][m]]
            goal2['metrics'][m] = dict(winners=winners, winner_count=len(winners),
                                       denominator=5, win_rate=len(winners)/5, separate_threshold=None)
        wins = sum(x['winner_count'] for x in goal2['metrics'].values())
        goal2.update(winning_cells=wins, win_rate=wins/10, numerical_conditions_met=wins>=7,
                     numerical_status='Met' if wins>=7 else 'Not met')
    absent = dict(coverage_complete=False, numerical_status='Not evaluated',
                  numerical_conditions_met=None, result_review='Pending', reason='outside extension scope')
    return dict(task_scope=SCOPE, required=27, completed=len(rows),
                coverage_complete=len(rows)==27,
                engineering_completion='Complete' if len(rows)==27 else 'Partial',
                scientific_gate='Pending result review' if len(rows)==27 else 'Not evaluated',
                result_review='Pending', numerical_status='Not evaluated',
                goals=dict(goal1=copy.deepcopy(absent), goal2=goal2, goal3=copy.deepcopy(absent)),
                fold6=dict(per_horizon=urban, equal_four_H_mean=urban_mean, replaces_goal1=False),
                EPF=epf, gate_policy=config['gate_policy'], F4_reused_without_new_run=False)
