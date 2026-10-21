"""Audited saved-result aggregation with exact, independent goal coverage."""
from pathlib import Path
import json
import math
from .artifacts import audit
from .gate_policy import validate_policy, validate_config_policy
from .training_config import validate
from utils.ch3_contract import digest

DOMAINS = ('PJM','NP','BE','FR','DE')
METRICS = ('mse', 'mae')

def checked_metrics(value):
    for name in METRICS:
        v = value[name]
        if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
            raise ValueError('nonfinite or invalid saved metric')
    return value

def aggregate(config, records, gate_policy=None):
    """Pure saved-metadata calculation, also used by the audited disk reader."""
    validate(config)
    configured = validate_config_policy(config)
    if gate_policy is not None:
        validate_policy(gate_policy)
        if gate_policy != configured:
            raise ValueError('gate policy differs from recorded startup contracts')
    tasks = {t['id']:t for t in config['tasks']}
    rows = {}
    for task, metrics in records:
        if task['id'] in rows:
            raise ValueError('duplicate result identity')
        if task['id'] not in tasks or task != tasks[task['id']] or metrics['task_id'] != task['id']:
            raise ValueError('foreign or mismatched exact summary task identity')
        rows[task['id']] = checked_metrics(metrics)
    if config['purpose'] == 'development':
        for t in tasks.values():
            expected = '|'.join((t['task'],'TimeMixer',t['dataset'],str(t['fold']),str(t['h']),str(t['seed'])))
            if t['baseline_cell_id'] != expected or (t['goal'] != 3 and t['baseline_pair_cell_id'] != expected):
                raise ValueError('wrong B pairing identity')
            b = config['existing_B'][expected]
            if b['round'] != 2 or b['profile_sha'] != t['baseline_profile_sha'] or b['metric_space'] != 'train-standardized' or b['review_accepted_by_user'] is not True:
                raise ValueError('wrong B paired profile/source')
            checked_metrics(b)

    def mean(items):
        return {m:math.fsum(v[m] for v in items)/len(items) for m in METRICS} if items else None
    def group(ids):
        present = sorted(set(ids) & rows.keys())
        return dict(count=len(present), required=len(ids), task_ids=present,
                    missing_task_ids=sorted(set(ids)-rows.keys()),
                    mean=mean([rows[k] for k in present]), complete=bool(ids) and set(ids)<=rows.keys())

    urban = {}
    bkeys = sorted({t['baseline_pair_cell_id'] for t in tasks.values() if t['goal']==1})
    bitems = [checked_metrics(config['existing_B'][k]) for k in bkeys if k in config['existing_B']]
    urban['B'] = dict(count=len(bitems), required=8, mean=mean(bitems), complete=len(bkeys)==len(bitems)==8, cell_ids=bkeys)
    for arm in ('U','V','UV'):
        urban[arm] = group([t['id'] for t in tasks.values() if t['goal']==1 and t['arm']==arm])
    # Goal1 is independently judged inside each fold. Goal3 retains the pooled
    # eight-cell input averages below; do not apply one shared Urban weight.
    foldwise = {}
    for fold in configured['urban']['folds']:
        selected = [t for t in tasks.values() if t['goal']==1 and t['fold']==fold]
        keys = sorted({t['baseline_pair_cell_id'] for t in selected})
        base = [config['existing_B'][k] for k in keys if k in config['existing_B']]
        item = dict(B=dict(count=len(base), required=4, mean=mean(base), complete=len(keys)==len(base)==4, cell_ids=keys))
        for arm in ('U','V','UV'):
            item[arm] = group([t['id'] for t in selected if t['arm']==arm])
        foldwise[str(fold)] = item
    inputs = {v:group([t['id'] for t in tasks.values() if t['dataset']=='UrbanEV' and t['arm']=='UV' and t['input_variant']==v]) for v in ('F1','F2','F3','F4')}
    domains = {}
    for ds in DOMAINS:
        selected = [t for t in tasks.values() if t['goal']==2 and t['dataset']==ds]
        entry = group([t['id'] for t in selected])
        base = [config['existing_B'][t['baseline_pair_cell_id']] for t in selected if t['baseline_pair_cell_id'] in config['existing_B']]
        entry.update(candidate=entry['mean'], B=mean(base))
        entry['complete'] &= len(base)==len(selected)
        domains[ds] = entry

    goals = {}
    coverages = {1:all(v['complete'] and v['count']==4 for f in foldwise.values() for v in f.values()),
                 3:all(v['complete'] and v['count']==8 for v in inputs.values()),
                 2:all(v['complete'] and v['required']==1 for ds,v in domains.items())}
    for goal in (1,2,3):
        state = dict(coverage_complete=coverages[goal], engineering_completion='Complete' if coverages[goal] else 'Partial', numerical_conditions_met=None,
                     numerical_status='Not evaluated', result_review='Pending', metrics={})
        if coverages[goal]:
            for metric in METRICS:
                if goal in (1,3):
                    if goal == 1:
                        folds = {}
                        for fold, entries in foldwise.items():
                            vals = {k:v['mean'][metric] for k,v in entries.items()}
                            conditions = {'UV<U<B':vals['UV']<vals['U']<vals['B'], 'UV<V<B':vals['UV']<vals['V']<vals['B']}
                            folds[fold] = dict(means=vals, conditions=conditions, conditions_met=all(conditions.values()))
                        state['metrics'][metric] = dict(folds=folds, conditions_met=all(f['conditions_met'] for f in folds.values()))
                    else:
                        vals = {k:v['mean'][metric] for k,v in inputs.items()}
                        conditions = {'F4<F2<F1':vals['F4']<vals['F2']<vals['F1'], 'F4<F3<F1':vals['F4']<vals['F3']<vals['F1']}
                        state['metrics'][metric] = dict(means=vals, conditions=conditions, conditions_met=all(conditions.values()))
                else:
                    winners = [ds for ds,v in domains.items() if v['candidate'][metric]<v['B'][metric]]
                    state['metrics'][metric] = dict(winner_count=len(winners), winners=winners, denominator=5, win_rate=len(winners)/5, separate_threshold=None)
            if goal == 2:
                wins = sum(v['winner_count'] for v in state['metrics'].values())
                state.update(winning_cells=wins,required_winning_cells=7,denominator=10,win_rate=wins/10)
                state['numerical_conditions_met'] = wins >= 7
            else:
                state['numerical_conditions_met'] = all(v['conditions_met'] for v in state['metrics'].values())
            state['numerical_status'] = 'Met' if state['numerical_conditions_met'] else 'Not met'
        goals['goal'+str(goal)] = state
    complete = set(rows)==set(tasks) and all(coverages.values())
    numerical = all(v['numerical_conditions_met'] for v in goals.values()) if complete else None
    return dict(required=len(tasks),completed=len(rows),coverage_complete=complete,
                engineering_completion='Complete' if set(rows)==set(tasks) else 'Partial',
                gate_policy=configured,goals=goals,numerical_conditions_met=numerical,
                numerical_status=('Met' if numerical else 'Not met') if complete else 'Not evaluated',
                scientific_gate='Pending result review' if complete else 'Not evaluated',result_review='Pending',
                Urban_module=urban,Urban_module_foldwise=foldwise,Urban_module_pooled_is_gate=False,Urban_inputs=inputs,nonUrban=domains,
                cross_dataset_raw_MSE_average=False,F4_reused_without_new_run=True)

def summarize(config, root, attempt, gate_policy=None):
    validate_config_policy(config)
    if gate_policy is not None:
        validate_policy(gate_policy)
    records = []
    if attempt is None:
        from .worker import RUN_NAMESPACE
        registry = (Path(root) if config['purpose']=='synthetic' else RUN_NAMESPACE)/'completed'
        paths = []
        if registry.exists():
            for p in registry.iterdir():
                if p.name.endswith('.tmp'):
                    continue
                value = json.loads(p.read_text())
                if value.get('status')=='completed' and value['receipt']['manifest']['identity']['config_sha']==digest(config):
                    paths.append(Path(value['receipt']['path']))
    else:
        folder = Path(root)/attempt/'runs'
        paths = list(folder.iterdir()) if folder.exists() else []
    for path in paths:
        receipt = audit(path)
        identity = receipt['manifest']['identity']
        task = identity['task']
        if identity['config_sha']!=digest(config) or receipt['manifest']['task_id']!=task['id']:
            raise ValueError('summary configuration/task identity')
        validate_policy(identity.get('gate_policy'))
        if identity['gate_policy']!=config['gate_policy']:
            raise ValueError('gate policy differs from recorded startup contracts')
        records.append((task,receipt['metrics']))
    return aggregate(config,records,gate_policy)
