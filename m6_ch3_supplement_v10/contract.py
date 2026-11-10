"""Exact finite analysis scope and file bindings, independent of production AUTH."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCOPE = 'm6_ch3_supplement_v10_v1'
PAIRS = ((3, 4), (7, 4), (5, 2), (5, 6))
MODELS = ('UV', 'AMD', 'DLinear', 'ModernTCN', 'PatchTST', 'TimeMixer', 'TimeXer', 'iTransformer')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def load_ref(ref):
    if sha(ref['path']) != ref['sha256']:
        raise ValueError('bound file changed: ' + ref['path'])
    return read(ref['path'])


def jobs():
    out = []
    for dataset in ('UrbanEV', 'PJM'):
        for model in MODELS:
            for batch in (1, 16):
                out.append(dict(id=f'E-{dataset}-{model}-b{batch}', phase='E', dataset=dataset, model=model, batch=batch))
    for phase, dataset in [('C', 'UrbanEV'), ('P', 'PJM')]:
        for model in ('ModernTCN', 'TimeMixer', 'UV'):
            out.append(dict(id=f'{phase}-{model}', phase=phase, dataset=dataset, model=model))
    for dataset in ('UrbanEV', 'PJM'):
        for G, R in PAIRS:
            out.append(dict(id=f'S-{dataset}-G{G}-R{R}', phase='S', dataset=dataset, model='UV', G=G, R=R))
    out.append(dict(id='O', phase='O'))
    return out


def validate(config):
    if config.get('scope') != SCOPE or config.get('jobs') != jobs():
        raise ValueError('exact analysis scope required')
    expected = [[j['id']] for j in jobs() if j['phase'] in ('E', 'C', 'P')]
    expected += [[j['id'] for j in jobs() if j['phase'] == 'S' and j['dataset'] == d] for d in ('UrbanEV', 'PJM')]
    expected += [['O']]
    if config['waves'] != expected or config['budget'] != dict(sensitivity_tasks=8, max_epochs=160, max_updates=617120, test=0, performance_conditions=32, performance_calls=17600, case_presentations=1008, pjm_full_validation_passes=3):
        raise ValueError('fixed queue/budget required')
    if config['execution_authorized'] is not False:
        raise ValueError('source config remains non-authorizing')
    from .continuation import is_retry2
    if config.get('attempt') == 'ch3-supplement-v10-retry2' or any(k in config for k in ('environment_ref', 'inherited_receipts_ref', 'execution_jobs', 'execution_waves')):
        if not is_retry2(config):
            raise ValueError('partial execution fields require the exact retry2 continuation')
    if config.get('technical_continuation'):
        validate_continuation(config)
    return config


def validate_continuation(config):
    """Dispatch the two explicitly scoped technical continuations."""
    from .continuation import is_retry2, validate_retry2
    if is_retry2(config):
        return validate_retry2(config)
    chain = config['technical_continuation']
    if chain.get('schema') != 'm6-ch3-gpu-pid-repair-retry1' or config['attempt'] != 'ch3-supplement-v10-retry1':
        raise ValueError('exact technical retry identity required')
    original = load_ref(chain['previous_config_ref'])
    if original['attempt'] != 'ch3-supplement-v10-first-r1':
        raise ValueError('unexpected previous attempt')
    allowed = {'source_binding', 'attempt', 'technical_continuation'}
    if {k: v for k, v in config.items() if k not in allowed} != {k: v for k, v in original.items() if k not in allowed}:
        raise ValueError('technical retry changed scientific/config fields')
    runtime = load_ref(chain['previous_runtime_ref'])
    auth = load_ref(chain['previous_auth_ref'])
    failure = load_ref(chain['previous_failure_ref'])
    if runtime['config_ref']['sha256'] != chain['previous_config_ref']['sha256'] or runtime['auth_ref'] != chain['previous_auth_ref']:
        raise ValueError('previous runtime/config/AUTH chain mismatch')
    if auth['config_sha256'] != chain['previous_config_ref']['sha256'] or not failure.get('traceback'):
        raise ValueError('missing bound prior technical failure')
    old_folder = Path(chain['previous_runtime_ref']['path']).parent
    if Path(chain['previous_failure_ref']['path']).parent != old_folder or Path(chain['case_selection_ref']['path']).parent != old_folder:
        raise ValueError('previous artifacts do not belong to one attempt')
    if list((old_folder/'receipts').glob('*.json')) or list(old_folder.glob('wave-*-complete.json')) or (old_folder/'complete.json').exists():
        raise ValueError('retry1 only applies to the zero-success failed batch')
    selection = load_ref(chain['case_selection_ref'])
    if selection['nodes'] != ['105', '408'] or selection['indices'] != [2, 54] or selection['H'] != 6 or selection['target_times'][0] != '2023-01-30 00:00:00' or selection['target_times'][-1] != '2023-02-05 23:00:00':
        raise ValueError('fixed original case selection required')


def verify_case_selection(config, selection):
    chain = config.get('technical_continuation')
    if chain and selection != load_ref(chain['case_selection_ref']):
        raise ValueError('case selection changed from the sealed original')


def validate_permission(config_path, auth):
    # Fail before any network, GPU, model or runtime creation.
    if auth.get('schema') != 'm6-ch3-supplement-launch-v1' or auth.get('scope') != SCOPE or auth.get('review_approved') is not True or not auth.get('commit'):
        raise PermissionError('reviewed supplement authorization required')
    config = validate(read(config_path))
    if auth['config_sha256'] != sha(config_path) or auth['jobs'] != jobs() or auth['budget'] != config['budget']:
        raise PermissionError('authorization scope/config/budget mismatch')
    if auth.get('resume_review_approved') or auth.get('resume_task_ids') or auth.get('previous_runtime_ref'):
        raise PermissionError('fresh batch only; resume needs a separate reviewed continuation')
    if auth['source_binding'] != config['source_binding']:
        raise PermissionError('source binding mismatch')
    if auth.get('gpu_uuid') != config['gpu_uuid'] or auth.get('global_lock') != config['global_lock']:
        raise PermissionError('GPU UUID/shared lock binding mismatch')
    resources = auth.get('resource_plan', {})
    from .continuation import is_retry2, execution_waves
    if resources.get('waves') != execution_waves(config) or resources.get('q_max') != 4 or resources.get('mandatory_probe') is not False:
        raise PermissionError('exact queue/resource acceptance required')
    if is_retry2(config):
        if resources.get('phase_concurrency') != config['phase_concurrency']:
            raise PermissionError('continuation phase concurrency mismatch')
        for key in ('environment_ref', 'inherited_receipts_ref', 'technical_continuation',
                    'execution_jobs', 'execution_waves', 'original_wave_indices',
                    'inherited_completed_jobs', 'continuation_budget'):
            if auth.get(key) != config[key]:
                raise PermissionError('continuation authorization mismatch: '+key)
        if auth.get('logical_waves') != config['waves']:
            raise PermissionError('logical wave authorization mismatch')
        load_ref(config['environment_ref'])
    for p, h in config['source_binding'].items():
        if sha(ROOT/p) != h:
            raise PermissionError('source changed: ' + p)
    load_ref(config['ledger_ref'])
    return config
