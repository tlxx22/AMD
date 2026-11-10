"""Only retry2: twelve sealed UrbanEV E results plus thirty-five new jobs."""
from pathlib import Path
from .contract import jobs, read, sha, load_ref

SCHEMA = 'm6-ch3-environment-partial-retry2-v1'
OLD_COMMIT = '3fd1edb7bacb1a8f99cb4609e92a13b8d92cd99b'
OLD_CONFIG_SHA = '10c2f749813f93fd458f73bb589ceb5a94497661ddb6e73bb105e416e4dba312'
OLD_RUNTIME_SHA = '341ac4a8ff3cea4413f021a6ffaba515a68d527fc3724ee23020f135f39d35c5'
CASE_SHA = '57219203ae9a6e00e348e2c2af5eee38796f4969e1c19e8374d1b7afabe10164'


def is_retry2(config):
    return config.get('technical_continuation', {}).get('schema') == SCHEMA


def execution_jobs(config):
    return config['execution_jobs'] if is_retry2(config) else config['jobs']


def execution_waves(config):
    return config['execution_waves'] if is_retry2(config) else config['waves']


def validate_partition(config):
    inherited = jobs()[:12]
    if config.get('inherited_completed_jobs') != inherited or config.get('execution_jobs') != jobs()[12:]:
        raise ValueError('retry2 requires exactly the first twelve inherited E jobs and ordered remaining 35')
    if config.get('execution_waves') != config['waves'][12:] or config.get('original_wave_indices') != list(range(13, 42)):
        raise ValueError('retry2 exact 29 execution waves / original indices required')
    expected = dict(inherited_jobs=12, execution_jobs=35, logical_jobs=47,
                    inherited_waves=12, execution_waves=29, logical_waves=41,
                    inherited_performance_calls=6600, inherited_performance_presentations=56100,
                    remaining_performance_calls=11000, remaining_performance_presentations=93500,
                    sensitivity_tasks=8, max_epochs=160, max_updates=617120,
                    case_presentations=1008, pjm_full_validation_passes=3, test=0)
    if config.get('continuation_budget') != expected:
        raise ValueError('retry2 budget/coverage changed')


def checked_receipt(folder, job, runtime_ref, receipt_ref):
    folder = Path(folder).resolve()
    if Path(runtime_ref['path']).resolve() != folder/'runtime.json' or Path(receipt_ref['path']).resolve() != folder/'receipts'/(job['id']+'.json'):
        raise ValueError('receipt/runtime path outside exact producer attempt')
    runtime = load_ref(runtime_ref)
    receipt = load_ref(receipt_ref)
    if receipt['job'] != job or receipt['runtime_sha256'] != runtime_ref['sha256']:
        raise ValueError('receipt job/runtime identity mismatch')
    from .queue import validate_result
    config = load_ref(runtime['config_ref'])
    auth = load_ref(runtime['auth_ref'])
    if runtime['commit'] != auth['commit'] or auth['config_sha256'] != runtime['config_ref']['sha256']:
        raise ValueError('producer AUTH/config/commit mismatch')
    validate_result(config, job, receipt['result'])
    prefix = 'plots' if job['phase'] == 'O' else job['id']
    actual = {str(p.relative_to(folder)) for p in (folder/prefix).rglob('*') if p.is_file()}
    if set(receipt['files']) != actual or not actual:
        raise ValueError('payload set mismatch')
    for name, expected in receipt['files'].items():
        target = (folder/name).resolve()
        if not target.is_relative_to(folder/prefix) or sha(target) != expected:
            raise ValueError('payload path/checksum changed')
    if job['phase'] != 'O' and read(folder/job['id']/'result.json') != receipt['result']:
        raise ValueError('receipt/result bytes disagree')
    return receipt, runtime


def validate_retry2(config):
    validate_partition(config)
    chain = config['technical_continuation']
    if config['attempt'] != 'ch3-supplement-v10-retry2':
        raise ValueError('only named retry2 is authorized by this control schema')
    if chain['previous_config_ref']['sha256'] != OLD_CONFIG_SHA or chain['previous_runtime_ref']['sha256'] != OLD_RUNTIME_SHA or chain['case_selection_ref']['sha256'] != CASE_SHA:
        raise ValueError('retry2 must trace the exact reviewed retry1')
    old = load_ref(chain['previous_config_ref'])
    allowed = {'attempt', 'source_binding', 'technical_continuation', 'environment_ref',
               'inherited_receipts_ref', 'inherited_completed_jobs', 'execution_jobs',
               'execution_waves', 'original_wave_indices', 'continuation_budget'}
    if {k: v for k, v in config.items() if k not in allowed} != {k: v for k, v in old.items() if k not in allowed}:
        raise ValueError('environment continuation changed scientific configuration')
    folder = Path(chain['previous_runtime_ref']['path']).parent
    if old['attempt'] != 'ch3-supplement-v10-retry1' or folder.resolve() != (Path(old['result_root'])/old['attempt']).resolve():
        raise ValueError('wrong previous attempt')
    runtime = load_ref(chain['previous_runtime_ref'])
    if runtime['commit'] != OLD_COMMIT or runtime['config_ref'] != chain['previous_config_ref'] or runtime['auth_ref'] != chain['previous_auth_ref']:
        raise ValueError('previous producer identity changed')
    failure = load_ref(chain['previous_failure_ref'])
    if Path(chain['previous_failure_ref']['path']).resolve() != folder.resolve()/'failure.json' or not failure.get('traceback'):
        raise ValueError('preserved technical failure required')
    if Path(chain['case_selection_ref']['path']).resolve() != folder.resolve()/'case-selection.json':
        raise ValueError('case must belong to old attempt')
    load_ref(chain['case_selection_ref'])
    index = load_ref(config['inherited_receipts_ref'])
    if index['schema'] != 'm6-supplement-inherited-twelve-v1' or index['producer_runtime_ref'] != chain['previous_runtime_ref']:
        raise ValueError('wrong inherited index/runtime')
    if [r['job'] for r in index['entries']] != jobs()[:12]:
        raise ValueError('inherited jobs missing/duplicate/reordered')
    if {p.stem for p in (folder/'receipts').glob('*.json')} != {j['id'] for j in jobs()[:12]}:
        raise ValueError('old successful set changed')
    if {p.name for p in folder.glob('wave-*-complete.json')} != {f'wave-{i}-complete.json' for i in range(1, 13)} or (folder/'complete.json').exists():
        raise ValueError('old wave coverage changed')
    old_environment = load_ref(index['producer_environment_ref'])
    new_environment = load_ref(config['environment_ref'])
    if old_environment['effective_settings'] != new_environment['effective_settings']:
        raise ValueError('preserved/new measurement settings incompatible')
    for i, item in enumerate(index['entries'], 1):
        receipt, _ = checked_receipt(folder, item['job'], chain['previous_runtime_ref'], item['receipt_ref'])
        if item['environment_ref'] != index['producer_environment_ref'] or item['payload'] != receipt['files']:
            raise ValueError('inherited environment/payload identity mismatch')
        if item['result_ref'] != dict(path=str(folder/item['job']['id']/'result.json'), sha256=receipt['files'][item['job']['id']+'/result.json']):
            raise ValueError('inherited result reference mismatch')
        wave = load_ref(item['wave_ref'])
        if Path(item['wave_ref']['path']).resolve() != folder.resolve()/f'wave-{i}-complete.json' or wave != dict(jobs=[item['job']['id']]):
            raise ValueError('inherited wave reference mismatch')
        r = receipt['result']
        if any(r.get(k) != v for k, v in old_environment['measurement_fields'].items()):
            raise ValueError('inherited performance settings changed')
        for boundary in ('before', 'after'):
            b = r['gpu_process_boundary'][boundary]
            if b['classification'] != 'self' or b['target_pids'] != [b['identity']['driver_pid']]:
                raise ValueError('inherited GPU identity boundary rejected')
    from moderntcn_input_ablation_v10.launcher import proc_start
    owner = load_ref(chain['previous_owner_ref'])
    child_rows = load_ref(chain['failed_children_ref'])
    if Path(chain['previous_owner_ref']['path']).resolve() != folder.resolve()/'owner.json' or Path(chain['failed_children_ref']['path']).resolve() != folder.resolve()/'children-wave-13.json':
        raise ValueError('previous process provenance path mismatch')
    if owner['runtime_sha256'] != chain['previous_runtime_ref']['sha256']:
        raise ValueError('previous owner/runtime mismatch')
    for record in [owner, *child_rows]:
        try:
            if proc_start(record['pid']) == record['start']:
                raise ValueError('previous owned process still active')
        except (FileNotFoundError, ProcessLookupError):
            pass
    return index


def resolve(config, folder, job):
    """No arbitrary path input: source selected only by the exact logical job."""
    if job not in config['jobs']:
        raise ValueError('unknown logical result')
    folder = Path(folder)
    inherited = is_retry2(config) and job in config['inherited_completed_jobs']
    if inherited:
        index = load_ref(config['inherited_receipts_ref'])
        found = [r for r in index['entries'] if r['job'] == job]
        if len(found) != 1:
            raise ValueError('missing/duplicate inherited source')
        item = found[0]
        runtime_ref = index['producer_runtime_ref']
        producer = Path(runtime_ref['path']).parent
        receipt_ref, environment_ref = item['receipt_ref'], item['environment_ref']
    else:
        if job not in execution_jobs(config):
            raise ValueError('job not admitted to new execution')
        producer = folder
        runtime_ref = dict(path=str(folder/'runtime.json'), sha256=sha(folder/'runtime.json'))
        receipt_ref = dict(path=str(folder/'receipts'/(job['id']+'.json')), sha256=sha(folder/'receipts'/(job['id']+'.json')))
        environment_ref = config.get('environment_ref')
    receipt, runtime = checked_receipt(producer, job, runtime_ref, receipt_ref)
    if not inherited and runtime['config_ref']['sha256'] != sha(runtime['config_ref']['path']):
        raise ValueError('new runtime/config changed')
    if not inherited and load_ref(runtime['config_ref']) != config:
        raise ValueError('foreign new producer configuration')
    if not inherited and is_retry2(config) and (runtime.get('environment_ref') != config['environment_ref'] or producer.resolve() != (Path(config['result_root'])/config['attempt']).resolve()):
        raise ValueError('foreign new producer environment/attempt')
    result_ref = None if job['phase'] == 'O' else dict(path=str(producer/job['id']/'result.json'), sha256=receipt['files'][job['id']+'/result.json'])
    lineage = dict(job=job, inherited=inherited, producer_attempt=producer.name,
                   producer_commit=runtime['commit'], runtime_ref=runtime_ref,
                   auth_ref=runtime['auth_ref'], config_ref=runtime['config_ref'],
                   environment_ref=environment_ref, receipt_ref=receipt_ref, result_ref=result_ref)
    return receipt['result'], lineage


def complete_summary(config, folder):
    expected = execution_jobs(config)
    if {p.stem for p in (Path(folder)/'receipts').glob('*.json')} != {j['id'] for j in expected}:
        raise ValueError('new receipt coverage mismatch')
    lineages = [resolve(config, folder, j)[1] for j in config['jobs']]
    waves = execution_waves(config)
    if {p.name for p in Path(folder).glob('wave-*-complete.json')} != {f'wave-{i}-complete.json' for i in range(1, len(waves)+1)}:
        raise ValueError('new wave coverage mismatch')
    for i, wave in enumerate(waves, 1):
        actual = read(Path(folder)/f'wave-{i}-complete.json')
        if actual['jobs'] != wave or actual.get('original_wave_index') != (config['original_wave_indices'][i-1] if is_retry2(config) else i):
            raise ValueError('new wave identity mismatch')
    if (Path(folder)/'STOP').exists() or (Path(folder)/'failure.json').exists():
        raise ValueError('failed/stopped attempt cannot complete')
    return dict(technical_complete=True, logical_jobs=47, logical_waves=41,
                inherited_completed=47-len(expected), execution_completed=len(expected),
                execution_waves_completed=len(waves), completed_by_phase=dict(E=32, C=3, P=3, S=8, O=1),
                fresh_sensitivity=8, test=0, sources=lineages,
                scientific_claim='no replacement of main table; old failure remains sealed')
