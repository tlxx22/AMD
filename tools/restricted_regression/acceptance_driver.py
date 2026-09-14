"""Sealed THLS stages: new methods, inherited CPU suite, then real-prefix probe."""
import ast
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import threading
import tempfile
import time
import traceback

import current_policy
from run_restricted import VERSION, TOOL_ROOT, read_json, static_inventory, ensure_unique_exact, verify_bundle
from resource_budget import CLOCK, SharedBudget, initialize


ACCEPTANCE_REVISION = 'thls-tmp-fixture-full-acceptance-approved-v1'
FIXTURE_LAYOUT_VERSION = 'execution-scoped-tmp-fixtures-v1'
REPAIR_ID = 'test_ddi_gelu_backward_layout.DDIGELUBackwardLayoutTests.test_cpu_cotangent_contiguous_production_boundary_and_analytic_reference'
CLOSED_ID = 'test_tsAMD_enhanced.AMDEnhancedTHLSTests.test_disabled_frozen_equivalence_and_isolated_construction_rng'
CARRIED = dict(forward=621, backward=153, adam=33)
NEW_LIMITS = dict(forward=824, backward=193, adam=49)
FINAL_RESERVATION = dict(forward=203, backward=40, adam=16)
LIFECYCLE_ID = 'test_runner.THLSRunnerContractTests.test_four_horizon_an_actual_lifecycles_checksums_summary_duplicates'
BUDGET_APPROVAL = Path('/public/home/yueweiting/大论文/amd-execution-evidence/m4/m4620-full-acceptance-k_6hmz09/authorization.json')
BUDGET_APPROVAL_SHA = '95721cfb14923d5a4a61628c83788e47ce17d9bd7e72405f420bc5c4c273196f'
EXPECTED_SOURCE = '16343e9298a9c4a1eb477c061f7b58552ab04b0706f751b8b83e42749b703a7f'


def validate_execution_authorization(repo):
    if hashlib.sha256(BUDGET_APPROVAL.read_bytes()).hexdigest() != BUDGET_APPROVAL_SHA:
        raise RuntimeError('explicit budget approval bytes mismatch')
    approval = read_json(BUDGET_APPROVAL)
    if (approval.get('authorized') is not True or approval.get('carried') != CARRIED
            or approval.get('limits') != NEW_LIMITS
            or approval.get('production_fingerprint') != EXPECTED_SOURCE):
        raise RuntimeError('explicit budget/source approval mismatch')
    paths = sorted([repo/'main.py', *(repo/'models').rglob('*.py'),
                    *(repo/'utils').glob('*.py')], key=lambda p:p.relative_to(repo).as_posix())
    digest = hashlib.sha256()
    for path in paths:
        name = path.relative_to(repo).as_posix().encode(); data = path.read_bytes()
        digest.update(len(name).to_bytes(4,'big')); digest.update(name)
        digest.update(len(data).to_bytes(8,'big')); digest.update(data)
    if len(paths) != 24 or digest.hexdigest() != EXPECTED_SOURCE:
        raise RuntimeError('approved repaired source fingerprint mismatch')
    source = approval['carried_source']
    raw = Path(source['path']).read_bytes()
    if (hashlib.sha256(raw).hexdigest() != source['sha256']
            or json.loads(raw)['budgets']['cumulative_new'] != CARRIED):
        raise RuntimeError('latest carried budget evidence mismatch')
    prior = approval['prior_budget_approval']
    raw = Path(prior['path']).read_bytes()
    if (hashlib.sha256(raw).hexdigest() != prior['sha256']
            or json.loads(raw)['limits'] != approval['prior_limits']):
        raise RuntimeError('existing approved limits mismatch')
    return approval


def validate_workload_reservation(carried, limits, reservation):
    required = {key: carried[key] + reservation[key] for key in reservation}
    deficits = {key: required[key] - limits[key] for key in required if required[key] > limits[key]}
    if deficits:
        raise RuntimeError('full rerun cumulative budget insufficient: ' + json.dumps(
            dict(carried=carried, reservation=reservation, required=required,
                 limits=limits, deficits=deficits), sort_keys=True))
    return required


def all_ids(inventory):
    repair = inventory.get('repair', [])
    if repair != [REPAIR_ID]:
        raise ValueError('current repair inventory must be exactly the one approved method')
    if len(inventory['new']) != 18 or len(inventory['inherited']) != 314:
        raise ValueError('original test sets cannot be reduced')
    return repair + inventory['new'] + inventory['inherited']


def group_counts(result, inventory):
    keys = ['passed','skipped','failure','error','blocked','unexecuted']
    terminals = result['terminals']
    return {group: {key:sum(terminals.get(test_id, 'unexecuted')==key for test_id in ids) for key in keys}
            for group,ids in [('repair',inventory['repair']),('thls_new',inventory['new']),('inherited',inventory['inherited'])]}


def validate_inputs(repo):
    bundle = verify_bundle()
    inventory = read_json(TOOL_ROOT / 'expected_test_ids.json')
    ensure_unique_exact(static_inventory(repo), all_ids(inventory))
    policy = read_json(TOOL_ROOT / 'restrictions.json')
    current_policy.validate(policy, inventory)
    return inventory, policy, bundle


def file_seal(repo):
    paths = [repo/'main.py', repo/'summarize_results.py', repo/'AGENTS.md',
             *(repo/'models').rglob('*.py'), *(repo/'utils').glob('*.py'),
             *(repo/'tests').glob('*.py'), *TOOL_ROOT.iterdir()]
    return {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def verify_seal(seal):
    for name, digest in seal.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('sealed source/test/tool changed: ' + name)


def write_new(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2); handle.write('\n')


def preflight(repo):
    inventory, policy, bundle = validate_inputs(Path(repo))
    return dict(status='current exact policy and static inventory passed; no load performed',
                tool_version=VERSION, policy_id=current_policy.POLICY_ID,
                historical_recovery_status=policy['recovery_status'],
                historical_real_asset=16, current_approved_real_asset=4, old_cpu_only=1,
                new_ids=len(inventory['new']), repair_ids=len(inventory['repair']), inherited_ids=len(inventory['inherited']),
                 acceptance_revision=ACCEPTANCE_REVISION, total_ids=len(all_ids(inventory)),
                bundle_sha256=bundle, business_imports=0, model_constructions=0)


def create_fixture_layout(evidence_root):
    """One fresh /tmp root per execution; evidence remains at its durable root."""
    evidence_root = Path(evidence_root).resolve()
    if not evidence_root.is_dir() or Path('/tmp') in evidence_root.parents:
        raise ValueError('persistent evidence root must already exist outside /tmp')
    temporary = Path(tempfile.mkdtemp(prefix='amd-thls-restricted-', dir='/tmp')).resolve()
    stages = {}
    for stage in ('new_cuda', 'inherited_cpu', 'real_prefix'):
        path = temporary / stage
        path.mkdir()
        stages[stage] = str(path)
    layout = dict(version=FIXTURE_LAYOUT_VERSION, fixture_execution_root=str(temporary),
                  stage_roots=stages, evidence_root=str(evidence_root),
                  lifecycle='normal test cleanup; no automatic deletion on failure')
    write_new(evidence_root / 'fixture-layout.json', layout)
    return layout


def output_tree_bytes(root, observations=None, *, transient=False):
    """Count current output; only vanished descendants of fixtures are benign.

    This is a sampled size, not an atomic filesystem snapshot. Never follow
    directory symlinks, or hide missing evidence roots and permission failures.
    """
    root = Path(root).absolute()
    fixtures = root if transient else root / 'fixtures'
    if not stat.S_ISDIR(root.stat(follow_symlinks=False).st_mode):
        raise NotADirectoryError(str(root))
    observations = observations if observations is not None else {}

    def vanished_fixture(error):
        if isinstance(error, FileNotFoundError) and error.filename is not None:
            path = Path(error.filename).absolute()
            if path != fixtures and fixtures in path.parents:
                observations['vanished_fixture_entries'] = observations.get('vanished_fixture_entries', 0) + 1
                return
        raise error

    size = 0
    for directory, _, names in os.walk(root, onerror=vanished_fixture, followlinks=False):
        for name in names:
            path = Path(directory) / name
            try:
                size += path.stat(follow_symlinks=False).st_size
            except FileNotFoundError as error:
                vanished_fixture(error)
    observations['successful_samples'] = observations.get('successful_samples', 0) + 1
    observations['output_peak_bytes'] = max(observations.get('output_peak_bytes', 0), size)
    return size


def worker(config):
    import restricted_io_guard as guard
    state = guard.require_installed()
    current_policy.require_scope(state, config['stage'])
    verify_seal(read_json(config['seal_file']))
    validate_execution_authorization(Path(config['repo']))
    if config.get('budget_authorization_sha256') != BUDGET_APPROVAL_SHA:
        raise RuntimeError('child budget authorization binding mismatch')
    if config['stage'] == 'new_cuda' and any(config['limits'][k] != v for k,v in NEW_LIMITS.items()):
        raise RuntimeError('child cumulative limits mismatch')
    from resource_budget import INSTANCE, attach_analysis_apis
    import torch
    report_path = Path(config['report_file'])
    if config['stage'] == 'real_prefix':
        from real_prefix_probe import run_probe
        return run_probe(config)
    import unittest
    from run_restricted import flatten, execute_cases
    inventory, policy, _ = validate_inputs(Path(config['repo']))
    if config['stage'] == 'new_cuda' and not torch.cuda.is_available():
        write_new(report_path, dict(status='Pending: required CUDA unavailable', tests_started=0))
        return 3
    print(json.dumps(dict(torch_version=torch.__version__, cuda_available=torch.cuda.is_available(),
          device=torch.cuda.get_device_name(0) if config['stage']=='new_cuda' else 'CPU-only'),ensure_ascii=False),flush=True)
    suite = unittest.defaultTestLoader.discover(str(Path(config['repo'])/'tests'), pattern='test_*.py')
    cases = list(flatten(suite)); ids = [t.id() for t in cases]
    ensure_unique_exact(ids, all_ids(inventory))
    chosen = (inventory['repair'] + [LIFECYCLE_ID] + [n for n in inventory['new'] if n!=LIFECYCLE_ID]
              if config['stage']=='new_cuda' else inventory['inherited'])
    by_id = {t.id():t for t in cases}
    attach_analysis_apis()
    selected = [by_id[x] for x in chosen]
    restrictions = current_policy.restrictions_for(policy, inventory, config['stage'])
    # This audit is independent of the workload counters. Read all children too.
    seen_denials = 0
    def new_denials():
        nonlocal seen_denials
        events = [json.loads(line) for line in Path(config['audit_log']).read_text().splitlines()]
        refused = [e for e in events if e['event'] == 'denied']
        fresh = refused[seen_denials:]
        seen_denials = len(refused)
        return fresh
    stop_monitor = threading.Event()
    monitor_error = []
    output_observations = {}
    fixture_observations = {}
    def monitor():
        while not stop_monitor.wait(.5):
            try:
                INSTANCE.sample(torch)
                size = output_tree_bytes(config['session_root'], output_observations)
                size += output_tree_bytes(config['fixture_root'], fixture_observations, transient=True)
                if size > config['limits']['output']:
                    raise RuntimeError('execution output limit exceeded')
            except BaseException as exc:
                # A reported business failure is already stopping the suite.
                if 'business test' not in str(exc):
                    monitor_error.append(str(exc)); INSTANCE.stop(exc)
                    os.kill(os.getpid(), signal.SIGALRM)
                return
    watchdog = threading.Thread(target=monitor,daemon=True); watchdog.start()
    try:
        with Path(config['journal_file']).open('x') as journal:
            result = execute_cases(selected, chosen, restrictions, journal,
                                   per_method_seconds=config['limits'].get('method_seconds'),
                                   failfast=True, before_test=INSTANCE.current,
                                   on_failure=lambda reason:INSTANCE.stop('business test: '+reason),
                                   denial_check=new_denials)
    finally:
        stop_monitor.set(); watchdog.join(timeout=2)
    audit = [json.loads(line) for line in Path(config['audit_log']).read_text().splitlines()]
    denied = [event for event in audit if event['event']=='denied']
    result.update(stage=config['stage'], policy_id=current_policy.POLICY_ID,
                  skip_basis={r['id']:r['basis'] for r in policy['current_policy']['entries'] if r['id'] in restrictions},
                  budget=read_json(config['budget_file']), unexpected_forbidden_access=len(denied),
                  monitor_errors=monitor_error, required_cuda_executed=config['stage']=='new_cuda' and result['success'],
                  groups=group_counts(result,inventory), acceptance_revision=ACCEPTANCE_REVISION)
    result['success'] = result['success'] and not denied and not monitor_error and not result['budget']['stopped']
    result['output_monitor'] = dict(evidence=output_observations, fixtures=fixture_observations,
                                    limit_applies_to='sum of evidence and current stage fixture bytes')
    write_new(report_path,result)
    print(json.dumps({k:result[k] for k in ['stage','counts','fixture_error_events','unexpected_forbidden_access','success']},ensure_ascii=False),flush=True)
    return 0 if result['success'] else 1


def run(repo, output):
    repo=Path(repo).resolve();output=Path(output).resolve()
    inventory,policy,bundle=validate_inputs(repo)
    approval=validate_execution_authorization(repo)
    validate_workload_reservation(CARRIED, NEW_LIMITS, FINAL_RESERVATION)
    output.mkdir(exist_ok=False)
    seal=file_seal(repo)
    reference_root=output/'unrepaired-frozen-reference';reference_root.mkdir()
    references={}
    for name in ('models/common.py','models/tsAMD.py','models/tsmoe.py'):
        content=subprocess.check_output(['git','show','amd_reproduced_baseline_v1:'+name],cwd=repo)
        path=reference_root/Path(name).name
        path.write_bytes(content);references[name]=str(path)
        seal[str(path)]=hashlib.sha256(content).hexdigest()
    write_new(output/'sealed-inputs.json',seal)
    prior=Path(approval['carried_source']['path'])
    reference_report=repo.parent/'amd-execution-evidence/m4/m4613-revin-affine-z0nwh7by/acceptance-conclusion.json'
    if hashlib.sha256(reference_report.read_bytes()).hexdigest()!='3d959ef48f8b6d741c6682aa98dca1f4de26f06b65b66d22e986296b3aab31e9':
        raise RuntimeError('accepted reference report checksum mismatch')
    proof=read_json(reference_report)['bias_reference']
    if proof['status']!='complete_CUDA_bias_reference_conditions_Passed':
        raise RuntimeError('independent RevIN reference conditions not satisfied')
    if hashlib.sha256((reference_report.parent/proof['path']).read_bytes()).hexdigest()!=proof['sha256']:
        raise RuntimeError('accepted RevIN reference checksum mismatch')
    # All stages use the same sealed bytes; no auto retry or version mixing.
    layout=create_fixture_layout(output)
    stages=[]
    try:
        for stage in ['new_cuda','inherited_cpu','real_prefix']:
            verify_seal(seal)
            session=output/stage;session.mkdir()
            fixture=Path(layout['stage_roots'][stage])
            limits=dict(rss=8*1024**3,reserved=4*1024**3,output=1024**3,
                        seconds=1800 if stage=='new_cuda' else
                                (600 if stage=='real_prefix' else 1800),method_seconds=180)
            limits.update(NEW_LIMITS if stage=='new_cuda' else
                          dict(forward=36,backward=4,adam=0) if stage=='real_prefix' else
                          dict(forward=None,backward=None,adam=None))
            budget=initialize(session/'budget.json',stage,limits)
            if stage=='new_cuda':
                with budget.state() as state:
                    state['counts']=dict(CARRIED);state['carried_counts']=dict(CARRIED)
                    state['carried_source']=dict(path=str(prior),sha256=hashlib.sha256(prior.read_bytes()).hexdigest())
                    state['static_this_sequence_reservation']=dict(FINAL_RESERVATION)
                    state['analysis_seconds_already_used']=0
                    state['explicit_budget_authorization_sha256']=BUDGET_APPROVAL_SHA
            config=dict(version=VERSION,repo=str(repo),tool_root=str(TOOL_ROOT),session_root=str(session),
                        audit_log=str(session/'audit.jsonl'),forbidden_roots=[str(repo/'data'),str(repo/'artifacts')],
                        stage=stage,access_policy='real_prefix_probe' if stage=='real_prefix' else 'synthetic_regression',
                        restriction_policy_id=current_policy.POLICY_ID,approval_sha256=current_policy.APPROVAL_SHA,
                        budget_file=str(session/'budget.json'),limits=limits,seal_file=str(output/'sealed-inputs.json'),
                        report_file=str(session/'report.json'),journal_file=str(session/'journal.jsonl'),
                        owner_pid=os.getpid(),business_bootstrap=True,
                        fixture_root=str(fixture),fixture_execution_root=layout['fixture_execution_root'],
                        fixture_layout_version=FIXTURE_LAYOUT_VERSION,
                         frozen_forward_reference=references,acceptance_revision=ACCEPTANCE_REVISION,
                        budget_authorization_sha256=BUDGET_APPROVAL_SHA,
                        production_fingerprint=EXPECTED_SOURCE)
            write_new(session/'guard-config.json',config)
            command=[sys.executable,'-B',str(TOOL_ROOT/'run_restricted.py'),'--worker-config',str(session/'guard-config.json')]
            write_new(session/'command.json',command)
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GIT_OPTIONAL_LOCKS='0',
                     PYTHONPATH=os.pathsep.join([str(TOOL_ROOT),str(repo/'tests'),str(repo)]),
                     AMD_RR_CONFIG=str(session/'guard-config.json'),
                     AMD_RR_CONFIG_SHA256=hashlib.sha256((session/'guard-config.json').read_bytes()).hexdigest(),
                     TMPDIR=str(fixture),CUDA_VISIBLE_DEVICES='' if stage=='inherited_cpu' else '0')
            start=CLOCK()
            with (session/'execution.log').open('x') as log:
                child=subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write_new(session/'process.json',dict(pid=child.pid,command=command,started_monotonic=start))
                try:
                    code=child.wait(timeout=limits['seconds'])
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    code=124
            row=dict(stage=stage,exit_code=code,elapsed_seconds=CLOCK()-start,
                     report_exists=(session/'report.json').exists(),budget=read_json(session/'budget.json'))
            stages.append(row);write_new(session/'exit.json',row)
            if code:
                break
        verify_seal(seal)
    except BaseException:
        write_new(output/'controller-failure.json',dict(traceback=traceback.format_exc()))
        raise
    finally:
        summary=dict(tool_version=VERSION,policy_id=current_policy.POLICY_ID,stages=stages,
                     full_inventory=dict(repair=inventory['repair'],new=inventory['new'],inherited=inventory['inherited']),
                      acceptance_revision=ACCEPTANCE_REVISION,carried_new_counts=CARRIED,
                     success=len(stages)==3 and all(row['exit_code']==0 for row in stages))
        write_new(output/'acceptance-result.json',summary)
    print(json.dumps({'success':summary['success'],'stages':[{'stage':s['stage'],'exit_code':s['exit_code'],'counts':s['budget']['counts']} for s in stages]},indent=2),flush=True)
    return 0 if summary['success'] else 1
