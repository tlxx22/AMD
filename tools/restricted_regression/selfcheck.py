"""No-model synthetic checks. Deliberate errors are NOT business test results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

from run_restricted import execute_cases, ensure_unique_exact, require_complete_policy, PolicyBlocked

ROOT = Path(__file__).resolve().parent
VERSION = 'restricted-regression-minimal-v3'


def ledger_selfcheck():
    calls = {}
    modules = []

    def module(name, broken=False):
        obj = types.ModuleType(name)
        modules.append(name)
        sys.modules[name] = obj
        if broken:
            def bad_setup():
                calls[name] = calls.get(name, 0) + 1
                raise RuntimeError('deliberate module fixture error')
            obj.setUpModule = bad_setup
        return obj

    full = module('rr_full_restricted_module', True)
    normal = module('rr_normal_dummy_module')
    bad_module = module('rr_bad_dummy_module', True)
    cleanup_module = module('rr_cleanup_dummy_module')
    def register_module_cleanup():
        def bad_cleanup():
            raise RuntimeError('deliberate module cleanup error')
        unittest.addModuleCleanup(bad_cleanup)
    cleanup_module.setUpModule = register_module_cleanup
    later_module = module('rr_later_dummy_module')

    class FullModule(unittest.TestCase):
        __module__ = full.__name__
        @classmethod
        def setUpClass(cls):
            calls['full_module_class'] = 1
            raise RuntimeError('must not execute')
        def test_one(self):
            raise AssertionError('restricted method executed')
        def test_two(self):
            raise AssertionError('restricted method executed')

    class FullClass(unittest.TestCase):
        __module__ = normal.__name__
        @classmethod
        def setUpClass(cls):
            calls['full_class'] = 1
            raise RuntimeError('must not execute')
        def test_one(self):
            raise AssertionError('restricted method executed')
        def test_two(self):
            raise AssertionError('restricted method executed')

    class Mixed(unittest.TestCase):
        __module__ = normal.__name__
        def setUp(self):
            calls['mixed_setup'] = calls.get('mixed_setup', 0) + 1
        def test_active(self):
            calls['mixed_active'] = 1
        def test_restricted(self):
            raise AssertionError('restricted method executed')

    class ClassError(unittest.TestCase):
        __module__ = normal.__name__
        @classmethod
        def setUpClass(cls):
            raise RuntimeError('deliberate class fixture error')
        def test_blocked(self):
            raise AssertionError('should be blocked')

    class CleanupError(unittest.TestCase):
        __module__ = normal.__name__
        def test_cleanup(self):
            def bad_cleanup():
                raise RuntimeError('deliberate method cleanup error')
            self.addCleanup(bad_cleanup)

    class ClassCleanupError(unittest.TestCase):
        __module__ = normal.__name__
        @classmethod
        def setUpClass(cls):
            def bad_cleanup():
                raise RuntimeError('deliberate class cleanup error')
            cls.addClassCleanup(bad_cleanup)
        def test_class_cleanup(self):
            pass

    class SubtestError(unittest.TestCase):
        __module__ = normal.__name__
        def test_subtests(self):
            with self.subTest(case='failure'):
                self.assertEqual(1, 2)
            with self.subTest(case='error'):
                raise RuntimeError('deliberate subtest error')

    class ModuleError(unittest.TestCase):
        __module__ = bad_module.__name__
        def test_blocked(self):
            raise AssertionError('should be blocked')

    class MethodSetupError(unittest.TestCase):
        __module__ = normal.__name__
        def setUp(self):
            raise RuntimeError('deliberate method setup error')
        def test_setup(self):
            raise AssertionError('must not execute')

    class ModuleCleanupError(unittest.TestCase):
        __module__ = cleanup_module.__name__
        def test_module_cleanup(self):
            pass

    class Later(unittest.TestCase):
        __module__ = later_module.__name__
        def test_active(self):
            calls['later_active'] = 1

    classes = [FullModule, FullClass, Mixed, ClassError, CleanupError, ClassCleanupError,
               SubtestError, MethodSetupError, ModuleError, ModuleCleanupError, Later]
    for cls in classes:
        cls.__qualname__ = cls.__name__
    cases = [t for cls in classes for t in unittest.defaultTestLoader.loadTestsFromTestCase(cls)]
    ids = [t.id() for t in cases]
    restrictions = {t.id(): 'selfcheck exact policy only' for t in cases
                    if type(t) in {FullModule, FullClass} or t.id().endswith('Mixed.test_restricted')}
    try:
        report = execute_cases(cases, ids, restrictions)
        assert calls.get(full.__name__, 0) == calls.get('full_module_class', 0) == calls.get('full_class', 0) == 0
        assert calls['mixed_setup'] == calls['mixed_active'] == calls['later_active'] == 1
        assert report['counts']['skipped'] == 5
        assert report['counts']['blocked'] == 2
        assert report['counts']['unexecuted'] == 0
        assert report['counts']['error'] == 3
        assert report['fixture_error_events'] == 4
        assert report['subtest_failure_events'] == report['subtest_error_events'] == 1
        assert not report['success']
        rejects = []
        for name, actual, expected in [('duplicate', ['x', 'x'], ['x']),
                                        ('missing', ['x'], ['x', 'y']),
                                        ('extra', ['x', 'y'], ['x']),
                                        ('duplicate_expected', ['x'], ['x', 'x'])]:
            try:
                ensure_unique_exact(actual, expected)
            except ValueError:
                rejects.append(name)
        assert len(rejects) == 4

        class AccidentalSkip(unittest.TestCase):
            def test_skip(self):
                raise unittest.SkipTest('unregistered runtime skip')
        skipped = AccidentalSkip('test_skip')
        skip_report = execute_cases([skipped], [skipped.id()], {})
        assert not skip_report['success'] and skip_report['anomalies']
        incomplete = json.loads((ROOT / 'restrictions.json').read_text())
        try:
            require_complete_policy(incomplete)
        except PolicyBlocked:
            restriction_block = True
        else:
            restriction_block = False
        assert restriction_block
        return dict(status='Passed', injected_fixture_report=report, calls=calls,
                    exact_id_rejections=rejects, unexpected_skip_rejected=not skip_report['success'],
                    historical_incomplete_check_retained_separately=True)
    finally:
        for name in modules:
            sys.modules.pop(name, None)


def guarded_child(session):
    import restricted_io_guard as guard
    guard.require_installed()
    session = Path(session)
    denied = []

    def must_deny(name, operation):
        try:
            operation()
        except guard.ForbiddenAccess:
            denied.append(name)
        else:
            raise AssertionError('guard did not reject: ' + name)

    allowed = session / 'allowed.txt'
    assert allowed.read_text() == 'synthetic allowed\n'
    must_deny('absolute_path', lambda: (session / 'forbidden/secret.txt').read_bytes())
    must_deny('relative_path', lambda: Path('forbidden/secret.txt').read_bytes())
    must_deny('symlink', lambda: (session / 'forbidden-link').read_bytes())
    dir_fd = int(os.environ['RR_ALLOWED_DIR_FD'])
    fd = os.open('allowed.txt', os.O_RDONLY, dir_fd=dir_fd)
    assert os.read(fd, 4096) == b'synthetic allowed\n'
    os.close(fd)
    secret_dir_fd = int(os.environ['RR_FORBIDDEN_DIR_FD'])
    must_deny('dir_fd', lambda: os.open('secret.txt', os.O_RDONLY, dir_fd=secret_dir_fd))
    must_deny('inherited_file_fd', lambda: os.read(int(os.environ['RR_FORBIDDEN_FILE_FD']), 4096))
    must_deny('legacy_checkpoint', lambda: (session / 'forbidden/old-weight.pt').read_bytes())

    read_fd, write_fd = os.pipe()
    with os.fdopen(write_fd, 'wb') as writer:
        writer.write(b'pipe payload')
    assert os.read(read_fd, 4096) == b'pipe payload'
    os.close(read_fd)

    bad_env = dict(os.environ)
    bad_env.pop('AMD_RR_CONFIG')
    must_deny('python_missing_guard', lambda: subprocess.run([sys.executable, '-B', '-c', 'pass'], env=bad_env))
    must_deny('python_disable_site', lambda: subprocess.run([sys.executable, '-S', '-c', 'pass']))
    must_deny('direct_content_reader', lambda: subprocess.run(['cat', 'forbidden/secret.txt']))
    must_deny('checksum_forbidden_target', lambda: subprocess.run(['/usr/bin/sha256sum', '-c', 'forbidden.sha256'], check=True))
    child_code = "from pathlib import Path; import restricted_io_guard as g; g.require_installed(); " + \
        "\ntry: Path('forbidden/secret.txt').read_bytes()\nexcept g.ForbiddenAccess: print('child denied with guard')\nelse: raise AssertionError('child bypass')\n"
    child = subprocess.run([sys.executable, '-B', '-c', child_code], capture_output=True, text=True, check=True)
    assert child.stdout.strip() == 'child denied with guard'
    checksum = subprocess.run(['sha256sum', '-c', 'allowed.sha256'], capture_output=True, text=True, check=True)
    assert 'OK' in checksum.stdout
    subprocess.run(['bash', '--version'], check=True, capture_output=True)
    subprocess.run(['/usr/bin/git', 'rev-parse', 'HEAD'], cwd=guard.require_installed()['repo'], check=True, capture_output=True)
    import platform
    platform_value = platform.platform()
    assert isinstance(platform_value, str) and platform_value
    assert (session / 'fresh.pt').read_bytes() == b'fresh synthetic checkpoint bytes'
    assert (session.parent / 'outside-root-synthetic.pt').read_bytes() == b'allowed by path, not extension'
    normal = subprocess.run(['/bin/echo', 'ordinary query'], check=True, capture_output=True, text=True)
    assert normal.stdout.strip() == 'ordinary query'
    caught_reports = []
    for use_child in (False, True):
        seen = len([json.loads(l) for l in (session/'audit.jsonl').read_text().splitlines()
                    if json.loads(l)['event']=='denied'])
        def new_denials():
            nonlocal seen
            all_events = [json.loads(l) for l in (session/'audit.jsonl').read_text().splitlines()]
            refused = [e for e in all_events if e['event']=='denied']
            new = refused[seen:]; seen = len(refused); return new
        called = []
        class Caught(unittest.TestCase):
            def test_1_caught(self):
                if use_child:
                    subprocess.run([sys.executable, '-B', '-c', child_code], check=True, capture_output=True)
                else:
                    try: (session/'forbidden/secret.txt').read_bytes()
                    except guard.ForbiddenAccess: pass
                called.append('method continued after catch')
            def test_2_must_not_run(self):
                called.append('wrong later method')
        cases=list(unittest.defaultTestLoader.loadTestsFromTestCase(Caught))
        report=execute_cases(cases,[t.id() for t in cases],{},failfast=True,denial_check=new_denials)
        assert report['counts']['error']==1 and report['counts']['unexecuted']==1
        assert report['counts']['passed']==0 and not report['success']
        assert called==['method continued after catch']
        caught_reports.append(dict(child=use_child,report=report,called=called))
    subprocess.run(['/bin/bash', '-n'], input=b'echo safe\n', check=True, capture_output=True)
    subprocess.run(['/bin/bash', '-n', 'syntax.sh'], check=True, capture_output=True)
    cleanup = session / 'cleanup'
    cleanup.mkdir()
    (cleanup / 'file').write_text('fresh synthetic')
    shutil.rmtree(cleanup)
    assert not cleanup.exists()
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    result = dict(status='Passed', deliberate_parent_denials=denied, python_child_guard_verified=True,
                  allowed_dir_fd=True, pipe_round_trip=True, syntax_checks=2,
                  checked_synthetic_checksum=True, synthetic_cleanup=True,
                  model_imports=0, real_forbidden_content_reads=0, platform=platform_value,
                  fresh_checkpoint_read=True, no_extension_blacklist=True, caught_reports=caught_reports)
    (session / 'guard-result.json').write_text(json.dumps(result, indent=2) + '\n')


def guard_selfcheck(output, repo):
    session = output / 'guard-cases'
    session.mkdir()
    (session / 'forbidden').mkdir()
    (session / 'forbidden/secret.txt').write_text('synthetic forbidden; not real data\n')
    (session / 'allowed.txt').write_text('synthetic allowed\n')
    (session / 'fresh.pt').write_bytes(b'fresh synthetic checkpoint bytes')
    (session / 'forbidden/old-weight.pt').write_bytes(b'synthetic forbidden checkpoint')
    (session.parent / 'outside-root-synthetic.pt').write_bytes(b'allowed by path, not extension')
    (session / 'syntax.sh').write_text('#!/bin/bash\necho safe\n')
    (session / 'forbidden-link').symlink_to(session / 'forbidden/secret.txt')
    allowed_hash = hashlib.sha256((session / 'allowed.txt').read_bytes()).hexdigest()
    forbidden_hash = hashlib.sha256((session / 'forbidden/secret.txt').read_bytes()).hexdigest()
    (session / 'allowed.sha256').write_text(allowed_hash + '  allowed.txt\n')
    (session / 'forbidden.sha256').write_text(forbidden_hash + '  forbidden/secret.txt\n')
    config = dict(version=VERSION, repo=str(repo), tool_root=str(ROOT), session_root=str(session),
                  audit_log=str(session / 'audit.jsonl'),
                  forbidden_roots=[str(repo / 'data'), str(repo / 'artifacts'), str(session / 'forbidden')])
    config_path = session / 'guard-config.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    env = dict(os.environ, AMD_RR_CONFIG=str(config_path),
               AMD_RR_CONFIG_SHA256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
               PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(session))
    fds = [os.open(session, os.O_RDONLY | os.O_DIRECTORY),
           os.open(session / 'forbidden', os.O_RDONLY | os.O_DIRECTORY),
           os.open(session / 'forbidden/secret.txt', os.O_RDONLY)]
    for name, fd in zip(['RR_ALLOWED_DIR_FD', 'RR_FORBIDDEN_DIR_FD', 'RR_FORBIDDEN_FILE_FD'], fds):
        env[name] = str(fd)
    command = [sys.executable, '-B', str(ROOT / 'selfcheck.py'), '--guard-child', str(session)]
    try:
        child = subprocess.run(command, env=env, cwd=session, pass_fds=fds, capture_output=True, text=True, timeout=30)
    finally:
        for fd in fds:
            os.close(fd)
    (output / 'guard-child.log').write_text(child.stdout + child.stderr)
    if child.returncode:
        raise RuntimeError('guard child failed: ' + str(child.returncode) + '\n' + child.stderr[-4000:])
    result = json.loads((session / 'guard-result.json').read_text())
    events = [json.loads(line) for line in (session / 'audit.jsonl').read_text().splitlines()]
    denials = [e for e in events if e['event'] == 'denied']
    installs = [e for e in events if e['event'] == 'installed']
    assert len(denials) == len(result['deliberate_parent_denials']) + 3
    assert all(e.get('test_id') for e in denials)
    assert len({e['pid'] for e in installs}) == 3
    bad_env = dict(env, AMD_RR_CONFIG_SHA256='0' * 64)
    bad = subprocess.run([sys.executable, '-B', '-c', "raise AssertionError('must not run')"],
                         env=bad_env, cwd=session, capture_output=True, text=True, timeout=10)
    (output / 'bootstrap-reject.log').write_text(bad.stdout + bad.stderr)
    assert bad.returncode == 86
    return dict(result, audit_deliberate_denials=len(denials), installed_processes=len(installs),
                corrupted_bootstrap_exit=bad.returncode, unexpected_forbidden_access=0)



def current_policy_budget_selfcheck(output, repo):
    from copy import deepcopy
    import current_policy as policy
    from resource_budget import initialize, SharedBudget, BudgetExceeded
    document=json.loads((ROOT/'restrictions.json').read_text())
    inventory=json.loads((ROOT/'expected_test_ids.json').read_text())
    rows=policy.validate(document,inventory)
    assert document['recovery_status']=='incomplete'
    assert len(policy.restrictions_for(document,inventory,'inherited_cpu'))==21
    assert policy.restrictions_for(document,inventory,'new_cuda')=={}
    rejected=[]
    cases=[]
    bad=deepcopy(document);bad['current_policy'].pop('approval');cases.append(('missing_approval',bad))
    bad=deepcopy(document);bad['current_policy']['approval']['ids'][0]='wrong';cases.append(('wrong_approval_ID',bad))
    bad=deepcopy(document);bad['current_policy']['entries'].append(deepcopy(rows[0]));cases.append(('extra_restriction',bad))
    bad=deepcopy(document)
    next(r for r in bad['current_policy']['entries'] if r['id'] in policy.CURRENT_IDS)['mode']='cpu_only'
    cases.append(('CPU_only_wrong_ID',bad))
    for name,bad in cases:
        try:policy.validate(bad,inventory)
        except (ValueError,KeyError):rejected.append(name)
        else:raise AssertionError('invalid current policy accepted: '+name)
    real_config=dict(stage='real_prefix',access_policy='real_prefix_probe',restriction_policy_id=policy.POLICY_ID,approval_sha256=policy.APPROVAL_SHA)
    try:policy.require_scope(real_config,'new_cuda')
    except ValueError:rejected.append('real_policy_borrowed_by_synthetic')
    else:raise AssertionError('real scope borrowed')
    session=output/'budget-cases';session.mkdir()
    limits=dict(forward=2,backward=1,adam=1)
    shared=initialize(session/'budget.json','selfcheck_dummy_only',limits)
    shared.current('dummy parent/child; no Torch import');shared.charge('forward')
    config=dict(version=VERSION,repo=str(repo),tool_root=str(ROOT),session_root=str(session),
                audit_log=str(session/'audit.jsonl'),forbidden_roots=[str(repo/'data'),str(repo/'artifacts')])
    path=session/'guard-config.json';path.write_text(json.dumps(config)+'\n')
    env=dict(os.environ,AMD_RR_CONFIG=str(path),AMD_RR_CONFIG_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
             PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1')
    code="from resource_budget import SharedBudget; import sys; from restricted_io_guard import require_installed; require_installed(); SharedBudget(sys.argv[1],dict(forward=2,backward=1,adam=1)).charge('forward'); assert 'torch' not in sys.modules"
    r=subprocess.run([sys.executable,'-B','-c',code,str(session/'budget.json')],env=env,cwd=session,capture_output=True,text=True,timeout=15)
    (session/'child.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
    try:shared.charge('forward')
    except BudgetExceeded:rejected.append('shared_counter_cap')
    else:raise AssertionError('counter exceeded')
    try:shared.charge('adam')
    except BudgetExceeded:rejected.append('work_after_stop')
    else:raise AssertionError('work continued after stop')
    s=json.loads((session/'budget.json').read_text());assert s['counts']==dict(forward=2,backward=0,adam=0)
    assert len(s['by_pid'])==2
    called=[]
    class FailFast(unittest.TestCase):
        def test_1_failure(self):self.fail('deliberate failure stops scheduling')
        def test_2_must_not_run(self):called.append(1)
    cases=list(unittest.defaultTestLoader.loadTestsFromTestCase(FailFast))
    report=execute_cases(cases,[t.id() for t in cases],{},failfast=True)
    assert not called and not report['success'] and report['counts']['failure']==1 and report['counts']['unexecuted']==1
    return dict(status='Passed',historical_recovery_still_incomplete=True,current_entries=len(rows),
                current_approved_entries=4,negatives_rejected=rejected,parent_child_dummy_counter=s,
                failfast_report=report,business_optimizer_steps=0,models_constructed=0)


def call_alias_selfcheck(output):
    import resource_budget as budget
    assert 'torch' not in sys.modules
    shared = budget.initialize(output / 'call-alias-budget.json', 'no_model_alias_selfcheck', dict(forward=4))
    previous = budget.INSTANCE
    budget.INSTANCE = shared
    calls = []

    class DummyModule:
        def __init__(self, child=None):
            self.child = child

        def _call_impl(self, value):
            calls.append(self.child is not None)
            if value is None:
                raise ValueError('deliberate dummy failure')
            return self.child(value) if self.child is not None else value + 1

        __call__ = _call_impl

    try:
        original_impl = DummyModule._call_impl
        budget.instrument_module_calls(DummyModule)
        wrapper = DummyModule.__call__
        budget.instrument_module_calls(DummyModule)
        assert DummyModule.__call__ is wrapper
        assert DummyModule._call_impl is original_impl
        leaf = DummyModule()
        root = DummyModule(leaf)
        assert root(10) == 11
        assert leaf(20) == 21
        try:
            root(None)
        except ValueError:
            pass
        else:
            raise AssertionError('dummy failure swallowed')
        assert root(30) == 31
        state = json.loads(shared.path.read_text())
        assert state['counts'] == dict(forward=4, backward=0, adam=0)
        assert len(calls) == 6
        assert getattr(budget.LOCAL, 'forward', 0) == 0
        assert 'torch' not in sys.modules
        return dict(status='Passed', dummy_outer_calls=4, dummy_inner_calls=2,
                    attempted_failure_counted=True, alias_and_idempotence_verified=True,
                    original_impl_unchanged=True, torch_imported=False,
                    actual_model_forward=0, backward=0, optimizer_steps=0)
    finally:
        budget.INSTANCE = previous


def carried_budget_selfcheck(output, repo):
    from acceptance_driver import CARRIED, NEW_LIMITS, FINAL_RESERVATION, validate_execution_authorization, validate_workload_reservation
    from resource_budget import initialize, BudgetExceeded
    validate_execution_authorization(repo)
    shared=initialize(output/'carried-dummy.json','no_model_carried_check',NEW_LIMITS)
    with shared.state() as state:state['counts']=dict(CARRIED)
    shared.charge('forward')
    config=dict(version=VERSION,repo=str(repo),tool_root=str(ROOT),session_root=str(output),
                audit_log=str(output/'carried-audit.jsonl'),forbidden_roots=[str(repo/'data'),str(repo/'artifacts')])
    path=output/'carried-guard-config.json';path.write_text(json.dumps(config)+'\n')
    env=dict(os.environ,AMD_RR_CONFIG=str(path),AMD_RR_CONFIG_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
             PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1')
    code="from resource_budget import SharedBudget; import sys,json; from restricted_io_guard import require_installed; require_installed(); SharedBudget(sys.argv[1],json.loads(sys.argv[2])).charge('forward'); assert 'torch' not in sys.modules"
    child=subprocess.run([sys.executable,'-B','-c',code,str(shared.path),json.dumps(NEW_LIMITS)],env=env,capture_output=True,text=True,timeout=15)
    assert child.returncode==0,child.stderr
    state=json.loads(shared.path.read_text())
    expected=dict(CARRIED);expected['forward']+=2
    assert state['counts']==expected and len(state['by_pid'])==2
    for kind in NEW_LIMITS:
        check=initialize(output/(kind+'-cap-dummy.json'),'no_model_cap_check',NEW_LIMITS)
        with check.state() as value:value['counts']=dict(NEW_LIMITS)
        try:check.charge(kind)
        except BudgetExceeded:pass
        else:raise AssertionError('cap did not refuse '+kind)
        assert json.loads(check.path.read_text())['counts']==NEW_LIMITS
    required={key:CARRIED[key]+FINAL_RESERVATION[key] for key in FINAL_RESERVATION}
    enough=all(required[key]<=NEW_LIMITS[key] for key in required)
    try:
        assert validate_workload_reservation(CARRIED,NEW_LIMITS,FINAL_RESERVATION)==required
    except RuntimeError:
        assert not enough
    else:
        assert enough
    try:validate_workload_reservation(NEW_LIMITS,NEW_LIMITS,dict(forward=1,backward=0,adam=0))
    except RuntimeError:pass
    else:raise AssertionError('insufficient full sequence was admitted')
    return dict(status='Passed',dummy_only=True,carried=CARRIED,limits=NEW_LIMITS,
                full_sequence_reservation=FINAL_RESERVATION,required=required,
                full_sequence_currently_fits=enough,

                dummy_parent_and_child_charge='forward counters only; no model',
                observed_dummy_state=state,business_forward=0,business_backward=0,business_Adam=0)


def output_monitor_selfcheck(output):
    import errno
    from unittest.mock import patch
    from acceptance_driver import output_tree_bytes
    assert 'torch' not in sys.modules
    root = output / 'output-monitor-case'
    fixtures = root / 'fixtures'
    gone_dir = fixtures / 'gone-directory'
    gone_dir.mkdir(parents=True)
    (gone_dir / 'payload').write_bytes(b'discarded')
    gone_file = fixtures / 'gone-file'
    gone_file.write_bytes(b'discarded too')
    (fixtures / 'kept').write_bytes(b'123')
    evidence = root / 'report'
    evidence.write_bytes(b'1234567')
    outside = output / 'outside-monitor'
    outside.mkdir()
    (outside / 'large').write_bytes(b'x' * 1024)
    (fixtures / 'linked-directory').symlink_to(outside, target_is_directory=True)
    scandir, path_stat = os.scandir, Path.stat
    removed = []

    def deleting_scandir(path):
        if not isinstance(path, int) and Path(path) == gone_dir and 'directory' not in removed:
            removed.append('directory')
            shutil.rmtree(gone_dir)
        return scandir(path)

    def deleting_stat(path, *args, **kwargs):
        if path == gone_file and 'file' not in removed:
            removed.append('file')
            path.unlink()
        return path_stat(path, *args, **kwargs)

    observations = {}
    with patch('os.scandir', side_effect=deleting_scandir), patch.object(Path, 'stat', deleting_stat):
        size = output_tree_bytes(root, observations)
    assert set(removed) == {'directory', 'file'}
    assert size == 10 and observations['vanished_fixture_entries'] == 2
    assert output_tree_bytes(root) == 10
    rejected = []
    for label, target, exception in [
        ('missing_root', root, FileNotFoundError(errno.ENOENT, 'injected missing root', str(root))),
        ('missing_evidence', evidence, FileNotFoundError(errno.ENOENT, 'injected missing evidence', str(evidence))),
        ('permission_error', fixtures / 'kept', PermissionError(errno.EACCES, 'injected permissions', str(fixtures / 'kept'))),
    ]:
        def failing_stat(path, *args, **kwargs):
            if path == target:
                raise exception
            return path_stat(path, *args, **kwargs)
        with patch.object(Path, 'stat', failing_stat):
            try:
                output_tree_bytes(root)
            except type(exception):
                rejected.append(label)
            else:
                raise AssertionError('monitor hid ' + label)
    def failing_scandir(path):
        if Path(path) == fixtures:
            raise PermissionError(errno.EACCES, 'injected walk permission failure', str(fixtures))
        return scandir(path)
    with patch('os.scandir', side_effect=failing_scandir):
        try:
            output_tree_bytes(root)
        except PermissionError:
            rejected.append('traversal_permission_error')
        else:
            raise AssertionError('walk permission failure hidden')
    assert 'torch' not in sys.modules
    return dict(status='Passed', actual_removed_fixture_entries=removed,
                sampled_bytes=size, observations=observations, rejected=rejected,
                directory_symlink_not_followed=True, torch_imported=False,
                forward=0, backward=0, Adam=0)


def fixture_child(config_path):
    """Use only synthetic bytes; a .pt suffix does not imply tensor loading."""
    import restricted_io_guard as guard
    state = guard.require_installed()
    config = json.loads(Path(config_path).read_text())
    root = Path(state['fixture_root']); evidence = Path(state['session_root'])
    denied = []

    def refusal(label, action):
        try:
            action()
        except guard.ForbiddenAccess:
            denied.append(label)
        else:
            raise AssertionError('forbidden fixture operation accepted: ' + label)

    assert (root / 'data' / 'toy.csv').read_text() == 'value\n1\n'
    checkpoint = Path(config['preview']['checkpoint_path'])
    assert checkpoint.read_bytes() == b'synthetic checkpoint sentinel; not tensor data'
    (checkpoint.parent / 'written-by-child').write_text('synthetic\n')
    (evidence / 'child-evidence.txt').write_text('durable evidence\n')
    refusal('protected_synthetic_file', lambda: (root / 'forbidden' / 'secret').read_bytes())
    refusal('protected_symlink', lambda: (root / 'secret-link').read_bytes())
    refusal('other_stage_tmp_read', lambda: (root.parent / 'real_prefix' / 'foreign.pt').read_bytes())
    refusal('other_stage_tmp_write', lambda: (root.parent / 'real_prefix' / 'new.pt').write_bytes(b'x'))
    # Use a descriptor acquired for the allowed fixture root, then resolve dir_fd.
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        refusal('dir_fd_protected_file', lambda: os.open('forbidden/secret', os.O_RDONLY, dir_fd=fd))
    finally:
        os.close(fd)
    fd = os.open(checkpoint, os.O_RDONLY)
    try:
        assert os.read(fd, 9) == b'synthetic'
    finally:
        os.close(fd)
    checksum = subprocess.run(['sha256sum', '-c', 'checksums.sha256'], cwd=checkpoint.parent,
                              capture_output=True, text=True, timeout=15)
    assert checksum.returncode == 0, checksum.stderr
    code = ("from pathlib import Path; import os,sys; from restricted_io_guard import require_installed,ForbiddenAccess; "
            "s=require_installed(); p=Path(s['fixture_root']); assert (p/'data/toy.csv').read_text()=='value\\n1\\n'; "
            "assert 'torch' not in sys.modules;\n"
            "try:(p.parent/'real_prefix/foreign.pt').read_bytes()\n"
            "except ForbiddenAccess:print('child-denied')\n"
            "else:raise AssertionError('child escaped fixture boundary')\n")
    child = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=15)
    assert child.returncode == 0 and 'child-denied' in child.stdout, child.stderr
    bad_env = dict(os.environ, TMPDIR='/tmp')
    refusal('changed_child_tmpdir', lambda: subprocess.run(
        [sys.executable, '-B', '-c', 'raise AssertionError("must not start")'], env=bad_env))
    assert 'torch' not in sys.modules and 'main' not in sys.modules
    (evidence / 'fixture-child-result.json').write_text(json.dumps(dict(
        status='Passed', deliberate_parent_refusals=denied, child_refusals=1,
        checkpoint_plain_bytes_read=True, system_checksum=checksum.stdout,
        business_imports=0, model=0, Dataset=0, optimizer=0)) + '\n')


def fixture_layout_selfcheck(output, repo):
    import ast
    from acceptance_driver import create_fixture_layout, output_tree_bytes
    case = output / 'fixture-layout-case'; case.mkdir()
    layout = create_fixture_layout(case)
    base = Path(layout['fixture_execution_root'])
    fixture = Path(layout['stage_roots']['inherited_cpu'])
    evidence = case / 'inherited_cpu'; evidence.mkdir()
    try:
        # Child directories passed to existing writers must not already exist.
        data = fixture / 'data'; source = fixture / 'source'; artifacts = fixture / 'artifacts'
        assert not data.exists() and not source.exists() and not artifacts.exists()
        for path in (data, source / 'models', artifacts, fixture / 'summary', fixture / 'forbidden'):
            path.mkdir(parents=True)
        (data / 'toy.csv').write_text('value\n1\n')
        (source / 'models' / 'synthetic.py').write_text('# source identity sentinel only\n')
        # Path-only previews, not synthetic dataset or model execution.
        parent = artifacts / 'thls-preview' / 'UrbanEV' / 'target_exogenous' / 'volume' / 'horizon_3' / 'fold_6' / 'seed_2024'
        staging = parent / '.synthetic-run.staging'; staging.mkdir(parents=True)
        completed = parent / 'synthetic-run'; completed.mkdir()
        checkpoint = staging / 'last.pt'
        checkpoint.write_bytes(b'synthetic checkpoint sentinel; not tensor data')
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        (staging / 'checksums.sha256').write_text(digest + '  last.pt\n')
        (fixture / 'forbidden' / 'secret').write_bytes(b'must not be read under guard')
        (fixture / 'secret-link').symlink_to(fixture / 'forbidden' / 'secret')
        (base / 'real_prefix' / 'foreign.pt').write_bytes(b'another stage, not authorized here')
        preview = dict(cli=['--data', str(data), '--artifact_root', str(artifacts), '--resume', str(staging)],
            scientific_path_fields={'preprocessing.data_path': str((data / 'toy.csv').resolve()),
                'preprocessing.data_root': str(data.resolve())}, source_root=str(source),
            source_files=[{'path': 'models/synthetic.py'}], artifact_root=str(artifacts),
            staging_path=str(staging), completed_path=str(completed), checkpoint_path=str(checkpoint),
            resume_path=str(staging), checksum_entries=['last.pt'], summary_input=str(artifacts),
            summary_output=str(fixture / 'summary' / 'results.csv'))
        assert '/public/home/' not in json.dumps(preview)
        assert Path(preview['source_files'][0]['path']).is_absolute() is False
        assert '/public/home/' in str(evidence) and Path('/tmp') not in evidence.parents
        for protected in (repo / 'data', repo / 'artifacts'):
            assert protected.resolve() not in fixture.parents and fixture not in protected.resolve().parents
        # Execute only the two actual pure path validators extracted by AST.
        validators = {}
        for filename, name in [('main.py', '_enhanced_run_id_from_staging'),
                               ('summarize_results.py', '_enhanced_path_identity')]:
            tree = ast.parse((repo / filename).read_bytes())
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            namespace = {'Path': Path}
            exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), namespace)
            validators[name] = namespace[name]
        assert validators['_enhanced_run_id_from_staging'](staging) == 'synthetic-run'
        try:
            validators['_enhanced_run_id_from_staging'](completed)
        except ValueError as exc:
            assert 'enhanced resume path must be the hidden staging directory' in str(exc)
        else:
            raise AssertionError('completed directory accepted')
        assert validators['_enhanced_path_identity'](artifacts / 'thls-preview', completed)['artifact_horizon'] == 3
        config = dict(version=VERSION, repo=str(repo), tool_root=str(ROOT), session_root=str(evidence),
            audit_log=str(evidence / 'audit.jsonl'), forbidden_roots=[str(repo / 'data'), str(repo / 'artifacts'), str(fixture / 'forbidden')],
            fixture_root=str(fixture), fixture_execution_root=str(base), fixture_layout_version=layout['version'],
            stage='inherited_cpu', access_policy='synthetic_regression', business_bootstrap=False, preview=preview)
        path = evidence / 'guard-config.json'; path.write_text(json.dumps(config) + '\n')
        env = dict(os.environ, AMD_RR_CONFIG=str(path), AMD_RR_CONFIG_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(fixture))
        child = subprocess.run([sys.executable, '-B', str(ROOT / 'selfcheck.py'), '--fixture-child', str(path)],
                               env=env, capture_output=True, text=True, timeout=30)
        (evidence / 'child.stdout').write_text(child.stdout); (evidence / 'child.stderr').write_text(child.stderr)
        assert child.returncode == 0, child.stderr
        result = json.loads((evidence / 'fixture-child-result.json').read_text())
        events = [json.loads(line) for line in (evidence / 'audit.jsonl').read_text().splitlines()]
        assert sum(x['event'] == 'denied' for x in events) == 7
        assert len({x['pid'] for x in events if x['event'] == 'installed'}) == 2
        # A broad /tmp configuration cannot become a trusted fixture scope.
        bad = dict(config, fixture_root='/tmp')
        bad_path = evidence / 'bad-guard-config.json'; bad_path.write_text(json.dumps(bad) + '\n')
        bad_env = dict(env, AMD_RR_CONFIG=str(bad_path), AMD_RR_CONFIG_SHA256=hashlib.sha256(bad_path.read_bytes()).hexdigest(), TMPDIR='/tmp')
        bad_child = subprocess.run([sys.executable, '-B', '-c', 'raise AssertionError("must not execute")'],
                                   env=bad_env, capture_output=True, text=True, timeout=15)
        (evidence / 'broad-scope.stderr').write_text(bad_child.stderr)
        assert bad_child.returncode == 86 and 'binding mismatch' in bad_child.stderr
        size = output_tree_bytes(fixture, transient=True)
        assert size > 0
        metadata = {str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in base.rglob('*') if p.is_file() and not p.is_symlink()}
        (evidence / 'fixture-metadata.sha256.json').write_text(json.dumps(metadata, indent=2) + '\n')
        (evidence / 'path-preview.json').write_text(json.dumps(preview, indent=2) + '\n')
        assert 'torch' not in sys.modules and 'main' not in sys.modules
        return dict(status='Passed', layout=layout, preview_kind='path-only; no dataset/model/runtime constructed',
            deliberate_denials=7, protected_content_reads=0, python_guarded_processes=2,
            broad_tmp_scope_rejected=True, durable_metadata=str(evidence / 'fixture-metadata.sha256.json'),
            sampled_fixture_bytes=size, child_result=result, forward=0, backward=0, Adam=0)
    finally:
        # Only this selfcheck's newly created synthetic tree; preserve durable evidence.
        shutil.rmtree(base)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo')
    parser.add_argument('--output')
    parser.add_argument('--guard-child')
    parser.add_argument('--fixture-child')
    args = parser.parse_args()
    if args.fixture_child:
        fixture_child(args.fixture_child)
        return 0
    if args.guard_child:
        guarded_child(args.guard_child)
        return 0
    output = Path(args.output)
    output.mkdir(exist_ok=False)
    result = dict(tool_version=VERSION, selfcheck_scope='synthetic files/dummy unittest/standard-library processes only')
    try:
        result['ledger'] = ledger_selfcheck()
        (output / 'ledger-report.json').write_text(json.dumps(result['ledger'], indent=2) + '\n')
        result['guard'] = guard_selfcheck(output, Path(args.repo).resolve())
        result['current_policy_budget'] = current_policy_budget_selfcheck(output, Path(args.repo).resolve())
        result['call_alias'] = call_alias_selfcheck(output)
        result['current_carried_budget'] = carried_budget_selfcheck(output, Path(args.repo).resolve())
        result['output_monitor'] = output_monitor_selfcheck(output)
        result['fixture_layout'] = fixture_layout_selfcheck(output, Path(args.repo).resolve())
        result['status'] = 'Passed'
    except BaseException:
        import traceback
        result['status'] = 'Failed'
        result['traceback'] = traceback.format_exc()
    result['business_tests_executed'] = 0
    (output / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'ledger'}, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'Passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
