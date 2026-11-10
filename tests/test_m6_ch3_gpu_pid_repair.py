"""CPU parser/control fixtures; real /proc only, never query or allocate CUDA."""
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

from m6_ch3_supplement_v10 import gpu_identity as g
from m6_ch3_supplement_v10.benchmark import measure
from m6_ch3_supplement_v10.contract import read, sha, write_new, validate, validate_permission, verify_case_selection

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path(os.environ['M6_PID_TEST_EVIDENCE'])


def raw(local=101, driver=901, start=777):
    fields = ['S'] + ['0']*18 + [str(start)] + ['0']*10
    return dict(local_pid=local, kernel=g.SUPPORTED_KERNEL, main_thread=True,
                status=f'Name:\tfixture\nPid:\t{local}\nTgid:\t{local}\n',
                stat=f'{local} (fixture) '+ ' '.join(fields),
                sched=f'fixture ({driver}, #threads: 1)\nraw-body\n',
                pid_namespace='pid:[4026534934]')


class PidRepair(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp(prefix=self._testMethodName+'-', dir=EVIDENCE))

    def boundary(self, output='', context=True, before=None, after=None, query_error=None, capture_error=None):
        existing = set(self.folder.glob('fixture-*.json'))
        before = before if before is not None else raw()
        after = after if after is not None else before
        snapshots = iter([before, after])
        def capture(dst):
            dst.update(copy.deepcopy(next(snapshots)))
            if capture_error:
                raise capture_error
            return dst
        with patch.object(g, 'capture_self', side_effect=capture), patch.object(g.subprocess, 'run') as query:
            if query_error:
                query.side_effect = query_error
            else:
                query.return_value = subprocess.CompletedProcess(g.COMMAND, 0, output, '')
            try:
                result = g.check_gpu_identity('GPU-target', self.folder, 'fixture', context)
            finally:
                created = set(self.folder.glob('fixture-*.json')) - existing
                self.assertEqual(len(created), 1)
                self.last_path = created.pop()
                self.query_calls = query.call_count
                for call in query.call_args_list:
                    self.assertEqual(call.args[0], g.COMMAND)
                    self.assertEqual(call.kwargs['timeout'], 60)
        return result

    def latest(self):
        return json.loads(self.last_path.read_text())

    def test_same_and_different_spaces(self):
        for local, driver in [(101, 101), (101, 901)]:
            result = self.boundary(f'{driver}, GPU-target\n777, GPU-other\n', before=raw(local, driver))
            self.assertEqual(result['target_pids'], [driver])
            data = self.latest()
            self.assertTrue(data['accepted'])
            self.assertEqual(data['identity']['local_pid'], local)
            self.assertEqual(data['identity']['driver_pid'], driver)
            self.assertIn('raw-body', data['before']['sched'])
            self.assertEqual(data['queries'][0]['returncode'], 0)
            self.assertEqual(len(data['records']), 2)

    def test_foreign_collision_and_missing_self(self):
        for output in ['901, GPU-target\n888, GPU-target\n', '901, GPU-target\n101, GPU-target\n', '101, GPU-target\n', '']:
            with self.assertRaises(RuntimeError):
                self.boundary(output)
            d = self.latest()
            self.assertFalse(d['accepted'])
            self.assertEqual(d['identity']['driver_pid'], 901)
            self.assertEqual(self.query_calls, 1)  # foreign load never retried
            if '101' in output:
                self.assertIn(101, d['foreign_pids'])

    def test_pre_context_empty_only(self):
        self.assertEqual(self.boundary('', context=False)['classification'], 'empty-before-context')
        for output in ['901, GPU-target\n', '888, GPU-target\n']:
            with self.assertRaises(RuntimeError):
                self.boundary(output, context=False)
            self.assertFalse(self.latest()['accepted'])

    def test_races_and_namespace_changes(self):
        for changed in [raw(start=778), raw(local=102), raw(driver=902), dict(raw(), pid_namespace='pid:[999]')]:
            with self.assertRaises(ValueError):
                self.boundary('901, GPU-target\n', after=changed)
            self.assertEqual(self.latest()['classification'], 'unknown')
            self.assertIn('changed', self.latest()['error'])

    def test_mapping_failures_save_partial_evidence(self):
        variants = [dict(raw(), sched='broken'), dict(raw(), kernel='6.1.0'),
                    dict(raw(), main_thread=False), dict(raw(), status='Pid: 101\nTgid: 102\n'),
                    dict(raw(), status=raw()['status']+'NSpid: 901 101\n'),
                    dict(raw(), stat='bad'), dict(raw(), pid_namespace='unknown')]
        for variant in variants:
            with self.assertRaises((ValueError, KeyError, IndexError)):
                self.boundary(before=variant)
            self.assertEqual(self.query_calls, 0)
            self.assertFalse(self.latest()['accepted'])
        for error in [PermissionError('proc permission'), FileNotFoundError('missing sched')]:
            with self.assertRaises(type(error)):
                self.boundary(capture_error=error)
            self.assertEqual(self.query_calls, 0)
            self.assertEqual(self.latest()['error_type'], type(error).__name__)

    def test_query_errors_and_timeout_evidence(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            self.boundary(query_error=subprocess.TimeoutExpired(g.COMMAND, 60, output=b'partial', stderr=b'timeout'))
        d = self.latest()
        self.assertEqual(self.query_calls, 3)
        self.assertEqual([q['stdout'] for q in d['queries']], ['partial']*3)
        for q in d['queries']:
            self.assertEqual(q['stderr'], 'timeout')
            self.assertIsNone(q['returncode'])
        with patch.object(g, 'capture_self', side_effect=lambda d: d.update(raw())), patch.object(g.subprocess, 'run', return_value=subprocess.CompletedProcess(g.COMMAND, 2, 'raw', 'driver error')) as query:
            with self.assertRaises(RuntimeError):
                g.check_gpu_identity('GPU-target', self.folder, 'nonzero', True)
            self.assertEqual(query.call_count, 1)
        d = read(next(self.folder.glob('nonzero-*.json')))
        self.assertEqual(d['queries'][0]['returncode'], 2)
        self.assertEqual(d['queries'][0]['stderr'], 'driver error')
        for output in ['N/A, GPU-target\n', '901 GPU-target\n', '901, GPU-target\n901, GPU-target\n']:
            with self.assertRaises(ValueError):
                self.boundary(output)
            self.assertEqual(self.latest()['classification'], 'unknown')

    def test_real_proc_parent_child(self):
        before, after = {}, {}
        g.capture_self(before)
        parent = g.parse_identity(before)
        code = 'import json; from m6_ch3_supplement_v10.gpu_identity import capture_self,parse_identity; a={};b={};capture_self(a);capture_self(b);assert parse_identity(a)==parse_identity(b);print(json.dumps(dict(before=a,after=b,identity=parse_identity(a))))'
        child = json.loads(subprocess.check_output([sys.executable, '-B', '-c', code], cwd=ROOT, text=True, timeout=10))
        g.capture_self(after)
        self.assertEqual(parent, g.parse_identity(after))
        self.assertNotEqual(parent['local_pid'], child['identity']['local_pid'])
        self.assertNotEqual(parent['driver_pid'], child['identity']['driver_pid'])
        self.assertEqual(parent['pid_namespace'], child['identity']['pid_namespace'])
        self.assertNotIn('NSpid:', before['status'])
        write_new(self.folder/'real-proc.json', dict(parent_before=before, parent_after=after, parent_identity=parent, child=child))

    def test_measurement_calls_and_boundary_order(self):
        events, callbacks = [], []
        def boundary(*args):
            events.append(args[2])
            return dict(classification='self')
        ticks = iter(range(10))
        def run_measurement():
            events.append('measurement-start')
            result = measure(lambda: callbacks.append(1), lambda: None, lambda: None, lambda: {}, 1, clock=lambda: next(ticks))
            events.append('measurement-end')
            return result
        with patch.object(g, 'check_gpu_identity', side_effect=boundary):
            r = g.measure_with_identity(run_measurement, 'GPU-target', self.folder)
        self.assertEqual(events, ['before-timing', 'measurement-start', 'measurement-end', 'after-timing'])
        self.assertEqual(len(callbacks), 550)
        self.assertEqual([x['calls'] for x in r['groups']], [100]*5)
        callbacks.clear()
        next_job = Mock()
        with patch.object(g, 'check_gpu_identity', side_effect=RuntimeError('foreign')):
            with self.assertRaises(RuntimeError):
                g.measure_with_identity(run_measurement, 'GPU-target', self.folder)
                next_job()
        self.assertEqual(callbacks, [])
        next_job.assert_not_called()
        write_new(self.folder/'callback-counts.json', dict(success_noop_callbacks=550, rejected_callbacks=0, real_model_forwards=0))

    def test_retry_binding_and_fixed_case(self):
        config = read(ROOT/'configs/m6_ch3_supplement_v10_retry1.json')
        validate(config)
        chain = config['technical_continuation']
        selection = read(chain['case_selection_ref']['path'])
        verify_case_selection(config, selection)
        target = self.folder/'case-selection.json'
        write_new(target, selection)
        self.assertEqual(sha(target), chain['case_selection_ref']['sha256'])
        changed = copy.deepcopy(selection)
        changed['nodes'].reverse()
        with self.assertRaises(ValueError):
            verify_case_selection(config, changed)
        for key, value in [('attempt', 'ch3-supplement-v10-retry2'), ('cpu_threads', 99), ('jobs', [])]:
            bad = dict(config, **{key: value})
            with self.assertRaises(ValueError):
                validate(bad)
        old_auth = read(chain['previous_auth_ref']['path'])
        with self.assertRaises(PermissionError):
            validate_permission(ROOT/'configs/m6_ch3_supplement_v10_retry1.json', old_auth)
        with self.assertRaises(PermissionError):
            validate_permission(chain['previous_config_ref']['path'], old_auth)
        with self.assertRaises(PermissionError):
            validate_permission('/not-read', dict(review_approved=False, commit=None))
        old = Path(chain['previous_runtime_ref']['path']).parent
        with self.assertRaises(FileExistsError):
            old.mkdir(exist_ok=False)

    def test_after_boundary_failure_preserves_measurement(self):
        calls = []
        with patch.object(g, 'check_gpu_identity', side_effect=[dict(classification='self'), RuntimeError('new foreign process')]):
            with self.assertRaises(RuntimeError):
                g.measure_with_identity(lambda: (calls.append(1) or dict(warmup_calls=50, timed_calls=500, fixture=True)), 'GPU-target', self.folder)
        self.assertEqual(calls, [1])
        self.assertEqual(read(self.folder/'measurement-completed.json')['timed_calls'], 500)
        self.assertFalse((self.folder/'receipt.json').exists())

    def test_production_wiring_and_owned_cleanup(self):
        # Execute only extracted CPU control nodes, NOT supervise/worker/preflight.
        queue = ast.parse((ROOT/'m6_ch3_supplement_v10/queue.py').read_text())
        fn = next(n for n in queue.body if isinstance(n, ast.FunctionDef) and n.name == 'supervise')
        loop = next(n for n in ast.walk(fn) if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple))
        while_index = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.While))
        nodes = loop.body[while_index:while_index+2]
        class Child:
            def __init__(self, code): self.returncode = code
            def poll(self): return self.returncode
        failed, active = Child(1), Child(None)
        children = [('failed', failed), ('active', active)]
        def sleep(_):
            self.assertTrue((self.folder/'STOP').exists())
            active.returncode = 1
        env = dict(children=children, folder=self.folder, write_new=write_new, time=Mock(sleep=sleep))
        with self.assertRaises(RuntimeError):
            exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), '<owned-control-fixture>', 'exec'), env)
        self.assertTrue((self.folder/'STOP').exists())
        # No signal/terminate/kill path in the repair or production queue.
        for filename in ['gpu_identity.py', 'queue.py']:
            tree = ast.parse((ROOT/'m6_ch3_supplement_v10'/filename).read_text())
            self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ('kill', 'killpg', 'terminate') for n in ast.walk(tree)))
        text = (ROOT/'m6_ch3_supplement_v10/readonly.py').read_text()
        body = text[text.index('def run('):]
        self.assertLess(body.index("'before-model-load'"), body.index('restore('))
        self.assertIn('measure_with_identity(', body)
        self.assertIn('verify_case_selection(config, selection)', text)
        self.assertIn('case-selection.json', text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
