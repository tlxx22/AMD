"""Parent-controller import/preflight and r2 isolation; synthetic, no model/GPU."""
import builtins
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, redirect_stdout
from unittest.mock import patch

import ch3_runner as runner
import m6_type1_followup_entry as entry
from tools.restricted_regression import run_restricted as restricted
from utils import ch3_moderntcn_etth1_recovery as r, ch3_type1_chain as q, ch3_type1_tasks as s
from utils.ch3_contract import ROOT, digest
from utils.ch3_native_recovery_records import bound, exclusive, ref, sha
from tests.test_m6_moderntcn_etth1_recovery import attempt

BUNDLE_SHA = 'e3ff46b98d4752f2f94cb1b6822e37b4b08b7585942fa9ca2ad55d03ac8096d9'
CLOSED_PARENT = '3c837ba6dd82043cb2dcffdd5263e55d5588444a'
SYNTHETIC_HEAD = 'f' * 40


def executable_fixture(template):
    value = copy.deepcopy(template)
    value.update(reviewed=True, execution_permitted=True, structure_frozen=True,
                 m6_authorized=True, budget_authorized=True,
                 closure_commit=SYNTHETIC_HEAD,
                 authorization_basis='synthetic CPU fixture only; never authorizes a real execution')
    return value


class RepoRoot(unittest.TestCase):
    def child(self, body):
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
        environment.pop('PYTHONPATH', None)
        check = ("import sys,json;from pathlib import Path;"
                 "assert str(Path('tools/restricted_regression').resolve()) not in sys.path;"
                 "assert 'run_restricted' not in sys.modules;\n")
        process = subprocess.run([q.PYTHON, '-B', '-c', check + body], cwd=ROOT,
                                 env=environment, capture_output=True, text=True, timeout=90)
        self.assertEqual(process.returncode, 0, process.stderr)
        return json.loads(process.stdout)

    def test_package_bundle_import_without_sys_path_injection(self):
        value = self.child(
            "from tools.restricted_regression.run_restricted import verify_bundle\n"
            "value=verify_bundle();assert 'torch' not in sys.modules\n"
            "print(json.dumps({'bundle_sha':value,'torch_imported':False}))\n")
        self.assertEqual(value, dict(bundle_sha=BUNDLE_SHA, torch_imported=False))

    def test_actual_parent_status_without_child_path_injection(self):
        value = self.child(
            "from utils.ch3_moderntcn_etth1_recovery import activate,status,RESULT,LOG\n"
            "activate();value=status();assert 'torch' not in sys.modules\n"
            "assert not RESULT.exists() and not LOG.exists()\n"
            "print(json.dumps({'state':value['state'],'gpu_ready':value['READY_FOR_GPU_EXECUTION'],"
            "'torch_imported':False,'new_output_created':False}))\n")
        self.assertFalse(value['gpu_ready'])
        self.assertFalse(value['torch_imported'])
        self.assertFalse(value['new_output_created'])


class ParentPreflight(unittest.TestCase):
    @contextmanager
    def fixture(self, code=None):
        from utils import ch3_ms_seal_recovery as ms, ch3_round2_amendment as amendment
        with tempfile.TemporaryDirectory(prefix='m6-import-preflight-') as temp, attempt(), ExitStack() as stack:
            root = Path(temp)
            result = root / 'result'
            source_dir = root / 'source'
            source_dir.mkdir()
            config = dict(synthetic=True)
            ids = ['synthetic-MS-' + str(i) for i in range(203)]
            verification = exclusive(source_dir / 'verification.json', dict(task_ids=ids, receipts={}))
            ms_ref = exclusive(source_dir / 'MS.json', dict(
                technical_complete=True, training_commit='df6a16403e10d51097db8c88829909c533d15652',
                task_ids=ids, receipts={}, source_verification_ref=verification))
            auth = exclusive(source_dir / 'producer-auth.json', dict(closure_commit=r.BASE))
            permit = dict(commit=r.BASE, protocol_sha=digest(config),
                          code=code if code is not None else {'tools/restricted_regression/bundle.sha256': '0' * 64})
            source = exclusive(source_dir / 'source.json', dict(producer_commit=r.BASE, refs=dict(
                config=exclusive(source_dir / 'config.json', config),
                producer_permit=exclusive(source_dir / 'producer.json', permit),
                controller=exclusive(source_dir / 'controller.json', dict(authorization=auth)),
                authorization=auth,
                failure=exclusive(source_dir / 'failure.json', dict(error="RuntimeError('owned child technical failure: M_BASE-probe')")),
                probe_failure=exclusive(source_dir / 'probe-failure.json', dict(error="ValueError('numeric gate failure; no resource fallback')")),
                MS_boundary=ms_ref, source_verification=verification,
                upstream_boundary=exclusive(source_dir / 'upstream.json', dict(boundaries={'MS': ms_ref})),
                exit_verification=exclusive(source_dir / 'exit.json', dict(instances=[])))))
            confirmation = exclusive(source_dir / 'confirmation.json', dict(worker_instances=[]))
            stack.enter_context(patch.object(r, 'SOURCE_REF', source))
            stack.enter_context(patch.object(r, 'evidence', return_value=dict(confirmation_ref=confirmation)))
            for module, name, value in [(s, 'RESULT', result), (ms, 'RESULT', result),
                                         (amendment, 'RESULT', result / 'round2-amendment'),
                                         (q, 'CONTROL', result / 'queue/controller'),
                                         (q, 'LOG', root / 'launcher.log'),
                                         (q, 'SESSION', 'fixture-m6-import-absent-' + str(os.getpid()))]:
                stack.enter_context(patch.object(module, name, value))
            stack.enter_context(patch.object(q, 'closure', return_value=SYNTHETIC_HEAD))
            remote = stack.enter_context(patch.object(q, 'verify_live_remote', return_value=SYNTHETIC_HEAD))
            # Only unrelated hardware/data queries are replaced. Source, inheritance,
            # bundle, status, readiness, validate_start and the public CLI stay real.
            stack.enter_context(patch.object(q, 'dynamic', return_value=dict(
                commit=SYNTHETIC_HEAD, protocol_sha='synthetic', code={}, environment={},
                hardware={}, source_states={})))
            guards = []
            for module, name in [(r, 'import_ms'), (r, 'load_seed'), (q, 'run'), (q, 'create_permit'),
                                 (ms, 'snapshot'), (ms, 'verify_source'),
                                 (runner, 'init_training'), (runner, 'update'), (runner, 'evaluate')]:
                guards.append(stack.enter_context(patch.object(
                    module, name, side_effect=AssertionError('preflight must not execute ' + name))))
            source_calls = stack.enter_context(patch.object(
                r, 'verify_registered_source', wraps=r.verify_registered_source))
            inheritance_calls = stack.enter_context(patch.object(
                r, 'verify_production_inheritance', wraps=r.verify_production_inheritance))
            bundle_calls = stack.enter_context(patch.object(
                restricted, 'verify_bundle', wraps=restricted.verify_bundle))
            template = q.start_template()
            approval = exclusive(root / 'synthetic-approval.json', executable_fixture(template))
            yield root, result, approval, stack, remote, source_calls, inheritance_calls, bundle_calls
            for guard in guards:
                guard.assert_not_called()
            self.assertFalse(result.exists())
            self.assertFalse((result / 'queue/controller').exists())
            self.assertFalse((result / 'queue/controller/upstream-technical-boundary.json').exists())
            self.assertFalse((result / 'probe').exists())
            self.assertFalse((root / 'launcher.log').exists())
            self.assertNotIn('torch', sys.modules)

    def public(self, approval, action='preflight'):
        output = io.StringIO()
        with patch.object(sys, 'argv', ['m6_moderntcn_etth1_recovery_entry.py', action,
                                      '--approval', approval['path'], '--approval-sha', approval['sha256']]), redirect_stdout(output):
            code = entry.cli()
        return code, json.loads(output.getvalue())

    def test_public_preflight_reaches_source_inheritance_bundle_read_only(self):
        with self.fixture() as (_, _, approval, _, remote, source, inheritance, bundle):
            code, value = self.public(approval)
            self.assertEqual(code, 0)
            self.assertEqual(value['blocked'], [])
            self.assertTrue(value['READY_TO_ARM_HANDOFF'])
            self.assertFalse(value['READY_FOR_GPU_EXECUTION'])
            self.assertGreaterEqual(source.call_count, 1)
            self.assertEqual(source.call_count, inheritance.call_count)
            self.assertEqual(inheritance.call_count, bundle.call_count)
            self.assertEqual(remote.call_count, 1)

    def test_public_dry_run_uses_same_parent_check_without_materialization(self):
        with self.fixture() as (_, _, approval, _, _, source, inheritance, bundle):
            code, value = self.public(approval, 'dry-run')
            self.assertEqual((code, value['blocked']), (0, []))
            self.assertGreater(bundle.call_count, 0)
            self.assertEqual(source.call_count, inheritance.call_count)

    def test_qualified_import_failure_is_blocked_not_silently_skipped(self):
        original = builtins.__import__

        def fail(name, *args, **kwargs):
            if name == 'tools.restricted_regression.run_restricted':
                raise ModuleNotFoundError('synthetic qualified bundle import failure')
            return original(name, *args, **kwargs)

        with self.fixture() as (_, _, approval, stack, remote, _, _, _):
            stack.enter_context(patch.object(builtins, '__import__', side_effect=fail))
            code, value = self.public(approval)
            self.assertEqual(code, 2)
            self.assertTrue(any('synthetic qualified bundle import failure' in x for x in value['blocked']))
            self.assertFalse(value['READY_TO_ARM_HANDOFF'])
            remote.assert_not_called()

    def test_corrupted_bundle_is_blocked_in_actual_bundle_verifier(self):
        with self.fixture() as (root, _, approval, stack, _, _, _, bundle):
            wrong = root / 'bad-bundle'
            wrong.mkdir()
            (wrong / 'unit.py').write_text('synthetic altered bundle\n')
            (wrong / 'bundle.sha256').write_text('0' * 64 + '  unit.py\n')
            stack.enter_context(patch.object(restricted, 'TOOL_ROOT', wrong))
            code, value = self.public(approval)
            self.assertEqual(code, 2)
            self.assertTrue(any('tool seal mismatch' in x for x in value['blocked']))
            self.assertGreater(bundle.call_count, 0)

    def test_illegal_producer_delta_is_blocked_before_start(self):
        with self.fixture(code={'utils/ch3_time_marks.py': '0' * 64}) as (_, _, approval, _, _, _, inheritance, bundle):
            code, value = self.public(approval)
            self.assertEqual(code, 2)
            self.assertTrue(any('unreviewed producer delta' in x for x in value['blocked']))
            self.assertGreater(inheritance.call_count, 0)
            bundle.assert_not_called()


class AttemptIsolation(unittest.TestCase):
    def test_r2_binding_keeps_scope_protocol_and_old_collection_sources(self):
        with attempt():
            self.assertEqual(r.ATTEMPT, 'M_BASE-ModernTCN-ETTh1-numeric-r2')
            self.assertEqual(r.SESSION, 'ch3-m6-m128-moderntcn-etth1-numeric-r2')
            self.assertEqual(r.RESULT.name, 'baseline-unified-v3-ms-seal-m128-recovery1-moderntcn-etth1-numeric-r2')
            self.assertEqual(r.LOG, r.REPAIR_PACKAGE / 'followup-launcher.log')
            self.assertEqual((q.CONTROL, q.LOG, q.SESSION), (r.RESULT / 'queue/controller', r.LOG, r.SESSION))
            self.assertEqual(s.ID, 'm6-baseline-type1-followup-v3-recovery1')
            self.assertEqual(s.PROTOCOL, 'baseline-type1-followup-v3-recovery1')
            self.assertEqual(q.start_template()['upstream_anchors']['execution_attempt'], r.ATTEMPT)
            self.assertEqual(q.start_template()['upstream_anchors']['parent_technical_failure_ref'], r.PARENT_TECHNICAL_FAILURE_REF)
            self.assertIn(r.SESSION, r.WRAPPER.read_text())
            self.assertIn(str(r.LOG), r.WRAPPER.read_text())
            self.assertNotIn('sys.path', r.WRAPPER.read_text())
            self.assertEqual(r.OLD_RESULT.name, 'baseline-unified-v3-ms-seal-m128-recovery1-probe-schema-r1')

    def test_r1_authorization_cannot_authorize_r2_even_at_same_commit(self):
        old = bound(ref(r.POLICY_PACKAGE / 'start-review.json'))
        self.assertEqual(old['closure_commit'], CLOSED_PARENT)
        with attempt(), patch.object(q, 'closure', return_value=CLOSED_PARENT):
            with self.assertRaisesRegex(PermissionError, 'exact unified authorization'):
                q.validate_start(old)

    def test_future_approval_requires_exact_repair_commit_and_attempt(self):
        with attempt(), patch.object(q, 'closure', return_value=SYNTHETIC_HEAD):
            value = executable_fixture(q.start_template())
            self.assertEqual(q.validate_start(value), value)
            for mutation in ('commit', 'attempt', 'parent'):
                bad = copy.deepcopy(value)
                if mutation == 'commit':
                    bad['closure_commit'] = CLOSED_PARENT
                elif mutation == 'attempt':
                    bad['upstream_anchors']['execution_attempt'] = 'M_BASE-ModernTCN-ETTh1-numeric-r1'
                else:
                    bad['upstream_anchors']['parent_technical_failure_ref']['sha256'] = '0' * 64
                with self.assertRaises(PermissionError):
                    q.validate_start(bad)
            c = bound(r.CONFIG_REF)
            with self.assertRaises(PermissionError):
                r.validate_permit_link(c, dict(execution_attempt='M_BASE-ModernTCN-ETTh1-numeric-r1',
                                              probe_recovery_ref=r.REUSE_REF), True)
            r.validate_permit_link(c, dict(execution_attempt=r.ATTEMPT, probe_recovery_ref=r.REUSE_REF), True)

    def test_r1_failure_artifacts_remain_bound_and_unchanged(self):
        value = bound(r.PARENT_TECHNICAL_FAILURE_REF)
        self.assertEqual(value['failure_state'], 'VERIFY_IMPORT_MS203_AND_SEAL')
        self.assertFalse(value['scientific_failure'])
        self.assertEqual((value['new_probe_measurements'], value['new_formal_runs']), (0, 0))
        self.assertEqual(value['new_attempt'], r.ATTEMPT)
        for path, row in value['retained_artifacts'].items():
            self.assertEqual(sha(path), row['sha256'], path)
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()
