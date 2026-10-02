"""V3 regression methods retained for the active V4 scope; zero models/updates/GPU."""
import copy
import hashlib
import json
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from utils import ch3_native_chain as chain
from utils import ch3_native_execution as execution
from utils import ch3_native_tasks as scope
from utils import ch3_m_tasks as source
from utils.ch3_contract import digest, profile


class ReplacementV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cs = chain.configs()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=scope.PACKAGE / 'fixtures')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.c = self.cs['tmark']
        self.t = next(t for t in self.c['tasks'] if t['model'] == 'TimeMixer' and t['dataset'] == 'NP')
        self.rule = scope.replacement_numeric_rule()

    def test_all_18_exact_uniform_rules(self):
        rows = self.c['native_replacement']['numeric_policies']
        self.assertEqual(set(rows), {m + '-' + d for m in scope.MODELS for d in scope.DOMAINS})
        for key, row in rows.items():
            with self.subTest(scope=key):
                self.assertEqual(row['rule'], self.rule)
                self.assertEqual(row['policy_sha256'], digest(self.rule))
                self.assertEqual(row['rule']['state_atol'], 1e-4)
                self.assertEqual(row['rule']['loss_atol'], 1e-6)
                self.assertEqual(row['rule']['metric_atol'], 1e-6)
                self.assertEqual(row['rule']['rtol'], 0.0)
                self.assertEqual(row['rule']['loss_rtol'], 0.0)
                self.assertIs(row['rule']['equal_nan'], False)

    def test_missing_None_or_wrong_rule_refused(self):
        key = 'TimeMixer-NP'
        for bad in ('missing', None, dict(self.rule, state_atol=1e-5)):
            with self.subTest(rule=bad):
                c = copy.deepcopy(self.c)
                if bad == 'missing': del c['native_replacement']['numeric_policies'][key]
                else: c['native_replacement']['numeric_policies'][key]['rule'] = bad
                with self.assertRaises(ValueError): scope.validate(c)

    def test_87_84_science_and_M_policy_bytes_unchanged(self):
        for rel in ('configs/ch3_formal_profiles.json', 'models/ch3_adapter.py', 'utils/ch3_time_marks.py', 'utils/ch3_data.py', 'utils/ch3_m_tasks.py'):
            before = subprocess.check_output(['git', '-C', str(scope.ROOT), 'show', scope.BASE + ':' + rel])
            self.assertEqual(before, (scope.ROOT / rel).read_bytes(), rel)
        old = json.loads(subprocess.check_output(['git', '-C', str(scope.ROOT), 'show', scope.BASE + ':configs/ch3_native_time_mark_profiles.json']))
        for t in self.c['tasks']:
            self.assertEqual(profile(old, t), profile(self.c, t))
        before = copy.deepcopy(old); after = copy.deepcopy(self.c)
        before['native_replacement'].pop('numeric_policies'); after['native_replacement'].pop('numeric_policies')
        self.assertEqual(before, after)
        mc = self.cs['m']
        self.assertEqual(len(mc['tasks']), 84)
        self.assertEqual(len(scope.probe_groups(mc)), 21)
        self.assertEqual(scope.context(mc)['caps'], dict(adam=1512, forward=2016, backward=1512))

    def test_fresh_seven_groups_and_independent_cost(self):
        plan = scope.plan(self.c)
        self.assertEqual((plan['computational_identities'], plan['representative_tasks']), (7, 25))
        self.assertEqual((plan['nominal_workers'], plan['max_workers']), (50, 73))
        self.assertEqual(plan['nominal_cost'], dict(adam=300, forward=416, backward=300))
        self.assertEqual(plan['caps'], dict(adam=438, forward=608, backward=438))
        groups = scope.probe_groups(self.c)
        self.assertEqual([g['planned_q'] for g in groups], [4, 4, 4, 4, 4, 4, 2])
        self.assertEqual([len(g['representatives']) for g in groups], [4, 4, 4, 4, 4, 3, 2])
        prior = source.bound(chain.PREPARATION['prior_execution'])
        self.assertEqual(prior['executions']['v2']['actual'], dict(adam=192, forward=272, backward=192))
        self.assertIs(prior['budget_refund'], False)
        self.assertEqual(prior['new_v4_starting_debit'], dict(adam=0, forward=0, backward=0))
        self.assertEqual(prior['historical_actual_plus_v4_max'], dict(adam=774, forward=1088, backward=774))

    def test_v2_permit_completion_paths_cannot_authorize_v3(self):
        self.assertEqual(scope.ID, 'm6-native-tmark-chain-v4')
        self.assertEqual(chain.SESSION, 'ch3-native-time-mark-chain-v4')
        old = source.PACKAGE / 'native-time-mark-chain-v2'
        a = json.loads((old / 'tmark-probe-permit.json').read_text())
        with patch.object(chain, 'binding', return_value={}):
            with self.assertRaises(ValueError): chain.validate_permit(self.c, a, True)
        fake = dict(purpose='native_successor_probe_complete_v1', scope='m6-native-tmark-chain-v2-tmark-probe', plan=scope.plan(self.c), execution_complete=True)
        with self.assertRaises(ValueError): execution.validate_probe_completion(self.c, fake)
        self.assertNotEqual(scope.context(self.c)['probe_root'].parent, old)
        self.assertNotEqual(chain.LOG.parent, old)

    def payload(self, name, value=0.0, changed='model.parameter', exact=0, nonfinite=False):
        paths = ('model.parameter', 'model.buffer', 'gradient.parameter', 'optimizer.exp_avg', 'optimizer.exp_avg_sq')
        entries = []; raw = bytearray()
        for key in paths:
            v = float('nan') if nonfinite and key == changed else value if key == changed else 0.0
            entries.append(dict(path=key, offset=len(raw), nbytes=8, numel=1, dtype='torch.float64', shape=[1], mode='bounded'))
            raw.extend(struct.pack('<d', v))
        for key in ('optimizer.step', 'model.nonfloating', 'optimizer.nonfloating', 'optimizer.param_groups'):
            entries.append(dict(path=key, offset=len(raw), nbytes=8, numel=1, dtype='torch.int64', shape=[1], mode='exact'))
            raw.extend(struct.pack('<q', exact if key == changed else 0))
        schema = self.root / (name + '.schema.json'); data = self.root / (name + '.raw')
        schema.write_text(json.dumps(dict(schema='explicit-synthetic-full-state', entries=entries), sort_keys=True))
        data.write_bytes(raw)
        return dict(schema_file=str(schema), schema_sha=source.sha(schema), data_file=str(data), data_sha=source.sha(data), bytes=len(raw))

    def traces(self, **kwargs):
        left = self.payload('serial'); right = self.payload('parallel', **kwargs)
        def trace(row):
            v = dict(id=self.t['id'], profile_sha=digest(profile(self.c, self.t)), initial='fixed-init', initial_rng='fixed-rng', batch_ids=list(range(6)), validation_tail='fixed-tail', steps=6, final_rng='fixed-final-rng', finite=True, final='final-summary', numeric_policy_sha=digest(self.rule), trajectory=[dict(step=i, loss=1.0, state='summary', optimizer='summary') for i in range(1, 7)], validation=dict(mse=1.0, mae=1.0, sse=2.0, sae=2.0, elements=2))
            v['full_numeric_trace'] = [dict(row, step=i) for i in range(1, 7)]
            v['M_full_state_trace'] = copy.deepcopy(v['full_numeric_trace'])
            return v
        return trace(left), trace(right)

    def compare(self, x, y):
        ctx = dict(scope.context(self.c), probe_root=self.root)
        with patch.object(scope, 'context', return_value=ctx): return execution.compare(self.c, self.t, x, y)

    def test_each_floating_category_at_limit_and_over_limit(self):
        for key in ('model.parameter', 'model.buffer', 'gradient.parameter', 'optimizer.exp_avg', 'optimizer.exp_avg_sq'):
            with self.subTest(tensor=key):
                x, y = self.traces(value=1e-4, changed=key)
                self.assertTrue(self.compare(x, y)['passed'])
                x, y = self.traces(value=1.0001e-4, changed=key)
                self.assertFalse(self.compare(x, y)['passed'])

    def test_loss_and_normalized_metric_over_limit(self):
        x, y = self.traces()
        y['trajectory'][0]['loss'] += 2e-6
        self.assertFalse(self.compare(x, y)['passed'])
        x, y = self.traces()
        y['validation'].update(mse=1.000002, sse=2.000004)
        self.assertFalse(self.compare(x, y)['passed'])

    def test_exact_identity_step_nonfloating_and_groups(self):
        for key in ('initial', 'initial_rng', 'batch_ids', 'validation_tail', 'final_rng', 'profile_sha'):
            with self.subTest(identity=key):
                x, y = self.traces(); y[key] = 'wrong'
                with self.assertRaises(ValueError): self.compare(x, y)
        for key in ('optimizer.step', 'model.nonfloating', 'optimizer.nonfloating', 'optimizer.param_groups'):
            with self.subTest(exact=key):
                x, y = self.traces(changed=key, exact=1)
                self.assertFalse(self.compare(x, y)['passed'])

    def test_schema_dtype_shape_nonfinite_and_artifact_SHA(self):
        for kind in ('shape', 'dtype', 'sha', 'finite'):
            with self.subTest(kind=kind):
                x, y = self.traces(nonfinite=kind == 'finite')
                if kind == 'sha': Path(y['full_numeric_trace'][0]['data_file']).write_bytes(b'tampered')
                elif kind != 'finite':
                    row = y['full_numeric_trace'][0]; p = Path(row['schema_file']); s = json.loads(p.read_text())
                    s['entries'][0][kind] = [2] if kind == 'shape' else 'torch.float32'
                    p.write_text(json.dumps(s)); new_sha = source.sha(p)
                    for points in (y['full_numeric_trace'], y['M_full_state_trace']):
                        for point in points: point['schema_sha'] = new_sha
                    self.assertFalse(self.compare(x, y)['passed']); continue
                with self.assertRaises(ValueError): self.compare(x, y)

    def test_TimeMixer_Urban_new_policy_and_old_policy_retained(self):
        from utils import ch3_urban_confirmation as urban
        t = next(t for t in self.c['tasks'] if t['model'] == 'TimeMixer' and t['dataset'] == 'UrbanEV')
        self.assertEqual(urban.policy(self.c, t), self.rule)
        self.assertEqual(urban.POLICY['state_atol'], 1e-4)
        self.assertEqual(scope.worker_counts(self.c, t), dict(adam=6, forward=10, backward=6))
        self.assertEqual(scope.worker_counts(self.c, self.t), dict(adam=6, forward=8, backward=6))
        self.assertEqual(scope.worker_counts(self.cs['m'], self.cs['m']['tasks'][0]), dict(adam=6, forward=8, backward=6))

    def test_TimeMixer_endpoint_replays_new_bound_not_historical_bound(self):
        from utils import ch3_urban_confirmation as urban
        from tests.test_m6_urban_confirmation import UrbanConfirmationTests
        t = next(t for t in self.c['tasks'] if t['model'] == 'TimeMixer' and t['dataset'] == 'UrbanEV')
        fixture = UrbanConfirmationTests('test_comparison_at_bound')
        fixture.c = self.c; fixture.spec = dict(task_ids=[t['id']])
        t, x, y, detail = fixture.pair()
        for trace in (x, y): trace['urban_confirmation']['policy_sha'] = digest(self.rule)
        detail['points'][1]['tensors'][0].update(max_abs_diff=1e-4, byte_equal=False)
        self.assertTrue(urban.compare_measured(self.c, t, x, y, detail)['passed'])
        detail['points'][1]['tensors'][0]['max_abs_diff'] = 1.0001e-4
        self.assertFalse(urban.compare_measured(self.c, t, x, y, detail)['passed'])
        detail['points'][1]['tensors'][0]['max_abs_diff'] = 0.0
        y['urban_confirmation']['evaluations']['6']['metrics']['mse'] += 2e-6
        self.assertFalse(urban.compare_measured(self.c, t, x, y, detail)['passed'])


class ReplacementFallbackV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.cs = chain.configs()

    def fixture(self, failures=None, numeric=True, index=0):
        from tests.test_m6_native_chain import SavedProbeTests
        tmp = tempfile.TemporaryDirectory(dir=scope.PACKAGE / 'fixtures')
        self.addCleanup(tmp.cleanup)
        fixture = SimpleNamespace(root=Path(tmp.name))
        fixture.c = copy.deepcopy(self.cs['tmark'])
        fixture.ctx = dict(scope.context(fixture.c), probe_root=fixture.root / 'probe', control=fixture.root / 'formal')
        fixture.ctx['probe_root'].mkdir()
        fixture.a = dict(commit='synthetic', protocol_sha=digest(fixture.c), code={}, environment={}, hardware={'cpu_affinity': []})
        (fixture.ctx['probe_root'] / 'approval.json').write_text(json.dumps(fixture.a))
        group = scope.probe_groups(fixture.c)[index]
        stack, attempts = SavedProbeTests.engine(fixture, failures)
        full_plan = scope.plan(fixture.c)
        stack.enter_context(patch.object(scope, 'plan', return_value=full_plan))
        stack.enter_context(patch.object(scope, 'probe_groups', return_value=[group]))
        if not numeric:
            stack.enter_context(patch.object(execution, 'compare', side_effect=lambda c, t, x, y: dict(passed=x is y)))
        return fixture, stack, attempts

    def test_new_scope_starts_every_reference_then_actual_q4(self):
        fixture, stack, attempts = self.fixture()
        with stack:
            r = execution.run_probe(fixture.c, fixture.a)
            self.assertEqual(attempts, ['serial'] * 4 + ['q4'])
            self.assertEqual(r['scope'], 'm6-native-tmark-chain-v4-tmark-probe')
            self.assertEqual(r['budget']['actual'], dict(adam=48, forward=64, backward=48))
            self.assertIs(r['budget']['refund'], False)

    def test_numeric_q4_stops_no_q2_and_cost_kept(self):
        fixture, stack, attempts = self.fixture(numeric=False)
        with stack:
            with self.assertRaises(ValueError): execution.run_probe(fixture.c, fixture.a)
            self.assertEqual(attempts, ['serial'] * 4 + ['q4'])
            failed = json.loads((fixture.ctx['probe_root'] / 'failure.json').read_text())
            self.assertEqual(failed['budget']['actual'], dict(adam=48, forward=64, backward=48))
            self.assertIs(failed['budget']['refund'], False)
            self.assertFalse((fixture.ctx['probe_root'] / 'complete.json').exists())

    def test_resource_q4_then_q2_retains_failed_cost(self):
        fixture, stack, attempts = self.fixture({'q4': 'resource'})
        with stack:
            r = execution.run_probe(fixture.c, fixture.a)
            self.assertEqual(attempts, ['serial'] * 4 + ['q4', 'q2', 'q2'])
            self.assertEqual(next(iter(r['decisions'].values()))['concurrency'], 2)
            self.assertEqual(r['budget']['actual'], dict(adam=72, forward=96, backward=72))

    def test_planned_TimeXer_q2_resource_only_legal_q1(self):
        fixture, stack, attempts = self.fixture({'q2': 'resource'}, index=6)
        with stack:
            r = execution.run_probe(fixture.c, fixture.a)
            self.assertEqual(attempts, ['serial', 'serial', 'q2'])
            self.assertEqual(next(iter(r['decisions'].values()))['concurrency'], 1)
            self.assertEqual(r['budget']['actual'], dict(adam=24, forward=32, backward=24))

    def test_business_failure_cannot_fallback(self):
        fixture, stack, attempts = self.fixture({'q4': 'business'})
        with stack:
            with self.assertRaises(RuntimeError): execution.run_probe(fixture.c, fixture.a)
            self.assertEqual(attempts, ['serial'] * 4 + ['q4'])


if __name__ == '__main__': unittest.main()
