"""V4 contract and historical isolation; synthetic/JSON evidence only."""
import copy
import json
import subprocess
import unittest
from unittest.mock import patch
from utils import ch3_native_chain as chain
from utils import ch3_native_tasks as scope
from utils import ch3_native_execution as execution
from utils import ch3_m_tasks as source
from utils.ch3_contract import digest, profile


class ReplacementV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cs = chain.configs()

    def test_all_18_v4_rules_are_explicit_and_exact(self):
        rule = scope.replacement_numeric_rule()
        self.assertEqual(rule['id'], 'native-tmark-replacement-fullfloat-atol1e-4-v4')
        expected = dict(kind='full_float_state', state_atol=1e-4, loss_atol=1e-6,
                        metric_atol=1e-6, loss_rtol=0., rtol=0., equal_nan=False)
        rows = self.cs['tmark']['native_replacement']['numeric_policies']
        self.assertEqual(set(rows), {m+'-'+d for m in scope.MODELS for d in scope.DOMAINS})
        for key, row in rows.items():
            with self.subTest(scope=key):
                self.assertEqual(row['rule'], rule)
                self.assertEqual(row['policy_sha256'], digest(rule))
                for field, value in expected.items():
                    self.assertEqual(row['rule'][field], value)
        self.assertIn('parameters/buffers, gradients, Adam exp_avg and exp_avg_sq', rule['floating_state'])
        self.assertEqual(rule['nonfloating_and_step'], 'exact')
        self.assertEqual(rule['initialization_rng_batches'], 'exact')
        self.assertIn('shape/dtype', rule['exact_state'])
        self.assertIn('task/profile/source/data', rule['exact_state'])

    def test_retained_v2_v3_neither_refunded_nor_admission(self):
        prior = source.bound(chain.PREPARATION['prior_execution'])
        amounts = {'v2': dict(adam=192, forward=272, backward=192),
                   'v3': dict(adam=144, forward=208, backward=144)}
        for version, expected in amounts.items():
            with self.subTest(version=version):
                row = prior['executions'][version]
                self.assertEqual(row['actual'], expected)
                self.assertIs(row['budget_refund'], False)
                self.assertIs(row['admission_for_v4'], False)
                budget = source.bound(row['budget'])
                self.assertEqual(budget['actual'], expected)
                self.assertEqual(budget['reserved'], expected)
                self.assertIs(budget['refund'], False)
                for key in ('failure', 'chain_failure'):
                    self.assertTrue(source.bound(row[key]))
        self.assertEqual(prior['historical_actual'], dict(adam=336, forward=480, backward=336))
        self.assertEqual(prior['new_v4_starting_debit'], dict(adam=0, forward=0, backward=0))
        self.assertEqual(prior['v4_caps'], dict(adam=438, forward=608, backward=438))
        self.assertEqual(prior['historical_actual_plus_v4_max'], dict(adam=774, forward=1088, backward=774))

    def test_v2_and_v3_permit_complete_and_admission_rejected(self):
        c = self.cs['tmark']
        for version in ('v2', 'v3'):
            with self.subTest(version=version):
                old = source.PACKAGE / ('native-time-mark-chain-'+version)
                permit = json.loads((old/'tmark-probe-permit.json').read_text())
                with patch.object(chain, 'binding', return_value={}):
                    with self.assertRaises(ValueError):
                        chain.validate_permit(c, permit, True)
                complete = dict(purpose='native_successor_probe_complete_v1',
                                scope='m6-native-tmark-chain-'+version+'-tmark-probe',
                                plan=scope.plan(c), execution_complete=True)
                with self.assertRaises(ValueError):
                    execution.validate_probe_completion(c, complete)
                with self.assertRaises(ValueError):
                    chain.validate_admission(c, dict(complete, reviewed=True))
        self.assertEqual(scope.ID, 'm6-native-tmark-chain-v4')
        self.assertEqual(chain.SESSION, 'ch3-native-time-mark-chain-v4')
        for c in self.cs.values():
            for probe in (True, False):
                self.assertEqual(chain.permit_path(c, probe).parent, scope.PACKAGE)

    def test_profiles_M_bytes_and_policies_unchanged(self):
        def old(rel):
            return subprocess.check_output(['git', '-C', str(scope.ROOT), 'show', scope.BASE+':'+rel])
        for rel in ('configs/ch3_formal_profiles.json', 'models/ch3_adapter.py',
                    'utils/ch3_time_marks.py', 'utils/ch3_data.py', 'utils/ch3_m_tasks.py',
                    'utils/ch3_native_execution.py', 'utils/ch3_urban_confirmation.py'):
            self.assertEqual(old(rel), (scope.ROOT/rel).read_bytes(), rel)
        before = json.loads(old('configs/ch3_native_time_mark_profiles.json'))
        after = copy.deepcopy(self.cs['tmark'])
        for t in after['tasks']:
            self.assertEqual(profile(before, t), profile(after, t), t['id'])
        before['native_replacement'].pop('numeric_policies')
        after['native_replacement'].pop('numeric_policies')
        self.assertEqual(before, after)
        m = self.cs['m']
        oldm = json.loads(old('configs/ch3_formal_profiles.json'))
        self.assertEqual(m['m_experiment']['numeric_policies'], oldm['m_experiment']['numeric_policies'])
        self.assertEqual(digest(m), '831163583475c86604286db561cc309b10b31a6d4df10f3b57c0a52f6c74b3fa')
        self.assertEqual(len(scope.probe_groups(m)), 21)
        self.assertEqual(scope.context(m)['caps'], dict(adam=1512, forward=2016, backward=1512))
        self.assertEqual(scope.plan(m)['run_budget'], dict(runs=84, run_epochs=840))
        self.assertEqual(scope.plan(m)['max_optimizer_steps'], 294790)

    def test_seven_groups_preserve_actual_combinations_and_fresh_debit(self):
        c = self.cs['tmark']
        plan = scope.plan(c)
        self.assertEqual(plan['run_budget'], dict(runs=87, run_epochs=1020))
        self.assertEqual(plan['max_optimizer_steps'], 3687530)
        self.assertEqual((plan['computational_identities'], plan['representative_tasks'],
                          plan['nominal_workers'], plan['max_workers']), (7, 25, 50, 73))
        self.assertEqual([len(g['representatives']) for g in plan['groups']], [4,4,4,4,4,3,2])
        self.assertEqual([g['planned_q'] for g in plan['groups']], [4,4,4,4,4,4,2])
        parent = json.loads((source.PACKAGE/'native-time-mark-chain-v3/replacement-plan.json').read_text())
        for x, y in zip(plan['groups'], parent['groups']):
            self.assertEqual({k:v for k,v in x.items() if k!='identities'},
                             {k:v for k,v in y.items() if k!='identities'})
        self.assertEqual(plan['planned_formal_waves'], parent['planned_formal_waves'])
        self.assertEqual(plan['nominal_cost'], dict(adam=300, forward=416, backward=300))
        for c in self.cs.values():
            for probe in (True,False):
                a = chain.approval_template(c, probe)
                self.assertEqual(a['already_charged'], dict(adam=0,forward=0,backward=0))
                self.assertIs(a['execution_permitted'], False)
                self.assertIs(a['reviewed'], False)
                self.assertIsNone(a['commit'])

    def test_historical_scale_and_clarified_v3_measurements(self):
        basis = source.bound(chain.PREPARATION['policy_basis'])
        self.assertIs(source.bound(basis['reviewed_resource_report'])['reviewed'], True)
        for field, value in dict(state_atol=1e-4,loss_atol=1e-6,metric_atol=1e-6,rtol=0).items():
            self.assertEqual(basis['historical_policy'][field], value)
        observed = source.bound(chain.PREPARATION['v3_observation'])['comparison']
        self.assertEqual(observed['state_max_abs'], 2.0210838556522503e-5)
        self.assertEqual(observed['loss_max_abs'], 0.)
        self.assertTrue(all(v==0 for v in observed['evaluation_diffs']['2'].values()))
        self.assertGreater(max(observed['evaluation_diffs']['6'].values()), 0)
        self.assertLess(max(observed['evaluation_diffs']['6'].values()), 1e-6)
        self.assertIs(observed['passed'], False)  # original v3 gate stays failed
