"""CPU/control fixtures only: no real checkpoint, data, GPU, or launch."""
import ast
import copy
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from m6_ch3_supplement_v10.observations import select_nodes, case_times, PositionErrors
from m6_ch3_supplement_v10.benchmark import measure
from m6_ch3_supplement_v10.contract import jobs, validate_permission, write_new, sha, SCOPE


class Controls(unittest.TestCase):
    def test_stage_completion(self):
        from m6_ch3_supplement_v10.queue import validate_result
        result = dict(warmup_calls=50, timed_calls=500, batch=1,
                      groups=[dict(group=i, calls=100, wall_seconds=1) for i in range(5)])
        job = dict(phase='E', batch=1)
        validate_result({}, job, result)
        bad = copy.deepcopy(result)
        bad['groups'][4]['group'] = 3
        with self.assertRaises(ValueError):
            validate_result({}, job, bad)
        with self.assertRaises(ValueError):
            validate_result({}, job, dict(result, timed_calls=400))
        config = dict(sensitivity_prototypes=dict(UrbanEV=dict(arithmetic=dict(train_batches=1, eval_batches=dict(validation=1)))))
        result = dict(epochs=2, updates=2, test_calls=0, status='validation_only_completed', routes=dict(train=2, validation=2), best_epoch=1, validation=dict(mse=.1, mae=.2))
        validate_result(config, dict(phase='S', dataset='UrbanEV'), result)
        with self.assertRaises(ValueError):
            validate_result(config, dict(phase='S', dataset='UrbanEV'), dict(result, test_calls=1))

    def test_measurement_process_boundary(self):
        import os
        import subprocess
        from unittest.mock import patch
        from m6_ch3_supplement_v10.readonly import exclusive_measurement_gpu
        with patch('subprocess.check_output', return_value=f'{os.getpid()}, GPU-fixture\n') as call:
            self.assertEqual(exclusive_measurement_gpu('GPU-fixture'), [os.getpid()])
            self.assertEqual(call.call_args.kwargs['timeout'], 60)
        with patch('subprocess.check_output', return_value='999999, GPU-fixture\n'):
            with self.assertRaises(RuntimeError):
                exclusive_measurement_gpu('GPU-fixture')
        with patch('subprocess.check_output', side_effect=subprocess.TimeoutExpired('fixture', 60)) as call:
            with self.assertRaises(subprocess.TimeoutExpired):
                exclusive_measurement_gpu('GPU-fixture')
            self.assertEqual(call.call_count, 3)
        with patch('subprocess.check_output', side_effect=subprocess.CalledProcessError(1, 'fixture')) as call:
            with self.assertRaises(subprocess.CalledProcessError):
                exclusive_measurement_gpu('GPU-fixture')
            self.assertEqual(call.call_count, 1)

    def test_owned_stop(self):
        import os
        from m6_ch3_supplement_v10.queue import owned_stop, proc_start
        root = Path('/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/tests/owned-control-fixture')
        root.mkdir(exist_ok=False)
        config = root/'synthetic-config.json'
        write_new(config, dict(result_root=str(root.parent), attempt=root.name, purpose='synthetic_control'))
        runtime = root/'runtime.json'
        write_new(runtime, dict(scope=SCOPE, synthetic=True, config_ref=dict(path=str(config), sha256=sha(config))))
        write_new(root/'owner.json', dict(pid=os.getpid(), start=proc_start(os.getpid()), runtime_sha256=sha(runtime), synthetic=True))
        owned_stop(root)
        self.assertTrue((root/'STOP').exists())
        owned_stop(root)

    def test_clock_and_counts(self):
        calls, sync = [], []
        ticks = iter(range(10))
        r = measure(lambda: calls.append(1), lambda: sync.append(1), lambda: None, lambda: {}, 16, clock=lambda: next(ticks))
        self.assertEqual(len(calls), 550)
        self.assertEqual(len(sync), 11)
        self.assertEqual(len(r['groups']), 5)
        self.assertEqual(r['median_batch_ms'], 10)
        self.assertEqual(r['median_sample_ms'], 10/16)

    def test_case_selection_and_clock(self):
        x = np.arange(200)[:, None] % 9 * np.arange(1, 13)[None, :] + 100
        a = select_nodes(x, list(range(12)))
        self.assertEqual(a, select_nodes(x, list(range(12))))
        self.assertEqual(len(set(a['indices'])), 2)
        t = case_times(pd.date_range('2023-01-01', periods=400, freq='h'))
        self.assertEqual(t['target_times'][0], '2023-01-02 00:00:00')
        self.assertEqual(pd.Timestamp(t['target_times'][0])-pd.Timestamp(t['origins'][0]), pd.Timedelta(hours=6))
        fallback = case_times(pd.date_range('2023-01-03', periods=168, freq='h'))
        self.assertIn('earliest', fallback['reason'])
        with self.assertRaises(ValueError):
            case_times(pd.date_range('2023-01-03', periods=167, freq='h'))

    def test_position_aggregation(self):
        x = np.arange(3*24, dtype=np.float32).reshape(3, 24, 1)/100
        m = PositionErrors(24)
        m.add(x[:1], np.zeros_like(x[:1]))
        m.add(x[1:], np.zeros_like(x[1:]))
        r = m.result()
        self.assertAlmostEqual(r['overall_mse'], float(np.square(x.astype(float)).mean()))
        self.assertAlmostEqual(r['overall_mae'], float(np.abs(x.astype(float)).mean()))
        with self.assertRaises(ValueError):
            m.add(x, np.zeros((3, 1, 1)))

    def test_scope_and_nonexecution(self):
        tasks = jobs()
        self.assertEqual(len({x['id'] for x in tasks}), len(tasks))
        self.assertEqual(sum(x['phase'] == 'E' for x in tasks), 32)
        self.assertEqual(sum(x['phase'] == 'S' for x in tasks), 8)
        with self.assertRaises(PermissionError):
            validate_permission('/does/not/exist', dict(review_approved=False, commit=None))
        with self.assertRaises(PermissionError):
            validate_permission('/does/not/exist', dict(schema='moderntcn-uv-launch-v2', review_approved=True, commit='old'))

    def test_validation_only_boundary(self):
        path = Path(__file__).parents[1]/'m6_ch3_supplement_v10/sensitivity.py'
        tree = ast.parse(path.read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        loaders = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'load']
        self.assertEqual(len(loaders), 1)
        self.assertFalse(any(k.arg == 'test_capability' for n in loaders for k in n.keywords))
        self.assertNotIn('test-claim', path.read_text())
        text = path.read_text()
        self.assertLess(text.index('loss_guard(loss)'), text.index('loss.backward()'))
        self.assertLess(text.index('gradients_collected(named)'), text.index('opt.step()'))
        self.assertLess(text.index('parameters_collected(named)'), text.index('scheduler.after_successful_update()'))


if __name__ == '__main__':
    unittest.main()
