"""One bounded synthetic short chain through the actual validation-only core."""
import copy
import json
from pathlib import Path
import unittest
import torch
from torch.utils.data import TensorDataset
from m6_ch3_supplement_v10.model import build
from m6_ch3_supplement_v10.sensitivity import fit
from utils.ch3_onecycle import configuration
from tests.moderntcn_v10_fixtures import profile


class Worker(unittest.TestCase):
    def test_short_chain(self):
        p = profile('UrbanEV', H=6)
        p['training'].update(batch=2, eval_batch=2, epochs=2, threads=1, scheduler=configuration(2, 1))
        m = build(p, 5, 2)
        x = torch.linspace(-.8, .9, 2*12*11).reshape(2, 12, 11)
        y = torch.zeros(2, 1, 1)
        sets = {k: TensorDataset(x, y) for k in ('train', 'validation')}
        out = Path('/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/tests/synthetic-worker-r2')
        identity = dict(purpose='synthetic_control', job=dict(id='synthetic-r2', G=5, R=2), analysis=m.analysis_identity,
                        execution_device=dict(device='cpu', gpu_uuid=None))
        r = fit(m, p, dict(arithmetic=dict(train_batches=1)), sets, identity, out, lambda: None, 'cpu')
        self.assertEqual(r['updates'], 2)
        self.assertEqual(r['test_calls'], 0)
        self.assertEqual(r['routes'], dict(train=2, validation=2))
        self.assertFalse(list(out.glob('*test*')))
        history = [json.loads(s) for s in (out/'history.jsonl').read_text().splitlines()]
        best = min(history, key=lambda h: h['validation']['mse'])
        self.assertEqual(r['best_epoch'], best['epoch'])
        self.assertEqual(r['validation'], best['validation'])
        with self.assertRaises(ValueError):
            fit(m, p, {}, dict(sets, test=sets['validation']), identity, out/'forbidden', lambda: None, 'cpu')
        with self.assertRaises(FileExistsError):
            fit(m, p, dict(arithmetic=dict(train_batches=1)), sets, identity, out, lambda: None, 'cpu')


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    unittest.main()
