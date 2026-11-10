"""Synthetic producer state only; no original trained state is read."""
import json
from pathlib import Path
import unittest
import torch
from moderntcn_dev_v10 import adapter
from m6_ch3_supplement_v10.readonly import restore
from m6_ch3_supplement_v10.contract import sha
from tests.moderntcn_v10_fixtures import profile, equal, open_branches


class Restore(unittest.TestCase):
    def test_synthetic_producer(self):
        p = profile('UrbanEV', H=6)
        a = adapter.build(p, use_u=True, use_v=True, fold=6)
        open_branches(a)
        a.eval()
        target = Path('/public/home/yueweiting/大论文/amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-preparation-r1/tests/readonly-synthetic.pt')
        if target.exists():
            raise FileExistsError(target)
        identity = dict(candidate_id='moderntcn-uv-dev-v10', purpose='synthetic_control')
        torch.save(dict(identity=identity, model=a.state_dict()), target)
        row = dict(kind='synthetic_control', identity=identity, profile=p, task=dict(fold=6), weights_ref=dict(path=str(target), sha256=sha(target)))
        b, predict = restore(row, 'cpu')
        equal(self, a.state_dict(), b.state_dict())
        x = torch.linspace(-.8, .9, 2*12*11).reshape(2, 12, 11)
        before = torch.get_rng_state().clone()
        with torch.inference_mode():
            equal(self, adapter.prediction(a, x), predict(x, None))
        equal(self, before, torch.get_rng_state())
        with self.assertRaises(ValueError):
            restore(dict(row, weights_ref=dict(path=str(target), sha256='wrong')), 'cpu')


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    unittest.main()
