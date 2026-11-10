"""Only synthetic tensors; no observation loader or saved real state access."""
import copy
import unittest
import torch
from m6_ch3_supplement_v10.model import build, validate_state
from moderntcn_dev_v10 import adapter
from tests.moderntcn_v10_fixtures import profile, equal, open_branches
from utils.ch3_onecycle import Schedule, configuration


class ModelTests(unittest.TestCase):
    def test_default_exact(self):
        for dataset, horizon, fold in [('UrbanEV', 6, 6), ('PJM', 24, 1)]:
            p = profile(dataset, H=horizon)
            a = adapter.build(p, use_u=True, use_v=True, task='MS', fold=fold)
            b = build(p, fold=fold)
            open_branches(a)
            open_branches(b)
            equal(self, a.state_dict(), b.state_dict())
            x = torch.linspace(-.8, .9, 2*p['T']*p['C']).reshape(2, p['T'], p['C'])
            optimizers = [torch.optim.Adam(m.parameters(), lr=.01, weight_decay=1e-7) for m in (a, b)]
            schedulers = [Schedule(o, configuration(20, 1)) for o in optimizers]
            for step in range(2):
                initial_rng = torch.get_rng_state().clone()
                results = []
                for m, o, sched in zip((a, b), optimizers, schedulers):
                    torch.set_rng_state(initial_rng)
                    m.train()
                    o.zero_grad(set_to_none=True)
                    y = adapter.prediction(m, x)
                    adapter.loss(y, torch.zeros_like(y)).backward()
                    results.append((y.detach().clone(), {n: None if v.grad is None else v.grad.clone() for n, v in m.named_parameters()}, torch.get_rng_state().clone()))
                    o.step()
                    sched.after_successful_update()
                equal(self, results[0], results[1])
                equal(self, a.state_dict(), b.state_dict())
                equal(self, optimizers[0].state_dict(), optimizers[1].state_dict())
                equal(self, schedulers[0].state_dict(), schedulers[1].state_dict())
            with torch.inference_mode():
                a.eval()
                b.eval()
                equal(self, adapter.prediction(a, x), adapter.prediction(b, x))

    def test_variants(self):
        for dataset, horizon, fold in [('UrbanEV', 6, 6), ('PJM', 24, 1)]:
            p = profile(dataset, H=horizon)
            reference = build(p, fold=fold)
            baseline = copy.deepcopy(reference.state_dict())
            expected_rng = torch.get_rng_state().clone()
            for G, R in [(3, 4), (7, 4), (5, 2), (5, 6)]:
                m = build(p, G, R, fold=fold)
                equal(self, expected_rng, torch.get_rng_state())
                for k, v in m.state_dict().items():
                    if not k.startswith(('u.', 'v.')):
                        equal(self, v, baseline[k])
                self.assertEqual(m.u.coef.shape[-1], G+3)
                self.assertEqual(m.v.T.shape[0], R)
                torch.testing.assert_close(m.v.T @ m.v.T.T, torch.eye(R), atol=2e-6, rtol=2e-5)
                q = torch.linspace(-8, 8, 2*m.u.I).reshape(2, m.u.I)
                basis = m.u.basis(q)
                torch.testing.assert_close(basis.sum(-1), torch.ones_like(q), atol=2e-6, rtol=2e-5)
                self.assertTrue(bool((basis >= 0).all()))
                open_branches(m)
                x = torch.linspace(-.8, .9, 2*p['T']*p['C']).reshape(2, p['T'], p['C'])
                y = adapter.prediction(m, x)
                adapter.loss(y, torch.zeros_like(y)).backward()
                for name in ('u.coef', 'v.A', 'v.W2.weight'):
                    self.assertGreater(float(dict(m.named_parameters())[name].grad.abs().sum()), 0)
                state = copy.deepcopy(m.state_dict())
                validate_state(m, state, m.analysis_identity)
                state['u.knots'][0] += .1
                with self.assertRaises(ValueError):
                    validate_state(m, state, m.analysis_identity)
                with self.assertRaises(ValueError):
                    validate_state(m, m.state_dict(), {'G': G, 'R': R})


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    unittest.main()
