"""THLS engineering contracts; only newly generated synthetic tensors."""
import copy
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

from models.modules.target_history_local_shape_residual import TargetHistoryLocalShapeResidual


def thls_precisions():
    # Required CUDA is deliberate: an external preflight must mark it Pending
    # when unavailable, rather than silently dropping it from this list.
    return (("cpu", torch.float32), ("cpu", torch.float64), ("cuda", torch.float32))


class TargetHistoryLocalShapeResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make(self, length, device, dtype):
        return TargetHistoryLocalShapeResidual(
            length, 3 if length == 12 else 5, 7 if length == 12 else 31
        ).to(device=device, dtype=dtype)

    def close(self, x, y):
        torch.testing.assert_close(x, y, rtol=0,
            atol=1e-12 if x.dtype == torch.float64 else 1e-6)

    def test_signed_features_boundaries_reference_and_gradient(self):
        cases = 0
        for device, dtype in thls_precisions():
            for length in (12, 512):
                model = self.make(length, device, dtype)
                t = torch.linspace(0, 1, length, device=device, dtype=dtype)
                for y in (torch.full_like(t, .25), t, (t > .5).to(dtype),
                          1e-10*torch.sin(t*8), .25+1e-6*torch.sin(t*8)):
                    y = torch.stack((y, -2*y))
                    reference = torch.zeros(2, 3, length, device=device, dtype=dtype)
                    reference[:, 0] = y
                    for k in range(1, length):
                        reference[:, 1, k] = y[:, k]-y[:, k-1]
                    for k in range(2, length):
                        reference[:, 2, k] = y[:, k]-2*y[:, k-1]+y[:, k-2]
                    features = model.compute_features(y)
                    self.close(features, reference)
                    self.assertTrue(torch.isfinite(features).all())
                    self.assertEqual(int(torch.count_nonzero(features[:, 1, :1])), 0)
                    self.assertEqual(int(torch.count_nonzero(features[:, 2, :2])), 0)
                    cases += 1
                y = torch.randn(2, length, device=device, dtype=dtype, requires_grad=True)
                gradient = torch.autograd.grad(model.compute_features(y)[0, 2, 2], y)[0]
                expected = torch.zeros_like(y); expected[0, :3] = y.new_tensor([1, -2, 1])
                self.assertTrue(torch.equal(gradient, expected))
        self.assertEqual(cases, 30)

    def test_reference_residual_and_components_single_dropout(self):
        for device, dtype in thls_precisions():
            model = self.make(12, device, dtype).train()
            y = torch.randn(2, 12, device=device, dtype=dtype)
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state() if device == "cuda" else None
            with mock.patch.object(model.dropout1, "forward", wraps=model.dropout1.forward) as d1, \
                    mock.patch.object(model.dropout2, "forward", wraps=model.dropout2.forward) as d2:
                parts = model.compute_components(y)
                self.assertEqual(d1.call_count, 1); self.assertEqual(d2.call_count, 1)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None: torch.cuda.set_rng_state(cuda_rng)
            output = model(y)
            self.assertTrue(torch.equal(output, parts["residual"]))
            self.assertTrue(torch.equal(output, model.eta*parts["delta"]))
            self.assertEqual(tuple(output.shape), (2, 1, 12))
            model.eval()
            features = model.compute_features(y)
            v = F.conv1d(features, model.input_projection.weight, model.input_projection.bias)
            v = F.conv1d(v, model.temporal_conv.large_branch.weight,
                         model.temporal_conv.large_branch.bias, padding=3, groups=8) + \
                F.conv1d(F.conv1d(features, model.input_projection.weight, model.input_projection.bias),
                         model.temporal_conv.small_branch.weight, model.temporal_conv.small_branch.bias,
                         padding=1, groups=8)
            v = F.layer_norm(v.transpose(1, 2), (8,), model.feature_norm.weight,
                             model.feature_norm.bias, 1e-5).transpose(1, 2)
            v = F.gelu(F.conv1d(v, model.ffn_expand.weight, model.ffn_expand.bias), approximate="none")
            v = F.conv1d(v, model.ffn_reduce.weight, model.ffn_reduce.bias)
            reference = model.eta*F.conv1d(v, model.output_projection.weight, model.output_projection.bias)
            self.close(model(y), reference)

    def test_nontrivial_sample_isolation_permutation_and_task_gradients(self):
        for device, dtype in thls_precisions():
            for length in (12, 512):
                model = self.make(length, device, dtype).eval()
                y = torch.randn(3, length, device=device, dtype=dtype, requires_grad=True)
                out = model(y)
                changed = y.detach().clone(); changed[0, length//2] += .7
                perturbed = model(changed)
                self.assertTrue(torch.equal(out[1:], perturbed[1:]))
                self.assertGreater(float((out[0]-perturbed[0]).abs().max()), 0)
                permutation = torch.tensor([2, 0, 1], device=device)
                self.close(model(y[permutation]), out[permutation])
                target = torch.sin(torch.arange(length, device=device, dtype=dtype)/3).expand(3, 1, -1)
                ((y[:, None, :]+out-target)**2).mean().backward()
                self.assertTrue(torch.isfinite(y.grad).all())
                groups = {}
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    groups.setdefault(name.split('.')[0], 0.)
                    groups[name.split('.')[0]] += float(parameter.grad.abs().sum())
                self.assertTrue(all(value > 0 for value in groups.values()), groups)
                print("THLS_TASK_GRADIENT", device, str(dtype), length, groups, flush=True)

    def test_parameter_counts_mac_and_nontrivial_deployment(self):
        for device, dtype in thls_precisions():
            for length, train_count, deploy_count, train_mac, deploy_mac in (
                    (12, 434, 402, 368, 344), (512, 642, 594, 576, 536)):
                model = self.make(length, device, dtype).train()
                with torch.no_grad(): model.eta.fill_(.013)
                original = copy.deepcopy(model.state_dict())
                deploy = model.to_deploy()
                self.assertTrue(model.training); self.assertFalse(deploy.training)
                self.assertFalse(model.deploy); self.assertTrue(deploy.deploy)
                self.assertEqual(sum(p.numel() for p in model.parameters()), train_count)
                self.assertEqual(sum(p.numel() for p in deploy.parameters()), deploy_count)
                for key in original: self.assertTrue(torch.equal(original[key], model.state_dict()[key]))
                self.assertEqual(3*8+8*(3+7 if length == 12 else 5+31)+8*16+16*8+8, train_mac)
                self.assertEqual(train_mac-8*(3 if length == 12 else 5), deploy_mac)
                model.eval(); y = torch.randn(2, length, device=device, dtype=dtype)
                before, after = model(y), deploy(y)
                self.close(before, after)
                self.assertGreater(float(before.abs().max()), 0)
                self.assertEqual(after.dtype, dtype); self.assertEqual(after.device.type, device)
                keys = set(deploy.state_dict()); deploy.switch_to_deploy()
                self.assertEqual(set(deploy.state_dict()), keys)
                self.assertTrue(any("reparam_branch" in key for key in keys))
                self.assertFalse(any("large_branch" in key or "small_branch" in key for key in keys))
                print("THLS_DEPLOY", device, str(dtype), length, train_count, deploy_count,
                      "max_abs", float((before-after).abs().max()), flush=True)

    def test_strict_restore_rejects_forms_keys_shapes_dtypes_and_nonfinite_atomically(self):
        model = self.make(12, "cpu", torch.float32)
        original = copy.deepcopy(model.state_dict())
        for kind in ("key", "shape", "dtype", "finite", "deploy"):
            state = copy.deepcopy(original); key = "input_projection.weight"
            if kind == "key": state.pop(key)
            elif kind == "shape": state[key] = state[key][:1]
            elif kind == "dtype": state[key] = state[key].double()
            elif kind == "finite": state[key][0, 0, 0] = float('nan')
            else: state = model.to_deploy().state_dict()
            with self.assertRaises(RuntimeError): model.load_state_dict(state)
            for key in original: self.assertTrue(torch.equal(original[key], model.state_dict()[key]))
        with self.assertRaises(ValueError): model.load_state_dict(original, strict=False)
        model.load_state_dict(original)

    def test_input_kernel_guards_and_initialization(self):
        for args in ((2, 1, 3), (12, 4, 7), (12, 3, 13), (12, 7, 3)):
            with self.assertRaises(ValueError): TargetHistoryLocalShapeResidual(*args)
        model = self.make(12, "cpu", torch.float32)
        self.assertAlmostEqual(float(model.eta), 1e-3, places=9)
        self.assertGreater(int(torch.count_nonzero(model.output_projection.weight)), 0)
        for name, value in model.named_parameters():
            if name.endswith('bias'): self.assertEqual(int(torch.count_nonzero(value)), 0)
        self.assertTrue(torch.equal(model.feature_norm.weight, torch.ones(8)))
        for bad in (torch.zeros(2, 1, 12), torch.empty(0, 12), torch.zeros(2, 11)):
            with self.assertRaises(ValueError): model.compute_features(bad)
        with self.assertRaises(TypeError): model.compute_features(torch.ones(2, 12, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
