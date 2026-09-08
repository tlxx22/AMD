"""Locked P2 mathematics and deployment checks using synthetic tensors only."""
import copy
import unittest
from unittest import mock

import numpy as np
import torch

from models.modules.local_change_gated_pmcr import LocalChangeGate, LocalChangeGatedPMCR
from models.modules.modern_conv_refinement import PeakPreservingModernConvRefinement


def precisions():
    result = [("cpu", torch.float32), ("cpu", torch.float64)]
    if torch.cuda.is_available():
        result.append(("cuda", torch.float32))
    return result


def inputs(length, dtype, device):
    u = torch.linspace(0, 1, length, dtype=dtype, device=device)
    factor = torch.tensor([[1, .7, 1.3], [.9, 1.2, .8]], dtype=dtype, device=device)[..., None]
    offset = torch.tensor([[.15, -.05, .3], [-.2, .1, .25]], dtype=dtype, device=device)[..., None]
    wave = torch.sin(2 * torch.pi * u) + .1 * u
    return [(.25 * torch.ones_like(u) * factor + offset),
            u * factor + offset, (u >= .5).to(dtype) * factor + offset,
            1e-10 * wave * factor, .25 + 1e-6 * wave * factor]


def perturb_gate(gate):
    with torch.no_grad():
        gate.conv2.weight.copy_(torch.tensor([.20, -.10, .15, .05],
                                            dtype=gate.conv2.weight.dtype,
                                            device=gate.conv2.weight.device).reshape(1, 4, 1))
        gate.conv2.bias.fill_(.03)


def rng_state(device):
    return (torch.get_rng_state().clone(),
            torch.cuda.get_rng_state(device).clone() if device == "cuda" else None)


def restore_rng(state, device):
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state(state[1], device)


class LocalChangeGatedPMCRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def close(self, a, b, dtype):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-12 if dtype == torch.float64 else 1e-6)

    def make(self, length, dtype, device):
        return LocalChangeGatedPMCR(8, 3 if length == 12 else 5,
                                   7 if length == 12 else 31).to(device=device, dtype=dtype)

    def test_reference_features_boundaries_and_non_detached_mean(self):
        for device, dtype in precisions():
            for length in (12, 512):
                with self.subTest(device=device, dtype=dtype, length=length):
                    cases = inputs(length, dtype, device)
                    cases += [torch.zeros(2, 3, length, dtype=dtype, device=device),
                              torch.arange(length, dtype=dtype, device=device).expand(2, 3, -1)]
                    for hidden in cases:
                        # Independent per-series NumPy recurrence, outside autograd.
                        x = hidden.cpu().numpy()
                        d1, d2 = np.zeros_like(x), np.zeros_like(x)
                        for t in range(1, length):
                            d1[..., t] = x[..., t] - x[..., t-1]
                        for t in range(2, length):
                            d2[..., t] = d1[..., t] - d1[..., t-1]
                        a = np.stack((np.abs(d1), np.abs(d2)), axis=2)
                        reference = a / (a + np.mean(a, axis=-1, keepdims=True) + 1e-6)
                        feature = LocalChangeGate.compute_features(hidden)
                        self.close(feature, torch.as_tensor(reference, device=device), dtype)
                        self.assertTrue(torch.isfinite(feature).all())
                        self.assertTrue(((feature >= 0) & (feature <= 1)).all())
                        self.assertEqual(torch.count_nonzero(feature[:, :, 0, :1]), 0)
                        self.assertEqual(torch.count_nonzero(feature[:, :, 1, :2]), 0)
                    ramp = torch.arange(length, device=device, dtype=dtype).reshape(1, 1, -1).requires_grad_()
                    f = LocalChangeGate.compute_features(ramp)
                    grad = torch.autograd.grad(f[0, 0, 0, 1], ramp)[0]
                    exact = -1 / (length * (1 + (length - 1) / length + 1e-6) ** 2)
                    self.close(grad[0, 0, -1], grad.new_tensor(exact), dtype)
                    self.assertNotEqual(float(grad[0, 0, -1]), 0)
        gate = LocalChangeGate()
        for bad in (torch.ones(1, 1, 2), torch.ones(1, 3), torch.empty(0, 1, 12)):
            with self.assertRaises(ValueError):
                gate(bad)
        with self.assertRaises(TypeError):
            gate(torch.ones(1, 1, 12, dtype=torch.long))

    def test_initial_output_input_and_body_gradient_equivalence(self):
        for device, dtype in precisions():
            maxima = [0., 0., 0.]
            pairs = 0
            for length in (12, 512):
                p2 = self.make(length, dtype, device)
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(2024)
                    v1 = PeakPreservingModernConvRefinement(
                        8, 3 if length == 12 else 5, 7 if length == 12 else 31).to(device=device, dtype=dtype)
                for key, value in v1.state_dict().items():
                    self.assertTrue(torch.equal(value, p2.body.state_dict()[key]), key)
                q = (.2 + .3 * torch.cos(3 * torch.pi * torch.linspace(
                    0, 1, length, device=device, dtype=dtype))).expand(2, 3, -1)
                for training in (False, True):
                    v1.train(training); p2.train(training)
                    for case in inputs(length, dtype, device):
                        x1, x2 = case.clone().requires_grad_(), case.clone().requires_grad_()
                        v1.zero_grad(set_to_none=True); p2.zero_grad(set_to_none=True)
                        state = rng_state(device)
                        y1 = v1(x1)
                        rng1 = rng_state(device)
                        restore_rng(state, device)
                        parts = p2.compute_components(x2)
                        y2 = parts["output"]
                        rng2 = rng_state(device)
                        self.assertTrue(torch.equal(rng1[0], rng2[0]))
                        if device == "cuda":
                            self.assertTrue(torch.equal(rng1[1], rng2[1]))
                        self.assertTrue(torch.equal(parts["gate"], torch.ones_like(x2)))
                        derivative = torch.autograd.grad(p2.compute_gate(x2).sum(), x2)[0]
                        self.assertEqual(torch.count_nonzero(derivative), 0)
                        ((y1 - q)**2).mean().backward()
                        ((y2 - q)**2).mean().backward()
                        self.close(y1, y2, dtype); self.close(x1.grad, x2.grad, dtype)
                        maxima[0] = max(maxima[0], float((y1-y2).abs().max()))
                        maxima[1] = max(maxima[1], float((x1.grad-x2.grad).abs().max()))
                        for (name, p1), (other, p2v) in zip(v1.named_parameters(), p2.body.named_parameters()):
                            self.assertEqual(name, other)
                            self.close(p1.grad, p2v.grad, dtype)
                            maxima[2] = max(maxima[2], float((p1.grad-p2v.grad).abs().max()))
                        self.assertEqual(torch.count_nonzero(p2.gate.conv1.weight.grad), 0)
                        self.assertEqual(torch.count_nonzero(p2.gate.conv1.bias.grad), 0)
                        pairs += 1
                # Same nondegenerate output MSE as the accepted synthetic specification.
                p2.train()
                x = inputs(length, dtype, device)[2].clone().requires_grad_()
                p2.zero_grad(set_to_none=True)
                ((p2(x)-q)**2).mean().backward()
                norms = [float(p.grad.norm()) for p in p2.gate.conv2.parameters()]
                self.assertTrue(all(np.isfinite(v) and v > 0 for v in norms))
                print("P2_INITIAL_GATE_GRAD", device, str(dtype), length, norms, flush=True)
            print("P2_EQUIVALENCE", device, str(dtype), "pairs", pairs, "max_errors", maxima, flush=True)

    def test_nontrivial_gate_independence_permutation_and_task_gradient(self):
        for device, dtype in precisions():
            for length in (12, 512):
                p2 = self.make(length, dtype, device).eval()
                perturb_gate(p2.gate)
                x = inputs(length, dtype, device)[2]
                changed = x.clone(); changed[1, 2, length//3] += .4
                before, after = p2.compute_components(x), p2.compute_components(changed)
                mask = torch.ones(2, 3, dtype=torch.bool, device=device); mask[1, 2] = False
                for name in before:
                    self.assertTrue(torch.isfinite(before[name]).all(), name)
                    self.close(before[name][mask], after[name][mask], dtype)
                    self.close(p2.compute_components(x[[1, 0]])[name], before[name][[1, 0]], dtype)
                    self.close(p2.compute_components(x[:, [2, 0, 1]])[name],
                               before[name][:, [2, 0, 1]], dtype)
                gate = before["gate"]
                self.assertTrue(((gate >= 0) & (gate <= 2)).all())
                self.assertGreater(float((gate-after["gate"]).abs().max()), 0)
                grad_x = x.clone().requires_grad_()
                weights = torch.linspace(.7, 1.3, length, dtype=dtype, device=device)
                vjp = torch.autograd.grad((p2.compute_gate(grad_x)*weights).sum(), grad_x)[0]
                self.assertGreater(float(vjp.norm()), 0)
                q = (.2 + .3*torch.cos(3*torch.pi*torch.linspace(0, 1, length, dtype=dtype, device=device)))
                p2.zero_grad(set_to_none=True)
                ((p2(grad_x)-q)**2).mean().backward()
                norm = float(p2.gate.conv1.weight.grad.norm())
                self.assertTrue(np.isfinite(norm) and norm > 0)
                print("P2_DYNAMIC_GATE", device, str(dtype), length,
                      "range", float(gate.min()), float(gate.max()),
                      "input_vjp", float(vjp.norm()), "first_layer_task_grad", norm, flush=True)

    def test_components_single_dropout_sample_and_analysis_semantics(self):
        p2 = self.make(12, torch.float64, "cpu").train()
        perturb_gate(p2.gate)
        hidden = inputs(12, torch.float64, "cpu")[2].requires_grad_()
        state = rng_state("cpu")
        with mock.patch.object(p2.body, "compute_delta", wraps=p2.body.compute_delta) as call:
            parts = p2.compute_components(hidden)
            self.assertEqual(call.call_count, 1)
        self.assertEqual(set(parts), {"base_delta", "gate", "effective_delta", "residual", "output"})
        for value in parts.values():
            self.assertEqual(value.shape, hidden.shape)
            self.assertTrue(value.requires_grad)
        self.assertTrue(p2.training)
        self.close(parts["effective_delta"], parts["gate"]*parts["base_delta"], torch.float64)
        self.close(parts["output"], hidden+parts["residual"], torch.float64)
        restore_rng(state, "cpu"); self.close(p2(hidden), parts["output"], torch.float64)
        restore_rng(state, "cpu"); self.close(p2.compute_delta(hidden), parts["effective_delta"], torch.float64)
        restore_rng(state, "cpu"); self.close(p2.compute_base_delta(hidden), parts["base_delta"], torch.float64)
        self.close(p2.compute_gate(hidden), parts["gate"], torch.float64)

    def test_nontrivial_dynamic_deploy_parameters_mac_dtype_and_device(self):
        for device, dtype in precisions():
            for length, train_count, deploy_count, body_macs in ((12, 451, 419, 352), (512, 659, 611, 560)):
                p2 = self.make(length, dtype, device).train()
                perturb_gate(p2.gate)
                source = {k: v.detach().clone() for k, v in p2.state_dict().items()}
                self.assertEqual(sum(p.numel() for p in p2.parameters()), train_count)
                self.assertEqual(sum(p.numel() for p in p2.gate.parameters()), 33)
                x = inputs(length, dtype, device)[2]
                counts = {"all": 0, "gate": 0}
                handles = []
                for name, module in p2.named_modules():
                    if isinstance(module, torch.nn.Conv1d):
                        def count(mod, args, out, name=name):
                            value = out.numel()*(mod.in_channels//mod.groups)*mod.kernel_size[0]
                            counts["all"] += value
                            if name.startswith("gate."):
                                counts["gate"] += value
                        handles.append(module.register_forward_hook(count))
                p2(x)
                for handle in handles: handle.remove()
                self.assertEqual(counts["gate"], 28*x.numel())
                self.assertEqual(counts["all"], (body_macs+28)*x.numel())
                deployed = p2.to_deploy()
                self.assertTrue(p2.training)
                self.assertFalse(p2.deploy)
                self.assertFalse(deployed.training)
                self.assertTrue(deployed.deploy)
                self.assertEqual(sum(p.numel() for p in deployed.parameters()), deploy_count)
                for key, value in source.items():
                    self.assertTrue(torch.equal(value, p2.state_dict()[key]), key)
                for param in deployed.parameters():
                    self.assertEqual(param.dtype, dtype); self.assertEqual(param.device.type, device)
                p2.eval()
                self.close(p2(x), deployed(x), dtype)
                self.close(p2.compute_base_delta(x), deployed.compute_base_delta(x), dtype)
                self.close(p2.compute_gate(x), deployed.compute_gate(x), dtype)
                altered = x.clone(); altered[1, 2, length//3] += .4
                self.assertGreater(float((deployed.compute_gate(x)-deployed.compute_gate(altered)).abs().max()), 0)
                keys = set(deployed.state_dict())
                self.assertTrue(any("reparam" in k for k in keys))
                self.assertFalse(any(".large_branch." in k or ".small_branch." in k for k in keys))
                self.assertEqual({k for k in keys if k.startswith("gate.")},
                                 {k for k in source if k.startswith("gate.")})
                saved = copy.deepcopy(deployed.state_dict())
                self.assertIs(deployed.switch_to_deploy(), deployed)
                for key, value in saved.items(): self.assertTrue(torch.equal(value, deployed.state_dict()[key]))
                print("P2_DEPLOY", device, str(dtype), length, train_count, deploy_count,
                      "gate_MAC", counts["gate"], "output_error", float((p2(x)-deployed(x)).abs().max()), flush=True)

    def test_same_form_state_restore_is_strict_and_atomic(self):
        model = self.make(12, torch.float32, "cpu")
        perturb_gate(model.gate)
        original = copy.deepcopy(model.state_dict())
        fresh = self.make(12, torch.float32, "cpu")
        fresh.load_state_dict(original)
        for key in original: self.assertTrue(torch.equal(fresh.state_dict()[key], original[key]))
        for kind in ("key", "shape", "dtype", "deploy"):
            bad = copy.deepcopy(original)
            key = "gate.conv1.weight"
            if kind == "key": bad.pop(key)
            elif kind == "shape": bad[key] = bad[key].reshape(-1)
            elif kind == "dtype": bad[key] = bad[key].double()
            else: bad = model.to_deploy().state_dict()
            with self.assertRaises(RuntimeError): model.load_state_dict(bad)
            for key in original: self.assertTrue(torch.equal(model.state_dict()[key], original[key]))
        with self.assertRaises(ValueError): model.load_state_dict(original, strict=False)
