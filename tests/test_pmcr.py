import math
import unittest

import torch

from models.modules.modern_conv_refinement import (
    PeakPreservingModernConvRefinement,
    ReparamLargeKernelDWConv,
)


class PMCRCoreTests(unittest.TestCase):
    def _module(self, *, dtype=torch.float32, device="cpu", dropout=0.0):
        return PeakPreservingModernConvRefinement(
            hidden_dim=8,
            kernel_small=3,
            kernel_large=7,
            dropout=dropout,
            gamma_init=1e-3,
        ).to(device=device, dtype=dtype)

    def test_shape_contract_for_short_and_long_sequences(self):
        torch.manual_seed(2101)
        module = self._module().eval()
        for shape in ((2, 3, 12), (1, 5, 96), (4, 2, 31)):
            with self.subTest(shape=shape):
                hidden = torch.randn(*shape)
                with torch.no_grad():
                    delta = module.compute_delta(hidden)
                    output = module(hidden)
                self.assertEqual(delta.shape, shape)
                self.assertEqual(output.shape, shape)

    def test_dtype_and_device_are_preserved(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))

        for device in devices:
            dtypes = (torch.float32, torch.float64) if device.type == "cpu" else (
                torch.float32,
            )
            for dtype in dtypes:
                with self.subTest(device=str(device), dtype=str(dtype)):
                    module = self._module(dtype=dtype, device=device).eval()
                    hidden = torch.randn(2, 4, 12, device=device, dtype=dtype)
                    with torch.no_grad():
                        delta = module.compute_delta(hidden)
                        output = module(hidden)
                    for value in (delta, output):
                        self.assertEqual(value.dtype, hidden.dtype)
                        self.assertEqual(value.device, hidden.device)

    def test_twelve_step_same_padding_preserves_every_temporal_length(self):
        module = self._module().eval()
        hidden = torch.randn(2, 3, 12)
        flattened = hidden.reshape(6, 1, 12)
        projected = module.input_projection(flattened)
        large = module.temporal_conv.large_branch(projected)
        small = module.temporal_conv.small_branch(projected)

        self.assertEqual(projected.shape, (6, 8, 12))
        self.assertEqual(large.shape, (6, 8, 12))
        self.assertEqual(small.shape, (6, 8, 12))
        self.assertEqual(module.compute_delta(hidden).shape, hidden.shape)
        self.assertEqual(module(hidden).shape, hidden.shape)

    def test_input_and_all_logical_parameter_groups_receive_gradients(self):
        torch.manual_seed(2102)
        module = self._module(dropout=0.0).train()
        hidden = torch.randn(2, 3, 12, requires_grad=True)
        weights = torch.linspace(-1.7, 2.3, hidden.numel()).reshape_as(hidden)
        delta = module.compute_delta(hidden)
        output = hidden + module.gamma_pmcr * delta
        loss = (output * weights).sum() + 0.031 * (delta.square() * weights.abs()).sum()
        loss.backward()

        self._assert_nonzero_finite_gradient("input", hidden.grad)
        groups = {
            "input_projection": tuple(module.input_projection.parameters()),
            "large_branch": tuple(module.temporal_conv.large_branch.parameters()),
            "small_branch": tuple(module.temporal_conv.small_branch.parameters()),
            "feature_norm": tuple(module.feature_norm.parameters()),
            "ffn_expand": tuple(module.ffn_expand.parameters()),
            "ffn_reduce": tuple(module.ffn_reduce.parameters()),
            "output_projection": tuple(module.output_projection.parameters()),
            "gamma_pmcr": (module.gamma_pmcr,),
        }
        for name, parameters in groups.items():
            with self.subTest(parameter_group=name):
                self.assertTrue(parameters)
                for index, parameter in enumerate(parameters):
                    self._assert_nonzero_finite_gradient(
                        f"{name}[{index}]", parameter.grad
                    )

    def _assert_nonzero_finite_gradient(self, name, gradient):
        self.assertIsNotNone(gradient, name)
        self.assertTrue(torch.isfinite(gradient).all().item(), name)
        self.assertGreater(gradient.abs().max().item(), 0.0, name)

    def test_forward_is_outer_residual_with_scaled_compute_delta(self):
        torch.manual_seed(2103)
        module = self._module(dropout=0.1).eval()
        hidden = torch.randn(2, 3, 12)
        with torch.no_grad():
            delta = module.compute_delta(hidden)
            expected = hidden + module.gamma_pmcr * delta
            actual = module(hidden)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_locked_initialization_contract(self):
        torch.manual_seed(2104)
        module = self._module()
        for projection in (
            module.input_projection,
            module.ffn_expand,
            module.ffn_reduce,
            module.output_projection,
        ):
            self.assertTrue(torch.equal(projection.bias, torch.zeros_like(projection.bias)))
        for branch in (
            module.temporal_conv.large_branch,
            module.temporal_conv.small_branch,
        ):
            self.assertTrue(torch.equal(branch.bias, torch.zeros_like(branch.bias)))
        self.assertTrue(
            torch.equal(module.feature_norm.weight, torch.ones_like(module.feature_norm.weight))
        )
        self.assertTrue(
            torch.equal(module.feature_norm.bias, torch.zeros_like(module.feature_norm.bias))
        )
        self.assertEqual(module.gamma_pmcr.ndim, 0)
        self.assertAlmostEqual(module.gamma_pmcr.item(), 1e-3, places=9)

    def test_parameter_validation_rejects_invalid_contracts(self):
        invalid_pmcr = (
            {"hidden_dim": 1, "kernel_small": 3, "kernel_large": 7},
            {"hidden_dim": 8, "kernel_small": 2, "kernel_large": 7},
            {"hidden_dim": 8, "kernel_small": 7, "kernel_large": 7},
            {"hidden_dim": 8, "kernel_small": 3, "kernel_large": 7, "dropout": 1.0},
            {"hidden_dim": 8, "kernel_small": 3, "kernel_large": 7, "dropout": math.nan},
            {"hidden_dim": 8, "kernel_small": 3, "kernel_large": 7, "gamma_init": 0.0},
            {"hidden_dim": 8, "kernel_small": 3, "kernel_large": 7, "gamma_init": math.inf},
        )
        for kwargs in invalid_pmcr:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    PeakPreservingModernConvRefinement(**kwargs)

        with self.assertRaisesRegex(ValueError, "channels"):
            ReparamLargeKernelDWConv(0, 3, 7)
        with self.assertRaisesRegex(ValueError, "odd"):
            ReparamLargeKernelDWConv(8, 3, 8)

    def test_input_shape_guards_are_explicit(self):
        module = self._module()
        with self.assertRaisesRegex(TypeError, "torch.Tensor"):
            module.compute_delta([1.0])
        with self.assertRaisesRegex(ValueError, "\[batch, variable, time\]"):
            module.compute_delta(torch.randn(2, 12))
        with self.assertRaisesRegex(ValueError, "non-empty"):
            module.compute_delta(torch.empty(0, 3, 12))


if __name__ == "__main__":
    unittest.main()
