import unittest

import torch

from models.modules.modern_conv_refinement import (
    PeakPreservingModernConvRefinement,
    ReparamLargeKernelDWConv,
)


class PMCRReparameterizationTests(unittest.TestCase):
    def _assert_equivalent(self, *, device, dtype, atol, rtol):
        torch.manual_seed(2301)
        module = PeakPreservingModernConvRefinement(
            hidden_dim=8,
            kernel_small=3,
            kernel_large=7,
            dropout=0.0,
            gamma_init=1e-3,
        ).to(device=device, dtype=dtype).eval()
        hidden = torch.randn(2, 4, 12, device=device, dtype=dtype)

        projected = module.input_projection(hidden.reshape(8, 1, 12))
        temporal_deploy = module.temporal_conv.to_deploy()
        full_deploy = module.to_deploy()
        with torch.no_grad():
            temporal_train = module.temporal_conv(projected)
            temporal_fused = temporal_deploy(projected)
            delta_train = module.compute_delta(hidden)
            delta_fused = full_deploy.compute_delta(hidden)
            output_train = module(hidden)
            output_fused = full_deploy(hidden)

        errors = {
            "temporal": (temporal_train - temporal_fused).abs().max().item(),
            "delta": (delta_train - delta_fused).abs().max().item(),
            "forward": (output_train - output_fused).abs().max().item(),
        }
        for name, left, right in (
            ("temporal", temporal_train, temporal_fused),
            ("delta", delta_train, delta_fused),
            ("forward", output_train, output_fused),
        ):
            with self.subTest(name=name, device=str(device), dtype=str(dtype)):
                torch.testing.assert_close(
                    left,
                    right,
                    atol=atol,
                    rtol=rtol,
                    msg=(
                        f"{name} reparameterization failed: "
                        f"max_abs_error={errors[name]:.12g}, "
                        f"dtype={dtype}, device={device}"
                    ),
                )
        print(
            "PMCR reparameterization "
            f"device={device} dtype={dtype} "
            + " ".join(f"{name}_max_abs={value:.12g}" for name, value in errors.items())
        )
        return errors

    def test_cpu_float64_equivalence(self):
        self._assert_equivalent(
            device=torch.device("cpu"),
            dtype=torch.float64,
            atol=1e-10,
            rtol=1e-8,
        )

    def test_cpu_and_optional_cuda_float32_equivalence(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            with self.subTest(device=str(device)):
                self._assert_equivalent(
                    device=device,
                    dtype=torch.float32,
                    atol=1e-6,
                    rtol=1e-5,
                )

    def test_equivalent_kernel_query_does_not_mutate_training_module(self):
        module = ReparamLargeKernelDWConv(8, 3, 7).double().eval()
        before = {key: value.clone() for key, value in module.state_dict().items()}
        kernel, bias = module.get_equivalent_kernel_bias()
        after = module.state_dict()

        self.assertEqual(kernel.shape, (8, 1, 7))
        self.assertEqual(bias.shape, (8,))
        self.assertEqual(kernel.dtype, torch.float64)
        self.assertFalse(module.deploy)
        self.assertEqual(before.keys(), after.keys())
        for key in before:
            self.assertTrue(torch.equal(before[key], after[key]), key)

    def test_switch_to_deploy_is_in_place_and_idempotent(self):
        module = ReparamLargeKernelDWConv(8, 3, 7).eval()
        hidden = torch.randn(2, 8, 12)
        with torch.no_grad():
            expected = module(hidden)
        returned = module.switch_to_deploy()
        first_state = {key: value.clone() for key, value in module.state_dict().items()}
        returned_again = module.switch_to_deploy()
        with torch.no_grad():
            actual = module(hidden)

        self.assertIs(returned, module)
        self.assertIs(returned_again, module)
        self.assertTrue(module.deploy)
        self.assertFalse(hasattr(module, "large_branch"))
        self.assertFalse(hasattr(module, "small_branch"))
        self.assertTrue(all("reparam_branch" in key for key in first_state))
        self.assertEqual(first_state.keys(), module.state_dict().keys())
        for key in first_state:
            self.assertTrue(torch.equal(first_state[key], module.state_dict()[key]))
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_to_deploy_preserves_original_and_returns_eval_copy(self):
        module = PeakPreservingModernConvRefinement(
            8, 3, 7, dropout=0.0, gamma_init=1e-3
        ).double().train()
        before = {key: value.clone() for key, value in module.state_dict().items()}
        hidden = torch.randn(2, 4, 12, dtype=torch.float64)
        with torch.no_grad():
            output_before = module(hidden)
        deployed = module.to_deploy()
        with torch.no_grad():
            output_after = module(hidden)

        self.assertTrue(module.training)
        self.assertFalse(module.deploy)
        self.assertFalse(deployed.training)
        self.assertTrue(deployed.deploy)
        self.assertEqual(before.keys(), module.state_dict().keys())
        for key in before:
            self.assertTrue(torch.equal(before[key], module.state_dict()[key]), key)
        self.assertTrue(torch.equal(output_before, output_after))
        self.assertTrue(
            any("large_branch" in key for key in module.state_dict())
        )
        self.assertTrue(
            any("small_branch" in key for key in module.state_dict())
        )
        self.assertFalse(
            any("reparam_branch" in key for key in module.state_dict())
        )
        self.assertTrue(
            any("reparam_branch" in key for key in deployed.state_dict())
        )
        self.assertFalse(
            any("large_branch" in key or "small_branch" in key for key in deployed.state_dict())
        )
        self.assertEqual(
            next(module.parameters()).dtype, next(deployed.parameters()).dtype
        )
        self.assertEqual(
            next(module.parameters()).device, next(deployed.parameters()).device
        )


if __name__ == "__main__":
    unittest.main()
