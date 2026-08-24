import unittest

import torch

from models.modules.target_exogenous_bridge import (
    TARGET_EXOGENOUS,
    TargetExogenousBridge,
)


class TargetExogenousBridgeTests(unittest.TestCase):
    @staticmethod
    def _module(dtype=torch.float32, dropout=0.0):
        return TargetExogenousBridge(
            seq_len=12,
            feature_num=4,
            task_mode=TARGET_EXOGENOUS,
            target_idx=0,
            aux_idx=(1, 2, 3),
            context_dim=8,
            num_heads=2,
            dropout=dropout,
            gamma_init=1e-3,
        ).to(dtype=dtype)

    def test_single_shape_dtype_device_and_attention_shape(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda", torch.cuda.current_device()))
        for device in devices:
            for dtype in (torch.float32, torch.float64):
                with self.subTest(device=str(device), dtype=str(dtype)):
                    model = self._module(dtype=dtype).to(device).eval()
                    hidden = torch.randn(2, 4, 12, dtype=dtype, device=device)
                    normalized = torch.randn(2, 12, 4, dtype=dtype, device=device)
                    output, context, weights = model(
                        hidden, normalized, need_weights=True
                    )
                    self.assertEqual(output.shape, hidden.shape)
                    self.assertEqual(context.shape, (2, 8))
                    self.assertEqual(weights.shape, (2, 2, 1, 3))
                    for tensor in (output, context, weights):
                        self.assertEqual(tensor.dtype, dtype)
                        self.assertEqual(tensor.device, device)

    def test_non_target_channels_are_elementwise_unchanged(self):
        torch.manual_seed(301)
        model = self._module().eval()
        hidden = torch.randn(3, 4, 12)
        normalized = torch.randn(3, 12, 4)
        output, _ = model(hidden, normalized)
        self.assertTrue(torch.equal(output[:, 1:, :], hidden[:, 1:, :]))
        self.assertGreater((output[:, 0, :] - hidden[:, 0, :]).abs().max().item(), 0)

    def test_input_and_all_logical_parameter_groups_receive_gradients(self):
        torch.manual_seed(302)
        model = self._module(dropout=0.0).train()
        hidden = torch.randn(3, 4, 12, requires_grad=True)
        normalized = torch.randn(3, 12, 4, requires_grad=True)
        output, context = model(hidden, normalized)
        output_weight = torch.linspace(0.1, 1.3, output.numel()).reshape_as(output)
        context_weight = torch.linspace(-0.4, 0.7, context.numel()).reshape_as(context)
        loss = (output * output_weight).sum() + (context * context_weight).sum()
        loss.backward()

        for name, tensor in (
            ("hidden", hidden),
            ("normalized_input", normalized),
        ):
            self.assertIsNotNone(tensor.grad, name)
            self.assertTrue(torch.isfinite(tensor.grad).all(), name)
            self.assertGreater(tensor.grad.abs().max().item(), 0, name)

        groups = {
            "query": ("query_projection", "query_norm"),
            "exogenous": ("exogenous_projection", "exogenous_norm"),
            "attention": ("cross_attention",),
            "output": ("output_projection",),
            "gamma": ("gamma_teb",),
        }
        named = dict(model.named_parameters())
        for group, prefixes in groups.items():
            members = [
                parameter
                for name, parameter in named.items()
                if any(name.startswith(prefix) for prefix in prefixes)
            ]
            self.assertTrue(members, group)
            self.assertTrue(all(parameter.grad is not None for parameter in members), group)
            self.assertTrue(
                all(torch.isfinite(parameter.grad).all() for parameter in members),
                group,
            )
            self.assertTrue(
                any(parameter.grad.abs().max().item() > 0 for parameter in members),
                group,
            )

    def test_shared_projector_and_no_variable_identity_parameters(self):
        model = self._module()
        self.assertIsInstance(model.exogenous_projection, torch.nn.Linear)
        names = tuple(name.lower() for name, _ in model.named_parameters())
        self.assertFalse(any("identity" in name or "embedding" in name for name in names))
        self.assertEqual(
            sum(name.startswith("exogenous_projection.") for name in names),
            2,
        )

    def test_auxiliary_token_permutation_preserves_context_and_output(self):
        torch.manual_seed(303)
        first = self._module().eval()
        second = TargetExogenousBridge(
            seq_len=12,
            feature_num=4,
            task_mode=TARGET_EXOGENOUS,
            target_idx=0,
            aux_idx=(3, 1, 2),
            context_dim=8,
            num_heads=2,
            dropout=0.0,
            gamma_init=1e-3,
        ).eval()
        second.load_state_dict(first.state_dict(), strict=True)
        hidden = torch.randn(2, 4, 12)
        normalized = torch.randn(2, 12, 4)
        out_a, context_a, weights_a = first(
            hidden, normalized, need_weights=True
        )
        out_b, context_b, weights_b = second(
            hidden, normalized, need_weights=True
        )
        self.assertTrue(torch.allclose(out_a, out_b, atol=1e-7, rtol=1e-6))
        self.assertTrue(torch.allclose(context_a, context_b, atol=1e-7, rtol=1e-6))
        self.assertTrue(
            torch.allclose(
                weights_b,
                weights_a[..., (2, 0, 1)],
                atol=1e-7,
                rtol=1e-6,
            )
        )

    def test_gamma_is_one_shared_unconstrained_scalar(self):
        model = self._module()
        self.assertIsInstance(model.gamma_teb, torch.nn.Parameter)
        self.assertEqual(model.gamma_teb.shape, torch.Size([]))
        self.assertAlmostEqual(model.gamma_teb.item(), 1e-3, places=9)
        self.assertTrue(model.gamma_teb.requires_grad)

    def test_parameter_and_input_contract_guards(self):
        kwargs = {
            "seq_len": 12,
            "feature_num": 4,
            "task_mode": TARGET_EXOGENOUS,
            "target_idx": 0,
            "aux_idx": (1, 2),
            "context_dim": 8,
            "num_heads": 2,
        }
        with self.assertRaisesRegex(ValueError, "divisible"):
            TargetExogenousBridge(**{**kwargs, "num_heads": 3})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            TargetExogenousBridge(**{**kwargs, "aux_idx": (1, 1)})
        with self.assertRaisesRegex(ValueError, "exclude"):
            TargetExogenousBridge(**{**kwargs, "aux_idx": (0, 1)})
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            TargetExogenousBridge(**{**kwargs, "aux_idx": (1, 4)})
        with self.assertRaisesRegex(ValueError, "dropout"):
            TargetExogenousBridge(**{**kwargs, "dropout": 1.0})
        for gamma in (0.0, 1e-2, -1e-3, float("nan"), float("inf")):
            with self.subTest(gamma_init=gamma):
                with self.assertRaisesRegex(ValueError, "fixed at 1e-3"):
                    TargetExogenousBridge(
                        **kwargs,
                        gamma_init=gamma,
                    )

        model = self._module()
        with self.assertRaisesRegex(ValueError, "hidden"):
            model(torch.randn(2, 12, 4), torch.randn(2, 12, 4))
        with self.assertRaisesRegex(ValueError, "normalized_input"):
            model(torch.randn(2, 4, 12), torch.randn(2, 11, 4))
        with self.assertRaisesRegex(TypeError, "same dtype"):
            model(
                torch.randn(2, 4, 12, dtype=torch.float32),
                torch.randn(2, 12, 4, dtype=torch.float64),
            )


if __name__ == "__main__":
    unittest.main()
