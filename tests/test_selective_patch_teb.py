import unittest

import torch

from models.modules.patch_conditioned_target_exogenous_bridge import (
    FIXED_SINUSOIDAL,
    RIGHT_ZERO_CROP,
    PatchConditionedTargetExogenousBridge,
)
from models.modules.selective_patch_target_exogenous_bridge import (
    GLOBAL_PREDICTION_ROLE_STATE_ONLY,
    PATCH_CONFIDENCE_GATE_SCALAR_POST_PROJECTION,
    PATCH_GATE_ACTIVATION_TWO_SIGMOID,
    PATCH_GATE_INIT_EXPLICIT_ZERO_IDENTITY,
    PATCH_GATE_INPUT_QUERY_AND_ATTENTION,
    SelectivePatchTargetExogenousBridge,
)
from models.modules.target_exogenous_bridge import TARGET_EXOGENOUS


class SelectivePatchTEBTests(unittest.TestCase):
    @staticmethod
    def _t2(*, seq_len=12, patch_size=3, context_dim=8, dropout=0.0):
        return PatchConditionedTargetExogenousBridge(
            seq_len=seq_len,
            feature_num=3,
            task_mode=TARGET_EXOGENOUS,
            target_idx=1,
            aux_idx=(2, 0),
            context_dim=context_dim,
            num_heads=2 if context_dim != 32 else 4,
            dropout=dropout,
            patch_size=patch_size,
            gamma_init=1e-3,
            padding_policy=RIGHT_ZERO_CROP,
            position_policy=FIXED_SINUSOIDAL,
        )

    @staticmethod
    def _t3(*, seq_len=12, patch_size=3, context_dim=8, dropout=0.0):
        return SelectivePatchTargetExogenousBridge(
            seq_len=seq_len,
            feature_num=3,
            task_mode=TARGET_EXOGENOUS,
            target_idx=1,
            aux_idx=(2, 0),
            context_dim=context_dim,
            num_heads=2 if context_dim != 32 else 4,
            dropout=dropout,
            patch_size=patch_size,
            gamma_init=1e-3,
            padding_policy=RIGHT_ZERO_CROP,
            position_policy=FIXED_SINUSOIDAL,
            patch_confidence_gate=(
                PATCH_CONFIDENCE_GATE_SCALAR_POST_PROJECTION
            ),
            patch_gate_input=PATCH_GATE_INPUT_QUERY_AND_ATTENTION,
            patch_gate_activation=PATCH_GATE_ACTIVATION_TWO_SIGMOID,
            patch_gate_init=PATCH_GATE_INIT_EXPLICIT_ZERO_IDENTITY,
            global_prediction_role=GLOBAL_PREDICTION_ROLE_STATE_ONLY,
        )

    def test_exact_parameter_counts_keys_and_identity_initialization(self):
        for seq_len, patch_size, expected in (
            (512, 32, 39426),
            (12, 3, 5541),
        ):
            with self.subTest(seq_len=seq_len, patch_size=patch_size):
                module = self._t3(
                    seq_len=seq_len,
                    patch_size=patch_size,
                    context_dim=32,
                )
                count = sum(p.numel() for p in module.parameters() if p.requires_grad)
                self.assertEqual(count, expected)
                self.assertEqual(
                    module.patch_confidence_gate_weight.numel()
                    + module.patch_confidence_gate_bias.numel(),
                    65,
                )
                self.assertTrue(torch.equal(
                    module.patch_confidence_gate_weight,
                    torch.zeros_like(module.patch_confidence_gate_weight),
                ))
                self.assertTrue(torch.equal(
                    module.patch_confidence_gate_bias,
                    torch.zeros_like(module.patch_confidence_gate_bias),
                ))
                query = torch.randn(2, module.num_patches, 32)
                attention = torch.randn_like(query)
                gate = module.compute_patch_confidence_gate(query, attention)
                self.assertTrue(torch.equal(gate, torch.ones_like(gate)))
                self.assertNotIn("fixed_sinusoidal_position", module.state_dict())
                keys = set(module.state_dict())
                self.assertFalse(any("beta_global" in key for key in keys))
                self.assertFalse(any("global_bridge" in key for key in keys))
                self.assertFalse(any("global_injection" in key for key in keys))
                self.assertFalse(any("patch_post" in key for key in keys))
                self.assertFalse(any("patch_ffn" in key for key in keys))

    def test_t2_base_and_cpu_cuda_rng_initialization_parity(self):
        seed = 6113
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        t2 = self._t2(context_dim=32, dropout=0.1)
        t2_cpu_rng = torch.get_rng_state().clone()
        t2_cuda_rng = (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        t3 = self._t3(context_dim=32, dropout=0.1)
        self.assertTrue(torch.equal(t2_cpu_rng, torch.get_rng_state()))
        if torch.cuda.is_available():
            self.assertEqual(len(t2_cuda_rng), len(torch.cuda.get_rng_state_all()))
            for expected, actual in zip(t2_cuda_rng, torch.cuda.get_rng_state_all()):
                self.assertTrue(torch.equal(expected, actual))

        t2_state = t2.state_dict()
        t3_state = t3.state_dict()
        extra = set(t3_state) - set(t2_state)
        self.assertEqual(extra, {
            "patch_confidence_gate_weight",
            "patch_confidence_gate_bias",
        })
        for key, value in t2_state.items():
            self.assertTrue(torch.equal(value, t3_state[key]), key)
        self.assertTrue(torch.equal(
            t2.fixed_sinusoidal_position,
            t3.fixed_sinusoidal_position,
        ))

    def test_initial_eval_and_train_forward_are_t2_equivalent(self):
        seed = 6121
        torch.manual_seed(seed)
        t2 = self._t2(dropout=0.1)
        torch.manual_seed(seed)
        t3 = self._t3(dropout=0.1)
        hidden = torch.randn(4, 3, 12)
        normalized = torch.randn(4, 12, 3)

        t2.eval()
        t3.eval()
        with torch.no_grad():
            t2_out = t2(hidden, normalized)
            t3_out = t3(hidden, normalized)
        for expected, actual in zip(t2_out, t3_out):
            self.assertTrue(torch.equal(expected, actual))

        t2.train()
        t3.train()
        forward_rng = torch.get_rng_state().clone()
        torch.set_rng_state(forward_rng)
        t2_out = t2(hidden, normalized)
        torch.set_rng_state(forward_rng)
        t3_out = t3(hidden, normalized)
        for expected, actual in zip(t2_out, t3_out):
            torch.testing.assert_close(expected, actual, rtol=1e-6, atol=1e-6)

    def test_post_projection_gate_controls_weight_and_bias(self):
        torch.manual_seed(6131)
        module = self._t3()
        with torch.no_grad():
            module.patch_output_projection.bias.fill_(2.75)
        query = torch.randn(2, 4, 8)
        attention = torch.randn_like(query)
        raw = module.compute_raw_patch_delta(attention)
        expected_gate = module.compute_patch_confidence_gate(query, attention)
        effective = module.compute_effective_patch_delta(query, attention)
        self.assertTrue(torch.equal(effective, expected_gate * raw))

        original = module.compute_patch_confidence_gate
        try:
            module.compute_patch_confidence_gate = (
                lambda q, a: a.new_zeros(*a.shape[:-1], 1)
            )
            raw_zero, gate_zero, effective_zero = (
                module.compute_patch_delta_components(query, attention)
            )
        finally:
            module.compute_patch_confidence_gate = original
        self.assertGreater(raw_zero.abs().max().item(), 0.0)
        self.assertTrue(torch.equal(gate_zero, torch.zeros_like(gate_zero)))
        self.assertTrue(torch.equal(
            effective_zero, torch.zeros_like(effective_zero)
        ))

    def test_forecast_gradient_trains_patch_gate_but_not_global_query(self):
        torch.manual_seed(6143)
        module = self._t3().train()
        hidden = torch.randn(3, 3, 12, requires_grad=True)
        normalized = torch.randn(3, 12, 3, requires_grad=True)
        hidden_out, _ = module(hidden, normalized)
        weight = torch.linspace(
            -0.8, 1.3, hidden_out.numel(), dtype=hidden_out.dtype
        ).reshape_as(hidden_out)
        (hidden_out * weight).sum().backward()

        nonzero_groups = (
            "patch_confidence_gate_weight",
            "patch_confidence_gate_bias",
            "patch_query_projection",
            "patch_query_norm",
            "exogenous_projection",
            "exogenous_norm",
            "cross_attention",
            "patch_output_projection",
            "gamma_teb",
        )
        parameters = dict(module.named_parameters())
        for group in nonzero_groups:
            selected = [
                (name, parameter)
                for name, parameter in parameters.items()
                if name == group or name.startswith(group + ".")
            ]
            self.assertTrue(selected, group)
            for name, parameter in selected:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
                self.assertGreater(parameter.grad.abs().max().item(), 0.0, name)

        for group in ("global_query_projection", "global_query_norm"):
            for name, parameter in parameters.items():
                if name.startswith(group + "."):
                    if parameter.grad is not None:
                        self.assertTrue(torch.equal(
                            parameter.grad, torch.zeros_like(parameter.grad)
                        ), name)

        module.zero_grad(set_to_none=True)
        _, context = module(hidden.detach(), normalized.detach())
        context_weight = torch.linspace(
            -0.7, 0.9, context.numel(), dtype=context.dtype
        ).reshape_as(context)
        (context * context_weight).sum().backward()
        for group in ("global_query_projection", "global_query_norm"):
            for name, parameter in module.named_parameters():
                if name.startswith(group + "."):
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertGreater(parameter.grad.abs().max().item(), 0.0, name)

    def test_single_target_shape_dtype_device_and_channel_contract(self):
        cases = [
            (torch.device("cpu"), torch.float32),
            (torch.device("cpu"), torch.float64),
        ]
        cuda_executed = False
        if torch.cuda.is_available():
            cases.append((torch.device("cuda"), torch.float32))
            cuda_executed = True
        for device, dtype in cases:
            with self.subTest(device=str(device), dtype=str(dtype)):
                torch.manual_seed(6151)
                module = self._t3(dropout=0.1).to(device=device, dtype=dtype).eval()
                hidden = torch.randn(2, 3, 12, device=device, dtype=dtype)
                normalized = torch.randn(2, 12, 3, device=device, dtype=dtype)
                with torch.no_grad():
                    hidden_out, context = module(hidden, normalized)
                self.assertEqual(hidden_out.shape, hidden.shape)
                self.assertEqual(context.shape, (2, 8))
                self.assertTrue(torch.equal(hidden_out[:, 0], hidden[:, 0]))
                self.assertTrue(torch.equal(hidden_out[:, 2], hidden[:, 2]))
                self.assertTrue(torch.isfinite(hidden_out).all().item())
                self.assertTrue(torch.isfinite(context).all().item())
        print(f"T3 CUDA float32 executed={cuda_executed}")

    def test_contract_mismatches_and_inherited_aux_guards(self):
        fields = {
            "patch_confidence_gate": "pre_projection",
            "patch_gate_input": "attention_only",
            "patch_gate_activation": "sigmoid",
            "patch_gate_init": "random",
            "global_prediction_role": "forecast_connected",
        }
        for field, value in fields.items():
            kwargs = dict(
                seq_len=12,
                feature_num=3,
                task_mode=TARGET_EXOGENOUS,
                target_idx=1,
                aux_idx=(2, 0),
                context_dim=8,
                num_heads=2,
                dropout=0.0,
                patch_size=3,
            )
            kwargs[field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "T3 contract mismatch"):
                    SelectivePatchTargetExogenousBridge(**kwargs)

        for aux_idx, error in (
            ((), ValueError),
            ((0, 0), ValueError),
            ((1, 0), ValueError),
            ((3,), ValueError),
            ((True,), TypeError),
        ):
            with self.subTest(aux_idx=aux_idx):
                with self.assertRaises(error):
                    SelectivePatchTargetExogenousBridge(
                        seq_len=12,
                        feature_num=3,
                        task_mode=TARGET_EXOGENOUS,
                        target_idx=1,
                        aux_idx=aux_idx,
                        context_dim=8,
                        num_heads=2,
                        dropout=0.0,
                        patch_size=3,
                    )


if __name__ == "__main__":
    unittest.main()
