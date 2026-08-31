import unittest

import torch
import torch.nn as nn

from models.modules.global_mediated_patch_target_exogenous_bridge import (
    GLOBAL_GATE_IDENTITY_INIT,
    GLOBAL_GATE_INPUT_CONTRACT,
    GLOBAL_GATE_SCALAR_PER_PATCH,
    GLOBAL_RESIDUAL_CONTRACT,
    PATCH_ATTENTION_RESIDUAL_NONE,
    GlobalMediatedPatchTargetExogenousBridge,
)
from models.modules.patch_conditioned_target_exogenous_bridge import (
    FIXED_SINUSOIDAL,
    RIGHT_ZERO_CROP,
    PatchConditionedTargetExogenousBridge,
)
from models.modules.target_exogenous_bridge import TARGET_EXOGENOUS


class _ZeroAttention(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, query, key, value, **kwargs):
        context = torch.zeros_like(query)
        weights = query.new_zeros(
            query.shape[0], self.num_heads, query.shape[1], key.shape[1]
        )
        return context, weights


class GlobalMediatedPatchTEBTests(unittest.TestCase):
    @staticmethod
    def _module(
        *, seq_len=12, patch_size=3, context_dim=8, dropout=0.0,
        target_idx=1, aux_idx=(2, 0),
    ):
        return GlobalMediatedPatchTargetExogenousBridge(
            seq_len=seq_len,
            feature_num=3,
            task_mode=TARGET_EXOGENOUS,
            target_idx=target_idx,
            aux_idx=aux_idx,
            context_dim=context_dim,
            num_heads=2 if context_dim != 32 else 4,
            dropout=dropout,
            patch_size=patch_size,
            gamma_init=1e-3,
            padding_policy=RIGHT_ZERO_CROP,
            position_policy=FIXED_SINUSOIDAL,
            global_residual=GLOBAL_RESIDUAL_CONTRACT,
            patch_attention_residual=PATCH_ATTENTION_RESIDUAL_NONE,
            global_gate=GLOBAL_GATE_SCALAR_PER_PATCH,
            global_gate_input=GLOBAL_GATE_INPUT_CONTRACT,
            global_gate_init=GLOBAL_GATE_IDENTITY_INIT,
            beta_global_init=1e-3,
        )

    def test_exact_parameter_counts_initialization_and_no_patch_modules(self):
        for seq_len, patch_size, expected in ((512, 32, 39491), (12, 3, 5606)):
            with self.subTest(seq_len=seq_len, patch_size=patch_size):
                module = self._module(
                    seq_len=seq_len,
                    patch_size=patch_size,
                    context_dim=32,
                )
                count = sum(p.numel() for p in module.parameters() if p.requires_grad)
                self.assertEqual(count, expected)
                self.assertEqual(count - (39361 if seq_len == 512 else 5476), 130)
                self.assertAlmostEqual(module.beta_global.item(), 1e-3, places=9)
                gate_input = torch.randn(2, module.num_patches, 64)
                self.assertTrue(torch.equal(
                    2 * torch.sigmoid(module.global_injection_gate(gate_input)),
                    torch.ones(2, module.num_patches, 1),
                ))
                self.assertNotIn("fixed_sinusoidal_position", module.state_dict())
                names = set(dict(module.named_modules()))
                self.assertNotIn("patch_post_norm", names)
                self.assertNotIn("patch_residual_norm", names)

    def test_global_residual_gate_and_context_are_exact(self):
        torch.manual_seed(5101)
        module = self._module().eval()
        patch_attention = torch.randn(2, module.num_patches, module.context_dim)
        global_query = torch.randn(2, 1, module.context_dim)
        global_attention = torch.randn(2, 1, module.context_dim)
        fused, context, gate = module._global_mediated_patch_context(
            patch_attention, global_query, global_attention
        )
        expected_global = module.global_bridge_norm(
            global_query + global_attention
        )
        self.assertTrue(torch.equal(context, expected_global.squeeze(1)))
        self.assertTrue(torch.equal(gate, torch.ones_like(gate)))
        torch.testing.assert_close(
            fused,
            patch_attention + module.beta_global * expected_global,
            rtol=0,
            atol=0,
        )

        zero_attention = torch.zeros_like(global_attention)
        _, zero_context, _ = module._global_mediated_patch_context(
            patch_attention, global_query, zero_attention
        )
        self.assertTrue(torch.equal(
            zero_context, module.global_bridge_norm(global_query).squeeze(1)
        ))
        changed_query = global_query.clone()
        changed_query[..., 0] += 1.0
        _, changed_query_context, _ = module._global_mediated_patch_context(
            patch_attention, changed_query, zero_attention
        )
        self.assertFalse(torch.equal(zero_context, changed_query_context))
        changed_attention = zero_attention.clone()
        changed_attention[..., 1] += 1.0
        _, changed_attention_context, _ = module._global_mediated_patch_context(
            patch_attention, global_query, changed_attention
        )
        self.assertFalse(torch.equal(zero_context, changed_attention_context))

    def test_zero_attention_beta_zero_proves_no_patch_query_residual(self):
        torch.manual_seed(5113)
        module = self._module().eval()
        module.cross_attention = _ZeroAttention(module.num_heads)
        with torch.no_grad():
            module.beta_global.zero_()
            module.patch_output_projection.bias.zero_()
        normalized = torch.randn(2, 12, 3)
        first = torch.randn(2, 3, 12)
        second = first.clone()
        second[:, 1, :] += torch.linspace(-2.0, 3.0, 12)
        with torch.no_grad():
            first_out, _ = module(first, normalized)
            second_out, _ = module(second, normalized)
        self.assertTrue(torch.equal(first_out - first, torch.zeros_like(first)))
        self.assertTrue(torch.equal(second_out - second, torch.zeros_like(second)))

    def test_beta_zero_matches_t2_hidden_and_patch_projection_input(self):
        torch.manual_seed(5129)
        t2 = PatchConditionedTargetExogenousBridge(
            seq_len=12,
            feature_num=3,
            task_mode=TARGET_EXOGENOUS,
            target_idx=1,
            aux_idx=(2, 0),
            context_dim=8,
            num_heads=2,
            dropout=0.0,
            patch_size=3,
        ).eval()
        t2g = self._module().eval()
        completed = t2g.state_dict()
        completed.update(t2.state_dict())
        t2g.load_state_dict(completed, strict=True)
        with torch.no_grad():
            t2g.beta_global.zero_()

        captured = {}
        hooks = [
            t2.patch_output_projection.register_forward_pre_hook(
                lambda _module, inputs: captured.__setitem__("t2", inputs[0].detach().clone())
            ),
            t2g.patch_output_projection.register_forward_pre_hook(
                lambda _module, inputs: captured.__setitem__("t2g", inputs[0].detach().clone())
            ),
        ]
        hidden = torch.randn(2, 3, 12)
        normalized = torch.randn(2, 12, 3)
        try:
            with torch.no_grad():
                t2_out, _ = t2(hidden, normalized)
                t2g_out, _ = t2g(hidden, normalized)
        finally:
            for hook in hooks:
                hook.remove()
        torch.testing.assert_close(t2g_out, t2_out, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(captured["t2g"], captured["t2"], rtol=1e-6, atol=1e-6)

    def test_forecast_only_loss_trains_global_and_patch_paths(self):
        torch.manual_seed(5147)
        module = self._module(dropout=0.0).train()
        hidden = torch.randn(3, 3, 12, requires_grad=True)
        normalized = torch.randn(3, 12, 3, requires_grad=True)
        hidden_out, _ = module(hidden, normalized)
        weight = torch.linspace(
            -1.3, 1.7, hidden_out.numel(), dtype=hidden_out.dtype
        ).reshape_as(hidden_out)
        (hidden_out * weight).sum().backward()
        groups = (
            "patch_query_projection",
            "patch_query_norm",
            "global_query_projection",
            "global_query_norm",
            "global_bridge_norm",
            "global_injection_gate",
            "exogenous_projection",
            "exogenous_norm",
            "cross_attention",
            "patch_output_projection",
            "beta_global",
            "gamma_teb",
        )
        parameters = dict(module.named_parameters())
        for group in groups:
            selected = [
                (name, parameter)
                for name, parameter in parameters.items()
                if name == group or name.startswith(group + ".")
            ]
            self.assertTrue(selected, group)
            for name, parameter in selected:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
                self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0, name)

    def test_shape_dtype_device_and_single_target_contract(self):
        cases = [
            (torch.device("cpu"), torch.float32, 12, 3, 2),
            (torch.device("cpu"), torch.float64, 12, 3, 2),
            (torch.device("cpu"), torch.float32, 512, 32, 1),
        ]
        cuda_executed = False
        if torch.cuda.is_available():
            cases.append((
                torch.device("cuda", torch.cuda.current_device()),
                torch.float32,
                12,
                3,
                2,
            ))
            cuda_executed = True
        for device, dtype, seq_len, patch_size, batch_size in cases:
            with self.subTest(
                device=str(device), dtype=str(dtype), seq_len=seq_len
            ):
                torch.manual_seed(5167)
                module = self._module(
                    seq_len=seq_len,
                    patch_size=patch_size,
                    dropout=0.1,
                ).to(device=device, dtype=dtype).eval()
                hidden = torch.randn(
                    batch_size, 3, seq_len, device=device, dtype=dtype
                )
                normalized = torch.randn(
                    batch_size, seq_len, 3, device=device, dtype=dtype
                )
                with torch.no_grad():
                    hidden_out, context = module(hidden, normalized)
                self.assertEqual(hidden_out.shape, hidden.shape)
                self.assertEqual(context.shape, (batch_size, 8))
                self.assertEqual(hidden_out.dtype, dtype)
                self.assertEqual(context.dtype, dtype)
                self.assertEqual(hidden_out.device, device)
                self.assertEqual(context.device, device)
                self.assertTrue(torch.equal(hidden_out[:, 0, :], hidden[:, 0, :]))
                self.assertTrue(torch.equal(hidden_out[:, 2, :], hidden[:, 2, :]))
                self.assertTrue(torch.isfinite(hidden_out).all().item())
                self.assertTrue(torch.isfinite(context).all().item())
        print(f"T2G CUDA float32 executed={cuda_executed}")


if __name__ == "__main__":
    unittest.main()
