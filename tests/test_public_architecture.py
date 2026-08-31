import math
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import DDI, MDM
from models.modules.global_mediated_patch_target_exogenous_bridge import (
    GlobalMediatedPatchTargetExogenousBridge,
)
from models.modules.patch_conditioned_target_exogenous_bridge import (
    FIXED_SINUSOIDAL,
    RIGHT_ZERO_CROP,
    PatchConditionedTargetExogenousBridge,
)
from models.modules.target_exogenous_bridge import (
    TARGET_EXOGENOUS,
    TargetExogenousBridge,
)
from models.tsAMD import AMD
from models.tsmoe import AMS, TopKGating


def _set_golden_gate(gating):
    with torch.no_grad():
        gating.gate.weight.copy_(
            torch.tensor(
                [
                    [0.2, -0.1, 0.3],
                    [-0.4, 0.5, 0.1],
                    [0.7, -0.2, -0.3],
                    [0.1, 0.2, -0.5],
                ],
                dtype=torch.float32,
            )
        )
        gating.gate.bias.copy_(torch.tensor([0.05, -0.1, 0.2, 0.0]))
        gating.w_noise.zero_()


def _legacy_fp32_ams_forward(module, x, time_embedding):
    """Reference the public FP32 allocation and reduction order."""
    batch_size = x.shape[0]
    feature_num = x.shape[1]
    x = torch.transpose(x, 0, 1)
    time_embedding = torch.transpose(time_embedding, 0, 1)

    output = torch.zeros(feature_num, batch_size, module.pred_len).to(x.device)
    loss = 0
    for i in range(feature_num):
        channel_input = x[i]
        gates = module.gating(time_embedding[i])
        expert_outputs = torch.zeros(
            module.num_experts, batch_size, module.pred_len
        ).to(x.device)
        for j in range(module.num_experts):
            expert_outputs[j, :, :] = module.experts[j](channel_input)
        expert_outputs = torch.transpose(expert_outputs, 0, 1)
        gates = gates.unsqueeze(-1).expand(-1, -1, module.pred_len)
        output[i, :, :] = (gates * expert_outputs).sum(1)
        importance = gates.sum(0)
        loss += module.loss_coef * module.cv_squared(importance)
    return torch.transpose(output, 0, 1), loss


class ArchitectureContractTests(unittest.TestCase):
    def test_ddi_consumes_mdm_output_u_and_selector_reuses_u(self):
        class AddConstantMDM(nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value
                self.seen = None
                self.produced = None

            def forward(self, value):
                self.seen = value.detach().clone()
                self.produced = value + self.value
                return self.produced

        class RecordingDDI(nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value
                self.seen = None

            def forward(self, value):
                self.seen = value.detach().clone()
                return value + self.value

        class RecordingAMS(nn.Module):
            def __init__(self):
                super().__init__()
                self.expert_input = None
                self.selector_input = None

            def forward(self, expert_input, selector_input):
                self.expert_input = expert_input.detach().clone()
                self.selector_input = selector_input.detach().clone()
                return expert_input, expert_input.new_zeros(())

        model = AMD(
            (4, 3), 4, n_block=2, dropout=0.0, patch=2, k=1, c=2,
            alpha=0.0, target_slice=slice(0, None), norm=False,
            layernorm=False,
        ).eval()
        mdm = AddConstantMDM(10.0)
        first_ddi = RecordingDDI(100.0)
        second_ddi = RecordingDDI(1000.0)
        ams = RecordingAMS()
        model.pastmixing = mdm
        model.fc_blocks = nn.ModuleList([first_ddi, second_ddi])
        model.moe = ams

        original = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        output, auxiliary_loss = model(original)
        x_bcl = original.transpose(1, 2)
        u = x_bcl + 10.0

        self.assertTrue(torch.equal(mdm.seen, x_bcl))
        self.assertTrue(torch.equal(mdm.produced, u))
        self.assertTrue(torch.equal(first_ddi.seen, u))
        self.assertTrue(torch.equal(second_ddi.seen, u + 100.0))
        self.assertTrue(torch.equal(ams.expert_input, u + 1100.0))
        self.assertTrue(torch.equal(ams.selector_input, u))
        self.assertTrue(torch.equal(output, (u + 1100.0).transpose(1, 2)))
        self.assertEqual(auxiliary_loss.item(), 0.0)

    def test_paper_close_operator_contract_is_frozen(self):
        model = AMD(
            (8, 7), 3, n_block=1, dropout=0.1, patch=4, k=1, c=2,
            alpha=1.0, target_slice=slice(0, None), norm=True,
            layernorm=True,
        )
        self.assertIsInstance(model.pastmixing.norm, nn.LayerNorm)
        self.assertEqual(model.pastmixing.norm.normalized_shape, (8,))
        block = model.fc_blocks[0]
        self.assertIsInstance(block.norm, nn.LayerNorm)
        self.assertEqual(block.norm.normalized_shape, (8,))
        self.assertIsInstance(block.norm1, nn.BatchNorm1d)
        self.assertIsInstance(block.norm2, nn.BatchNorm1d)
        self.assertEqual(block.norm1.num_features, 4 * 7)
        self.assertEqual(block.norm2.num_features, 4 * 7)
        self.assertEqual(block.ff_dim, max(32, 2 ** math.ceil(math.log2(7))))
        self.assertEqual(block.ff_dim, 32)
        self.assertEqual(block.fc_block[0].in_features, 7)
        self.assertEqual(block.fc_block[0].out_features, 32)
        self.assertEqual(block.fc_block[3].in_features, 32)
        self.assertEqual(block.fc_block[3].out_features, 7)

        wide_block = DDI(
            (4, 33), dropout=0.0, patch=2, alpha=1.0, layernorm=False
        )
        self.assertEqual(wide_block.ff_dim, 64)

        self.assertEqual(model.moe.num_experts, 8)
        self.assertEqual(model.moe.top_k, 2)
        self.assertEqual(model.moe.gating.gate.in_features, 8)
        self.assertEqual(model.moe.gating.gate.out_features, 8)
        self.assertEqual(len(model.moe.experts), 8)
        expert = model.moe.experts[0].net
        self.assertEqual(expert[0].in_features, 8)
        self.assertEqual(expert[0].out_features, 2048)
        self.assertEqual(expert[-1].out_features, 3)

    def test_layernorm_false_keeps_released_internal_batch_norm(self):
        model = AMD(
            (8, 7), 3, n_block=1, dropout=0.0, patch=4, k=1, c=2,
            alpha=1.0, target_slice=slice(0, None), norm=False,
            layernorm=False,
        )
        self.assertFalse(hasattr(model.pastmixing, "norm"))
        block = model.fc_blocks[0]
        self.assertFalse(hasattr(block, "norm"))
        self.assertIsInstance(block.norm1, nn.BatchNorm1d)
        self.assertIsInstance(block.norm2, nn.BatchNorm1d)

    def test_layernorm_uses_each_channels_last_sequence_dimension(self):
        x = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [101.0, 102.0, 103.0, 104.0]]]
        )
        expected = F.layer_norm(x, (4,))
        cross_channel = F.layer_norm(x, (2, 4))

        mdm = MDM((4, 2), k=0, c=2, layernorm=True).train()
        mdm_output = mdm(x)
        torch.testing.assert_close(mdm_output, expected)
        self.assertFalse(torch.allclose(mdm_output, cross_channel))
        torch.testing.assert_close(
            mdm_output.mean(dim=-1), torch.zeros(1, 2), atol=5e-6, rtol=0
        )

        # With one full-length patch, DDI applies only its entry LayerNorm.
        ddi = DDI(
            (4, 2), dropout=0.0, patch=4, alpha=0.0, layernorm=True
        ).train()
        torch.testing.assert_close(ddi(x), expected)

    def test_selector_is_dense_simplex_and_horizon_shared(self):
        gating = TopKGating(3, 4, top_k=2).eval()
        _set_golden_gate(gating)
        inputs = torch.tensor(
            [[1.0, -2.0, 0.5], [-0.25, 0.75, 2.0]], dtype=torch.float32
        )
        actual = gating(inputs)
        expected = torch.tensor(
            [
                [0.0277403705, 0.0010230321, 0.9694588184, 0.0017777547],
                [0.4182974100, 0.5618477464, 0.0106211230, 0.0092337029],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(actual.sum(1), torch.ones(2), rtol=0, atol=1e-6)
        self.assertTrue(torch.all(actual > 0).item())

        expanded = actual.unsqueeze(-1).expand(-1, -1, 5)
        self.assertTrue(torch.equal(expanded[:, :, 0], expanded[:, :, -1]))

    def test_public_selector_auxiliary_loss_golden(self):
        module = AMS(
            (3, 2), pred_len=4, ff_dim=5, dropout=0.0,
            num_experts=4, top_k=2,
        ).eval()
        _set_golden_gate(module.gating)
        x = torch.tensor(
            [
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
                [[-0.3, 0.2, 0.5], [0.8, -0.4, 0.1]],
            ],
            dtype=torch.float32,
        )
        time_embedding = torch.tensor(
            [
                [[1.0, -2.0, 0.5], [-0.25, 0.75, 2.0]],
                [[0.2, 0.4, -0.3], [1.2, -0.2, 0.7]],
                [[-0.8, 0.6, 0.9], [0.3, 0.2, -0.5]],
            ],
            dtype=torch.float32,
        )
        _, loss = module(x, time_embedding)
        self.assertAlmostEqual(loss.item(), 0.8828197718, places=6)

        manual = 0
        for channel in range(2):
            gates = module.gating(time_embedding[:, channel, :])
            gates = gates.unsqueeze(-1).expand(-1, -1, module.pred_len)
            manual += module.loss_coef * module.cv_squared(gates.sum(0))
        self.assertTrue(torch.equal(loss, manual))

    def test_dtype_aware_allocation_preserves_public_fp32_results(self):
        torch.manual_seed(17)
        module = AMS(
            (4, 2), pred_len=3, ff_dim=7, dropout=0.0,
            num_experts=4, top_k=2,
        ).eval()
        x = torch.randn(3, 2, 4, dtype=torch.float32)
        time_embedding = torch.randn(3, 2, 4, dtype=torch.float32)
        expected_output, expected_loss = _legacy_fp32_ams_forward(
            module, x, time_embedding
        )
        actual_output, actual_loss = module(x, time_embedding)
        self.assertTrue(torch.equal(actual_output, expected_output))
        self.assertTrue(torch.equal(actual_loss, expected_loss))

        double_module = module.double()
        double_output, _ = double_module(x.double(), time_embedding.double())
        self.assertEqual(double_output.dtype, torch.float64)

    def test_ams_gradients_reach_all_public_parameters(self):
        torch.manual_seed(23)
        module = AMS(
            (4, 2), pred_len=3, ff_dim=7, dropout=0.0,
            num_experts=4, top_k=2,
        ).train()
        output, auxiliary_loss = module(
            torch.randn(3, 2, 4), torch.randn(3, 2, 4)
        )
        (output.square().mean() + auxiliary_loss).backward()
        for name, parameter in module.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)

    def test_valid_variant_forward_shapes(self):
        cases = (
            dict(norm=True, layernorm=True, alpha=0.0),
            dict(norm=False, layernorm=False, alpha=1.0),
        )
        for case in cases:
            with self.subTest(**case):
                model = AMD(
                    (8, 3), 4, n_block=1, dropout=0.0, patch=4,
                    k=1, c=2, target_slice=slice(0, None), **case
                ).eval()
                output, auxiliary_loss = model(torch.randn(2, 8, 3))
                self.assertEqual(output.shape, (2, 4, 3))
                self.assertEqual(auxiliary_loss.ndim, 0)
                self.assertTrue(torch.isfinite(output).all().item())
                self.assertTrue(torch.isfinite(auxiliary_loss).item())

    def test_constructor_guards_reject_unsupported_shapes(self):
        invalid_mdm = (
            dict(input_shape=(8, 3), k=-1, c=2),
            dict(input_shape=(8, 3), k=1, c=0),
            dict(input_shape=(8, 3), k=4, c=2),
        )
        for kwargs in invalid_mdm:
            with self.subTest(module="MDM", **kwargs):
                with self.assertRaises(ValueError):
                    MDM(**kwargs)

        for patch in (0, 3, 9):
            with self.subTest(module="DDI", patch=patch):
                with self.assertRaises(ValueError):
                    DDI((8, 3), patch=patch)

        for kwargs in (
            {"alpha": float("nan")},
            {"alpha": float("inf")},
            {"dropout": float("nan")},
        ):
            with self.subTest(module="DDI", **kwargs):
                with self.assertRaises(ValueError):
                    DDI((8, 3), patch=4, **kwargs)

        for top_k in (0, 5):
            with self.subTest(module="TopKGating", top_k=top_k):
                with self.assertRaises(ValueError):
                    TopKGating(8, 4, top_k=top_k)

    def test_forward_guards_report_shape_and_pair_mismatches(self):
        mdm = MDM((8, 3), k=1, c=2).eval()
        with self.assertRaisesRegex(ValueError, "expects"):
            mdm(torch.randn(2, 8, 3))

        ddi = DDI((8, 3), patch=4).eval()
        with self.assertRaisesRegex(ValueError, "expects"):
            ddi(torch.randn(2, 3, 7))

        ams = AMS((8, 3), 4).eval()
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            ams(torch.randn(2, 3, 8), torch.randn(2, 2, 8))

        model = AMD(
            (8, 3), 4, n_block=1, dropout=0.0, patch=4, k=1,
            c=2, alpha=0.0, target_slice=slice(0, None),
        ).eval()
        with self.assertRaisesRegex(ValueError, "expects"):
            model(torch.randn(2, 3, 8))

    def test_batch_size_one_training_error_is_explicit(self):
        model = AMD(
            (8, 3), 4, n_block=1, dropout=0.0, patch=4, k=1,
            c=2, alpha=0.0, target_slice=slice(0, None),
            norm=True, layernorm=True,
        ).train()
        with self.assertRaisesRegex(ValueError, "training batch size >= 2"):
            model(torch.randn(1, 8, 3))

        no_internal_batch_norm_execution = AMD(
            (4, 3), 4, n_block=1, dropout=0.0, patch=4, k=1,
            c=2, alpha=1.0, target_slice=slice(0, None),
            norm=False, layernorm=True,
        ).train()
        output, auxiliary_loss = no_internal_batch_norm_execution(
            torch.randn(1, 4, 3)
        )
        self.assertEqual(output.shape, (1, 4, 3))
        self.assertTrue(torch.isfinite(output).all().item())
        self.assertTrue(torch.isfinite(auxiliary_loss).item())

    def test_t2_is_an_independent_lightweight_public_bridge(self):
        self.assertFalse(
            issubclass(PatchConditionedTargetExogenousBridge, TargetExogenousBridge)
        )
        module = PatchConditionedTargetExogenousBridge(
            seq_len=12,
            feature_num=3,
            task_mode=TARGET_EXOGENOUS,
            target_idx=0,
            aux_idx=(1, 2),
            context_dim=32,
            num_heads=4,
            dropout=0.1,
            patch_size=3,
            padding_policy=RIGHT_ZERO_CROP,
            position_policy=FIXED_SINUSOIDAL,
        )
        self.assertEqual(
            set(dict(module.named_children())),
            {
                "patch_query_projection",
                "patch_query_norm",
                "global_query_projection",
                "global_query_norm",
                "exogenous_projection",
                "exogenous_norm",
                "cross_attention",
                "patch_output_projection",
            },
        )
        self.assertFalse(
            any(
                isinstance(child, (nn.TransformerEncoder, nn.Embedding))
                for child in module.modules()
            )
        )
        self.assertFalse(any("position" in key for key in module.state_dict()))


    def test_t2g_is_a_minimal_t2_extension_without_patch_residual_modules(self):
        self.assertTrue(issubclass(
            GlobalMediatedPatchTargetExogenousBridge,
            PatchConditionedTargetExogenousBridge,
        ))
        module = GlobalMediatedPatchTargetExogenousBridge(
            seq_len=12,
            feature_num=3,
            task_mode=TARGET_EXOGENOUS,
            target_idx=0,
            aux_idx=(1, 2),
            context_dim=32,
            num_heads=4,
            dropout=0.1,
            patch_size=3,
        )
        children = set(dict(module.named_children()))
        self.assertEqual(
            children,
            {
                "patch_query_projection",
                "patch_query_norm",
                "global_query_projection",
                "global_query_norm",
                "exogenous_projection",
                "exogenous_norm",
                "cross_attention",
                "patch_output_projection",
                "global_bridge_norm",
                "global_injection_gate",
            },
        )
        self.assertNotIn("patch_post_norm", children)
        self.assertNotIn("patch_residual_norm", children)
        self.assertFalse(any("position" in key for key in module.state_dict()))


if __name__ == "__main__":
    unittest.main()
