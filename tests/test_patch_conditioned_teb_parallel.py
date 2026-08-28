import unittest

import torch
import torch.nn as nn

from models.modules.patch_conditioned_target_exogenous_bridge import (
    FIXED_SINUSOIDAL,
    RIGHT_ZERO_CROP,
    PatchConditionedTargetExogenousBridge,
)
from models.modules.target_exogenous_bridge import PARALLEL_MULTIVARIATE


class _RecordingVectorizedAttention(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.calls = 0
        self.query_shape = None
        self.key_shape = None
        self.mask = None

    def forward(self, query, key, value, *, attn_mask, **kwargs):
        self.calls += 1
        self.query_shape = tuple(query.shape)
        self.key_shape = tuple(key.shape)
        self.mask = attn_mask.detach().clone()
        weights = query.new_zeros(
            query.shape[0], self.num_heads, query.shape[1], key.shape[1]
        )
        return query, weights


class PatchConditionedTEBParallelTests(unittest.TestCase):
    @staticmethod
    def _module(*, feature_num=3, target_idx=1, seq_len=5, patch_size=3):
        return PatchConditionedTargetExogenousBridge(
            seq_len=seq_len,
            feature_num=feature_num,
            task_mode=PARALLEL_MULTIVARIATE,
            target_idx=target_idx,
            aux_idx=(),
            context_dim=8,
            num_heads=2,
            dropout=0.0,
            patch_size=patch_size,
            gamma_init=1e-3,
            padding_policy=RIGHT_ZERO_CROP,
            position_policy=FIXED_SINUSOIDAL,
        )

    @staticmethod
    def _formal_context(module, hidden, normalized):
        batch_size = hidden.shape[0]
        patch_query = module._patch_queries(module._patchify(hidden))
        global_query = module.global_query_norm(
            module.global_query_projection(hidden)
        ).unsqueeze(2)
        query = torch.cat((patch_query, global_query), dim=2).reshape(
            batch_size,
            module.feature_num * (module.num_patches + 1),
            module.context_dim,
        )
        exogenous = module.exogenous_norm(
            module.exogenous_projection(normalized.transpose(1, 2))
        )
        context, _ = module.cross_attention(
            query,
            exogenous,
            exogenous,
            attn_mask=module._parallel_attention_mask(hidden.device),
            need_weights=False,
        )
        return context.reshape(
            batch_size,
            module.feature_num,
            module.num_patches + 1,
            module.context_dim,
        )

    def test_mask_shape_owner_contract_and_diagonal_attention_zero(self):
        torch.manual_seed(4201)
        module = self._module().eval()
        hidden = torch.randn(2, 3, 5)
        normalized = torch.randn(2, 5, 3)
        hidden_out, context, weights = module(
            hidden, normalized, need_weights=True
        )

        queries_per_variable = module.num_patches + 1
        expected_owner = torch.arange(3).repeat_interleave(queries_per_variable)
        expected_mask = torch.arange(3).unsqueeze(0).eq(
            expected_owner.unsqueeze(1)
        )
        actual_mask = module._parallel_attention_mask(torch.device("cpu"))
        self.assertTrue(torch.equal(actual_mask, expected_mask))
        self.assertEqual(actual_mask.shape, (3 * queries_per_variable, 3))
        self.assertEqual(weights.shape, (2, 2, 3 * queries_per_variable, 3))
        masked = actual_mask.unsqueeze(0).unsqueeze(0).expand_as(weights)
        self.assertTrue(torch.equal(weights.masked_select(masked), torch.zeros_like(
            weights.masked_select(masked)
        )))
        torch.testing.assert_close(
            weights.sum(dim=-1),
            torch.ones_like(weights.sum(dim=-1)),
            rtol=0,
            atol=1e-6,
        )
        self.assertEqual(hidden_out.shape, hidden.shape)
        self.assertEqual(context.shape, (2, 8))
        self.assertTrue(torch.isfinite(weights).all().item())

    def test_one_vectorized_attention_call_updates_every_variable(self):
        torch.manual_seed(4211)
        module = self._module().eval()
        recorder = _RecordingVectorizedAttention(module.num_heads)
        module.cross_attention = recorder
        with torch.no_grad():
            module.patch_output_projection.weight.zero_()
            module.patch_output_projection.weight[:, 0] = 1.0
            module.patch_output_projection.bias.fill_(0.25)
        hidden = torch.randn(2, 3, 5)
        normalized = torch.randn(2, 5, 3)
        hidden_out, _, _ = module(hidden, normalized, need_weights=True)

        query_count = 3 * (module.num_patches + 1)
        self.assertEqual(recorder.calls, 1)
        self.assertEqual(recorder.query_shape, (2, query_count, 8))
        self.assertEqual(recorder.key_shape, (2, 3, 8))
        self.assertEqual(recorder.mask.shape, (query_count, 3))
        for variable in range(3):
            self.assertGreater(
                (hidden_out[:, variable, :] - hidden[:, variable, :])
                .abs()
                .max()
                .item(),
                0.0,
            )

    def test_target_idx_only_selects_global_context(self):
        torch.manual_seed(4229)
        first = self._module(target_idx=0).eval()
        second = self._module(target_idx=2).eval()
        second.load_state_dict(first.state_dict(), strict=True)
        hidden = torch.randn(2, 3, 5)
        normalized = torch.randn(2, 5, 3)

        with torch.no_grad():
            first_out, first_context = first(hidden, normalized)
            second_out, second_context = second(hidden, normalized)
            context_all = self._formal_context(first, hidden, normalized)
            global_all = context_all[:, :, first.num_patches, :]
        self.assertTrue(torch.equal(first_out, second_out))
        torch.testing.assert_close(first_context, global_all[:, 0, :])
        torch.testing.assert_close(second_context, global_all[:, 2, :])

    def test_variable_permutation_equivariance_and_shared_parameters(self):
        torch.manual_seed(4241)
        original = self._module(feature_num=3, target_idx=1).eval()
        permutation = (2, 0, 1)
        permuted_target = permutation.index(1)
        permuted = self._module(
            feature_num=3, target_idx=permuted_target
        ).eval()
        permuted.load_state_dict(original.state_dict(), strict=True)
        hidden = torch.randn(2, 3, 5)
        normalized = torch.randn(2, 5, 3)

        with torch.no_grad():
            original_out, original_context = original(hidden, normalized)
            permuted_out, permuted_context = permuted(
                hidden[:, permutation, :], normalized[:, :, permutation]
            )
        torch.testing.assert_close(
            permuted_out,
            original_out[:, permutation, :],
            rtol=1e-6,
            atol=1e-7,
        )
        torch.testing.assert_close(
            permuted_context, original_context, rtol=1e-6, atol=1e-7
        )

        wider = self._module(feature_num=5, target_idx=3)
        self.assertEqual(
            [(name, tuple(value.shape)) for name, value in original.state_dict().items()],
            [(name, tuple(value.shape)) for name, value in wider.state_dict().items()],
        )

    def test_own_exogenous_token_is_masked_but_affects_other_variables(self):
        torch.manual_seed(4253)
        module = self._module().eval()
        hidden = torch.randn(2, 3, 5)
        normalized = torch.randn(2, 5, 3)
        modified = normalized.clone()
        modified[:, :, 1] += torch.tensor(
            [0.0, 3.0, -2.0, 5.0, -4.0]
        ).reshape(1, 5)

        with torch.no_grad():
            before, _ = module(hidden, normalized)
            after, _ = module(hidden, modified)
        self.assertTrue(torch.equal(before[:, 1, :], after[:, 1, :]))
        changed_other = torch.stack(
            (
                (before[:, 0, :] - after[:, 0, :]).abs().max(),
                (before[:, 2, :] - after[:, 2, :]).abs().max(),
            )
        )
        self.assertGreater(changed_other.max().item(), 0.0)

    def test_parallel_c1_and_aux_contract_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least two variables"):
            self._module(feature_num=1, target_idx=0)
        with self.assertRaisesRegex(ValueError, "aux_idx must be empty"):
            PatchConditionedTargetExogenousBridge(
                seq_len=5,
                feature_num=3,
                task_mode=PARALLEL_MULTIVARIATE,
                target_idx=1,
                aux_idx=(0,),
                context_dim=8,
                num_heads=2,
                dropout=0.0,
                patch_size=3,
            )


if __name__ == "__main__":
    unittest.main()
