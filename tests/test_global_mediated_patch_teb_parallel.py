import unittest

import torch
import torch.nn as nn

from models.modules.global_mediated_patch_target_exogenous_bridge import (
    GlobalMediatedPatchTargetExogenousBridge,
)
from models.modules.target_exogenous_bridge import PARALLEL_MULTIVARIATE


class _RecordingAttention(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.calls = 0
        self.query_shape = None
        self.mask = None

    def forward(self, query, key, value, *, attn_mask, **kwargs):
        self.calls += 1
        self.query_shape = tuple(query.shape)
        self.mask = attn_mask.detach().clone()
        weights = query.new_zeros(
            query.shape[0], self.num_heads, query.shape[1], key.shape[1]
        )
        return query, weights


class GlobalMediatedPatchTEBParallelTests(unittest.TestCase):
    @staticmethod
    def _module(*, feature_num=3, target_idx=1):
        return GlobalMediatedPatchTargetExogenousBridge(
            seq_len=5,
            feature_num=feature_num,
            task_mode=PARALLEL_MULTIVARIATE,
            target_idx=target_idx,
            aux_idx=(),
            context_dim=8,
            num_heads=2,
            dropout=0.0,
            patch_size=3,
        )

    def test_one_vectorized_attention_owner_mask_and_gate_shape(self):
        torch.manual_seed(5201)
        module = self._module().eval()
        recorder = _RecordingAttention(module.num_heads)
        module.cross_attention = recorder
        gate_shapes = []
        hook = module.global_injection_gate.register_forward_hook(
            lambda _module, _inputs, output: gate_shapes.append(tuple(output.shape))
        )
        hidden = torch.randn(2, 3, 5)
        normalized = torch.randn(2, 5, 3)
        try:
            hidden_out, context, weights = module(
                hidden, normalized, need_weights=True
            )
        finally:
            hook.remove()
        query_count = 3 * (module.num_patches + 1)
        self.assertEqual(recorder.calls, 1)
        self.assertEqual(recorder.query_shape, (2, query_count, 8))
        self.assertEqual(recorder.mask.shape, (query_count, 3))
        expected_owner = torch.arange(3).repeat_interleave(module.num_patches + 1)
        expected_mask = torch.arange(3).unsqueeze(0).eq(expected_owner.unsqueeze(1))
        self.assertTrue(torch.equal(recorder.mask, expected_mask))
        self.assertEqual(gate_shapes, [(2, 3, module.num_patches, 1)])
        self.assertEqual(hidden_out.shape, hidden.shape)
        self.assertEqual(context.shape, (2, 8))
        self.assertEqual(weights.shape, (2, 2, query_count, 3))

    def test_real_attention_masks_self_key_and_updates_all_variables(self):
        torch.manual_seed(5219)
        module = self._module().eval()
        hidden = torch.randn(2, 3, 5)
        normalized = torch.randn(2, 5, 3)
        with torch.no_grad():
            hidden_out, _, weights = module(hidden, normalized, need_weights=True)
        mask = module._parallel_attention_mask(hidden.device)
        expanded = mask.unsqueeze(0).unsqueeze(0).expand_as(weights)
        self.assertTrue(torch.equal(
            weights.masked_select(expanded),
            torch.zeros_like(weights.masked_select(expanded)),
        ))
        for variable in range(3):
            self.assertGreater(
                (hidden_out[:, variable, :] - hidden[:, variable, :]).abs().max().item(),
                0.0,
            )

    def test_target_idx_only_selects_global_bridge_context(self):
        torch.manual_seed(5231)
        first = self._module(target_idx=0).eval()
        second = self._module(target_idx=2).eval()
        second.load_state_dict(first.state_dict(), strict=True)
        hidden = torch.randn(2, 3, 5)
        normalized = torch.randn(2, 5, 3)
        with torch.no_grad():
            first_out, first_context = first(hidden, normalized)
            second_out, second_context = second(hidden, normalized)
        self.assertTrue(torch.equal(first_out, second_out))
        self.assertFalse(torch.equal(first_context, second_context))

    def test_variable_permutation_equivariance(self):
        torch.manual_seed(5249)
        original = self._module(target_idx=1).eval()
        permutation = (2, 0, 1)
        permuted = self._module(target_idx=permutation.index(1)).eval()
        permuted.load_state_dict(original.state_dict(), strict=True)
        hidden = torch.randn(2, 3, 5)
        normalized = torch.randn(2, 5, 3)
        with torch.no_grad():
            original_out, original_context = original(hidden, normalized)
            permuted_out, permuted_context = permuted(
                hidden[:, permutation, :], normalized[:, :, permutation]
            )
        torch.testing.assert_close(
            permuted_out, original_out[:, permutation, :], rtol=1e-6, atol=1e-7
        )
        torch.testing.assert_close(
            permuted_context, original_context, rtol=1e-6, atol=1e-7
        )

    def test_parallel_c1_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least two variables"):
            self._module(feature_num=1, target_idx=0)


if __name__ == "__main__":
    unittest.main()
