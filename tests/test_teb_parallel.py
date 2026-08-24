import unittest
from unittest import mock

import torch

from models.modules.target_exogenous_bridge import (
    PARALLEL_MULTIVARIATE,
    TargetExogenousBridge,
)


class ParallelTargetExogenousBridgeTests(unittest.TestCase):
    @staticmethod
    def _module(target_idx=1):
        return TargetExogenousBridge(
            seq_len=12,
            feature_num=4,
            task_mode=PARALLEL_MULTIVARIATE,
            target_idx=target_idx,
            aux_idx=(),
            context_dim=8,
            num_heads=2,
            dropout=0.0,
            gamma_init=1e-3,
        )

    def test_parallel_shape_diagonal_mask_and_context_anchor(self):
        torch.manual_seed(311)
        model = self._module(target_idx=1).eval()
        hidden = torch.randn(2, 4, 12)
        normalized = torch.randn(2, 12, 4)
        output, context, weights = model(
            hidden, normalized, need_weights=True
        )
        self.assertEqual(output.shape, (2, 4, 12))
        self.assertEqual(context.shape, (2, 8))
        self.assertEqual(weights.shape, (2, 2, 4, 4))
        diagonal = weights.diagonal(dim1=-2, dim2=-1)
        self.assertTrue(torch.equal(diagonal, torch.zeros_like(diagonal)))
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(
            torch.allclose(
                weights.sum(dim=-1),
                torch.ones_like(weights.sum(dim=-1)),
                atol=1e-7,
                rtol=1e-6,
            )
        )

    def test_parallel_float32_preserves_exact_cpu_or_cuda_device(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda", torch.cuda.current_device()))
        for device in devices:
            with self.subTest(device=str(device)):
                model = self._module().to(device=device, dtype=torch.float32).eval()
                hidden = torch.randn(2, 4, 12, device=device)
                normalized = torch.randn(2, 12, 4, device=device)
                output, context, weights = model(
                    hidden,
                    normalized,
                    need_weights=True,
                )
                for tensor in (output, context, weights):
                    self.assertEqual(tensor.dtype, torch.float32)
                    self.assertEqual(tensor.device, device)
                    self.assertTrue(torch.isfinite(tensor).all())

    def test_parallel_uses_one_vectorized_attention_call_and_updates_all_variables(self):
        torch.manual_seed(312)
        model = self._module().eval()
        hidden = torch.randn(2, 4, 12)
        normalized = torch.randn(2, 12, 4)
        with mock.patch.object(
            model.cross_attention,
            "forward",
            wraps=model.cross_attention.forward,
        ) as attention:
            output, _ = model(hidden, normalized)
        self.assertEqual(attention.call_count, 1)
        diagonal_mask = attention.call_args.kwargs["attn_mask"]
        self.assertEqual(diagonal_mask.shape, (4, 4))
        self.assertEqual(diagonal_mask.dtype, torch.bool)
        self.assertTrue(torch.equal(
            diagonal_mask, torch.eye(4, dtype=torch.bool)
        ))
        per_variable_change = (output - hidden).abs().amax(dim=(0, 2))
        self.assertTrue(torch.all(per_variable_change > 0))

    def test_target_idx_changes_only_exported_context_not_parallel_prediction(self):
        torch.manual_seed(313)
        first = self._module(target_idx=0).eval()
        second = self._module(target_idx=3).eval()
        second.load_state_dict(first.state_dict(), strict=True)
        hidden = torch.randn(2, 4, 12)
        normalized = torch.randn(2, 12, 4)
        out_a, context_a = first(hidden, normalized)
        out_b, context_b = second(hidden, normalized)
        self.assertTrue(torch.equal(out_a, out_b))
        self.assertFalse(torch.equal(context_a, context_b))

    def test_variable_permutation_equivariance(self):
        torch.manual_seed(314)
        model = self._module(target_idx=0).eval()
        hidden = torch.randn(2, 4, 12)
        normalized = torch.randn(2, 12, 4)
        permutation = torch.tensor([2, 0, 3, 1])
        output, _, weights = model(hidden, normalized, need_weights=True)
        permuted_output, _, permuted_weights = model(
            hidden[:, permutation, :],
            normalized[:, :, permutation],
            need_weights=True,
        )
        self.assertTrue(
            torch.allclose(
                permuted_output,
                output[:, permutation, :],
                atol=1e-7,
                rtol=1e-6,
            )
        )
        expected_weights = weights[:, :, permutation, :][:, :, :, permutation]
        self.assertTrue(
            torch.allclose(
                permuted_weights,
                expected_weights,
                atol=1e-7,
                rtol=1e-6,
            )
        )

    def test_parallel_rejects_single_variable_and_manual_aux_subset(self):
        common = {
            "seq_len": 12,
            "task_mode": PARALLEL_MULTIVARIATE,
            "target_idx": 0,
            "context_dim": 8,
            "num_heads": 2,
        }
        with self.assertRaisesRegex(
            ValueError, "Parallel TEB requires at least two variables"
        ):
            TargetExogenousBridge(
                **common,
                feature_num=1,
                aux_idx=(),
            )
        with self.assertRaisesRegex(ValueError, "aux_idx must be empty"):
            TargetExogenousBridge(
                **common,
                feature_num=4,
                aux_idx=(1,),
            )


if __name__ == "__main__":
    unittest.main()
