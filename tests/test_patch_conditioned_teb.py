import math
import unittest

import torch
import torch.nn as nn

from models.modules.patch_conditioned_target_exogenous_bridge import (
    FIXED_SINUSOIDAL,
    RIGHT_ZERO_CROP,
    PatchConditionedTargetExogenousBridge,
)
from models.modules.target_exogenous_bridge import TARGET_EXOGENOUS


class _RecordingAttention(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.key = None

    def forward(self, query, key, value, **kwargs):
        self.key = key.detach().clone()
        weights = query.new_zeros(
            query.shape[0], self.num_heads, query.shape[1], key.shape[1]
        )
        return query, weights


class PatchConditionedTEBTests(unittest.TestCase):
    @staticmethod
    def _module(
        *,
        seq_len=12,
        patch_size=3,
        feature_num=3,
        context_dim=8,
        target_idx=1,
        aux_idx=(2, 0),
        dropout=0.0,
    ):
        return PatchConditionedTargetExogenousBridge(
            seq_len=seq_len,
            feature_num=feature_num,
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
        )

    def test_short_long_shape_dtype_and_device_contract(self):
        cases = [
            (torch.device("cpu"), torch.float32, 12, 3, 2),
            (torch.device("cpu"), torch.float64, 12, 3, 3),
            (torch.device("cpu"), torch.float32, 512, 32, 1),
        ]
        cuda_executed = False
        if torch.cuda.is_available():
            cases.append((
                torch.device("cuda", torch.cuda.current_device()), torch.float32, 12, 3, 2
            ))
            cuda_executed = True

        for device, dtype, seq_len, patch_size, batch_size in cases:
            with self.subTest(
                device=str(device),
                dtype=str(dtype),
                seq_len=seq_len,
                patch_size=patch_size,
            ):
                torch.manual_seed(4101)
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
                self.assertTrue(torch.isfinite(hidden_out).all().item())
                self.assertTrue(torch.isfinite(context).all().item())

        print(f"T2 CUDA float32 executed={cuda_executed}")

    def test_right_zero_patchify_crop_and_fixed_position_contract(self):
        non_divisible = self._module(seq_len=5, patch_size=3)
        target = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        patches = non_divisible._patchify(target)
        expected = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]]])
        self.assertTrue(torch.equal(patches, expected))
        self.assertTrue(torch.equal(patches.reshape(1, -1)[:, :5], target))
        self.assertEqual(non_divisible.num_patches, math.ceil(5 / 3))
        self.assertEqual(non_divisible.pad_len, 1)

        divisible = self._module(seq_len=12, patch_size=3)
        target_divisible = torch.arange(12, dtype=torch.float32).reshape(1, 12)
        patches_divisible = divisible._patchify(target_divisible)
        self.assertEqual(patches_divisible.shape, (1, 4, 3))
        self.assertTrue(
            torch.equal(patches_divisible.reshape(1, -1), target_divisible)
        )
        self.assertEqual(divisible.pad_len, 0)

        repeated = self._module(seq_len=5, patch_size=3)
        self.assertTrue(
            torch.equal(
                non_divisible.fixed_sinusoidal_position,
                repeated.fixed_sinusoidal_position,
            )
        )
        self.assertNotIn("fixed_sinusoidal_position", non_divisible.state_dict())
        self.assertEqual(
            non_divisible.double().fixed_sinusoidal_position.dtype,
            torch.float64,
        )

    def test_exact_parameter_counts_and_gamma(self):
        cases = ((512, 32, 39361), (12, 3, 5476))
        for seq_len, patch_size, expected in cases:
            with self.subTest(seq_len=seq_len, patch_size=patch_size):
                module = self._module(
                    seq_len=seq_len,
                    patch_size=patch_size,
                    context_dim=32,
                )
                count = sum(
                    parameter.numel()
                    for parameter in module.parameters()
                    if parameter.requires_grad
                )
                formula = (
                    2 * patch_size * 32
                    + 2 * seq_len * 32
                    + 4 * 32 * 32
                    + 13 * 32
                    + patch_size
                    + 1
                )
                self.assertEqual(count, expected)
                self.assertEqual(count, formula)
                self.assertEqual(module.gamma_teb.ndim, 0)
                self.assertAlmostEqual(module.gamma_teb.item(), 1e-3, places=9)
                self.assertNotIn(
                    "fixed_sinusoidal_position", set(module.state_dict())
                )

    def test_single_target_updates_only_target_and_preserves_aux_order(self):
        torch.manual_seed(4113)
        module = self._module(target_idx=1, aux_idx=(2, 0)).eval()
        recorder = _RecordingAttention(module.num_heads)
        module.cross_attention = recorder
        hidden = torch.randn(2, 3, 12)
        normalized = torch.randn(2, 12, 3)

        hidden_out, context, weights = module(
            hidden, normalized, need_weights=True
        )
        expected_auxiliary = normalized[:, :, (2, 0)].transpose(1, 2)
        expected_key = module.exogenous_norm(
            module.exogenous_projection(expected_auxiliary)
        )
        self.assertTrue(torch.equal(recorder.key, expected_key))
        self.assertTrue(torch.equal(hidden_out[:, 0, :], hidden[:, 0, :]))
        self.assertTrue(torch.equal(hidden_out[:, 2, :], hidden[:, 2, :]))
        self.assertGreater(
            (hidden_out[:, 1, :] - hidden[:, 1, :]).abs().max().item(),
            0.0,
        )
        self.assertEqual(context.shape, (2, 8))
        self.assertEqual(weights.shape, (2, 2, 5, 2))

    def test_input_and_all_logical_parameter_group_gradients_are_nonzero(self):
        torch.manual_seed(4127)
        module = self._module(dropout=0.0).train()
        hidden = torch.randn(3, 3, 12, requires_grad=True)
        normalized = torch.randn(3, 12, 3, requires_grad=True)
        hidden_out, context = module(hidden, normalized)
        hidden_weight = torch.linspace(
            0.2, 1.7, hidden_out.numel(), dtype=hidden_out.dtype
        ).reshape_as(hidden_out)
        context_weight = torch.linspace(
            -0.7, 0.9, context.numel(), dtype=context.dtype
        ).reshape_as(context)
        loss = (hidden_out * hidden_weight).sum() + (context * context_weight).sum()
        loss.backward()

        for name, tensor in (("hidden", hidden), ("normalized", normalized)):
            self.assertIsNotNone(tensor.grad, name)
            self.assertTrue(torch.isfinite(tensor.grad).all().item(), name)
            self.assertGreater(tensor.grad.abs().max().item(), 0.0, name)

        groups = (
            "patch_query_projection",
            "patch_query_norm",
            "global_query_projection",
            "global_query_norm",
            "exogenous_projection",
            "exogenous_norm",
            "cross_attention",
            "patch_output_projection",
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
                self.assertGreater(parameter.grad.abs().max().item(), 0.0, name)

    def test_constructor_and_forward_guards(self):
        invalid_aux = (
            ((), ValueError),
            ((0, 0), ValueError),
            ((1, 0), ValueError),
            ((3,), ValueError),
            ((True,), TypeError),
        )
        for aux_idx, error in invalid_aux:
            with self.subTest(aux_idx=aux_idx):
                with self.assertRaises(error):
                    self._module(aux_idx=aux_idx)

        invalid_kwargs = (
            {"seq_len": 0},
            {"patch_size": 13},
            {"context_dim": 7},
            {"dropout": 1.0},
        )
        for updates in invalid_kwargs:
            with self.subTest(updates=updates):
                kwargs = {
                    "seq_len": 12,
                    "patch_size": 3,
                    "context_dim": 8,
                    "dropout": 0.0,
                }
                kwargs.update(updates)
                with self.assertRaises(ValueError):
                    self._module(**kwargs)

        for field, value in (
            ("gamma_init", 0.0),
            ("padding_policy", "replicate"),
            ("position_policy", "learnable"),
        ):
            kwargs = dict(
                seq_len=12,
                feature_num=3,
                task_mode=TARGET_EXOGENOUS,
                target_idx=1,
                aux_idx=(0, 2),
                context_dim=8,
                num_heads=2,
                dropout=0.0,
                patch_size=3,
                gamma_init=1e-3,
                padding_policy=RIGHT_ZERO_CROP,
                position_policy=FIXED_SINUSOIDAL,
            )
            kwargs[field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    PatchConditionedTargetExogenousBridge(**kwargs)

        module = self._module()
        with self.assertRaisesRegex(ValueError, "hidden expects"):
            module(torch.randn(2, 3, 11), torch.randn(2, 12, 3))
        with self.assertRaisesRegex(ValueError, "normalized_input"):
            module(torch.randn(2, 3, 12), torch.randn(2, 11, 3))
        with self.assertRaisesRegex(TypeError, "floating-point"):
            module(
                torch.ones(2, 3, 12, dtype=torch.int64),
                torch.ones(2, 12, 3, dtype=torch.int64),
            )


if __name__ == "__main__":
    unittest.main()
