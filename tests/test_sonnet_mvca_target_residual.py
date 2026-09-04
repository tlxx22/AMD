"""Permanent regression tests for the M4 Sonnet S2 production module."""

import inspect
import math
import unittest
from unittest import mock

import torch
import torch.nn as nn

from models.modules.sonnet_mvca_target_residual import (
    SONNET_ALPHA,
    SONNET_ATTENTION_DROPOUT,
    SONNET_D_MODEL,
    SONNET_EPSILON,
    SONNET_GAMMA_INIT,
    SONNET_N_ATOMS,
    PaperDefinedMVCA,
    SonnetMVCATargetResidual,
)


URBAN_FEATURES = (
    "volume",
    "e_price",
    "s_price",
    "Ta",
    "P",
    "h",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_weekend",
)
ETTM1_FEATURES = (
    "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"
)


def _module(
    *,
    seq_len=12,
    feature_num=11,
    target_idx=0,
    ordered_aux_idx=tuple(range(1, 11)),
    feature_schema=URBAN_FEATURES,
    **overrides,
):
    kwargs = {
        "seq_len": seq_len,
        "feature_num": feature_num,
        "task_mode": "target_exogenous",
        "target_idx": target_idx,
        "ordered_aux_idx": ordered_aux_idx,
        "feature_schema": feature_schema,
        "schema_fingerprint": "permanent-test-schema",
        "d_model": SONNET_D_MODEL,
        "n_atoms": SONNET_N_ATOMS,
        "alpha": SONNET_ALPHA,
        "epsilon": SONNET_EPSILON,
        "attention_dropout": SONNET_ATTENTION_DROPOUT,
        "gamma_init": SONNET_GAMMA_INIT,
    }
    kwargs.update(overrides)
    return SonnetMVCATargetResidual(**kwargs)


class SonnetJointEmbeddingTests(unittest.TestCase):
    def test_gather_and_latent_orders_are_aux_then_target(self):
        module = _module()
        values = torch.arange(2 * 12 * 11, dtype=torch.float32).reshape(2, 12, 11)
        auxiliary, target = module.gather_sources(values)
        self.assertTrue(torch.equal(auxiliary, values[:, :, 1:]))
        self.assertTrue(torch.equal(target, values[:, :, 0:1]))
        embedding, aux_embedding, target_embedding = module.joint_embedding(values)
        self.assertEqual(aux_embedding.shape, (2, 12, 32))
        self.assertEqual(target_embedding.shape, (2, 12, 32))
        self.assertTrue(torch.equal(embedding[:, :, :32], aux_embedding))
        self.assertTrue(torch.equal(embedding[:, :, 32:], target_embedding))
        self.assertEqual(module.source_idx, (*range(1, 11), 0))
        self.assertIsNotNone(module.aux_embedding.bias)
        self.assertIsNotNone(module.target_embedding.bias)
        self.assertNotIn("alpha", dict(module.named_parameters()))

    def test_no_second_normalization_or_joint_dropout(self):
        module = _module()
        normalizations = [
            name
            for name, child in module.named_modules()
            if isinstance(
                child,
                (nn.LayerNorm, nn.BatchNorm1d, nn.InstanceNorm1d),
            )
        ]
        dropouts = [
            name for name, child in module.named_modules()
            if isinstance(child, nn.Dropout)
        ]
        self.assertEqual(normalizations, [])
        self.assertEqual(dropouts, ["mvca.attention_dropout"])

    def test_invalid_alpha_task_schema_and_indices_are_rejected(self):
        cases = (
            ({"alpha": 0.25}, ValueError, "alpha"),
            ({"task_mode": "parallel_multivariate"}, ValueError, "only"),
            ({"ordered_aux_idx": ()}, ValueError, "non-empty"),
            ({"ordered_aux_idx": (1, True)}, TypeError, "non-bool"),
            ({"ordered_aux_idx": (1, 1)}, ValueError, "duplicate"),
            ({"ordered_aux_idx": (1, 11)}, ValueError, "out-of-range"),
            ({"ordered_aux_idx": (0, 1)}, ValueError, "exclude"),
            ({"feature_schema": URBAN_FEATURES[:-1]}, ValueError, "contain"),
            (
                {"feature_schema": (*URBAN_FEATURES[:-1], URBAN_FEATURES[-2])},
                ValueError,
                "duplicate",
            ),
            ({"schema_fingerprint": ""}, ValueError, "fingerprint"),
        )
        for overrides, error, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(error, message):
                    _module(**overrides)


class SonnetWaveletTests(unittest.TestCase):
    def test_formula_shapes_and_runtime_grid_for_T12_and_T512(self):
        for seq_len, features, target_idx, aux_idx in (
            (12, URBAN_FEATURES, 0, tuple(range(1, 11))),
            (512, ETTM1_FEATURES, 6, tuple(range(6))),
        ):
            with self.subTest(seq_len=seq_len):
                module = _module(
                    seq_len=seq_len,
                    feature_num=len(features),
                    target_idx=target_idx,
                    ordered_aux_idx=aux_idx,
                    feature_schema=features,
                )
                x = torch.randn(1, seq_len, len(features))
                embedding, _, _ = module.joint_embedding(x)
                atoms = module.wavelet_atoms(embedding)
                time = torch.linspace(0, 1, seq_len).view(1, 1, seq_len)
                parameters = module.freq_params
                reference = torch.exp(
                    -parameters[:, :, 0, None] * time.square()
                ) * torch.cos(
                    parameters[:, :, 1, None] * time
                    + parameters[:, :, 2, None] * time.square()
                )
                torch.testing.assert_close(atoms, reference, rtol=0, atol=0)
                projected = module.wavelet_transform(embedding, atoms)
                self.assertEqual(projected.shape, (1, 8, seq_len, 64))
                torch.testing.assert_close(
                    projected,
                    embedding.unsqueeze(1)
                    * atoms.permute(1, 2, 0).unsqueeze(0),
                    rtol=0,
                    atol=0,
                )

    def test_freq_params_initialization_state_keys_and_parameter_golden(self):
        torch.manual_seed(812)
        module = _module(
            seq_len=512,
            feature_num=7,
            target_idx=6,
            ordered_aux_idx=tuple(range(6)),
            feature_schema=ETTM1_FEATURES,
        )
        self.assertEqual(module.freq_params.shape, (64, 8, 3))
        self.assertTrue(torch.isfinite(module.freq_params).all())
        self.assertGreater(torch.count_nonzero(module.freq_params), 0)
        expected_keys = {
            "gamma_sonnet",
            "freq_params",
            "aux_embedding.weight",
            "aux_embedding.bias",
            "target_embedding.weight",
            "target_embedding.bias",
            "mvca.qkv_projection.weight",
            "mvca.qkv_projection.bias",
            "mvca.residual_mlp.0.weight",
            "mvca.residual_mlp.0.bias",
            "mvca.residual_mlp.2.weight",
            "mvca.residual_mlp.2.bias",
            "mvca.output_projection.weight",
            "mvca.output_projection.bias",
            "readout.weight",
            "readout.bias",
        }
        self.assertEqual(set(module.state_dict()), expected_keys)
        self.assertEqual(sum(p.numel() for p in module.parameters()), 26850)
        urban = _module()
        self.assertEqual(sum(p.numel() for p in urban.parameters()), 26978)
        self.assertNotEqual(torch.count_nonzero(module.readout.weight), 0)
        self.assertTrue(torch.equal(module.readout.bias, torch.zeros(1)))


class PaperDefinedMVCATests(unittest.TestCase):
    def test_requires_wavelet_space_and_retains_shape(self):
        mvca = PaperDefinedMVCA()
        with self.assertRaisesRegex(ValueError, "wavelet-space"):
            mvca(torch.randn(2, 12, 64))
        output = mvca(torch.randn(2, 8, 12, 64))
        self.assertEqual(output.shape, (2, 8, 12, 64))

    def test_latent_rfft_coherence_scale_softmax_and_value_reference(self):
        torch.manual_seed(901)
        mvca = PaperDefinedMVCA().eval()
        projected = torch.randn(2, 8, 12, 64)
        original_rfft = torch.fft.rfft
        with mock.patch("torch.fft.rfft", wraps=original_rfft) as patched:
            output, details = mvca.forward_with_diagnostics(projected)
        self.assertEqual(patched.call_count, 2)
        self.assertTrue(all(call.kwargs["dim"] == -1 for call in patched.call_args_list))
        query = details["query"]
        key = details["key"]
        qf = original_rfft(query, dim=-1)
        kf = original_rfft(key, dim=-1)
        csd = (qf * torch.conj(kf)).mean(dim=-1)
        pqq = (qf * torch.conj(qf)).mean(dim=-1).real
        pkk = (kf * torch.conj(kf)).mean(dim=-1).real
        coherence = csd.abs().square() / (pqq * pkk + 1e-6)
        probabilities = torch.softmax(coherence / math.sqrt(64), dim=2)
        weighted = probabilities.unsqueeze(-1) * details["value"]
        hidden = weighted + mvca.residual_mlp(weighted)
        reference = mvca.output_projection(hidden)
        self.assertEqual(details["query_frequency"].dtype, torch.complex64)
        torch.testing.assert_close(details["coherence"], coherence)
        torch.testing.assert_close(details["probabilities"], probabilities)
        torch.testing.assert_close(details["weighted_value"], weighted)
        torch.testing.assert_close(output, reference)
        torch.testing.assert_close(
            probabilities.sum(dim=2),
            torch.ones_like(probabilities.sum(dim=2)),
        )
        self.assertEqual(details["weighted_value"].shape, (2, 8, 12, 64))
        self.assertNotIn("var_attn", dict(mvca.named_parameters()))
        self.assertFalse(hasattr(mvca, "var_attn"))
        self.assertNotIn("clamp", inspect.getsource(PaperDefinedMVCA.coherence))
        self.assertIsNotNone(mvca.qkv_projection.bias)
        self.assertEqual(len(mvca.residual_mlp), 3)
        self.assertIsNotNone(mvca.output_projection.bias)


class SonnetResidualAndGradientTests(unittest.TestCase):
    def test_target_only_writeback_gamma_and_exact_zero_target_delta_identity(self):
        torch.manual_seed(117)
        module = _module().eval()
        x = torch.randn(2, 12, 11)
        x[:, :, 0] = 0
        output, details = module.forward_with_diagnostics(x)
        self.assertAlmostEqual(module.gamma_sonnet.item(), 1e-3, places=9)
        self.assertTrue(torch.equal(output[:, :, 1:], x[:, :, 1:]))
        self.assertTrue(
            torch.equal(
                output[:, :, 0:1] - x[:, :, 0:1],
                details["gated_delta"],
            )
        )
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreater(torch.count_nonzero(details["delta"]), 0)

    def test_first_backward_reaches_every_logical_group(self):
        torch.manual_seed(2024)
        module = _module().train()
        output = module(torch.randn(4, 12, 11))
        output[:, :, 0].square().mean().backward()
        groups = (
            "aux_embedding",
            "target_embedding",
            "freq_params",
            "mvca.qkv_projection",
            "mvca.residual_mlp",
            "mvca.output_projection",
            "readout",
            "gamma_sonnet",
        )
        for group in groups:
            gradients = [
                parameter.grad
                for name, parameter in module.named_parameters()
                if name == group or name.startswith(group + ".")
            ]
            self.assertTrue(gradients, group)
            for gradient in gradients:
                self.assertIsNotNone(gradient, group)
                self.assertTrue(torch.isfinite(gradient).all(), group)
                self.assertGreater(torch.count_nonzero(gradient), 0, group)

    def test_zero_gamma_or_zero_readout_cuts_upstream_gradient(self):
        upstream_prefixes = (
            "aux_embedding",
            "target_embedding",
            "freq_params",
            "mvca.",
        )
        for mode in ("zero_gamma", "zero_readout"):
            with self.subTest(mode=mode):
                torch.manual_seed(44)
                module = _module().eval()
                with torch.no_grad():
                    if mode == "zero_gamma":
                        module.gamma_sonnet.zero_()
                    else:
                        module.readout.weight.zero_()
                        module.readout.bias.zero_()
                output = module(torch.randn(3, 12, 11))
                output[:, :, 0].sum().backward()
                for name, parameter in module.named_parameters():
                    if name.startswith(upstream_prefixes):
                        gradient = parameter.grad
                        self.assertTrue(
                            gradient is None or torch.count_nonzero(gradient) == 0,
                            f"{mode}:{name}",
                        )


if __name__ == "__main__":
    unittest.main()
