import unittest

import torch
import torch.nn as nn
from models.common import RevIN

from models.modules.global_mediated_patch_target_exogenous_bridge import (
    GLOBAL_GATE_IDENTITY_INIT,
    GLOBAL_GATE_INPUT_CONTRACT,
    GLOBAL_GATE_SCALAR_PER_PATCH,
    GLOBAL_MEDIATED_PATCH_V1,
    GLOBAL_RESIDUAL_CONTRACT,
    PATCH_ATTENTION_RESIDUAL_NONE,
    GlobalMediatedPatchTargetExogenousBridge,
)
from models.modules.patch_conditioned_target_exogenous_bridge import (
    FIXED_SINUSOIDAL,
    PATCH_CONDITIONED_V1,
    RIGHT_ZERO_CROP,
    PatchConditionedTargetExogenousBridge,
)
from models.modules.selective_patch_target_exogenous_bridge import (
    GLOBAL_PREDICTION_ROLE_STATE_ONLY,
    PATCH_CONFIDENCE_GATE_SCALAR_POST_PROJECTION,
    PATCH_GATE_ACTIVATION_TWO_SIGMOID,
    PATCH_GATE_INIT_EXPLICIT_ZERO_IDENTITY,
    PATCH_GATE_INPUT_QUERY_AND_ATTENTION,
    SELECTIVE_PATCH_V1,
    SelectivePatchTargetExogenousBridge,
)
from models.modules.target_exogenous_bridge import (
    PARALLEL_MULTIVARIATE,
    TARGET_EXOGENOUS,
    TargetExogenousBridge,
)
from models.tsAMD import AMD
from models.tsAMD_enhanced import AMDEnhanced, GLOBAL_TEB_V1


def _capture_torch_rng_state():
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    }


def _restore_torch_rng_state(state):
    torch.set_rng_state(state["cpu"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _max_abs_error(left, right):
    return (left - right).abs().max().item()


class AMDEnhancedM0BTests(unittest.TestCase):
    def _model_kwargs(self):
        return {
            "input_shape": (4, 2),
            "pred_len": 3,
            "n_block": 1,
            "dropout": 0.1,
            "patch": 2,
            "k": 1,
            "c": 2,
            "alpha": 0.0,
            "target_slice": slice(0, None),
            "norm": True,
            "layernorm": True,
        }

    def test_frozen_weights_are_strictly_compatible(self):
        base = AMD(**self._model_kwargs())
        enhanced = AMDEnhanced(
            **self._model_kwargs(), target_idx=1, teb_context_dim=5
        )

        incompatible = enhanced.load_state_dict(base.state_dict(), strict=True)

        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertEqual(base.state_dict().keys(), enhanced.state_dict().keys())
        self.assertFalse(enhanced.use_pmcr)
        self.assertIsNone(enhanced.pmcr)

    def test_prediction_and_moe_loss_are_equivalent_on_cpu_and_cuda(self):
        original_rng = _capture_torch_rng_state()
        try:
            devices = [torch.device("cpu")]
            if torch.cuda.is_available():
                devices.append(torch.device("cuda"))

            for device in devices:
                with self.subTest(device=str(device)):
                    torch.manual_seed(20240815)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(20240815)

                    base = AMD(**self._model_kwargs()).to(device).eval()
                    enhanced = AMDEnhanced(
                        **self._model_kwargs(),
                        target_idx=1,
                        teb_context_dim=5,
                    ).to(device).eval()
                    enhanced.load_state_dict(base.state_dict(), strict=True)
                    x = torch.randn(2, 4, 2, device=device)

                    shared_rng = _capture_torch_rng_state()
                    _restore_torch_rng_state(shared_rng)
                    with torch.no_grad():
                        base_pred, base_moe = base(x)

                    _restore_torch_rng_state(shared_rng)
                    with torch.no_grad():
                        pass_pred, pass_moe = enhanced(
                            x, return_state_source=False
                        )

                    _restore_torch_rng_state(shared_rng)
                    with torch.no_grad():
                        state_pred, state_moe, state_source = enhanced(
                            x, return_state_source=True
                        )

                    errors = {
                        "pass_pred": _max_abs_error(base_pred, pass_pred),
                        "pass_moe": _max_abs_error(base_moe, pass_moe),
                        "state_pred": _max_abs_error(base_pred, state_pred),
                        "state_moe": _max_abs_error(base_moe, state_moe),
                    }
                    for name, error in errors.items():
                        self.assertLess(error, 1e-6, name)

                    self.assertEqual(state_source.shape, (2, 13))
                    self.assertEqual(state_source.dtype, x.dtype)
                    self.assertEqual(state_source.device, x.device)
                    self.assertTrue(
                        torch.equal(
                            state_source[:, -5:],
                            torch.zeros_like(state_source[:, -5:]),
                        )
                    )

                    print(
                        "M0-B equivalence "
                        f"device={device} "
                        + " ".join(
                            f"{name}_max_abs={error:.9g}"
                            for name, error in errors.items()
                        )
                    )
        finally:
            _restore_torch_rng_state(original_rng)

    def test_state_source_is_v_then_u_mdm_then_fixed_zero_context(self):
        class AddConstantMDM(nn.Module):
            def forward(self, value):
                return value + 10.0

        class AddConstantDDI(nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, value):
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

        model = AMDEnhanced(
            (4, 3),
            4,
            n_block=2,
            dropout=0.0,
            patch=2,
            k=1,
            c=2,
            alpha=0.0,
            target_slice=slice(0, None),
            norm=False,
            layernorm=False,
            target_idx=1,
            teb_context_dim=3,
        ).eval()
        model.pastmixing = AddConstantMDM()
        model.fc_blocks = nn.ModuleList(
            [AddConstantDDI(100.0), AddConstantDDI(1000.0)]
        )
        ams = RecordingAMS()
        model.moe = ams

        original = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        prediction, moe_loss, state_source = model(
            original, return_state_source=True
        )

        x_ch = original.transpose(1, 2)
        u_mdm = x_ch + 10.0
        v = u_mdm + 1100.0
        expected_state = torch.cat(
            (
                v[:, 1, :],
                u_mdm[:, 1, :],
                v.new_zeros((v.shape[0], 3)),
            ),
            dim=-1,
        )

        self.assertTrue(torch.equal(ams.expert_input, v))
        self.assertTrue(torch.equal(ams.selector_input, u_mdm))
        self.assertTrue(torch.equal(prediction, v.transpose(1, 2)))
        self.assertEqual(moe_loss.item(), 0.0)
        self.assertTrue(torch.equal(state_source, expected_state))
        self.assertEqual(state_source.shape, (2, 11))

    def test_m0b_contract_guards(self):
        with self.assertRaisesRegex(ValueError, "target_idx"):
            AMDEnhanced(
                **self._model_kwargs(), target_idx=2, teb_context_dim=5
            )
        with self.assertRaisesRegex(ValueError, "teb_context_dim"):
            AMDEnhanced(
                **self._model_kwargs(), target_idx=1, teb_context_dim=0
            )

        model = AMDEnhanced(
            **self._model_kwargs(), target_idx=1, teb_context_dim=5
        )
        with self.assertRaisesRegex(TypeError, "return_state_source"):
            model(torch.randn(2, 4, 2), return_state_source=1)

    def test_pmcr_enabled_forward_routing_and_state_source(self):
        class AddConstantMDM(nn.Module):
            def forward(self, value):
                return value + 10.0

        class AddConstantDDI(nn.Module):
            def forward(self, value):
                return value + 100.0

        class AddConstantPMCR(nn.Module):
            def forward(self, value):
                return value + 1000.0

        class RecordingAMS(nn.Module):
            def __init__(self):
                super().__init__()
                self.expert_input = None
                self.selector_input = None

            def forward(self, expert_input, selector_input):
                self.expert_input = expert_input.detach().clone()
                self.selector_input = selector_input.detach().clone()
                return expert_input, expert_input.new_tensor(7.0)

        model = AMDEnhanced(
            (4, 3),
            4,
            n_block=1,
            dropout=0.0,
            patch=2,
            k=1,
            c=2,
            alpha=0.0,
            target_slice=slice(0, None),
            norm=False,
            layernorm=False,
            target_idx=1,
            teb_context_dim=3,
            use_pmcr=True,
            pmcr_hidden_dim=8,
            pmcr_kernel_small=1,
            pmcr_kernel_large=3,
            pmcr_dropout=0.0,
        ).eval()
        model.pastmixing = AddConstantMDM()
        model.fc_blocks = nn.ModuleList([AddConstantDDI()])
        model.pmcr = AddConstantPMCR()
        ams = RecordingAMS()
        model.moe = ams

        original = torch.arange(24, dtype=torch.float64).reshape(2, 4, 3)
        prediction_plain, loss_plain = model(original, return_state_source=False)
        prediction, moe_loss, state_source = model(
            original, return_state_source=True
        )

        x_ch = original.transpose(1, 2)
        u_mdm = x_ch + 10.0
        v_pmcr = u_mdm + 1100.0
        expected_state = torch.cat(
            (
                v_pmcr[:, 1, :],
                u_mdm[:, 1, :],
                v_pmcr.new_zeros((2, 3)),
            ),
            dim=-1,
        )

        self.assertTrue(torch.equal(ams.expert_input, v_pmcr))
        self.assertTrue(torch.equal(ams.selector_input, u_mdm))
        self.assertTrue(torch.equal(prediction, v_pmcr.transpose(1, 2)))
        self.assertTrue(torch.equal(prediction_plain, prediction))
        self.assertTrue(torch.equal(loss_plain, moe_loss))
        self.assertTrue(torch.equal(state_source, expected_state))
        self.assertEqual(state_source.shape, (2, 11))
        self.assertEqual(state_source.dtype, original.dtype)
        self.assertEqual(state_source.device, original.device)
        self.assertTrue(torch.equal(state_source[:, -3:], torch.zeros_like(state_source[:, -3:])))

    def test_pmcr_enabled_module_parameters_and_shapes(self):
        model = AMDEnhanced(
            input_shape=(8, 2),
            pred_len=3,
            n_block=1,
            dropout=0.0,
            patch=2,
            k=1,
            c=2,
            alpha=0.0,
            target_slice=slice(0, None),
            norm=True,
            layernorm=True,
            target_idx=1,
            teb_context_dim=5,
            use_pmcr=True,
            pmcr_hidden_dim=8,
            pmcr_kernel_small=3,
            pmcr_kernel_large=7,
            pmcr_dropout=0.0,
        ).eval()
        self.assertIsNotNone(model.pmcr)
        self.assertTrue(any(key.startswith("pmcr.") for key in model.state_dict()))
        with torch.no_grad():
            prediction, moe_loss, state_source = model(
                torch.randn(2, 8, 2), return_state_source=True
            )
        self.assertEqual(prediction.shape, (2, 3, 2))
        self.assertEqual(moe_loss.ndim, 0)
        self.assertEqual(state_source.shape, (2, 21))

    def test_backbone_importer_uses_exact_pmcr_only_allowlist(self):
        base = AMD(**self._model_kwargs()).eval()
        enabled = AMDEnhanced(
            **self._model_kwargs(),
            target_idx=1,
            teb_context_dim=5,
            use_pmcr=True,
            pmcr_hidden_dim=8,
            pmcr_kernel_small=1,
            pmcr_kernel_large=3,
            pmcr_dropout=0.0,
        ).eval()
        pmcr_before = {
            key: value.clone()
            for key, value in enabled.state_dict().items()
            if key.startswith("pmcr.")
        }
        incompatible = enabled.load_amd_backbone_state_dict(base.state_dict())
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        for key, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, enabled.state_dict()[key]), key)
        for key, value in pmcr_before.items():
            self.assertTrue(torch.equal(value, enabled.state_dict()[key]), key)

        missing_backbone = base.state_dict().copy()
        missing_backbone.pop(next(iter(missing_backbone)))
        with self.assertRaisesRegex(RuntimeError, "missing_non_pmcr"):
            enabled.load_amd_backbone_state_dict(missing_backbone)

        unexpected = base.state_dict().copy()
        unexpected["unexpected.weight"] = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            enabled.load_amd_backbone_state_dict(unexpected)

        restored = AMDEnhanced(
            **self._model_kwargs(),
            target_idx=1,
            teb_context_dim=5,
            use_pmcr=True,
            pmcr_hidden_dim=8,
            pmcr_kernel_small=1,
            pmcr_kernel_large=3,
            pmcr_dropout=0.0,
        )
        complete = restored.load_state_dict(enabled.state_dict(), strict=True)
        self.assertEqual(complete.missing_keys, [])
        self.assertEqual(complete.unexpected_keys, [])

        disabled = AMDEnhanced(
            **self._model_kwargs(), target_idx=1, teb_context_dim=5
        )
        strict = disabled.load_amd_backbone_state_dict(base.state_dict())
        self.assertEqual(strict.missing_keys, [])
        self.assertEqual(strict.unexpected_keys, [])

    def test_pmcr_configuration_requires_explicit_valid_values(self):
        with self.assertRaisesRegex(ValueError, "explicit"):
            AMDEnhanced(
                **self._model_kwargs(),
                target_idx=1,
                teb_context_dim=5,
                use_pmcr=True,
            )
        with self.assertRaisesRegex(ValueError, "<= seq_len"):
            AMDEnhanced(
                **self._model_kwargs(),
                target_idx=1,
                teb_context_dim=5,
                use_pmcr=True,
                pmcr_hidden_dim=8,
                pmcr_kernel_small=3,
                pmcr_kernel_large=7,
            )
        with self.assertRaisesRegex(TypeError, "use_pmcr"):
            AMDEnhanced(
                **self._model_kwargs(),
                target_idx=1,
                teb_context_dim=5,
                use_pmcr=1,
            )


class AMDEnhancedM3Tests(unittest.TestCase):
    @staticmethod
    def _backbone_kwargs():
        return {
            "input_shape": (4, 3),
            "pred_len": 2,
            "n_block": 1,
            "dropout": 0.0,
            "patch": 4,
            "k": 0,
            "c": 2,
            "alpha": 0.0,
            "target_slice": None,
            "norm": False,
            "layernorm": False,
        }

    def _enhanced(
        self,
        *,
        use_pmcr=False,
        use_teb=False,
        task_mode=TARGET_EXOGENOUS,
        norm=False,
    ):
        kwargs = self._backbone_kwargs()
        kwargs["norm"] = norm
        return AMDEnhanced(
            **kwargs,
            target_idx=1,
            teb_context_dim=4,
            task_mode=task_mode,
            aux_idx=(0, 2) if task_mode == TARGET_EXOGENOUS else (),
            use_pmcr=use_pmcr,
            pmcr_hidden_dim=4 if use_pmcr else None,
            pmcr_kernel_small=1 if use_pmcr else None,
            pmcr_kernel_large=3 if use_pmcr else None,
            pmcr_dropout=0.0,
            use_teb=use_teb,
            teb_heads=2,
            teb_dropout=0.0,
            teb_gamma_init=1e-3,
        )

    def test_pmcr_teb_four_switch_matrix_and_state_keys(self):
        torch.manual_seed(3103)
        x = torch.randn(2, 4, 3)
        for use_pmcr, use_teb in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            with self.subTest(use_pmcr=use_pmcr, use_teb=use_teb):
                model = self._enhanced(
                    use_pmcr=use_pmcr,
                    use_teb=use_teb,
                ).eval()
                keys = set(model.state_dict())
                self.assertEqual(model.pmcr is not None, use_pmcr)
                self.assertEqual(model.teb is not None, use_teb)
                self.assertEqual(any(key.startswith("pmcr.") for key in keys), use_pmcr)
                self.assertEqual(any(key.startswith("teb.") for key in keys), use_teb)

                with torch.no_grad():
                    prediction, moe_loss, state_source = model(
                        x, return_state_source=True
                    )
                self.assertEqual(prediction.shape, (2, 2, 1))
                self.assertEqual(moe_loss.ndim, 0)
                self.assertEqual(state_source.shape, (2, 12))
                context = state_source[:, 8:]
                if use_teb:
                    self.assertGreater(context.abs().max().item(), 0.0)
                else:
                    self.assertTrue(torch.equal(context, torch.zeros_like(context)))

    def test_final_routing_and_state_source_use_teb_output_only_for_experts(self):
        class AddConstantMDM(nn.Module):
            def forward(self, value):
                return value + 10.0

        class AddConstantDDI(nn.Module):
            def forward(self, value):
                return value + 100.0

        class AddConstantPMCR(nn.Module):
            def forward(self, value):
                return value + 1000.0

        class RecordingTEB(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden = None
                self.normalized_input = None

            def forward(self, *, hidden, normalized_input):
                self.hidden = hidden.detach().clone()
                self.normalized_input = normalized_input.detach().clone()
                return hidden + 10000.0, hidden.new_full((hidden.shape[0], 4), 7.0)

        class RecordingAMS(nn.Module):
            def __init__(self):
                super().__init__()
                self.expert_input = None
                self.selector_input = None

            def forward(self, expert_input, selector_input):
                self.expert_input = expert_input.detach().clone()
                self.selector_input = selector_input.detach().clone()
                return expert_input, expert_input.new_tensor(5.0)

        model = self._enhanced(use_pmcr=True, use_teb=True).eval()
        model.pastmixing = AddConstantMDM()
        model.fc_blocks = nn.ModuleList([AddConstantDDI()])
        model.pmcr = AddConstantPMCR()
        teb = RecordingTEB()
        model.teb = teb
        ams = RecordingAMS()
        model.moe = ams

        x = torch.arange(24, dtype=torch.float64).reshape(2, 4, 3)
        prediction, moe_loss, state_source = model(x, return_state_source=True)
        x_ch = x.transpose(1, 2)
        u_mdm = x_ch + 10.0
        v_local = u_mdm + 1100.0
        v_final = v_local + 10000.0

        self.assertTrue(torch.equal(teb.hidden, v_local))
        self.assertTrue(torch.equal(teb.normalized_input, x))
        self.assertTrue(torch.equal(ams.expert_input, v_final))
        self.assertTrue(torch.equal(ams.selector_input, u_mdm))
        self.assertTrue(torch.equal(prediction, v_final[:, 1:2, :].transpose(1, 2)))
        self.assertEqual(moe_loss.item(), 5.0)
        expected_state = torch.cat(
            (
                v_final[:, 1, :],
                u_mdm[:, 1, :],
                v_final.new_full((2, 4), 7.0),
            ),
            dim=-1,
        )
        self.assertTrue(torch.equal(state_source, expected_state))

    def test_full_channel_revin_denorm_precedes_formal_task_selection(self):
        class FixedAMS(nn.Module):
            def __init__(self, normalized_prediction):
                super().__init__()
                self.register_buffer("normalized_prediction", normalized_prediction)

            def forward(self, expert_input, selector_input):
                prediction = self.normalized_prediction.expand(
                    expert_input.shape[0], -1, -1
                )
                return prediction, expert_input.new_zeros(())

        x = torch.tensor(
            [
                [[1.0, 10.0, -5.0], [2.0, 14.0, -1.0],
                 [5.0, 22.0, 7.0], [8.0, 30.0, 15.0]],
                [[3.0, -2.0, 40.0], [7.0, 2.0, 44.0],
                 [11.0, 10.0, 52.0], [15.0, 18.0, 60.0]],
            ]
        )
        normalized_bch = torch.tensor(
            [[[0.5, -0.5], [1.0, -1.0], [1.5, -1.5]]]
        )

        target_model = self._enhanced(norm=True).eval()
        with torch.no_grad():
            target_model.rev_norm.affine_weight.copy_(torch.tensor([2.0, 3.0, 4.0]))
            target_model.rev_norm.affine_bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
        target_model.pastmixing = nn.Identity()
        target_model.fc_blocks = nn.ModuleList()
        target_model.moe = FixedAMS(normalized_bch)

        prediction, _ = target_model(x)
        mean = x.mean(dim=1, keepdim=True)
        scale = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        normalized_bhc = normalized_bch.expand(2, -1, -1).transpose(1, 2)
        expected_all = (
            (normalized_bhc - target_model.rev_norm.affine_bias)
            / (target_model.rev_norm.affine_weight + 1e-10)
            * scale
            + mean
        )
        self.assertEqual(prediction.shape, (2, 2, 1))
        torch.testing.assert_close(prediction, expected_all[:, :, 1:2])

        parallel_model = self._enhanced(
            task_mode=PARALLEL_MULTIVARIATE,
            norm=True,
        ).eval()
        with torch.no_grad():
            parallel_model.rev_norm.affine_weight.copy_(
                target_model.rev_norm.affine_weight
            )
            parallel_model.rev_norm.affine_bias.copy_(target_model.rev_norm.affine_bias)
        parallel_model.pastmixing = nn.Identity()
        parallel_model.fc_blocks = nn.ModuleList()
        parallel_model.moe = FixedAMS(normalized_bch)
        parallel_prediction, _ = parallel_model(x)
        self.assertEqual(parallel_prediction.shape, (2, 2, 3))
        torch.testing.assert_close(parallel_prediction, expected_all)

    def test_u1_target_output_is_frozen_amd_slice_without_enhancement_state(self):
        target_idx = 2
        kwargs = self._backbone_kwargs()
        frozen_kwargs = dict(kwargs)
        frozen_kwargs["target_slice"] = slice(None)
        frozen = AMD(**frozen_kwargs).eval()
        u1 = AMDEnhanced(
            **kwargs,
            target_idx=target_idx,
            teb_context_dim=4,
            task_mode=TARGET_EXOGENOUS,
            aux_idx=(0, 1),
            use_pmcr=False,
            use_teb=False,
            teb_gamma_init=1e-3,
        ).eval()
        u1.load_state_dict(frozen.state_dict(), strict=True)
        self.assertIsNone(u1.pmcr)
        self.assertIsNone(u1.teb)
        self.assertFalse(
            any(
                key.startswith(("pmcr.", "teb."))
                for key in u1.state_dict()
            )
        )

        torch.manual_seed(591)
        x = torch.randn(2, 4, 3)
        shared_rng = _capture_torch_rng_state()
        _restore_torch_rng_state(shared_rng)
        with torch.no_grad():
            frozen_prediction, frozen_moe = frozen(x)
        _restore_torch_rng_state(shared_rng)
        with torch.no_grad():
            u1_prediction, u1_moe = u1(x)
        expected = frozen_prediction[:, :, target_idx : target_idx + 1]
        self.assertEqual(u1_prediction.shape, (2, kwargs["pred_len"], 1))
        self.assertLess(_max_abs_error(u1_prediction, expected), 1e-6)
        self.assertLess(_max_abs_error(u1_moe, frozen_moe), 1e-6)

    def test_formal_off_off_revin_parity_matrix(self):
        cases = [
            (torch.device("cpu"), torch.float32, True, True, 0, PARALLEL_MULTIVARIATE),
            (torch.device("cpu"), torch.float32, False, None, 2, TARGET_EXOGENOUS),
            (torch.device("cpu"), torch.float64, True, False, 2, TARGET_EXOGENOUS),
            (torch.device("cpu"), torch.float64, False, None, 0, PARALLEL_MULTIVARIATE),
        ]
        if torch.cuda.is_available():
            cuda = torch.device("cuda", torch.cuda.current_device())
            cases.extend(
                [
                    (cuda, torch.float32, True, True, 0, TARGET_EXOGENOUS),
                    (cuda, torch.float32, True, False, 2, PARALLEL_MULTIVARIATE),
                    (cuda, torch.float32, False, None, 0, TARGET_EXOGENOUS),
                ]
            )

        max_prediction_error = 0.0
        max_moe_error = 0.0
        for device, dtype, norm, affine, target_idx, task_mode in cases:
            with self.subTest(
                device=str(device),
                dtype=str(dtype),
                norm=norm,
                affine=affine,
                target_idx=target_idx,
                task_mode=task_mode,
            ):
                kwargs = self._backbone_kwargs()
                kwargs["norm"] = norm
                frozen_kwargs = dict(kwargs)
                frozen_kwargs["target_slice"] = slice(None)
                formal_kwargs = dict(kwargs)
                formal_kwargs["target_slice"] = None
                frozen = AMD(**frozen_kwargs)
                formal = AMDEnhanced(
                    **formal_kwargs,
                    target_idx=target_idx,
                    teb_context_dim=4,
                    task_mode=task_mode,
                    aux_idx=(
                        tuple(index for index in range(3) if index != target_idx)
                        if task_mode == TARGET_EXOGENOUS
                        else ()
                    ),
                    use_pmcr=False,
                    use_teb=False,
                    teb_gamma_init=1e-3,
                )
                if norm and affine is False:
                    frozen.rev_norm = RevIN(3, affine=False)
                    formal.rev_norm = RevIN(3, affine=False)
                frozen = frozen.to(device=device, dtype=dtype).eval()
                formal = formal.to(device=device, dtype=dtype).eval()
                formal.load_state_dict(frozen.state_dict(), strict=True)
                torch.manual_seed(3411)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(3411)
                x = torch.randn(2, 4, 3, device=device, dtype=dtype)
                shared_rng = _capture_torch_rng_state()
                _restore_torch_rng_state(shared_rng)
                with torch.no_grad():
                    frozen_prediction, frozen_moe = frozen(x)
                expected_prediction = (
                    frozen_prediction
                    if task_mode == PARALLEL_MULTIVARIATE
                    else frozen_prediction[:, :, target_idx : target_idx + 1]
                )
                _restore_torch_rng_state(shared_rng)
                with torch.no_grad():
                    formal_prediction, formal_moe = formal(
                        x,
                        return_state_source=False,
                    )
                _restore_torch_rng_state(shared_rng)
                with torch.no_grad():
                    state_prediction, state_moe, _ = formal(
                        x,
                        return_state_source=True,
                    )
                for observed_prediction, observed_moe in (
                    (formal_prediction, formal_moe),
                    (state_prediction, state_moe),
                ):
                    prediction_error = _max_abs_error(
                        expected_prediction,
                        observed_prediction,
                    )
                    moe_error = _max_abs_error(frozen_moe, observed_moe)
                    max_prediction_error = max(max_prediction_error, prediction_error)
                    max_moe_error = max(max_moe_error, moe_error)
                    self.assertLess(prediction_error, 1e-6)
                    self.assertLess(moe_error, 1e-6)
        print(
            "formal RevIN off/off parity "
            f"prediction_max_abs={max_prediction_error:.9g} "
            f"moe_max_abs={max_moe_error:.9g}"
        )

    def test_teb_gamma_is_fixed_across_amd_enhanced_public_api(self):
        for gamma in (0.0, 1e-2, -1e-3, float("nan"), float("inf")):
            for use_teb in (False, True):
                with self.subTest(gamma=gamma, use_teb=use_teb):
                    with self.assertRaisesRegex(ValueError, "fixed at 1e-3"):
                        AMDEnhanced(
                            **self._backbone_kwargs(),
                            target_idx=1,
                            teb_context_dim=4,
                            task_mode=TARGET_EXOGENOUS,
                            aux_idx=(0, 2),
                            use_teb=use_teb,
                            teb_heads=2,
                            teb_dropout=0.0,
                            teb_gamma_init=gamma,
                        )
    def test_t2_integration_preserves_routing_and_fixed_state_source_shape(self):
        kwargs = self._backbone_kwargs()
        model = AMDEnhanced(
            **kwargs,
            target_idx=1,
            teb_context_dim=32,
            task_mode=TARGET_EXOGENOUS,
            aux_idx=(0, 2),
            use_pmcr=False,
            use_teb=True,
            teb_heads=4,
            teb_dropout=0.1,
            teb_gamma_init=1e-3,
            teb_architecture=PATCH_CONDITIONED_V1,
            teb_patch_size=2,
            teb_patch_padding=RIGHT_ZERO_CROP,
            teb_patch_position=FIXED_SINUSOIDAL,
        ).eval()
        self.assertIsInstance(model.teb, PatchConditionedTargetExogenousBridge)

        captured = {}

        def capture_u(_module, _inputs, output):
            captured["u_mdm"] = output.detach().clone()

        def capture_teb(_module, _inputs, output):
            captured["v_final"] = output[0].detach().clone()
            captured["context"] = output[1].detach().clone()

        handles = (
            model.pastmixing.register_forward_hook(capture_u),
            model.teb.register_forward_hook(capture_teb),
        )
        try:
            torch.manual_seed(3451)
            x = torch.randn(2, 4, 3)
            with torch.no_grad():
                prediction, moe_loss, state_source = model(
                    x, return_state_source=True
                )
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(prediction.shape, (2, 2, 1))
        self.assertEqual(moe_loss.ndim, 0)
        self.assertEqual(state_source.shape, (2, 2 * 4 + 32))
        self.assertTrue(
            torch.equal(state_source[:, :4], captured["v_final"][:, 1, :])
        )
        self.assertTrue(
            torch.equal(state_source[:, 4:8], captured["u_mdm"][:, 1, :])
        )
        self.assertTrue(torch.equal(state_source[:, 8:], captured["context"]))

    def test_t2g_integration_preserves_routing_and_state_source_shape(self):
        model = AMDEnhanced(
            **self._backbone_kwargs(),
            target_idx=1,
            teb_context_dim=32,
            task_mode=TARGET_EXOGENOUS,
            aux_idx=(0, 2),
            use_pmcr=False,
            use_teb=True,
            teb_heads=4,
            teb_dropout=0.1,
            teb_gamma_init=1e-3,
            teb_architecture=GLOBAL_MEDIATED_PATCH_V1,
            teb_patch_size=2,
            teb_patch_padding=RIGHT_ZERO_CROP,
            teb_patch_position=FIXED_SINUSOIDAL,
            teb_global_residual=GLOBAL_RESIDUAL_CONTRACT,
            teb_patch_attention_residual=PATCH_ATTENTION_RESIDUAL_NONE,
            teb_global_gate=GLOBAL_GATE_SCALAR_PER_PATCH,
            teb_global_gate_input=GLOBAL_GATE_INPUT_CONTRACT,
            teb_global_gate_init=GLOBAL_GATE_IDENTITY_INIT,
            teb_beta_global_init=1e-3,
        ).eval()
        self.assertIsInstance(
            model.teb, GlobalMediatedPatchTargetExogenousBridge
        )
        captured = {}
        handles = (
            model.pastmixing.register_forward_hook(
                lambda _module, _inputs, output: captured.__setitem__(
                    "u_mdm", output.detach().clone()
                )
            ),
            model.teb.register_forward_hook(
                lambda _module, _inputs, output: captured.update({
                    "v_final": output[0].detach().clone(),
                    "context": output[1].detach().clone(),
                })
            ),
        )
        try:
            torch.manual_seed(3463)
            with torch.no_grad():
                prediction, moe_loss, state_source = model(
                    torch.randn(2, 4, 3), return_state_source=True
                )
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(prediction.shape, (2, 2, 1))
        self.assertEqual(moe_loss.ndim, 0)
        self.assertEqual(state_source.shape, (2, 2 * 4 + 32))
        self.assertTrue(torch.equal(
            state_source[:, :4], captured["v_final"][:, 1, :]
        ))
        self.assertTrue(torch.equal(
            state_source[:, 4:8], captured["u_mdm"][:, 1, :]
        ))
        self.assertTrue(torch.equal(state_source[:, 8:], captured["context"]))

    def test_t3_integration_preserves_routing_and_state_source_shape(self):
        model = AMDEnhanced(
            **self._backbone_kwargs(),
            target_idx=1,
            teb_context_dim=32,
            task_mode=TARGET_EXOGENOUS,
            aux_idx=(0, 2),
            use_pmcr=False,
            use_teb=True,
            teb_heads=4,
            teb_dropout=0.1,
            teb_gamma_init=1e-3,
            teb_architecture=SELECTIVE_PATCH_V1,
            teb_patch_size=2,
            teb_patch_padding=RIGHT_ZERO_CROP,
            teb_patch_position=FIXED_SINUSOIDAL,
            teb_patch_confidence_gate=(
                PATCH_CONFIDENCE_GATE_SCALAR_POST_PROJECTION
            ),
            teb_patch_gate_input=PATCH_GATE_INPUT_QUERY_AND_ATTENTION,
            teb_patch_gate_activation=PATCH_GATE_ACTIVATION_TWO_SIGMOID,
            teb_patch_gate_init=PATCH_GATE_INIT_EXPLICIT_ZERO_IDENTITY,
            teb_global_prediction_role=GLOBAL_PREDICTION_ROLE_STATE_ONLY,
        ).eval()
        self.assertIsInstance(model.teb, SelectivePatchTargetExogenousBridge)
        self.assertFalse(any(
            token in key
            for key in model.teb.state_dict()
            for token in ("beta_global", "global_bridge", "global_injection")
        ))
        captured = {}
        handles = (
            model.pastmixing.register_forward_hook(
                lambda _module, _inputs, output: captured.__setitem__(
                    "u_mdm", output.detach().clone()
                )
            ),
            model.teb.register_forward_hook(
                lambda _module, _inputs, output: captured.update({
                    "v_final": output[0].detach().clone(),
                    "context": output[1].detach().clone(),
                })
            ),
        )
        try:
            torch.manual_seed(3467)
            with torch.no_grad():
                prediction, moe_loss, state_source = model(
                    torch.randn(2, 4, 3), return_state_source=True
                )
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(prediction.shape, (2, 2, 1))
        self.assertEqual(moe_loss.ndim, 0)
        self.assertEqual(state_source.shape, (2, 2 * 4 + 32))
        self.assertTrue(torch.equal(
            state_source[:, :4], captured["v_final"][:, 1, :]
        ))
        self.assertTrue(torch.equal(
            state_source[:, 4:8], captured["u_mdm"][:, 1, :]
        ))
        self.assertTrue(torch.equal(state_source[:, 8:], captured["context"]))

    def test_global_v1_public_class_and_state_keys_remain_patch_free(self):
        global_model = self._enhanced(use_teb=True)
        self.assertEqual(global_model.teb_architecture, GLOBAL_TEB_V1)
        self.assertIsInstance(global_model.teb, TargetExogenousBridge)
        global_keys = {
            key[len("teb."):]
            for key in global_model.state_dict()
            if key.startswith("teb.")
        }
        expected = {
            "gamma_teb", "query_projection.weight", "query_projection.bias",
            "query_norm.weight", "query_norm.bias",
            "exogenous_projection.weight", "exogenous_projection.bias",
            "exogenous_norm.weight", "exogenous_norm.bias",
            "cross_attention.in_proj_weight", "cross_attention.in_proj_bias",
            "cross_attention.out_proj.weight", "cross_attention.out_proj.bias",
            "output_projection.weight", "output_projection.bias",
        }
        self.assertEqual(global_keys, expected)
        self.assertFalse(any("patch" in key for key in global_keys))


    def test_checkpoint_rejections_never_pollute_parameters(self):
        baseline = dict(AMD(**self._backbone_kwargs()).state_dict())
        pmcr_source = dict(self._enhanced(use_pmcr=True).state_dict())
        teb_source = dict(self._enhanced(use_teb=True).state_dict())
        target = self._enhanced(use_pmcr=True, use_teb=True)

        partial_pmcr = dict(pmcr_source)
        partial_pmcr.pop(next(key for key in partial_pmcr if key.startswith("pmcr.")))
        partial_teb = dict(teb_source)
        partial_teb.pop(next(key for key in partial_teb if key.startswith("teb.")))
        unexpected = dict(baseline)
        unexpected["unexpected.weight"] = torch.ones(1)
        missing_backbone = dict(baseline)
        missing_backbone.pop(next(iter(missing_backbone)))
        shape_mismatch = dict(baseline)
        shape_key = next(
            key for key, value in shape_mismatch.items()
            if torch.is_tensor(value) and value.ndim > 0
        )
        expected = shape_mismatch[shape_key]
        shape_mismatch[shape_key] = expected.new_zeros(
            (*expected.shape[:-1], expected.shape[-1] + 1)
        )

        cases = (
            ("partial_pmcr", partial_pmcr, "pmcr_only", RuntimeError, "missing_source_keys"),
            ("partial_teb", partial_teb, "teb_only", RuntimeError, "missing_source_keys"),
            ("unexpected", unexpected, "baseline", RuntimeError, "unexpected"),
            ("missing_backbone", missing_backbone, "baseline", RuntimeError, "missing_source_keys"),
            ("shape_mismatch", shape_mismatch, "baseline", RuntimeError, "tensor contract"),
            ("wrong_source_kind", baseline, "unknown", ValueError, "source_kind"),
        )
        for name, state, source_kind, exception, message in cases:
            with self.subTest(case=name):
                before = {
                    key: value.detach().clone()
                    for key, value in target.state_dict().items()
                }
                with self.assertRaisesRegex(exception, message):
                    target.load_enhancement_state_dict(
                        state,
                        source_kind=source_kind,
                    )
                for key, value in before.items():
                    self.assertTrue(torch.equal(value, target.state_dict()[key]), key)

        before = {
            key: value.detach().clone()
            for key, value in target.state_dict().items()
        }
        with self.assertRaisesRegex(TypeError, "source_kind"):
            target.load_enhancement_state_dict(baseline)
        for key, value in before.items():
            self.assertTrue(torch.equal(value, target.state_dict()[key]), key)
    def test_checkpoint_source_kind_matrix_and_rejections(self):
        torch.manual_seed(3317)
        baseline = AMD(**self._backbone_kwargs()).state_dict()
        off = self._enhanced()
        pmcr_only = self._enhanced(use_pmcr=True)
        teb_only = self._enhanced(use_teb=True)
        full = self._enhanced(use_pmcr=True, use_teb=True)

        for name, target in (
            ("off", off),
            ("pmcr_only", pmcr_only),
            ("teb_only", teb_only),
            ("full", full),
        ):
            with self.subTest(source="baseline", target=name):
                result = target.load_enhancement_state_dict(
                    baseline, source_kind="baseline"
                )
                self.assertEqual(result.missing_keys, [])
                self.assertEqual(result.unexpected_keys, [])

        full_from_pmcr = self._enhanced(use_pmcr=True, use_teb=True)
        result = full_from_pmcr.load_enhancement_state_dict(
            pmcr_only.state_dict(), source_kind="pmcr_only"
        )
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])

        full_from_teb = self._enhanced(use_pmcr=True, use_teb=True)
        result = full_from_teb.load_enhancement_state_dict(
            teb_only.state_dict(), source_kind="teb_only"
        )
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])

        strict_copy = self._enhanced(use_pmcr=True, use_teb=True)
        result = strict_copy.load_state_dict(full.state_dict(), strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])

        partial_pmcr = dict(pmcr_only.state_dict())
        partial_pmcr.pop(next(key for key in partial_pmcr if key.startswith("pmcr.")))
        with self.assertRaisesRegex(RuntimeError, "missing_source_keys"):
            full.load_enhancement_state_dict(
                partial_pmcr, source_kind="pmcr_only"
            )

        partial_teb = dict(teb_only.state_dict())
        partial_teb.pop(next(key for key in partial_teb if key.startswith("teb.")))
        with self.assertRaisesRegex(RuntimeError, "missing_source_keys"):
            full.load_enhancement_state_dict(partial_teb, source_kind="teb_only")

        unexpected = dict(baseline)
        unexpected["not_allowed.weight"] = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            full.load_enhancement_state_dict(unexpected, source_kind="baseline")

        missing_backbone = dict(baseline)
        missing_backbone.pop(next(iter(missing_backbone)))
        before = {key: value.clone() for key, value in full.state_dict().items()}
        with self.assertRaisesRegex(RuntimeError, "missing_source_keys"):
            full.load_enhancement_state_dict(
                missing_backbone, source_kind="baseline"
            )
        for key, value in before.items():
            self.assertTrue(torch.equal(value, full.state_dict()[key]), key)


class AMDEnhancedSonnetS2Tests(unittest.TestCase):
    def _kwargs(self, *, enabled, norm=True, n_block=1, pred_len=1):
        return {
            "input_shape": (12, 11),
            "pred_len": pred_len,
            "n_block": n_block,
            "dropout": 0.1,
            "patch": 4,
            "k": 1,
            "c": 2,
            "alpha": 0.0,
            "target_slice": None,
            "norm": norm,
            "layernorm": True,
            "target_idx": 0,
            "teb_context_dim": 32,
            "task_mode": TARGET_EXOGENOUS,
            "aux_idx": tuple(range(1, 11)),
            "use_sonnet_mvca": enabled,
            "sonnet_feature_schema": (
                "volume", "e_price", "s_price", "Ta", "P", "h",
                "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
                "is_weekend",
            ),
            "sonnet_schema_fingerprint": "sonnet-test-schema",
            "module_init_seed": 2024,
        }

    def test_module_off_has_no_keys_and_exact_frozen_amd_parity(self):
        base_kwargs = {
            key: value
            for key, value in self._kwargs(enabled=False).items()
            if key in {
                "input_shape", "pred_len", "n_block", "dropout", "patch",
                "k", "c", "alpha", "norm", "layernorm",
            }
        }
        torch.manual_seed(71)
        base = AMD(**base_kwargs, target_slice=slice(0, 1)).eval()
        control = AMDEnhanced(**self._kwargs(enabled=False)).eval()
        result = control.load_state_dict(base.state_dict(), strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        self.assertIsNone(control.sonnet_mvca)
        self.assertFalse(
            any(key.startswith("sonnet_mvca.") for key in control.state_dict())
        )
        x = torch.randn(3, 12, 11)
        with torch.no_grad():
            base_prediction, base_moe = base(x)
            prediction, moe, state = control(x, return_state_source=True)
        self.assertTrue(torch.equal(prediction, base_prediction))
        self.assertTrue(torch.equal(moe, base_moe))
        self.assertTrue(torch.equal(state[:, -32:], torch.zeros_like(state[:, -32:])))

    def test_route_is_after_revin_before_mdm_and_reaches_ddi_and_ams(self):
        class AddTarget(nn.Module):
            def __init__(self):
                super().__init__()
                self.input = None

            def forward(self, value):
                self.input = value.detach().clone()
                output = value.clone()
                output[:, :, 0] = output[:, :, 0] + 3.0
                return output

        class RecordMDM(nn.Module):
            def __init__(self):
                super().__init__()
                self.input = None

            def forward(self, value):
                self.input = value.detach().clone()
                return value + 10.0

        class RecordDDI(nn.Module):
            def __init__(self):
                super().__init__()
                self.input = None

            def forward(self, value):
                self.input = value.detach().clone()
                return value + 20.0

        class RecordAMS(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = None
                self.selector = None

            def forward(self, experts, selector):
                self.experts = experts.detach().clone()
                self.selector = selector.detach().clone()
                return experts, experts.new_zeros(())

        candidate = AMDEnhanced(
            **self._kwargs(enabled=True, pred_len=12)
        ).eval()
        sonnet = AddTarget()
        mdm = RecordMDM()
        ddi = RecordDDI()
        ams = RecordAMS()
        candidate.sonnet_mvca = sonnet
        candidate.pastmixing = mdm
        candidate.fc_blocks = nn.ModuleList([ddi])
        candidate.moe = ams

        x = torch.randn(2, 12, 11)
        mean = x.mean(dim=1, keepdim=True)
        stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        normalized = (x - mean) / stdev
        _, _, state = candidate(x, return_state_source=True)
        self.assertTrue(torch.equal(sonnet.input, normalized))
        enhanced_ch = normalized.transpose(1, 2).clone()
        enhanced_ch[:, 0, :] += 3.0
        self.assertTrue(torch.equal(mdm.input, enhanced_ch))
        self.assertTrue(torch.equal(ddi.input, enhanced_ch + 10.0))
        self.assertTrue(torch.equal(ams.selector, enhanced_ch + 10.0))
        self.assertTrue(torch.equal(ams.experts, ddi.input + 20.0))
        self.assertTrue(torch.equal(state[:, :12], ams.experts[:, 0]))
        self.assertTrue(torch.equal(state[:, 12:24], ams.selector[:, 0]))
        self.assertTrue(torch.equal(state[:, 24:], torch.zeros_like(state[:, 24:])))

    def test_isolated_initialization_preserves_backbone_global_rng_and_first_batch(self):
        original = _capture_torch_rng_state()
        try:
            torch.manual_seed(2024)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(2024)
            initial = _capture_torch_rng_state()

            _restore_torch_rng_state(initial)
            control_generator = torch.Generator().manual_seed(2024)
            control = AMDEnhanced(**self._kwargs(enabled=False))
            control_after = _capture_torch_rng_state()
            control_generator_state = control_generator.get_state().clone()

            _restore_torch_rng_state(initial)
            candidate_generator = torch.Generator().manual_seed(2024)
            candidate = AMDEnhanced(**self._kwargs(enabled=True))
            candidate_after = _capture_torch_rng_state()
            candidate_generator_state = candidate_generator.get_state().clone()

            control_common = control.state_dict()
            candidate_common = {
                key: value
                for key, value in candidate.state_dict().items()
                if not key.startswith("sonnet_mvca.")
            }
            self.assertEqual(control_common.keys(), candidate_common.keys())
            for key in control_common:
                self.assertTrue(
                    torch.equal(control_common[key], candidate_common[key]), key
                )
            self.assertTrue(torch.equal(control_after["cpu"], candidate_after["cpu"]))
            if control_after["cuda"] is not None:
                self.assertEqual(
                    len(control_after["cuda"]), len(candidate_after["cuda"])
                )
                for left, right in zip(
                    control_after["cuda"], candidate_after["cuda"]
                ):
                    self.assertTrue(torch.equal(left, right))
            self.assertTrue(
                torch.equal(control_generator_state, candidate_generator_state)
            )

            data = torch.arange(80)
            control_loader = torch.utils.data.DataLoader(
                data,
                batch_size=8,
                shuffle=True,
                generator=control_generator,
            )
            candidate_loader = torch.utils.data.DataLoader(
                data,
                batch_size=8,
                shuffle=True,
                generator=candidate_generator,
            )
            self.assertTrue(
                torch.equal(next(iter(control_loader)), next(iter(candidate_loader)))
            )
        finally:
            _restore_torch_rng_state(original)

    def test_same_structure_restore_and_atomic_rejections(self):
        torch.manual_seed(1)
        source = AMDEnhanced(**self._kwargs(enabled=True))
        torch.manual_seed(2)
        target = AMDEnhanced(**self._kwargs(enabled=True))
        result = target.load_state_dict(source.state_dict(), strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, target.state_dict()[key]), key)

        cases = {}
        partial = dict(source.state_dict())
        partial.pop("sonnet_mvca.freq_params")
        cases["partial"] = partial
        wrong_dtype = dict(source.state_dict())
        wrong_dtype["sonnet_mvca.freq_params"] = wrong_dtype[
            "sonnet_mvca.freq_params"
        ].double()
        cases["dtype"] = wrong_dtype
        unexpected = dict(source.state_dict())
        unexpected["teb.unexpected"] = torch.ones(1)
        cases["cross_source"] = unexpected
        control = AMDEnhanced(**self._kwargs(enabled=False))
        cases["control_candidate"] = dict(control.state_dict())

        for name, state in cases.items():
            with self.subTest(name=name):
                before = {
                    key: value.detach().clone()
                    for key, value in target.state_dict().items()
                }
                with self.assertRaisesRegex(RuntimeError, "strict checkpoint"):
                    target.load_state_dict(state, strict=True)
                for key, value in before.items():
                    self.assertTrue(torch.equal(value, target.state_dict()[key]), key)
        with self.assertRaisesRegex(ValueError, "strict=True"):
            target.load_state_dict(source.state_dict(), strict=False)
        with self.assertRaisesRegex(RuntimeError, "forbids source import"):
            target.load_enhancement_state_dict(
                source.state_dict(), source_kind="baseline"
            )

    def test_parallel_f0_and_mixed_modules_are_rejected(self):
        parallel = self._kwargs(enabled=True)
        parallel.update({"task_mode": PARALLEL_MULTIVARIATE, "aux_idx": ()})
        with self.assertRaisesRegex(ValueError, "target_exogenous"):
            AMDEnhanced(**parallel)

        f0 = self._kwargs(enabled=True)
        f0.update({
            "input_shape": (12, 1),
            "aux_idx": (),
            "sonnet_feature_schema": ("volume",),
        })
        with self.assertRaisesRegex(ValueError, "non-empty"):
            AMDEnhanced(**f0)

        for switch in ("use_cce", "use_pmcr", "use_teb"):
            with self.subTest(switch=switch):
                mixed = self._kwargs(enabled=True)
                mixed[switch] = True
                with self.assertRaisesRegex(ValueError, "requires CCE, PMCR, and TEB"):
                    AMDEnhanced(**mixed)



class AMDEnhancedP2Tests(unittest.TestCase):
    """C routing and initialization; standalone synthetic inputs only."""

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @staticmethod
    def _kwargs(length=12):
        urban = length == 12
        features, target = (11, 0) if urban else (7, 6)
        common = dict(input_shape=(length, features), pred_len=1 if urban else 96,
                      n_block=1, dropout=.1, patch=length, k=0, c=2, alpha=.5,
                      norm=True, layernorm=True)
        enhanced = dict(target_slice=None, target_idx=target, teb_context_dim=16,
                        task_mode=TARGET_EXOGENOUS,
                        aux_idx=tuple(i for i in range(features) if i != target))
        body = dict(pmcr_hidden_dim=8, pmcr_kernel_small=3 if urban else 5,
                    pmcr_kernel_large=7 if urban else 31, pmcr_body_init_seed=2024)
        return common, enhanced, body

    def test_abc_rng_body_generator_first_batch_frozen_equivalence_and_route(self):
        from unittest import mock
        import main as runner
        from test_runner import _assert_nested_equal
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for length in (12, 512):
                common, enhanced, body = self._kwargs(length)
                additions = [{}, dict(use_pmcr=True, **body),
                             dict(use_pmcr_p2=True, pmcr_gate_init_seed=2025, **body)]
                models, rngs, batches = [], [], []
                for extra in additions:
                    runner.set_seed(2024)
                    generator = torch.Generator().manual_seed(2024)
                    initial = generator.get_state().clone()
                    # Constructor must not touch either eager or delayed CUDA seeds.
                    with mock.patch.object(torch.cuda, "manual_seed", side_effect=AssertionError("CUDA seed touched")), \
                            mock.patch.object(torch.cuda, "manual_seed_all", side_effect=AssertionError("CUDA seed touched")):
                        model = AMDEnhanced(**common, **enhanced, **extra)
                    self.assertTrue(torch.equal(initial, generator.get_state()))
                    models.append(model)
                    rngs.append(runner.capture_rng_state())
                    data = torch.arange(4*length*common["input_shape"][1], dtype=torch.float32).reshape(
                        4, length, common["input_shape"][1])
                    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data),
                        batch_size=2, shuffle=True, generator=generator, drop_last=True)
                    batches.append(next(iter(loader)))
                a, b, c = models
                _assert_nested_equal(self, rngs[0], rngs[1])
                _assert_nested_equal(self, rngs[0], rngs[2])
                _assert_nested_equal(self, batches[0], batches[1])
                _assert_nested_equal(self, batches[0], batches[2])
                for model in (b, c):
                    _assert_nested_equal(self, a.state_dict(), {k: v for k, v in model.state_dict().items()
                        if not k.startswith(("pmcr.", "pmcr_p2."))})
                _assert_nested_equal(self, b.pmcr.state_dict(), c.pmcr_p2.body.state_dict())
                self.assertIsNone(a.pmcr); self.assertIsNone(c.pmcr)
                self.assertIsNone(a.pmcr_p2); self.assertIsNone(b.pmcr_p2)
                self.assertFalse(c.use_pmcr)
                runner.set_seed(2024)
                target = enhanced["target_idx"]
                frozen = AMD(**common, target_slice=slice(target, target+1))
                _assert_nested_equal(self, a.state_dict(), frozen.state_dict())
                a, b, c, frozen = [m.to(device).eval() for m in (a, b, c, frozen)]
                x = torch.randn(2, length, common["input_shape"][1],
                                generator=torch.Generator().manual_seed(8)).to(device)
                observed = {}
                handles = [
                    c.pastmixing.register_forward_hook(lambda m, ins, out: observed.update(u=out.detach().clone())),
                    c.fc_blocks[-1].register_forward_hook(lambda m, ins, out: observed.update(ddi=out.detach().clone())),
                    c.pmcr_p2.register_forward_pre_hook(lambda m, ins: observed.update(p2_input=ins[0].detach().clone())),
                    c.pmcr_p2.register_forward_hook(lambda m, ins, out: observed.update(p2_output=out.detach().clone())),
                    c.moe.register_forward_pre_hook(lambda m, ins: observed.update(
                        experts=ins[0].detach().clone(), selector=ins[1].detach().clone())),
                ]
                try:
                    with torch.no_grad():
                        pa, ma = a(x); pf, mf = frozen(x)
                        self.assertTrue(torch.equal(pa, pf)); self.assertTrue(torch.equal(ma, mf))
                        state = _capture_torch_rng_state()
                        pb, mb = b(x)
                        _restore_torch_rng_state(state)
                        pc, mc, source = c(x, return_state_source=True)
                    self.assertTrue(torch.equal(pb, pc)); self.assertTrue(torch.equal(mb, mc))
                    self.assertTrue(torch.equal(observed["ddi"], observed["p2_input"]))
                    self.assertTrue(torch.equal(observed["experts"], observed["p2_output"]))
                    self.assertTrue(torch.equal(observed["selector"], observed["u"]))
                    self.assertGreater(float((observed["p2_output"]-observed["ddi"]).abs().max()), 0)
                    expected = torch.cat((observed["p2_output"][:, target], observed["u"][:, target],
                                          x.new_zeros((2, 16))), dim=-1)
                    self.assertTrue(torch.equal(source, expected))
                    self.assertEqual(source.shape, (2, 2*length+16))
                finally:
                    for handle in handles: handle.remove()
                print("P2_AMD_FAIRNESS_ROUTE", device, length,
                      "common/body/RNG/generator/first_batch=exact B_C_prediction_error=0", flush=True)

    def test_c_strict_state_and_combination_boundaries(self):
        from copy import deepcopy
        common, enhanced, body = self._kwargs()
        kwargs = dict(**common, **enhanced, **body, use_pmcr_p2=True, pmcr_gate_init_seed=2025)
        for change in ({"use_pmcr": True}, {"use_cce": True}, {"use_teb": True},
                       {"use_sonnet_mvca": True}, {"task_mode": PARALLEL_MULTIVARIATE},
                       {"pmcr_gate_init_seed": 2024}, {"pmcr_body_init_seed": 2025}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                AMDEnhanced(**{**kwargs, **change})
        model = AMDEnhanced(**kwargs)
        source = deepcopy(model.state_dict())
        for kind in ("key", "shape", "dtype"):
            bad = deepcopy(source)
            key = "pmcr_p2.gate.conv1.weight"
            if kind == "key": bad.pop(key)
            elif kind == "shape": bad[key] = bad[key].reshape(-1)
            else: bad[key] = bad[key].double()
            with self.assertRaises(RuntimeError): model.load_state_dict(bad)
            for key, value in source.items(): self.assertTrue(torch.equal(value, model.state_dict()[key]))
        with self.assertRaises(ValueError): model.load_state_dict(source, strict=False)
        backbone = {k: v for k, v in source.items() if not k.startswith("pmcr_p2.")}
        with self.assertRaises(RuntimeError): model.load_amd_backbone_state_dict(backbone)
        for key, value in source.items(): self.assertTrue(torch.equal(value, model.state_dict()[key]))

def _unrepaired_frozen_amd_class():
    """Exact immutable Git sources; import them without altering active modules."""
    import builtins
    import hashlib
    import json
    import os
    from pathlib import Path
    import subprocess
    import types
    expected = {
        "models/common.py": "570f47c3a7db3b5156e4e95df65b81aa13c5a0a741a61f1bb0798ab1ec1a3afb",
        "models/tsAMD.py": "fa72cdbe34348364344c0d9c0755668a82d22f6a37ee061c7ece93ecfaf90ba1",
        "models/tsmoe.py": "d6c7888410dc64c3514c76cf4f2720b99c11773b0011780afaac76ca98aee0f1",
    }
    config_path = os.environ.get("AMD_RR_CONFIG")
    paths = json.loads(Path(config_path).read_text())["frozen_forward_reference"] if config_path else None
    sources = {name: Path(paths[name]).read_bytes() if paths else subprocess.check_output(
        ["git", "show", "amd_reproduced_baseline_v1:"+name], cwd=Path(__file__).resolve().parents[1])
        for name in expected}
    assert all(hashlib.sha256(sources[n]).hexdigest()==h for n,h in expected.items())
    modules = {}
    original_import = builtins.__import__
    def reference_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level==0 and name in modules:
            return modules[name]
        return original_import(name,globals,locals,fromlist,level)
    for name in ("models.common", "models.tsmoe", "models.tsAMD"):
        module = types.ModuleType("_unrepaired_frozen_"+name.replace(".","_"))
        module.__dict__["__builtins__"] = dict(vars(builtins), __import__=reference_import)
        source_name=name.replace(".","/")+".py"
        exec(compile(sources[source_name], "git:amd_reproduced_baseline_v1:"+source_name, "exec"),module.__dict__)
        modules[name]=module
    return modules["models.tsAMD"].AMD


def _closed_gradient_numerics(left, right):
    if left is None or right is None:
        same = left is None and right is None
        return dict(none_equal=same, both_none=same, bitwise_equal=same, finite=same)
    a, b = left.detach(), right.detach()
    metadata = dict(none_equal=True, shape_equal=a.shape == b.shape,
                    dtype_equal=a.dtype == b.dtype, device_equal=a.device == b.device)
    if not all(metadata.values()):
        return dict(metadata, bitwise_equal=False, finite=False)
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    result = dict(metadata, bitwise_equal=torch.equal(a, b), finite=finite,
        different_elements=int((a != b).sum()), elements=a.numel(),
        left_nan=int(torch.isnan(a).sum()), right_nan=int(torch.isnan(b).sum()),
        left_inf=int(torch.isinf(a).sum()), right_inf=int(torch.isinf(b).sum()),
        max_abs=None, max_rel=None, max_ulp=None)
    if finite:
        ac, bc = a.to(device='cpu', dtype=torch.float64), b.to(device='cpu', dtype=torch.float64)
        difference = (ac-bc).abs(); scale = torch.maximum(ac.abs(), bc.abs())
        nonzero = scale != 0
        result['max_abs'] = float(difference.max()) if a.numel() else 0.
        result['max_rel'] = float((difference[nonzero]/scale[nonzero]).max()) if bool(nonzero.any()) else 0.
        if a.dtype == torch.float32:
            def ordered_bits(t):
                bits = t.to('cpu').contiguous().view(torch.int32).to(torch.int64)
                return torch.where(bits < 0, -(bits & 0x7fffffff), bits)
            ulp = (ordered_bits(a)-ordered_bits(b)).abs()
            result['max_ulp'] = int(ulp.max()) if a.numel() else 0
            result['approved_symmetric_bound_satisfied'] = bool((difference <= 1e-7+1e-6*scale).all())
    return result


def _assert_closed_parameter_gradient(testcase, name, left, right):
    """Role/dtype-scoped cross-entry exception; every other gradient remains bitwise."""
    if left is None or right is None:
        testcase.assertIs(left, right, name+' None pattern')
        return
    testcase.assertEqual(left.shape, right.shape, name+' shape')
    testcase.assertEqual(left.dtype, right.dtype, name+' dtype')
    testcase.assertEqual(left.device, right.device, name+' device')
    testcase.assertTrue(bool(torch.isfinite(left).all() and torch.isfinite(right).all()), name+' finite')
    if name in {'rev_norm.affine_weight', 'rev_norm.affine_bias'} and left.dtype == torch.float32:
        a = left.detach().to(device='cpu', dtype=torch.float64)
        b = right.detach().to(device='cpu', dtype=torch.float64)
        testcase.assertTrue(bool(((a-b).abs() <= 1e-7+1e-6*torch.maximum(a.abs(), b.abs())).all()),
                            name+' approved symmetric numerical bound')
    else:
        testcase.assertTrue(torch.equal(left, right), name+' bitwise gradient')


def _closed_report_json_safe(value):
    """Keep every observed nonfinite value reportable without accepting it."""
    import math
    if isinstance(value, dict):
        return {key:_closed_report_json_safe(item) for key,item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_closed_report_json_safe(item) for item in value]
    if hasattr(value, 'item'):
        return _closed_report_json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite':'NaN' if math.isnan(value) else ('+Infinity' if value > 0 else '-Infinity')}
    return value


class AMDEnhancedTHLSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @staticmethod
    def _kwargs(length=12, enabled=False):
        features, target = (11, 0) if length == 12 else (7, 6)
        common = dict(input_shape=(length, features), pred_len=1 if length == 12 else 96,
            n_block=1, dropout=.1, patch=12 if length == 12 else 16, k=0, c=2,
            alpha=.5, norm=True, layernorm=True)
        options = dict(target_slice=None, target_idx=target, teb_context_dim=32,
            task_mode=TARGET_EXOGENOUS, aux_idx=tuple(i for i in range(features) if i != target),
            local_shape_contract_declared=True, use_target_history_local_shape=enabled)
        if enabled:
            options.update(local_shape_kernel_small=3 if length == 12 else 5,
                           local_shape_kernel_large=7 if length == 12 else 31,
                           local_shape_init_seed=2024)
        return common, options

    def test_original_single_revin_target_route_selector_and_state(self):
        for device in ('cpu', 'cuda'):
            for length in (12, 512):
                common, options = self._kwargs(length, True)
                model = AMDEnhanced(**common, **options).to(device).eval()
                target = options['target_idx']; observed = {}; norm_calls = []
                x = torch.randn(2, length, common['input_shape'][1], device=device)
                original_x = x.clone()
                def norm_hook(module, inputs, output):
                    if inputs[1] == 'norm':
                        norm_calls.append(1); observed['z'] = output.detach().clone()
                handles = [model.rev_norm.register_forward_hook(norm_hook),
                    model.pastmixing.register_forward_pre_hook(lambda m, ins: observed.update(mdm_input=ins[0].detach().clone())),
                    model.pastmixing.register_forward_hook(lambda m, ins, out: observed.update(u=out.detach().clone())),
                    model.fc_blocks[-1].register_forward_hook(lambda m, ins, out: observed.update(ddi=out.detach().clone())),
                    model.target_history_local_shape.register_forward_pre_hook(lambda m, ins: observed.update(y=ins[0].detach().clone())),
                    model.target_history_local_shape.register_forward_hook(lambda m, ins, out: observed.update(residual=out.detach().clone())),
                    model.moe.register_forward_pre_hook(lambda m, ins: observed.update(v=ins[0].detach().clone(), selector=ins[1].detach().clone()))]
                try:
                    with torch.no_grad(): prediction, auxiliary, state = model(x, return_state_source=True)
                finally:
                    for handle in handles: handle.remove()
                self.assertEqual(len(norm_calls), 1)
                self.assertTrue(torch.equal(x, original_x))
                self.assertTrue(torch.equal(observed['y'], observed['z'][:, :, target]))
                self.assertTrue(torch.equal(observed['mdm_input'], observed['z'].transpose(1, 2)))
                aux = list(options['aux_idx'])
                self.assertTrue(torch.equal(observed['v'][:, aux], observed['ddi'][:, aux]))
                self.assertTrue(torch.equal(observed['v'][:, target:target+1],
                    observed['ddi'][:, target:target+1]+observed['residual']))
                self.assertTrue(torch.equal(observed['selector'], observed['u']))
                expected = torch.cat((observed['v'][:, target], observed['u'][:, target],
                                      x.new_zeros(2, 32)), -1)
                self.assertTrue(torch.equal(state, expected))
                self.assertEqual(state.shape, (2, 2*length+32))
                self.assertTrue(torch.isfinite(prediction).all()); self.assertTrue(torch.isfinite(auxiliary))

    def test_disabled_frozen_equivalence_and_isolated_construction_rng(self):
        import main as runner
        from unittest import mock
        from test_runner import _assert_nested_equal
        for device in ('cpu', 'cuda'):
            for length in (12, 512):
                common, options = self._kwargs(length)
                target = options['target_idx']; models = []; rngs = []; batches = []
                for enabled in (False, True):
                    runner.set_seed(2024)
                    generator = torch.Generator().manual_seed(2024)
                    before_generator = generator.get_state().clone()
                    _, extra = self._kwargs(length, enabled)
                    with mock.patch.object(torch.cuda, 'manual_seed', side_effect=AssertionError('CUDA seed touched')), \
                            mock.patch.object(torch.cuda, 'manual_seed_all', side_effect=AssertionError('CUDA seed touched')):
                        models.append(AMDEnhanced(**common, **extra))
                    self.assertTrue(torch.equal(before_generator, generator.get_state()))
                    rngs.append(runner.capture_rng_state())
                    sample = torch.arange(4*length*common['input_shape'][1]).reshape(4, length, -1).float()
                    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(sample),
                        batch_size=2, shuffle=True, drop_last=True, generator=generator)
                    batches.append(next(iter(loader)))
                a, n = models
                _assert_nested_equal(self, rngs[0], rngs[1]); _assert_nested_equal(self, batches[0], batches[1])
                _assert_nested_equal(self, a.state_dict(), {k:v for k,v in n.state_dict().items()
                    if not k.startswith('target_history_local_shape.')})
                self.assertIsNone(a.target_history_local_shape)
                self.assertIsNone(n.pmcr); self.assertIsNone(n.pmcr_p2)
                runner.set_seed(2024)
                frozen = AMD(**common, target_slice=slice(target, target+1))
                _assert_nested_equal(self, a.state_dict(), frozen.state_dict())
                a.to(device).eval(); frozen.to(device).eval()
                x = torch.randn(2, length, common['input_shape'][1], device=device, requires_grad=True)
                xf = x.detach().clone().requires_grad_()
                pa, aa = a(x); pf, af = frozen(xf)
                self.assertTrue(torch.equal(pa, pf)); self.assertTrue(torch.equal(aa, af))
                # Forward only: exact prepatch frozen sources, not repaired common imports.
                saved_rng = runner.capture_rng_state()
                try:
                    runner.set_seed(2024)
                    original = _unrepaired_frozen_amd_class()(**common, target_slice=slice(target,target+1)).to(device).eval()
                    _assert_nested_equal(self, frozen.state_dict(), original.state_dict())
                    runner.restore_rng_state(saved_rng)
                    po, ao = original(x.detach().clone().requires_grad_())
                    _assert_nested_equal(self, saved_rng, runner.capture_rng_state())
                    self.assertTrue(torch.equal(pa,po)); self.assertTrue(torch.equal(aa,ao))
                    _assert_nested_equal(self, frozen.state_dict(), original.state_dict())
                    self.assertTrue(all(not m.training for m in original.modules()))
                    forward_reference = dict(prediction_bitwise=torch.equal(pa,po),moe_bitwise=torch.equal(aa,ao),
                        state_equal=True,eval_mode=True,rng_unchanged=True,backward_executed=False)
                finally:
                    runner.restore_rng_state(saved_rng)
                del original,po,ao
                (pa.square().mean()+aa).backward(); (pf.square().mean()+af).backward()
                import json
                input_report = _closed_gradient_numerics(x.grad,xf.grad)
                parameters = {key:_closed_gradient_numerics(p.grad,dict(frozen.named_parameters())[key].grad)
                              for key,p in a.named_parameters()}
                # Persist all already-computed values before the grouped assertions.
                import hashlib
                import os
                import tempfile
                from pathlib import Path
                import numpy as np
                guard_config = os.environ.get('AMD_RR_CONFIG')
                if guard_config:
                    raw_root = Path(json.loads(Path(guard_config).read_text())['session_root'])/'closed-gradients'
                    raw_root.mkdir(exist_ok=True)
                else:
                    scratch = tempfile.TemporaryDirectory(prefix='thls-closed-gradients-')
                    self.addCleanup(scratch.cleanup)
                    raw_root = Path(scratch.name)
                raw_path = raw_root/(device+'-T'+str(length)+'.npz')
                arrays = {}
                for side, model, leaf in [('A', a, x), ('AMD', frozen, xf)]:
                    for name, parameter in [('input', leaf), *model.named_parameters()]:
                        if parameter.grad is not None:
                            arrays[side+'/'+name] = parameter.grad.detach().clone().cpu().numpy()
                with raw_path.open('xb') as handle:
                    np.savez(handle, **arrays)
                raw_record = dict(path=str(raw_path), sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                                  arrays=list(arrays), none_patterns='parameter_gradients and input_gradient records')
                del arrays
                affine_values = {name:dict(
                    left=None if dict(a.named_parameters())[name].grad is None else dict(a.named_parameters())[name].grad.detach().cpu().tolist(),
                    right=None if dict(frozen.named_parameters())[name].grad is None else dict(frozen.named_parameters())[name].grad.detach().cpu().tolist(),
                    different_indices=None if dict(a.named_parameters())[name].grad is None or dict(frozen.named_parameters())[name].grad is None else
                        (dict(a.named_parameters())[name].grad != dict(frozen.named_parameters())[name].grad).nonzero().cpu().tolist())
                    for name in ('rev_norm.affine_weight', 'rev_norm.affine_bias')}
                report = dict(device=device,length=length,prediction_bitwise=torch.equal(pa,pf),moe_bitwise=torch.equal(aa,af),
                    unrepaired_frozen_forward=forward_reference,backward_reference="AMD with repaired shared common.py",
                    input_gradient=input_report,parameter_gradients=parameters,
                    cross_entry_exception="float32 RevIN affine_weight/affine_bias only; symmetric atol=1e-7 rtol=1e-6",
                    affine_original_values=affine_values,raw_gradient_snapshot=raw_record,
                    all_parameter_gradients_bitwise=all(v['bitwise_equal'] for v in parameters.values()))
                payload = json.dumps(_closed_report_json_safe(report),allow_nan=False)
                with raw_path.with_suffix('.json').open('x') as handle:
                    handle.write(payload+'\n')
                print("THLS_CLOSED_EQUIVALENCE_RESULT "+payload,flush=True)
                self.assertTrue(input_report['finite']); self.assertTrue(all(v['finite'] for v in parameters.values()))
                self.assertTrue(torch.equal(x.grad, xf.grad))
                left_gradients = {k:p.grad for k,p in a.named_parameters()}
                right_gradients = {k:p.grad for k,p in frozen.named_parameters()}
                self.assertEqual(set(left_gradients), set(right_gradients))
                for name in left_gradients:
                    _assert_closed_parameter_gradient(self, name, left_gradients[name], right_gradients[name])

    def test_target_loss_reaches_saved_history_affine_and_branch(self):
        for device in ('cpu', 'cuda'):
            common, options = self._kwargs(enabled=True)
            model = AMDEnhanced(**common, **options).to(device).eval()
            x = torch.randn(2, 12, 11, device=device, requires_grad=True); captured = {}
            def remember(module, inputs):
                inputs[0].retain_grad(); captured['y'] = inputs[0]
            handle = model.target_history_local_shape.register_forward_pre_hook(remember)
            try:
                prediction, _ = model(x)
                target = torch.tensor([.7, -.3], device=device).reshape_as(prediction)
                (prediction-target).square().mean().backward()
            finally: handle.remove()
            self.assertIsNotNone(captured['y'].grad)
            self.assertGreater(float(captured['y'].grad.abs().sum()), 0)
            self.assertTrue(torch.isfinite(x.grad).all())
            for name, parameter in model.rev_norm.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            groups = {}
            for name, parameter in model.target_history_local_shape.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                groups.setdefault(name.split('.')[0], 0.)
                groups[name.split('.')[0]] += float(parameter.grad.abs().sum())
            self.assertTrue(all(v > 0 for v in groups.values()), groups)
            print('THLS_AMD_TARGET_GRADIENT', device, groups, flush=True)

    def test_combination_norm_shape_and_atomic_restore_contracts(self):
        from copy import deepcopy
        common, options = self._kwargs(enabled=True); kwargs = dict(**common, **options)
        for change in ({'use_pmcr': True}, {'use_pmcr_p2': True}, {'use_sonnet_mvca': True},
                       {'use_cce': True}, {'use_teb': True}, {'norm': False}, {'layernorm': False},
                       {'task_mode': PARALLEL_MULTIVARIATE}, {'target_idx': 11},
                       {'local_shape_init_seed': 2025}, {'target_slice': slice(0, 1)}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                AMDEnhanced(**{**kwargs, **change})
        model = AMDEnhanced(**kwargs); original = deepcopy(model.state_dict())
        key = 'target_history_local_shape.input_projection.weight'
        for kind in ('key','shape','dtype','finite','deploy'):
            bad = deepcopy(original)
            if kind == 'key': bad.pop(key)
            elif kind == 'shape': bad[key] = bad[key][:1]
            elif kind == 'dtype': bad[key] = bad[key].double()
            elif kind == 'finite': bad[key][0,0,0] = float('nan')
            else:
                bad = {k:v for k,v in bad.items() if not k.startswith('target_history_local_shape.')}
                bad.update({'target_history_local_shape.'+k:v
                    for k,v in model.target_history_local_shape.to_deploy().state_dict().items()})
            with self.assertRaises(RuntimeError): model.load_state_dict(bad)
            for name in original: self.assertTrue(torch.equal(original[name], model.state_dict()[name]))
        with self.assertRaises(ValueError): model.load_state_dict(original, strict=False)


if __name__ == "__main__":
    unittest.main()


class AMDEnhancedTHLSETTm1Tests(unittest.TestCase):
    def test_four_horizon_cpu_cuda_factory_fairness_full_horizon_and_gradients(self):
        import gc
        import json
        from copy import deepcopy
        import main as runner
        from test_runner import THLSETTm1Fixture, _assert_nested_equal
        old_threads=torch.get_num_threads();torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads,old_threads)
        f=THLSETTm1Fixture(self);records=[]
        self.assertTrue(torch.cuda.is_available(),'required CUDA is unavailable')
        for device in ('cpu','cuda'):
            for h in (96,192,336,720):
                states=[];rngs=[];batches=[];generators=[]
                for enabled in (False,True):
                    args=runner.prepare_args(f.args(h,enabled));generator=torch.Generator().manual_seed(2024)
                    runtime=f.runtime(args,generator);batch=next(iter(runtime.train_data))
                    batches.append(tuple(t.clone() for t in batch));generators.append(generator.get_state().clone())
                    runner.set_seed(2024);model=runner._build_model(args,runtime).to(device).eval()
                    states.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()
                                   if not k.startswith('target_history_local_shape.')})
                    rngs.append(runner.capture_rng_state());x,y=(t.to(device) for t in batch)
                    prediction,aux=model(x)
                    self.assertEqual(tuple(prediction.shape),(2,h,1));self.assertTrue(bool(torch.isfinite(prediction).all()))
                    self.assertTrue(bool(torch.isfinite(aux).all()))
                    adapted=runner._prediction_for_loss(prediction,y,task_mode=runner.TARGET_EXOGENOUS)
                    self.assertEqual(adapted.shape,y.shape);loss=(adapted-y).square().mean()+aux
                    gradients={}
                    if enabled:
                        loss.backward()
                        for name,p in model.named_parameters():
                            if p.grad is not None:
                                self.assertTrue(bool(torch.isfinite(p.grad).all()),name)
                                gradients[name]=float(p.grad.detach().double().norm())
                        for name,p in model.target_history_local_shape.named_parameters():
                            self.assertIsNotNone(p.grad,name);self.assertTrue(bool(torch.isfinite(p.grad).all()),name)
                        for prefix in ('input_projection','temporal_conv','feature_norm','ffn_expand','ffn_reduce','output_projection','eta'):
                            self.assertTrue(any(name.startswith('target_history_local_shape.'+prefix) for name in gradients),prefix)
                        self.assertEqual(model.target_history_local_shape.temporal_conv.small_branch.kernel_size,(5,))
                        self.assertEqual(model.target_history_local_shape.temporal_conv.large_branch.kernel_size,(31,))
                    else:
                        self.assertIsNone(model.target_history_local_shape)
                        ref_args=deepcopy(args);ref_args.implementation_variant=runner.BASELINE_IMPLEMENTATION_VARIANT
                        reference=runner._build_model(ref_args,runtime).to(device).eval()
                        reference.load_state_dict(model.state_dict(),strict=True)
                        ref_prediction,ref_aux=reference(x)
                        self.assertTrue(torch.equal(prediction,ref_prediction))
                        self.assertTrue(torch.equal(aux,ref_aux));del reference
                    records.append(dict(device=device,horizon=h,enabled=enabled,shape=list(prediction.shape),
                        loss=float(loss.detach()),finite=True,gradient_norms=gradients))
                    del model,prediction,aux,adapted,loss,x,y;gc.collect()
                _assert_nested_equal(self,states[0],states[1],'common_initialization')
                _assert_nested_equal(self,rngs[0],rngs[1],'construction_RNG')
                _assert_nested_equal(self,batches[0],batches[1],'first_batch')
                _assert_nested_equal(self,generators[0],generators[1],'train_generator')
        print('ETTM1_ACCEPTANCE '+json.dumps(dict(kind='CPU_CUDA_factory',cases=records)))


class SonnetTHLSIntegrationTests(unittest.TestCase):
    def setUp(self):
        import main as runner
        from test_runner import SonnetTHLSFixture
        self.runner=runner;self.fixture=SonnetTHLSFixture(self)
        self.old_threads=torch.get_num_threads();torch.set_num_threads(4)
        self.addCleanup(torch.set_num_threads,self.old_threads)

    def _model(self, args, device, dtype):
        from types import SimpleNamespace
        self.runner.set_seed(2024)
        model=self.runner._build_model(args,SimpleNamespace(n_feature=len(args.feature_names),target_slice=slice(args.target_idx,args.target_idx+1)))
        return model.to(device=device,dtype=dtype).eval()

    def test_pre_sonnet_history_routing_fairness_and_disabled_paths(self):
        import json
        from copy import deepcopy
        from test_runner import _assert_nested_equal
        r=self.runner;f=self.fixture;reports=[]
        self.assertTrue(torch.cuda.is_available(),'required CUDA unavailable')
        for dataset in ('UrbanEV','ETTm1'):
            f.dataset=dataset
            for device,dtype in (('cpu',torch.float32),('cpu',torch.float64),('cuda',torch.float32)):
                args=r.prepare_args(f.args());t=args.seq_len;ci=args.target_idx;c=len(args.feature_names);aux_idx=list(args.aux_idx)
                # Already initialized CUDA is included in the constructor RNG comparison.
                torch.cuda.get_rng_state_all()
                joint=self._model(args,device,dtype);joint_initial=deepcopy(joint.state_dict())
                construction=r.capture_rng_state();events={};handles=[]
                def snapshot(name,value):events.setdefault(name,[]).append(value.detach().clone())
                def revin(module,inputs,output):
                    if inputs[1]=='norm':
                        snapshot('z',output);events['z_storage']=output.untyped_storage().data_ptr()
                def sonnet(module,inputs,output):
                    snapshot('s2_in',inputs[0]);snapshot('s2_out',output)
                def history(module,inputs):
                    y=inputs[0];snapshot('history',y)
                    events['history_grad_fn']=y.grad_fn is not None
                    events['history_storage']=y.untyped_storage().data_ptr()
                def mdm(module,inputs,output):snapshot('mdm_in',inputs[0]);snapshot('u',output)
                def ddi(module,inputs,output):snapshot('v',output)
                def ams(module,inputs):snapshot('expert',inputs[0]);snapshot('selector',inputs[1])
                handles.extend([joint.rev_norm.register_forward_hook(revin),joint.sonnet_mvca.register_forward_hook(sonnet),
                    joint.target_history_local_shape.register_forward_pre_hook(history),
                    joint.pastmixing.register_forward_hook(mdm),joint.fc_blocks[-1].register_forward_hook(ddi),
                    joint.moe.register_forward_pre_hook(ams)])
                generator=torch.Generator().manual_seed(2024)
                x=torch.randn((2,t,c),generator=generator,dtype=dtype).to(device).requires_grad_()
                x_initial=x.detach().clone();rng=r.capture_rng_state()
                prediction,aux,state=joint(x,return_state_source=True)
                (prediction.square().mean()+aux).backward()
                for handle in handles:handle.remove()
                self.assertEqual(tuple(state.shape),(2,2*t+32));self.assertTrue(torch.equal(x,x_initial))
                for name in ('z','s2_in','s2_out','history','u','v','selector','expert'):
                    self.assertEqual(len(events[name]),1,name)
                self.assertTrue(torch.equal(events['z'][0],events['s2_in'][0]))
                self.assertTrue(torch.equal(events['history'][0],events['z'][0][:,:,ci]))
                self.assertTrue(events['history_grad_fn'])
                self.assertNotEqual(events['history_storage'],events['z_storage'])
                self.assertTrue(torch.equal(events['s2_out'][0][:,:,aux_idx],events['z'][0][:,:,aux_idx]))
                self.assertTrue(torch.equal(events['mdm_in'][0],events['s2_out'][0].transpose(1,2)))
                self.assertTrue(torch.equal(events['expert'][0][:,aux_idx,:],events['v'][0][:,aux_idx,:]))
                self.assertTrue(torch.equal(events['selector'][0],events['u'][0]))
                self.assertTrue(torch.equal(state[:,:t],events['expert'][0][:,ci,:]))
                self.assertTrue(torch.equal(state[:,t:2*t],events['u'][0][:,ci,:]))
                self.assertEqual(torch.count_nonzero(state[:,2*t:]).item(),0)
                groups={}
                for name,p in joint.named_parameters():
                    if name.startswith(('sonnet_mvca.','target_history_local_shape.')):
                        self.assertIsNotNone(p.grad,name);self.assertTrue(bool(torch.isfinite(p.grad).all()),name)
                        group='.'.join(name.split('.')[:2]);groups.setdefault(group,0.)
                        groups[group]+=float(p.grad.double().abs().sum())
                self.assertTrue(all(value>0 for value in groups.values()),groups)
                _assert_nested_equal(self,joint_initial,joint.state_dict())
                reports.append(dict(device=device,dtype=str(dtype),joint_gradient_groups=groups))
                # Each comparison uses a separately constructed existing production factory entrance.
                for use_s2,use_thls in ((True,False),(False,True),(False,False)):
                    newargs=deepcopy(args);newargs.use_sonnet_mvca=use_s2
                    newargs.use_target_history_local_shape=use_thls
                    newargs.local_shape_init_seed=2024 if use_thls else None
                    if not use_s2:
                        for name in ("module_init_seed","sonnet_d_model","sonnet_n_atoms","sonnet_alpha",
                                     "sonnet_epsilon","sonnet_attention_dropout","sonnet_gamma_init"):
                            setattr(newargs,name,None)
                    oldargs=deepcopy(newargs)
                    if use_s2:
                        oldargs.implementation_variant=r.SONNET_IMPLEMENTATION_VARIANT
                        oldargs.development_protocol_id=r.SONNET_DEVELOPMENT_PROTOCOL
                        oldargs.ablation_id=r.SONNET_CANDIDATE_ABLATION_ID;oldargs.train_epochs=10
                    else:
                        oldargs.implementation_variant=r.THLS_IMPLEMENTATION_VARIANT
                        oldargs.development_protocol_id=(r.THLS_ETTM1_DEVELOPMENT_PROTOCOL if f.dataset=="ETTm1" else r.THLS_DEVELOPMENT_PROTOCOL)
                        oldargs.ablation_id=r.THLS_ABLATION_ID if use_thls else r.THLS_CONTROL_ABLATION_ID
                        for name in ('module_init_seed','sonnet_d_model','sonnet_n_atoms','sonnet_alpha',
                                     'sonnet_epsilon','sonnet_attention_dropout','sonnet_gamma_init'):
                            setattr(oldargs,name,None)
                    oldargs=r.prepare_args(oldargs)
                    a=self._model(newargs,device,dtype);rng_a=r.capture_rng_state()
                    b=self._model(oldargs,device,dtype);rng_b=r.capture_rng_state()
                    _assert_nested_equal(self,rng_a,rng_b)
                    _assert_nested_equal(self,a.state_dict(),b.state_dict())
                    if use_s2 or use_thls:
                        _assert_nested_equal(self,construction,rng_a)
                        for name,value in a.state_dict().items():self.assertTrue(torch.equal(value,joint_initial[name]),name)
                        if not use_thls:self.assertIsNone(a.target_history_local_shape)
                        if not use_s2:self.assertIsNone(a.sonnet_mvca)
                    ga=torch.Generator().manual_seed(2024);gb=torch.Generator().manual_seed(2024)
                    _assert_nested_equal(self,ga.get_state(),gb.get_state())
                    # Actual Dataset order/target values and dedicated generator, no extra forward.
                    from contextlib import nullcontext
                    with (f.base._guard() if f.dataset=="UrbanEV" else nullcontext()):runtime=f.runtime(args,ga)
                    dataset=runtime.train_data.dataset
                    first_a=[dataset[i] for i in (0,1)];first_b=[dataset[i] for i in (0,1)]
                    _assert_nested_equal(self,first_a,first_b)
                    xa=x_initial.clone().requires_grad_();xb=x_initial.clone().requires_grad_()
                    state_a=deepcopy(a.state_dict());state_b=deepcopy(b.state_dict())
                    r.restore_rng_state(rng);pa,ma=a(xa);(pa.square().mean()+ma).backward()
                    r.restore_rng_state(rng);pb,mb=b(xb);(pb.square().mean()+mb).backward()
                    comparisons={}
                    for name,left,right in [('prediction',pa,pb),('moe',ma,mb),('input_gradient',xa.grad,xb.grad),
                        *((name,p.grad,dict(b.named_parameters())[name].grad) for name,p in a.named_parameters())]:
                        if left is None or right is None:
                            comparisons[name]=dict(none=(left is None,right is None),equal=left is right)
                        else:
                            comparisons[name]=dict(equal=torch.equal(left,right),finite=bool(torch.isfinite(left).all() and torch.isfinite(right).all()),
                                max_abs=float((left.double()-right.double()).abs().max()),
                                unequal=int(torch.count_nonzero(left!=right)),shape=list(left.shape),
                                dtype=str(left.dtype),left=left.detach().cpu().reshape(-1).tolist() if not torch.equal(left,right) else None,
                                right=right.detach().cpu().reshape(-1).tolist() if not torch.equal(left,right) else None)
                    row=dict(dataset=f.dataset,device=device,dtype=str(dtype),s2=use_s2,thls=use_thls,comparisons=comparisons)
                    reports.append(row);f.record('routing',reports)
                    for name,result in comparisons.items():
                        self.assertTrue(result['equal'],name+': '+json.dumps(result))
                        self.assertTrue(result.get('finite',True),name)
                    _assert_nested_equal(self,state_a,a.state_dict());_assert_nested_equal(self,state_b,b.state_dict())
                    del a,b,xa,xb
                del joint

    def test_joint_train_deploy_boundary_and_equivalence(self):
        from copy import deepcopy
        from test_runner import _assert_nested_equal
        f=self.fixture;r=self.runner;rows=[]
        self.assertTrue(torch.cuda.is_available(),'required CUDA unavailable')
        for dataset in ('UrbanEV','ETTm1'):
            f.dataset=dataset
            for device,dtype in (('cpu',torch.float32),('cpu',torch.float64),('cuda',torch.float32)):
                args=r.prepare_args(f.args());t=args.seq_len;c=len(args.feature_names)
                a=self._model(args,device,dtype);b=deepcopy(a)
                b.target_history_local_shape.switch_to_deploy()
                _assert_nested_equal(self,a.sonnet_mvca.state_dict(),b.sonnet_mvca.state_dict())
                if f.dataset=='UrbanEV':
                    self.assertEqual(sum(p.numel() for p in a.parameters()),257624)
                    self.assertEqual(sum(p.numel() for p in b.parameters()),257592)
                self.assertEqual(sum(p.numel() for p in a.target_history_local_shape.parameters()),434 if t==12 else 642)
                self.assertEqual(sum(p.numel() for p in b.target_history_local_shape.parameters()),402 if t==12 else 594)
                x=torch.randn((2,t,c),generator=torch.Generator().manual_seed(2024),dtype=dtype).to(device)
                with torch.no_grad():pa,ma=a(x);pb,mb=b(x)
                self.assertTrue(bool(torch.isfinite(pa).all() and torch.isfinite(pb).all()))
                torch.testing.assert_close(pa,pb,rtol=0,atol=1e-12 if dtype==torch.float64 else 1e-6)
                self.assertTrue(torch.equal(ma,mb))
                snapshot=deepcopy(a.state_dict())
                with self.assertRaises(RuntimeError):a.load_state_dict(b.state_dict(),strict=True)
                _assert_nested_equal(self,snapshot,a.state_dict())
                rows.append(dict(device=device,dtype=str(dtype),max_abs=float((pa-pb).abs().max())))
        f.record('deploy',rows)
