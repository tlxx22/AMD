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


if __name__ == "__main__":
    unittest.main()
