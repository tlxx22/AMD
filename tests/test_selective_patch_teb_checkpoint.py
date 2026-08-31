import unittest

import torch

from models.modules.global_mediated_patch_target_exogenous_bridge import (
    GLOBAL_GATE_IDENTITY_INIT,
    GLOBAL_GATE_INPUT_CONTRACT,
    GLOBAL_GATE_SCALAR_PER_PATCH,
    GLOBAL_MEDIATED_PATCH_V1,
    GLOBAL_RESIDUAL_CONTRACT,
    PATCH_ATTENTION_RESIDUAL_NONE,
)
from models.modules.patch_conditioned_target_exogenous_bridge import (
    FIXED_SINUSOIDAL,
    PATCH_CONDITIONED_V1,
    RIGHT_ZERO_CROP,
)
from models.modules.selective_patch_target_exogenous_bridge import (
    GLOBAL_PREDICTION_ROLE_STATE_ONLY,
    PATCH_CONFIDENCE_GATE_SCALAR_POST_PROJECTION,
    PATCH_GATE_ACTIVATION_TWO_SIGMOID,
    PATCH_GATE_INIT_EXPLICIT_ZERO_IDENTITY,
    PATCH_GATE_INPUT_QUERY_AND_ATTENTION,
    SELECTIVE_PATCH_V1,
)
from models.modules.target_exogenous_bridge import TARGET_EXOGENOUS
from models.tsAMD import AMD
from models.tsAMD_enhanced import AMDEnhanced, GLOBAL_TEB_V1


class SelectivePatchTEBCheckpointTests(unittest.TestCase):
    @staticmethod
    def _backbone_kwargs():
        return {
            "input_shape": (12, 3),
            "pred_len": 2,
            "n_block": 1,
            "dropout": 0.0,
            "patch": 12,
            "k": 0,
            "c": 2,
            "alpha": 0.0,
            "target_slice": None,
            "norm": False,
            "layernorm": False,
        }

    def _model(self, architecture=SELECTIVE_PATCH_V1, patch_size=3):
        candidate = {}
        if architecture in {
            PATCH_CONDITIONED_V1,
            GLOBAL_MEDIATED_PATCH_V1,
            SELECTIVE_PATCH_V1,
        }:
            candidate.update({
                "teb_patch_size": patch_size,
                "teb_patch_padding": RIGHT_ZERO_CROP,
                "teb_patch_position": FIXED_SINUSOIDAL,
            })
        if architecture == GLOBAL_MEDIATED_PATCH_V1:
            candidate.update({
                "teb_global_residual": GLOBAL_RESIDUAL_CONTRACT,
                "teb_patch_attention_residual": PATCH_ATTENTION_RESIDUAL_NONE,
                "teb_global_gate": GLOBAL_GATE_SCALAR_PER_PATCH,
                "teb_global_gate_input": GLOBAL_GATE_INPUT_CONTRACT,
                "teb_global_gate_init": GLOBAL_GATE_IDENTITY_INIT,
                "teb_beta_global_init": 1e-3,
            })
        if architecture == SELECTIVE_PATCH_V1:
            candidate.update({
                "teb_patch_confidence_gate": (
                    PATCH_CONFIDENCE_GATE_SCALAR_POST_PROJECTION
                ),
                "teb_patch_gate_input": PATCH_GATE_INPUT_QUERY_AND_ATTENTION,
                "teb_patch_gate_activation": PATCH_GATE_ACTIVATION_TWO_SIGMOID,
                "teb_patch_gate_init": PATCH_GATE_INIT_EXPLICIT_ZERO_IDENTITY,
                "teb_global_prediction_role": GLOBAL_PREDICTION_ROLE_STATE_ONLY,
            })
        return AMDEnhanced(
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
            teb_architecture=architecture,
            **candidate,
        )

    @staticmethod
    def _snapshot(model):
        return {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    def assertStateUnchanged(self, before, model):
        after = model.state_dict()
        self.assertEqual(list(before), list(after))
        for key, value in before.items():
            self.assertTrue(torch.equal(value, after[key]), key)

    def test_same_structure_strict_restore_and_strict_false_rejection(self):
        torch.manual_seed(6301)
        source = self._model()
        target = self._model()
        result = target.load_state_dict(source.state_dict(), strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, target.state_dict()[key]), key)

        before = self._snapshot(target)
        with self.assertRaisesRegex(ValueError, "strict=True"):
            target.load_state_dict(source.state_dict(), strict=False)
        self.assertStateUnchanged(before, target)

    def test_partial_gate_unexpected_and_shape_failures_are_atomic(self):
        source = dict(self._model().state_dict())
        partial = dict(source)
        partial.pop("teb.patch_confidence_gate_bias")
        unexpected = dict(source)
        unexpected["teb.unexpected.weight"] = torch.ones(1)
        shape = dict(source)
        shape["teb.patch_confidence_gate_weight"] = torch.zeros(1, 63)
        for name, state, message in (
            ("partial", partial, "missing"),
            ("unexpected", unexpected, "unexpected"),
            ("shape", shape, "shape"),
        ):
            with self.subTest(case=name):
                target = self._model()
                before = self._snapshot(target)
                with self.assertRaisesRegex(RuntimeError, message):
                    target.load_state_dict(state, strict=True)
                self.assertStateUnchanged(before, target)

    def test_patch_mismatch_is_rejected_before_parameter_write(self):
        source = self._model(patch_size=3)
        target = self._model(patch_size=4)
        before = self._snapshot(target)
        with self.assertRaisesRegex(RuntimeError, "shape"):
            target.load_state_dict(source.state_dict(), strict=True)
        self.assertStateUnchanged(before, target)

    def test_global_t2_t2g_and_t3_cross_loads_are_rejected_atomically(self):
        models = {
            "global": self._model(GLOBAL_TEB_V1),
            "t2": self._model(PATCH_CONDITIONED_V1),
            "t2g": self._model(GLOBAL_MEDIATED_PATCH_V1),
            "t3": self._model(SELECTIVE_PATCH_V1),
        }
        for source_name, source in models.items():
            for target_name, target in models.items():
                if source_name == target_name:
                    continue
                with self.subTest(source=source_name, target=target_name):
                    before = self._snapshot(target)
                    with self.assertRaisesRegex(
                        RuntimeError, "strict checkpoint contract"
                    ):
                        target.load_state_dict(source.state_dict(), strict=True)
                    self.assertStateUnchanged(before, target)

    def test_t3_rejects_every_source_kind_importer_without_pollution(self):
        target = self._model()
        baseline = AMD(**self._backbone_kwargs()).state_dict()
        for source_kind in ("baseline", "pmcr_only", "teb_only"):
            with self.subTest(source_kind=source_kind):
                before = self._snapshot(target)
                with self.assertRaisesRegex(RuntimeError, "from-scratch"):
                    target.load_enhancement_state_dict(
                        baseline, source_kind=source_kind
                    )
                self.assertStateUnchanged(before, target)

    def test_constructor_rejects_missing_wrong_and_foreign_contracts(self):
        base = dict(
            architecture=SELECTIVE_PATCH_V1,
            patch_size=3,
        )
        for field, value in (
            ("teb_patch_confidence_gate", "pre_projection"),
            ("teb_patch_gate_input", "attention_only"),
            ("teb_patch_gate_activation", "sigmoid"),
            ("teb_patch_gate_init", "random"),
            ("teb_global_prediction_role", "forecast_connected"),
        ):
            model_kwargs = {
                "teb_patch_size": 3,
                "teb_patch_padding": RIGHT_ZERO_CROP,
                "teb_patch_position": FIXED_SINUSOIDAL,
                "teb_patch_confidence_gate": (
                    PATCH_CONFIDENCE_GATE_SCALAR_POST_PROJECTION
                ),
                "teb_patch_gate_input": PATCH_GATE_INPUT_QUERY_AND_ATTENTION,
                "teb_patch_gate_activation": PATCH_GATE_ACTIVATION_TWO_SIGMOID,
                "teb_patch_gate_init": PATCH_GATE_INIT_EXPLICIT_ZERO_IDENTITY,
                "teb_global_prediction_role": GLOBAL_PREDICTION_ROLE_STATE_ONLY,
            }
            model_kwargs[field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "contract mismatch"):
                    AMDEnhanced(
                        **self._backbone_kwargs(),
                        target_idx=1,
                        teb_context_dim=32,
                        task_mode=TARGET_EXOGENOUS,
                        aux_idx=(0, 2),
                        use_teb=True,
                        teb_architecture=base["architecture"],
                        teb_heads=4,
                        teb_dropout=0.1,
                        **model_kwargs,
                    )

        with self.assertRaisesRegex(ValueError, "does not accept T2G"):
            AMDEnhanced(
                **self._backbone_kwargs(),
                target_idx=1,
                teb_context_dim=32,
                task_mode=TARGET_EXOGENOUS,
                aux_idx=(0, 2),
                use_teb=True,
                teb_architecture=SELECTIVE_PATCH_V1,
                teb_heads=4,
                teb_dropout=0.1,
                teb_patch_size=3,
                teb_patch_padding=RIGHT_ZERO_CROP,
                teb_patch_position=FIXED_SINUSOIDAL,
                teb_patch_confidence_gate=(
                    PATCH_CONFIDENCE_GATE_SCALAR_POST_PROJECTION
                ),
                teb_patch_gate_input=PATCH_GATE_INPUT_QUERY_AND_ATTENTION,
                teb_patch_gate_activation=PATCH_GATE_ACTIVATION_TWO_SIGMOID,
                teb_patch_gate_init=PATCH_GATE_INIT_EXPLICIT_ZERO_IDENTITY,
                teb_global_prediction_role=GLOBAL_PREDICTION_ROLE_STATE_ONLY,
                teb_beta_global_init=1e-3,
            )


if __name__ == "__main__":
    unittest.main()
