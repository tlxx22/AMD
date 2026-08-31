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
from models.modules.target_exogenous_bridge import TARGET_EXOGENOUS
from models.tsAMD import AMD
from models.tsAMD_enhanced import AMDEnhanced, GLOBAL_TEB_V1


class GlobalMediatedPatchTEBCheckpointTests(unittest.TestCase):
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

    def _model(self, *, architecture=GLOBAL_MEDIATED_PATCH_V1, patch_size=3):
        patch = {}
        if architecture in {PATCH_CONDITIONED_V1, GLOBAL_MEDIATED_PATCH_V1}:
            patch = {
                "teb_patch_size": patch_size,
                "teb_patch_padding": RIGHT_ZERO_CROP,
                "teb_patch_position": FIXED_SINUSOIDAL,
            }
        t2g = {}
        if architecture == GLOBAL_MEDIATED_PATCH_V1:
            t2g = {
                "teb_global_residual": GLOBAL_RESIDUAL_CONTRACT,
                "teb_patch_attention_residual": PATCH_ATTENTION_RESIDUAL_NONE,
                "teb_global_gate": GLOBAL_GATE_SCALAR_PER_PATCH,
                "teb_global_gate_input": GLOBAL_GATE_INPUT_CONTRACT,
                "teb_global_gate_init": GLOBAL_GATE_IDENTITY_INIT,
                "teb_beta_global_init": 1e-3,
            }
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
            **patch,
            **t2g,
        )

    @staticmethod
    def _snapshot(model):
        return {key: value.detach().clone() for key, value in model.state_dict().items()}

    def assertStateUnchanged(self, before, model):
        after = model.state_dict()
        self.assertEqual(list(before), list(after))
        for key, value in before.items():
            self.assertTrue(torch.equal(value, after[key]), key)

    def test_same_structure_strict_restore_and_strict_false_rejection(self):
        torch.manual_seed(5301)
        source = self._model()
        target = self._model()
        result = target.load_state_dict(source.state_dict(), strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        before = self._snapshot(target)
        with self.assertRaisesRegex(ValueError, "strict=True"):
            target.load_state_dict(source.state_dict(), strict=False)
        self.assertStateUnchanged(before, target)

    def test_partial_new_unexpected_and_shape_failures_are_atomic(self):
        state = dict(self._model().state_dict())
        partial = dict(state)
        partial.pop("teb.beta_global")
        unexpected = dict(state)
        unexpected["teb.unexpected.weight"] = torch.ones(1)
        wrong_shape = dict(state)
        wrong_shape["teb.global_injection_gate.weight"] = torch.zeros(1, 63)
        for name, candidate, message in (
            ("partial", partial, "missing"),
            ("unexpected", unexpected, "unexpected"),
            ("shape", wrong_shape, "shape"),
        ):
            with self.subTest(name=name):
                target = self._model()
                before = self._snapshot(target)
                with self.assertRaisesRegex(RuntimeError, message):
                    target.load_state_dict(candidate, strict=True)
                self.assertStateUnchanged(before, target)

    def test_global_t2_and_t2g_cross_loads_are_rejected_atomically(self):
        models = {
            "global": self._model(architecture=GLOBAL_TEB_V1),
            "t2": self._model(architecture=PATCH_CONDITIONED_V1),
            "t2g": self._model(),
        }
        for source_name, target_name in (
            ("global", "t2g"),
            ("t2", "t2g"),
            ("t2g", "t2"),
            ("t2g", "global"),
        ):
            with self.subTest(source=source_name, target=target_name):
                target = models[target_name]
                before = self._snapshot(target)
                with self.assertRaisesRegex(RuntimeError, "strict checkpoint contract"):
                    target.load_state_dict(models[source_name].state_dict(), strict=True)
                self.assertStateUnchanged(before, target)

    def test_t2g_rejects_every_source_kind_importer_without_pollution(self):
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

    def test_patch_and_contract_mismatches_are_rejected_at_construction(self):
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            AMDEnhanced(
                **self._backbone_kwargs(),
                target_idx=1,
                teb_context_dim=32,
                task_mode=TARGET_EXOGENOUS,
                aux_idx=(0, 2),
                use_pmcr=False,
                use_teb=True,
                teb_heads=4,
                teb_dropout=0.1,
                teb_architecture=GLOBAL_MEDIATED_PATCH_V1,
                teb_patch_size=3,
                teb_patch_padding=RIGHT_ZERO_CROP,
                teb_patch_position=FIXED_SINUSOIDAL,
                teb_global_residual=GLOBAL_RESIDUAL_CONTRACT,
                teb_patch_attention_residual=PATCH_ATTENTION_RESIDUAL_NONE,
                teb_global_gate=GLOBAL_GATE_SCALAR_PER_PATCH,
                teb_global_gate_input=GLOBAL_GATE_INPUT_CONTRACT,
                teb_global_gate_init=GLOBAL_GATE_IDENTITY_INIT,
                teb_beta_global_init=0.0,
            )


if __name__ == "__main__":
    unittest.main()
