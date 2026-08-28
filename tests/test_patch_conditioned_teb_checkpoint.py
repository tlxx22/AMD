import unittest

import torch

from models.modules.patch_conditioned_target_exogenous_bridge import (
    FIXED_SINUSOIDAL,
    PATCH_CONDITIONED_V1,
    RIGHT_ZERO_CROP,
    PatchConditionedTargetExogenousBridge,
)
from models.modules.target_exogenous_bridge import (
    TARGET_EXOGENOUS,
    TargetExogenousBridge,
)
from models.tsAMD import AMD
from models.tsAMD_enhanced import AMDEnhanced, GLOBAL_TEB_V1


class PatchConditionedTEBCheckpointTests(unittest.TestCase):
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

    def _model(self, *, architecture=PATCH_CONDITIONED_V1, patch_size=3):
        patch_kwargs = {}
        if architecture == PATCH_CONDITIONED_V1:
            patch_kwargs = {
                "teb_patch_size": patch_size,
                "teb_patch_padding": RIGHT_ZERO_CROP,
                "teb_patch_position": FIXED_SINUSOIDAL,
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
            **patch_kwargs,
        )

    def assertStateUnchanged(self, before, model):
        after = model.state_dict()
        self.assertEqual(list(before), list(after))
        for key, value in before.items():
            self.assertTrue(torch.equal(value, after[key]), key)

    @staticmethod
    def _snapshot(model):
        return {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    def test_same_structure_strict_restore_and_t2_strict_false_rejection(self):
        torch.manual_seed(4301)
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

    def test_partial_unexpected_and_shape_failures_are_atomic(self):
        torch.manual_seed(4313)
        source_state = dict(self._model().state_dict())
        partial = dict(source_state)
        partial.pop(next(key for key in partial if key.startswith("teb.patch_query")))
        unexpected = dict(source_state)
        unexpected["teb.unexpected.weight"] = torch.ones(1)
        shape_mismatch = dict(source_state)
        key = "teb.patch_query_projection.weight"
        shape_mismatch[key] = shape_mismatch[key].new_zeros(32, 4)

        for name, state, message in (
            ("partial", partial, "missing"),
            ("unexpected", unexpected, "unexpected"),
            ("shape", shape_mismatch, "shape"),
        ):
            with self.subTest(case=name):
                target = self._model()
                before = self._snapshot(target)
                with self.assertRaisesRegex(RuntimeError, message):
                    target.load_state_dict(state, strict=True)
                self.assertStateUnchanged(before, target)

    def test_patch_size_mismatch_is_rejected_before_parameter_write(self):
        source = self._model(patch_size=3)
        target = self._model(patch_size=4)
        before = self._snapshot(target)
        with self.assertRaisesRegex(RuntimeError, "shape"):
            target.load_state_dict(source.state_dict(), strict=True)
        self.assertStateUnchanged(before, target)

    def test_global_and_t2_cross_loads_are_rejected_atomically(self):
        torch.manual_seed(4337)
        global_model = self._model(architecture=GLOBAL_TEB_V1)
        t2_model = self._model()

        before_t2 = self._snapshot(t2_model)
        with self.assertRaisesRegex(RuntimeError, "strict checkpoint contract"):
            t2_model.load_state_dict(global_model.state_dict(), strict=True)
        self.assertStateUnchanged(before_t2, t2_model)

        before_global = self._snapshot(global_model)
        with self.assertRaisesRegex(RuntimeError, "strict checkpoint contract"):
            global_model.load_state_dict(t2_model.state_dict(), strict=True)
        self.assertStateUnchanged(before_global, global_model)

    def test_t2_rejects_every_source_kind_importer_without_pollution(self):
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

    def test_position_and_padding_mismatches_are_rejected_at_construction(self):
        common = dict(
            seq_len=12,
            feature_num=3,
            task_mode=TARGET_EXOGENOUS,
            target_idx=1,
            aux_idx=(0, 2),
            context_dim=32,
            num_heads=4,
            dropout=0.1,
            patch_size=3,
            gamma_init=1e-3,
        )
        with self.assertRaisesRegex(ValueError, "padding_policy"):
            PatchConditionedTargetExogenousBridge(
                **common,
                padding_policy="replicate",
                position_policy=FIXED_SINUSOIDAL,
            )
        with self.assertRaisesRegex(ValueError, "position_policy"):
            PatchConditionedTargetExogenousBridge(
                **common,
                padding_policy=RIGHT_ZERO_CROP,
                position_policy="learnable",
            )

    def test_global_v1_state_keys_remain_patch_free_and_exact(self):
        module = TargetExogenousBridge(
            seq_len=12,
            feature_num=3,
            task_mode=TARGET_EXOGENOUS,
            target_idx=1,
            aux_idx=(0, 2),
            context_dim=32,
            num_heads=4,
            dropout=0.1,
        )
        expected = {
            "gamma_teb",
            "query_projection.weight",
            "query_projection.bias",
            "query_norm.weight",
            "query_norm.bias",
            "exogenous_projection.weight",
            "exogenous_projection.bias",
            "exogenous_norm.weight",
            "exogenous_norm.bias",
            "cross_attention.in_proj_weight",
            "cross_attention.in_proj_bias",
            "cross_attention.out_proj.weight",
            "cross_attention.out_proj.bias",
            "output_projection.weight",
            "output_projection.bias",
        }
        self.assertEqual(set(module.state_dict()), expected)
        self.assertFalse(any("patch" in key for key in module.state_dict()))


if __name__ == "__main__":
    unittest.main()
