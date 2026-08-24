import unittest

import torch

from models.tsAMD_enhanced import AMDEnhanced


class TEBDisabledZeroContextTests(unittest.TestCase):
    @staticmethod
    def _kwargs(feature_num=3):
        return {
            "input_shape": (4, feature_num),
            "pred_len": 2,
            "n_block": 0,
            "dropout": 0.0,
            "patch": 2,
            "k": 0,
            "c": 2,
            "alpha": 0.0,
            "target_slice": None,
            "norm": True,
            "layernorm": True,
            "target_idx": 0,
            "teb_context_dim": 8,
            "teb_heads": 2,
            "teb_dropout": 0.0,
        }

    def test_target_exogenous_off_allows_empty_aux_and_returns_zero_context(self):
        model = AMDEnhanced(
            **self._kwargs(),
            task_mode="target_exogenous",
            aux_idx=(),
            use_teb=False,
        ).eval()
        self.assertIsNone(model.teb)
        self.assertFalse(any(key.startswith("teb.") for key in model.state_dict()))
        x = torch.randn(2, 4, 3)
        with torch.no_grad():
            plain_prediction, plain_loss = model(x)
            prediction, loss, state = model(x, return_state_source=True)
        self.assertTrue(torch.equal(plain_prediction, prediction))
        self.assertTrue(torch.equal(plain_loss, loss))
        self.assertEqual(prediction.shape, (2, 2, 1))
        self.assertEqual(state.shape, (2, 16))
        self.assertTrue(torch.equal(state[:, -8:], torch.zeros_like(state[:, -8:])))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_target_exogenous_on_rejects_empty_aux_exactly(self):
        for feature_num in (3, 1):
            with self.subTest(feature_num=feature_num), self.assertRaisesRegex(
                ValueError,
                "^TEB requires at least one auxiliary variable\\.$",
            ):
                AMDEnhanced(
                    **self._kwargs(feature_num=feature_num),
                    task_mode="target_exogenous",
                    aux_idx=(),
                    use_teb=True,
                )

    def test_parallel_single_variable_policy(self):
        with self.assertRaisesRegex(
            ValueError,
            "^Parallel TEB requires at least two variables\\.$",
        ):
            AMDEnhanced(
                **self._kwargs(feature_num=1),
                task_mode="parallel_multivariate",
                aux_idx=(),
                use_teb=True,
            )

        disabled = AMDEnhanced(
            **self._kwargs(feature_num=1),
            task_mode="parallel_multivariate",
            aux_idx=(),
            use_teb=False,
        ).eval()
        prediction, _, state = disabled(
            torch.randn(2, 4, 1),
            return_state_source=True,
        )
        self.assertEqual(prediction.shape, (2, 2, 1))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(torch.equal(state[:, -8:], torch.zeros_like(state[:, -8:])))

    def test_formal_task_mode_rejects_legacy_target_slice(self):
        kwargs = self._kwargs()
        kwargs["target_slice"] = slice(0, 1)
        with self.assertRaisesRegex(ValueError, "target_slice=None"):
            AMDEnhanced(
                **kwargs,
                task_mode="target_exogenous",
                aux_idx=(1,),
                use_teb=True,
            )


if __name__ == "__main__":
    unittest.main()
