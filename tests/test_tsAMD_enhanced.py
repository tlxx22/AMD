import unittest

import torch
import torch.nn as nn

from models.tsAMD import AMD
from models.tsAMD_enhanced import AMDEnhanced


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


if __name__ == "__main__":
    unittest.main()
