import copy
import unittest

import torch
from torch.nn import functional as F

from models.modules.cross_correlation_embedding import (
    FEATURE_SCHEMA_ORDER,
    IDENTITY_RESIDUAL_DELTA_V1,
    ORDERED_AUX_THEN_TARGET,
    PARALLEL_MULTIVARIATE,
    TARGET_EXOGENOUS,
    ZERO_SAME,
    CrossCorrelationEmbedding,
)
from models.tsAMD_enhanced import AMDEnhanced


def _rng_state():
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": (
            [value.clone() for value in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    }


def _assert_rng_equal(test_case, left, right):
    test_case.assertTrue(torch.equal(left["cpu"], right["cpu"]))
    if left["cuda"] is None:
        test_case.assertIsNone(right["cuda"])
    else:
        test_case.assertEqual(len(left["cuda"]), len(right["cuda"]))
        for expected, observed in zip(left["cuda"], right["cuda"]):
            test_case.assertTrue(torch.equal(expected, observed))


def _model(*, use_cce, task_mode=TARGET_EXOGENOUS, target_idx=1):
    return AMDEnhanced(
        input_shape=(4, 3),
        pred_len=2,
        n_block=1,
        dropout=0.0,
        patch=4,
        k=0,
        c=2,
        alpha=0.0,
        target_slice=None,
        norm=True,
        layernorm=False,
        target_idx=target_idx,
        teb_context_dim=4,
        task_mode=task_mode,
        aux_idx=(2, 0) if task_mode == TARGET_EXOGENOUS else (),
        use_cce=use_cce,
        cce_kernel_size=3,
        cce_lambda_init=0.1,
        cce_padding_policy=ZERO_SAME,
        cce_input_order_policy=(
            ORDERED_AUX_THEN_TARGET
            if task_mode == TARGET_EXOGENOUS
            else FEATURE_SCHEMA_ORDER
        ),
        cce_parameterization_policy=IDENTITY_RESIDUAL_DELTA_V1,
        cce_feature_schema=("a", "b", "c") if use_cce else None,
        cce_schema_fingerprint="fixture-schema" if use_cce else None,
        use_pmcr=False,
        use_teb=False,
    )


class CrossCorrelationEmbeddingTests(unittest.TestCase):
    def test_cpu_cuda_dtype_shape_identity_and_analysis_interface(self):
        cases = [
            (torch.device("cpu"), torch.float32),
            (torch.device("cpu"), torch.float64),
        ]
        if torch.cuda.is_available():
            cases.append((torch.device("cuda"), torch.float32))

        for device, dtype in cases:
            with self.subTest(device=str(device), dtype=str(dtype)):
                module = CrossCorrelationEmbedding(
                    feature_num=4,
                    task_mode=TARGET_EXOGENOUS,
                    target_idx=2,
                    aux_idx=(3, 0, 1),
                ).to(device=device, dtype=dtype)
                value = torch.randn(2, 17, 4, device=device, dtype=dtype)
                x_ch = value.transpose(1, 2).contiguous()
                output = module(x_ch)
                self.assertEqual(output.shape, x_ch.shape)
                self.assertEqual(output.dtype, dtype)
                self.assertEqual(output.device, module.delta_weight.device)
                self.assertTrue(torch.equal(output, x_ch))
                self.assertEqual(
                    module.effective_lambda().item(),
                    module.rho.new_tensor(0.1).item(),
                )
                delta = module.compute_ungated_delta(x_ch)
                self.assertEqual(delta.shape, (2, 1, 17))
                self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
                kernel, bias = module.equivalent_crosslinear_kernel()
                self.assertEqual(kernel.shape, (1, 4, 3))
                self.assertEqual(bias.shape, (1,))
                expected = F.conv1d(
                    x_ch[:, (3, 0, 1, 2), :], kernel, bias, padding=1
                )
                self.assertTrue(torch.equal(expected, x_ch[:, 2:3, :]))

    def test_constructor_is_cpu_and_all_cuda_rng_neutral(self):
        torch.manual_seed(8127)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(8127)
        before = _rng_state()
        CrossCorrelationEmbedding(
            feature_num=7,
            task_mode=TARGET_EXOGENOUS,
            target_idx=6,
            aux_idx=(0, 1, 2, 3, 4, 5),
        )
        CrossCorrelationEmbedding(
            feature_num=7,
            task_mode=PARALLEL_MULTIVARIATE,
            target_idx=6,
            aux_idx=(),
        )
        after = _rng_state()
        _assert_rng_equal(self, before, after)

    def test_target_positions_order_and_untouched_channels(self):
        for target_idx in (0, 2, 4):
            aux_idx = tuple(
                index for index in (4, 3, 2, 1, 0) if index != target_idx
            )
            with self.subTest(target_idx=target_idx, aux_idx=aux_idx):
                module = CrossCorrelationEmbedding(
                    feature_num=5,
                    task_mode=TARGET_EXOGENOUS,
                    target_idx=target_idx,
                    aux_idx=aux_idx,
                )
                with torch.no_grad():
                    module.delta_weight[0, 0, 1] = 10.0
                x_ch = torch.stack(
                    [
                        torch.full((2, 6), float(index + 1))
                        for index in range(5)
                    ],
                    dim=1,
                )
                output = module(x_ch)
                expected_target = (
                    x_ch[:, target_idx, :] + x_ch[:, aux_idx[0], :]
                )
                self.assertTrue(
                    torch.equal(output[:, target_idx, :], expected_target)
                )
                for channel in range(5):
                    if channel != target_idx:
                        self.assertTrue(
                            torch.equal(
                                output[:, channel, :], x_ch[:, channel, :]
                            )
                        )
                self.assertEqual(module.source_idx, (*aux_idx, target_idx))

    def test_parallel_shape_risk_dimensions_and_parameter_count(self):
        parallel = CrossCorrelationEmbedding(
            feature_num=5,
            task_mode=PARALLEL_MULTIVARIATE,
            target_idx=3,
            aux_idx=(),
        )
        value = torch.randn(2, 5, 19)
        self.assertTrue(torch.equal(parallel(value), value))
        self.assertEqual(
            sum(parameter.numel() for parameter in parallel.parameters()),
            3 * 5 * 5 + 5 + 1,
        )

        for feature_num, sequence_length in ((7, 12), (7, 512), (321, 12)):
            with self.subTest(C=feature_num, T=sequence_length):
                module = CrossCorrelationEmbedding(
                    feature_num=feature_num,
                    task_mode=TARGET_EXOGENOUS,
                    target_idx=feature_num - 1,
                    aux_idx=tuple(range(feature_num - 1)),
                )
                x_ch = torch.randn(1, feature_num, sequence_length)
                self.assertTrue(torch.equal(module(x_ch), x_ch))
                self.assertEqual(
                    sum(parameter.numel() for parameter in module.parameters()),
                    3 * feature_num + 2,
                )

    def test_zero_same_padding_and_equivalent_kernel(self):
        module = CrossCorrelationEmbedding(
            feature_num=2,
            task_mode=TARGET_EXOGENOUS,
            target_idx=1,
            aux_idx=(0,),
        )
        with torch.no_grad():
            module.delta_weight[0, 0, :] = 10.0
            module.delta_bias.zero_()
        x_ch = torch.tensor([[[1.0, 2.0, 3.0], [7.0, 7.0, 7.0]]])
        output = module(x_ch)
        self.assertTrue(torch.equal(output[:, 0, :], x_ch[:, 0, :]))
        torch.testing.assert_close(
            output[:, 1, :],
            torch.tensor([[10.0, 13.0, 12.0]]),
            rtol=0,
            atol=0,
        )
        kernel, bias = module.equivalent_crosslinear_kernel()
        expected_target = F.conv1d(
            x_ch[:, (0, 1), :], kernel, bias, padding=1
        )
        torch.testing.assert_close(
            output[:, 1:2, :], expected_target, rtol=0, atol=0
        )

    def test_contract_guards_and_no_internal_normalization(self):
        invalid = (
            ({"feature_num": True}, (ValueError, TypeError)),
            ({"target_idx": True}, (ValueError, TypeError)),
            ({"aux_idx": (True,)}, TypeError),
            ({"aux_idx": (0, 0)}, ValueError),
            ({"aux_idx": (3,)}, ValueError),
            ({"aux_idx": (1,)}, ValueError),
            ({"aux_idx": ()}, ValueError),
            ({"kernel_size": 5}, ValueError),
            ({"lambda_init": 0.2}, ValueError),
            ({"padding_policy": "reflect"}, ValueError),
            ({"parameterization_policy": "random"}, ValueError),
            ({"input_order_policy": FEATURE_SCHEMA_ORDER}, ValueError),
        )
        base = {
            "feature_num": 3,
            "task_mode": TARGET_EXOGENOUS,
            "target_idx": 1,
            "aux_idx": (0, 2),
        }
        for overrides, exception in invalid:
            with self.subTest(overrides=overrides):
                kwargs = {**base, **overrides}
                with self.assertRaises(exception):
                    CrossCorrelationEmbedding(**kwargs)

        with self.assertRaises(ValueError):
            CrossCorrelationEmbedding(
                feature_num=1,
                task_mode=PARALLEL_MULTIVARIATE,
                target_idx=0,
            )
        with self.assertRaises(ValueError):
            CrossCorrelationEmbedding(
                feature_num=3,
                task_mode=PARALLEL_MULTIVARIATE,
                target_idx=0,
                aux_idx=(1,),
            )
        module = CrossCorrelationEmbedding(**base)
        with torch.no_grad():
            module.rho.fill_(-1000.0)
            self.assertEqual(module.effective_lambda().item(), 0.0)
            module.rho.fill_(1000.0)
            self.assertEqual(module.effective_lambda().item(), 1.0)
            module.rho.zero_()
        self.assertEqual(list(module.modules()), [module])
        self.assertEqual(list(module.named_buffers()), [])
        self.assertFalse(
            any("norm" in name.lower() for name, _ in module.named_modules())
        )


class CrossCorrelationEmbeddingAMDIntegrationTests(unittest.TestCase):
    def _paired_models(self):
        torch.manual_seed(9182)
        control = _model(use_cce=False)
        torch.manual_seed(9182)
        candidate = _model(use_cce=True)
        return control, candidate

    def test_initial_prediction_moe_state_and_off_path_are_exact(self):
        control, candidate = self._paired_models()
        self.assertIsNone(control.cce)
        self.assertFalse(
            any(key.startswith("cce.") for key in control.state_dict())
        )
        self.assertEqual(
            {key for key in candidate.state_dict() if key.startswith("cce.")},
            {"cce.delta_weight", "cce.delta_bias", "cce.rho"},
        )
        for key, value in control.state_dict().items():
            self.assertTrue(torch.equal(value, candidate.state_dict()[key]), key)

        x = torch.randn(3, 4, 3)
        control.eval()
        candidate.eval()
        with torch.no_grad():
            control_result = control(x, return_state_source=True)
            candidate_result = candidate(x, return_state_source=True)
        for expected, observed in zip(control_result, candidate_result):
            self.assertTrue(torch.equal(expected, observed))
        state_source = candidate_result[2]
        self.assertTrue(
            torch.equal(
                state_source[:, 8:], torch.zeros_like(state_source[:, 8:])
            )
        )

    def test_first_backward_and_step_then_second_rho_gradient(self):
        control, candidate = self._paired_models()
        control.train()
        candidate.train()
        x = torch.randn(3, 4, 3)
        target = torch.randn(3, 2, 1)
        criterion = torch.nn.MSELoss()
        control_optimizer = torch.optim.Adam(
            control.parameters(), lr=3e-5, weight_decay=1e-7
        )
        candidate_optimizer = torch.optim.Adam(
            candidate.parameters(), lr=3e-5, weight_decay=1e-7
        )

        control_optimizer.zero_grad()
        candidate_optimizer.zero_grad()
        shared_rng = _rng_state()
        control_prediction, control_aux = control(x)
        torch.set_rng_state(shared_rng["cpu"])
        if shared_rng["cuda"] is not None:
            torch.cuda.set_rng_state_all(shared_rng["cuda"])
        candidate_prediction, candidate_aux = candidate(x)
        self.assertTrue(torch.equal(control_prediction, candidate_prediction))
        self.assertTrue(torch.equal(control_aux, candidate_aux))
        control_loss = criterion(control_prediction, target) + control_aux
        candidate_loss = criterion(candidate_prediction, target) + candidate_aux
        control_loss.backward()
        candidate_loss.backward()

        aux_gradient = candidate.cce.delta_weight.grad[:, :2, :]
        self.assertTrue(bool(torch.isfinite(aux_gradient).all()))
        self.assertGreater(aux_gradient.abs().max().item(), 0.0)
        self.assertEqual(candidate.cce.rho.grad.item(), 0.0)
        control_parameters = dict(control.named_parameters())
        candidate_parameters = dict(candidate.named_parameters())
        for name, expected in control_parameters.items():
            observed = candidate_parameters[name]
            if expected.grad is None:
                self.assertIsNone(observed.grad, name)
            else:
                self.assertTrue(torch.equal(expected.grad, observed.grad), name)

        delta_before = candidate.cce.delta_weight.detach().clone()
        control_optimizer.step()
        candidate_optimizer.step()
        for key, expected in control.state_dict().items():
            self.assertTrue(torch.equal(expected, candidate.state_dict()[key]), key)
        self.assertFalse(torch.equal(delta_before, candidate.cce.delta_weight))
        self.assertEqual(candidate.cce.rho.item(), 0.0)

        candidate_optimizer.zero_grad()
        prediction, auxiliary = candidate(x)
        (criterion(prediction, target) + auxiliary).backward()
        self.assertTrue(bool(torch.isfinite(candidate.cce.rho.grad)))
        self.assertNotEqual(candidate.cce.rho.grad.item(), 0.0)

    def test_cce_source_importer_exact_allowlist_and_atomic_rejection(self):
        torch.manual_seed(11)
        source = _model(use_cce=False)
        torch.manual_seed(29)
        target = _model(use_cce=True)
        source_state = copy.deepcopy(source.state_dict())
        contract = target.cce_source_import_contract()
        result = target.load_cce_source_state_dict(
            source_state, source_contract=contract
        )
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        for key, value in source_state.items():
            self.assertTrue(torch.equal(value, target.state_dict()[key]), key)
        for key in ("cce.delta_weight", "cce.delta_bias", "cce.rho"):
            observed = target.state_dict()[key]
            self.assertTrue(torch.equal(observed, torch.zeros_like(observed)))

        strict_copy = _model(use_cce=True)
        strict_copy.load_state_dict(target.state_dict(), strict=True)
        with self.assertRaisesRegex(ValueError, "strict=True"):
            strict_copy.load_state_dict(target.state_dict(), strict=False)
        with self.assertRaisesRegex(RuntimeError, "missing"):
            strict_copy.load_state_dict(source_state, strict=True)

        bad_states = []
        missing = copy.deepcopy(source_state)
        missing.pop(next(iter(missing)))
        bad_states.append(missing)
        partial_cce = copy.deepcopy(source_state)
        partial_cce["cce.rho"] = torch.zeros(())
        bad_states.append(partial_cce)
        shape = copy.deepcopy(source_state)
        shape_key = next(
            key for key, value in shape.items() if value.ndim > 0
        )
        shape[shape_key] = torch.zeros((*shape[shape_key].shape, 1))
        bad_states.append(shape)
        dtype = copy.deepcopy(source_state)
        dtype_key = next(
            key for key, value in dtype.items() if value.is_floating_point()
        )
        dtype[dtype_key] = dtype[dtype_key].double()
        bad_states.append(dtype)

        bad_contracts = []
        for field, value in (
            ("task_mode", PARALLEL_MULTIVARIATE),
            ("feature_schema", ("b", "a", "c")),
            ("target_idx", 0),
            ("aux_idx", (0, 2)),
            ("schema_fingerprint", "different"),
            ("input_order_policy", FEATURE_SCHEMA_ORDER),
        ):
            changed = dict(contract)
            changed[field] = value
            bad_contracts.append(changed)

        for bad_state in bad_states:
            before = copy.deepcopy(target.state_dict())
            with self.assertRaises(RuntimeError):
                target.load_cce_source_state_dict(
                    bad_state, source_contract=contract
                )
            for key, value in before.items():
                self.assertTrue(
                    torch.equal(value, target.state_dict()[key]), key
                )
        for bad_contract in bad_contracts:
            before = copy.deepcopy(target.state_dict())
            with self.assertRaises(RuntimeError):
                target.load_cce_source_state_dict(
                    source_state, source_contract=bad_contract
                )
            for key, value in before.items():
                self.assertTrue(
                    torch.equal(value, target.state_dict()[key]), key
                )


if __name__ == "__main__":
    unittest.main()
