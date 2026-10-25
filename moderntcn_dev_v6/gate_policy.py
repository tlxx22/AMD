"""Versioned reporting contract; never changes loss or validation selection."""
import copy
import json

_POLICY = {'schema': 'moderntcn-uv-v6-fold6-ms-gate-v1', 'metrics': ['mse', 'mae'], 'comparison': 'strict_less_unrounded', 'urban': {'folds': [6], 'horizons': [3, 6, 9, 12]}, 'goal1': {'aggregation': 'single_fold6_equal_four_H', 'arms': ['UV','V','U'], 'conditions': 'Urban_OK AND EPF_OK', 'Urban_OK': 'MSE_UV<MSE_U and MSE_UV<MSE_V and MAE_UV<MAE_U and MAE_UV<MAE_V on equal four H', 'EPF_OK': 'exists SAME market with all four UV<U and UV<V strict comparisons for MSE and MAE', 'required_evaluation_cells': 27, 'missing_status': 'Not evaluated', 'ties_are_wins': False, 'no_B_chain': True, 'no_every_H_gate': True, 'coverage_separate_from_contribution': True}, 'goal2': {'reference': 'paired_round2_cfg_adapted_v1', 'datasets': ['PJM', 'NP', 'BE', 'FR', 'DE'], 'horizons': [24], 'denominator': 10, 'minimum_strict_winning_cells': 7, 'separate_metric_threshold': False, 'ties_are_wins': False, 'missing_reduce_denominator': False, 'cross_dataset_raw_average': False}, 'goal3': {'aggregation': 'single_fold6_equal_four_H', 'conditions': 'F4<F2<F1 and F4<F3<F1 for each metric', 'F4_reuse': 'exact_same_candidate_revision', 'F2_F3_order': False}}

def policy():
    return copy.deepcopy(_POLICY)

def validate_policy(value):
    if not isinstance(value, dict) or json.dumps(value, sort_keys=True) != json.dumps(_POLICY, sort_keys=True):
        raise ValueError('dual gate policy differs from approved startup contracts')
    return value

def validate_config_policy(config):
    if 'main_metric' in config or 'gate_main_metric' in config:
        raise ValueError('legacy single metric startup contracts are not dual gate policies')
    return validate_policy(config.get('gate_policy'))
