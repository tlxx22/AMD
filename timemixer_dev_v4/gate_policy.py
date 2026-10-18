"""Versioned reporting contract; never changes loss or validation selection."""
import copy
import json

_POLICY = {
    'schema': 'timemixer-uv-dual-gate-v2',
    'metrics': ['mse', 'mae'],
    'comparison': 'strict_less_unrounded',
    'urban': {'folds': [1, 4], 'horizons': [3, 6, 9, 12]},
    'goal1': {'aggregation': 'each_fold_independently', 'within_fold_weight': 'equal_1_over_4',
              'cross_fold_compensation': False, 'conditions': 'UV<U<B and UV<V<B for each metric in both folds'},
    'goal3': {'aggregation': 'pooled_two_folds_four_horizons', 'weight': 'equal_1_over_8',
              'conditions': 'F4<F2<F1 and F4<F3<F1 for each metric', 'F4_reuse': 'exact_same_candidate_task_identity'},
    'goal2': {'scheme': 'A', 'minimum_wins_per_metric': 8, 'datasets': 11,
              'independent_winner_sets': True, 'intersection_requirement': False,
              'M_horizons': [96, 192, 336, 720], 'EPF_horizons': [24],
              'within_dataset_weight': 'equal', 'cross_dataset_raw_average': False},
}

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
