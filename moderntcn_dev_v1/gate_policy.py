"""Versioned reporting contract; never changes loss or validation selection."""
import copy
import json

_POLICY = {
    'schema': 'moderntcn-uv-ms-gate-v3',
    'metrics': ['mse', 'mae'],
    'comparison': 'strict_less_unrounded',
    'urban': {'folds': [1, 4], 'horizons': [3, 6, 9, 12]},
    'goal1': {'aggregation': 'each_fold_independently', 'within_fold_weight': 'equal_1_over_4',
              'cross_fold_compensation': False, 'conditions': 'UV<U<B and UV<V<B for each metric in both folds'},
    'goal3': {'aggregation': 'pooled_two_folds_four_horizons', 'weight': 'equal_1_over_8',
              'conditions': 'F4<F2<F1 and F4<F3<F1 for each metric', 'F4_reuse': 'exact_same_candidate_task_identity'},
    'goal2': {'reference':'paired_round2_B_only','datasets':['PJM','NP','BE','FR','DE'],
              'horizons':[24],'cell_axes':['dataset','H','metric'], 'denominator':10,
              'minimum_strict_winning_cells':7,'threshold_fraction':0.7,
              'separate_metric_threshold':False,'ties_are_wins':False,
              'missing_reduce_denominator':False,'cross_dataset_raw_average':False},
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
