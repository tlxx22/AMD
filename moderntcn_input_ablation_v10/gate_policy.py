"""Goal3 only; completed module goals are separately attributed references."""
import copy
_POLICY=dict(schema='moderntcn-v10-input-goal3-v1',metrics=['mse','mae'],comparison='strict_less_unrounded',dataset='UrbanEV',fold=6,horizons=[3,6,9,12],variants=['F1','F2','F3','F4'],arm='UV',aggregation='equal_four_H_per_metric',conditions='F4<F2<F1 and F4<F3<F1',F2_F3_order=False,per_H_gate=False,required_cells=16,fresh=12,F4_equivalent_references=4,missing_status='Not evaluated',ties_are_wins=False,goal1_goal2='separate_completed_module_result_reference_not_recomputed')
def policy():return copy.deepcopy(_POLICY)
def validate_policy(value):
    if value!=_POLICY:raise ValueError('input Goal3 policy changed')
    return value

def validate_config_policy(c):
    if 'main_metric' in c or 'gate_main_metric' in c:raise ValueError('no single-metric policy')
    return validate_policy(c.get('gate_policy'))
