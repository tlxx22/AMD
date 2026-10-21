"""Non-executing structural identity; deliberately contains no training recipe."""
from dataclasses import dataclass
import json
from .modules import receiver_indices
from .source import binding, implementation_binding

CANDIDATE_ID = 'timemixer-uv-dev-v6'
REVISION = dict(u='xlinear_vgm_direction_preserving_native_carrier_v6_r4',
                v='timebridge_stable_time_residual_head_rms_qk_v6_r4',
                fusion='finest_post_pdm_v_then_native_head_u')
MODULE_CONSTANTS = dict(u_basis_count=8, module_dropout=0., history_embedding='shared_Linear_L_D',
                        aux_aggregation='exclusive_prefix_suffix_for_M_ordered_slots_for_MS',
                        target_condition='diag_kappa_times_A_mix', gate_range=[0.,2.], gate_shared_over_P=True,
                        u_interface='norm_preserving_relative_gate_direction', u_norm_axis='D', u_local_dtype='float64', u_added_epsilon=False, diagnostic_sample_limit=8,
                        no_aux='identity_skip_entire_controller', native_bias_scaled=False,
                        qk_dim=32, heads=4, qk_head_dim=8, relation_MA=3, relation_axis='L',
                        relation_padding='replicate_first_left2', relation_detach=True, relation_formula='((G-Gprev1)+(G-Gprev2))/3',
                        qk_normalization='per_token_per_head_mean_square', qk_eps=1e-6, attention_min_dtype='float32',
                        value='raw_G_coordinate_groups', value_projection=False, output_projection=False,
                        alpha='0.5*tanh(a)', alpha_bound=.5,
                        zero_layers=['U.output.weight_bias','V.a'], initialization='torch_Linear_defaults_except_zero_layers')
INSERTION = dict(u='finest_native_temporal_Linear_before_native_variable_projection',
                 v='finest_after_last_PDM_before_future_multi_mixing')


def validate_revision(value):
    if (value.get('candidate_id') != CANDIDATE_ID or value.get('structural_revision') != REVISION
            or value.get('module_constants') != MODULE_CONSTANTS or value.get('insertion_point') != INSERTION
            or value.get('information_scope') != 'loader_history_and_current_PDM_native_carrier_only'
            or value.get('fusion') != 'finest_post_pdm_v_then_native_head_u'
            or any(k in value for k in ('v_context_mode','kappa_U','kappa_V','prototype_count','score_dims_U','post_author_denorm'))):
        raise ValueError('v6 r4 XLinear basis gate / TimeBridge true-time raw-value revision required')
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


@dataclass(frozen=True)
class StructuralSpec:
    encoded: str

    @property
    def value(self):
        return json.loads(self.encoded)

    @classmethod
    def from_profile(cls, profile, *, use_u, use_v, task=None, fold=None):
        if profile.get('model') != 'TimeMixer' or type(use_u) is not bool or type(use_v) is not bool:
            raise ValueError('TimeMixer profile and boolean switches required')
        p = json.loads(canonical(profile))
        if p['dataset'] not in ('UrbanEV','ETTh1','ETTh2','ETTm1','ETTm2','Weather','Exchange','PJM','NP','BE','FR','DE'):
            raise ValueError('dataset outside M6-Timemixer scope')
        mode = task or p.get('task') or 'MS'
        if mode != 'MS':
            raise ValueError('r4 retains only approved MS profiles; historical M tasks retired')
        receiver_indices(p['C'], mode, p['target_idx'], p['aux_idx'])
        s = p['structure']
        if type(s['d_model']) is not int or s['d_model'] < 1 or s['d_model'] % 4:
            raise ValueError('registered D must be divisible by four for raw Value groups')
        if s['channel_independence'] != (1 if mode == 'M' else 0):
            raise ValueError('candidate supports MS/CI=0 and M/CI=1')
        if s.get('use_future_temporal_feature') != 0:
            raise ValueError('future inputs outside candidate contract')
        if s['down_sampling_method'] != 'avg' or s['down_sampling_layers'] < 0 or s['e_layers'] < 1:
            raise ValueError('unsupported inherited profile')
        if s['decomp_method'] != 'moving_avg' or s['moving_avg'] % 2 != 1:
            raise ValueError('candidate uses inherited odd moving-average decomposition')
        if any(type(p[k]) is not int or p[k] < 1 for k in ('T', 'C', 'pred_len')):
            raise ValueError('invalid L/C/P')
        if p['T'] % (s['down_sampling_window'] ** s['down_sampling_layers']):
            raise ValueError('profile scales must divide L exactly')
        features = p['features']
        if len(features) != p['C'] or len(set(features)) != p['C']:
            raise ValueError('feature identity mismatch')
        if p['dataset'] == 'UrbanEV':
            from utils.feature_schema import get_feature_schema
            if p['input_variant'] not in ('F1', 'F2', 'F3', 'F4') or tuple(features) != get_feature_schema(p['input_variant']).feature_names:
                raise ValueError('UrbanEV canonical schema mismatch')
            if (mode, p['T'], p['pred_len'], p['target_idx']) != ('MS', 12, 1, 0) or p['H'] not in (3, 6, 9, 12):
                raise ValueError('UrbanEV label_horizon must remain separate from P=1')
        options = dict(s, task_name='long_term_forecast', seq_len=p['T'], pred_len=p['pred_len'],
                       enc_in=p['C'], dec_in=p['C'], c_out=p['C'], label_len=0, features='M')
        value = dict(candidate_id=CANDIDATE_ID, use_u=use_u, use_v=use_v, task=mode,
                     structural_revision=REVISION,
                     insertion_point=INSERTION, information_scope='loader_history_and_current_PDM_native_carrier_only',
                     fusion='finest_post_pdm_v_then_native_head_u',
                     dataset=p['dataset'], fold=fold if fold is not None else p.get('fold'),
                     input_variant=p['input_variant'], features=features, target_idx=p['target_idx'], aux_idx=p['aux_idx'],
                     L=p['T'], C=p['C'], pred_len=p['pred_len'], label_horizon=p['H'] if p['dataset']=='UrbanEV' else None,
                     native_options=options, time_mark=p.get('time_mark'),
                     module_init_seed=2024, backbone_seed=2024,
                     module_constants=MODULE_CONSTANTS,
                     sources=binding(), implementation=implementation_binding(),
                     training_protocol='round2_complete_onecycle_v3', execution_authorized=False)
        return cls(canonical(value))
