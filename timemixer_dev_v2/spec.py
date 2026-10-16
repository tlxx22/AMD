"""Non-executing structural identity; deliberately contains no training recipe."""
from dataclasses import dataclass
import json
from .modules import receiver_indices
from .source import binding, implementation_binding

CANDIDATE_ID = 'timemixer-uv-dev-v2'
REVISION = dict(u='normalized_history_std_tanh_v2', v_ci0='pdm_plus_target_pre_enc_trend', v_ci1='pdm_only')


def validate_revision(value):
    mode = 'pdm_only' if value['native_options']['channel_independence'] else 'pdm_plus_target_pre_enc_trend'
    if value.get('candidate_id') != CANDIDATE_ID or value.get('structural_revision') != REVISION or value.get('v_context_mode') != mode:
        raise ValueError('v2 candidate/structural revision/trend mode required')
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
        receiver_indices(p['C'], mode, p['target_idx'], p['aux_idx'])
        s = p['structure']
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
                     v_context_mode='pdm_only' if s['channel_independence'] else 'pdm_plus_target_pre_enc_trend',
                     dataset=p['dataset'], fold=fold if fold is not None else p.get('fold'),
                     input_variant=p['input_variant'], features=features, target_idx=p['target_idx'], aux_idx=p['aux_idx'],
                     L=p['T'], C=p['C'], pred_len=p['pred_len'], label_horizon=p['H'] if p['dataset']=='UrbanEV' else None,
                     native_options=options, time_mark=p.get('time_mark'),
                     module_init_seed=2024, backbone_seed=2024,
                     module_constants=dict(d_u=32, d_c=16, d_v=16, eps=1e-5, rho=0.5, gamma_init=0.01),
                     sources=binding(), implementation=implementation_binding(),
                     training_protocol='round2_complete_onecycle_v3', execution_authorized=False)
        return cls(canonical(value))
