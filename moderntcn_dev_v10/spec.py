"""Explicit MS schema, native effective configuration and version identity."""
from dataclasses import dataclass
import json
from .source import binding,implementation_binding
CANDIDATE_ID='moderntcn-uv-dev-v10'
REVISION={'u': 'horizon_coupled_native_linear_preserving_cubic_spline_readout_v10', 'v': 'target_state_conditioned_low_rank_temporal_filter_post_ffn1_v3', 'fusion': 'native_v3_block_then_interpolated_mean_spline_target_head_before_denorm_v10'}
MODULE_CONSTANTS={'u_degree': 3, 'u_grid_intervals': 5, 'u_basis_count': 8, 'u_knots_count': 12, 'u_knots_formula': '-1+(arange(12)-3)*(2.0/5.0)', 'u_coordinate': 'tanh(final_post_block_skip_target_H_D_major_N_minor)', 'u_parameter': 'coef[K,D*N,8]', 'u_init': 'zeros_no_random_draws', 'u_scale': '1/(D*N)', 'u_base': 'original_shared_native_linear_and_bias_unchanged', 'u_readout': 'q=F.linear(flatten_basis,flatten_coef)/I; delta=F.linear(q,M); P1 original v9 forward', 'u_target_only': True, 'u_grid_update': False, 'u_extra_dropout': False, 'native_norm_calls': 1, 'native_denorm_calls': 1, 'native_head_calls': 1, 'u_addition': 'normalized_target_after_native_head_before_denorm', 'v_router_history': 'original_Z0', 'v_state_history': 'original_X', 'v_filter_input': 'original_post_FFN1_h', 'v_rank': 4, 'v_router_width': 16, 'module_dropout': 0, 'u_seed_offset': 17011, 'v_seed_offset': 17021, 'statistics_dtype': 'float32', 'gelu_approximate': 'none', 'v_target_state_features': ['mean', 'log1p_population_variance'], 'v_state_projection_shape': [16, 2], 'v_state_projection_init': 'zeros', 'downsample_ratio': 4, 'K_rule': 'max(1,(P+3)//4)', 'interpolation': 't=p*(K-1)/(P-1); floor capped K-2; endpoint aligned noncircular linear', 'P1_implementation': 'moderntcn_dev_v9.modules.SplineTargetReadout; no M buffer or extra operation'}
def canonical(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def validate_revision(v):
    if v.get('candidate_id')!=CANDIDATE_ID or v.get('structural_revision')!=REVISION or v.get('module_constants')!=MODULE_CONSTANTS:raise ValueError('ModernTCN v10 structural identity required')
    from .modules import interpolation_identity
    if v.get('horizon_coupling')!=interpolation_identity(v['pred_len'],v['D']*v['N']):raise ValueError('fixed horizon/grid identity changed')
    return v

@dataclass(frozen=True)
class StructuralSpec:
    encoded:str
    @property
    def value(self):return json.loads(self.encoded)
    @classmethod
    def from_profile(cls,p,*,use_u,use_v,task=None,fold=None):
        if p['model']!='ModernTCN' or (task or p.get('task','MS'))!='MS' or type(use_u)is not bool or type(use_v)is not bool:raise ValueError('ModernTCN MS boolean switches required')
        C,L,P=p['C'],p['T'],p['pred_len'];y=p['target_idx'];aux=p['aux_idx'];s=p['structure']
        if not 0<=y<C or len(set(aux))!=len(aux) or any(j==y or not 0<=j<C for j in aux):raise ValueError('target/ordered aux schema')
        if len(p['features'])!=C or len(set(p['features']))!=C:raise ValueError('feature schema')
        if aux!=[i for i in range(C) if i!=y]:raise ValueError('all remaining ordered variables required')
        if s['num_blocks']!=[1] or s['dims']!=[64]*4 or s['small_kernel_merged'] or s['use_multi_scale'] or s['decomposition'] or s['individual']:raise ValueError('single unfused native block required')
        if (s['revin'],s['affine'],s['subtract_last'],s['head_dropout'])!=(1,0,0,0):raise ValueError('native normalization/head contract')
        variant=p.get('backbone_variant','cfg_adapted_v1');urban=p['dataset']=='UrbanEV'
        expected=(1,1,7,3,1) if urban and variant=='cfg_adapted_v1' else (8,4,51,5,8 if variant=='paper_C1_r8_v1' else 1)
        if (s['patch_size'],s['patch_stride'],s['large_size'][0],s['small_size'][0],s['ffn_ratio'])!=expected:raise ValueError('effective native configuration mismatch')
        if variant != 'cfg_adapted_v1' or s['dropout']!=(.1 if urban else .3):raise ValueError('backbone variant/dropout')
        if urban:
            from utils.feature_schema import get_feature_schema
            if tuple(p['features'])!=get_feature_schema(p['input_variant']).feature_names or (L,P,y)!=(12,1,0) or p['H'] not in (3,6,9,12):raise ValueError('Urban schema/L/P/H')
        elif p['dataset'] not in ('PJM','NP','BE','FR','DE') or (L,P,C,y)!=(96,24,3,2):raise ValueError('EPF schema')
        if p.get('time_mark') is not None:raise ValueError('ModernTCN second-round native path has no marks argument')
        opts=dict(s,task_name='long_term_forecast',seq_len=L,pred_len=P,enc_in=C,dec_in=C,c_out=C,label_len=0,features='M')
        N=L//s['patch_stride']
        from .modules import interpolation_identity
        horizon=interpolation_identity(P,64*N)
        return cls(canonical(dict(candidate_id=CANDIDATE_ID,structural_revision=REVISION,module_constants=MODULE_CONSTANTS,horizon_coupling=horizon,task='MS',dataset=p['dataset'],fold=fold,input_variant=p['input_variant'],features=p['features'],target_idx=y,aux_idx=aux,L=L,C=C,D=64,N=N,Kv=min(s['large_size'][0],2*N-1),pred_len=P,label_horizon=p['H'],time_mark=None,use_u=use_u,use_v=use_v,native_options=opts,backbone_variant=variant,module_init_seed=2024,sources=binding(),implementation=implementation_binding(),execution_authorized=False)))
