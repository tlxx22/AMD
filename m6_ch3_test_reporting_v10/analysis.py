"""Pure statistics, frozen time rules, and explicitly nonrepresentative display ranking."""
import numpy as np
import pandas as pd
from m6_ch3_supplement_v10.observations import PositionErrors, case_times


def preselect(old, node_order):
    mu=np.asarray(old['train_mean']);sd=np.asarray(old['train_std'])
    valid=np.isfinite(mu)&np.isfinite(sd)&(mu>0)
    bounds=np.quantile(mu[valid],[.25,.75])
    candidates=np.flatnonzero(valid&(mu>=bounds[0])&(mu<=bounds[1])).tolist()
    if candidates!=old['eligible'] or bounds.tolist()!=old['mean_bounds'] or len(candidates)<10 or len(mu)!=len(node_order):
        raise ValueError('sealed train-only eligibility changed')
    return np.random.Generator(np.random.PCG64(2024)).choice(np.asarray(candidates),10,replace=False).tolist()


def case_plan(row, pre):
    a=row['test_arithmetic'];p=row['profile'];nodes=len(row['data']['node_order']);start,end=a['endpoints'][1:]
    count=end-start-p['T']-p['H']+1
    first=pd.Timestamp(row['data']['time_first'])+pd.Timedelta(hours=start+p['T']+p['H']-1)
    times=pd.date_range(first,periods=count,freq='h')
    plan=case_times(times,H=6)
    lookup={str(t):i for i,t in enumerate(times)}
    indices=[lookup[t]*nodes+n for n in pre['display_indices'] for t in plan['target_times']]
    if len(indices)!=1680 or max(indices)>=a['full_test_samples']:raise ValueError('case test window bounds')
    plan.update(dataset_indices=indices,nodes=pre['display_nodes'],node_indices=pre['display_indices'],split='test',sample_order='node-major/target-time-minor')
    return plan


def sample_keys(row, indices):
    p=row['profile'];a=row['test_arithmetic'];urban=p['dataset']=='UrbanEV'
    nodes=len(row['data']['node_order']) if urban else 1
    ids=np.asarray(indices,dtype=np.int64);win=ids//nodes;node=ids%nodes
    origin=(a['endpoints'][1]+win+p['T']-1) if urban else (a['endpoints'][1]+win-1)
    targets=origin[:,None]+(np.array([p['H']]) if urban else np.arange(1,p['pred_len']+1))[None,:]
    if targets.min()<a['endpoints'][1] or targets.max()>=a['endpoints'][2]:raise ValueError('test labels outside split')
    base=np.datetime64(row['data']['time_first'],'ns')
    return dict(indices=ids,node_indices=node,origin_record_indices=origin,target_record_indices=targets,
                origin_times=base+origin.astype('timedelta64[h]'),target_times=base+targets.astype('timedelta64[h]'))


def physical(row, prediction, truth, node_indices):
    s=row['data']['scaler'];urban=row['profile']['dataset']=='UrbanEV';idx=row['profile']['target_idx']
    scale=np.asarray(s['volume_scale'])[node_indices,None,None] if urban else s['scale'][idx]
    mean=np.asarray(s['volume_mean'])[node_indices,None,None] if urban else s['mean'][idx]
    return prediction*scale+mean,truth*scale+mean


def rank_cases(rows):
    if len(rows)!=10 or len({r['node_index'] for r in rows})!=10:raise ValueError('exact ten unique nodes')
    ranked=[]
    for row in rows:
        r=dict(row);rel=[];abs_by={'mse':[],'mae':[]};wins=0
        for model in ('ModernTCN','TimeMixer'):
            for metric in ('mse','mae'):
                b=float(r[model+'_'+metric]);u=float(r['UV_'+metric])
                if not np.isfinite([b,u]).all() or min(b,u)<0:raise ValueError('finite nonnegative case errors')
                delta=b-u;v=delta/b if b!=0 else None
                r[model+'_'+metric+'_absolute_improvement']=delta
                r[model+'_'+metric+'_relative_improvement']=v
                r[model+'_'+metric+'_win']=u<b;wins+=int(u<b);abs_by[metric].append(delta)
                if v is not None:rel.append(v)
        r.update(strict_wins=wins,relative_mean=float(np.mean(rel)) if rel else None,NA_count=4-len(rel),
                 mse_absolute_mean=float(np.mean(abs_by['mse'])),mae_absolute_mean=float(np.mean(abs_by['mae'])))
        ranked.append(r)
    # Absolute ties remain within each metric's units; no MSE+MAE unit mixture.
    ranked.sort(key=lambda r:(-r['strict_wins'],-r['relative_mean'] if r['relative_mean'] is not None else float('inf'),-r['mse_absolute_mean'],-r['mae_absolute_mean'],r['node_index']))
    for i,r in enumerate(ranked):r.update(display_rank=i+1,selected_for_plot=i<4)
    return ranked


def compare(actual, original, policy):
    out={}
    for k in ('mse','mae'):
        x=actual['overall_'+k];y=original[k];err=abs(x-y)
        out[k]=dict(original=y,measured=x,absolute_error=err,atol=policy['atol'],rtol=policy['rtol'],passed=err<=policy['atol']+policy['rtol']*abs(y))
    return out
