"""Metadata-only windows and exact cache unions; explicit display selections only."""
import hashlib
import json
import numpy as np
import pandas as pd
from m6_ch3_test_reporting_v10.analysis import sample_keys, physical, PositionErrors

MODELS=('ModernTCN','TimeMixer','TimeXer','AMD','PatchTST','UV')
MARKETS=('PJM','NP','BE','FR','DE')
HS=(3,6,9,12)
TEN=('208','330','333','570','969','1083','1113','1120','1162','1173')


def digest(x):
    return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def index_sha(ids):
    return hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest()


def urban_plan(row,missing_only=False):
    p=row['profile'];a=row['test_arithmetic'];nodes=row['data']['node_order'];H=p['H']
    if H not in HS or len(nodes)!=275 or p['T']!=12 or p['pred_len']!=1 or p['dataset']!='UrbanEV':raise ValueError('fixed UrbanEV shape')
    if missing_only and H!=6:raise ValueError('only H6 has inherited ten-node subset')
    ts=pd.date_range('2023-02-13',periods=168,freq='h');base=pd.Timestamp(row['data']['time_first'])
    rec=((ts-base)/pd.Timedelta(hours=1)).astype('int64').tolist()
    chosen=[i for i,n in enumerate(nodes) if not missing_only or n not in TEN]
    if len(chosen)!=(265 if missing_only else 275) or not set(TEN)<=set(nodes):raise ValueError('node coverage')
    win=[r-H-a['endpoints'][1]-p['T']+1 for r in rec]
    if min(win)<0 or max(win)>=(a['full_test_samples']//275):raise ValueError('test week outside legal windows')
    ids=[w*275+n for n in chosen for w in win]
    keys=sample_keys(row,ids)
    if not np.array_equal(keys['target_record_indices'].reshape(len(chosen),168),np.tile(rec,(len(chosen),1))):raise ValueError('target alignment')
    if not np.all(keys['target_record_indices'][:,0]-keys['origin_record_indices']==H):raise ValueError('origin H')
    result=dict(dataset='UrbanEV',split='test',H=H,P=1,nodes=[nodes[i] for i in chosen],node_indices=chosen,target_times=[str(t) for t in ts],origin_times=[str(t-pd.Timedelta(hours=H)) for t in ts],dataset_indices=ids,order='node-major/target-time-minor',subset='remaining265' if missing_only else 'all275',index_sha256=index_sha(ids))
    result['sample_set_sha256']=digest({k:v for k,v in result.items() if k!='dataset_indices'})
    return result


def validate_cache(row,arrays,indices):
    keys=sample_keys(row,indices)
    for k,v in keys.items():
        if k not in arrays or not np.array_equal(arrays[k],v):raise ValueError('cache key mismatch: '+k)
    shape=(len(indices),row['profile']['pred_len'],1)
    for k in ('prediction','truth'):
        if arrays[k].shape!=shape or arrays[k].dtype!=np.float32 or not np.isfinite(arrays[k]).all():raise ValueError('prediction/truth shape/dtype/finite')
    pp,tt=physical(row,arrays['prediction'],arrays['truth'],keys['node_indices'])
    if not np.array_equal(pp,arrays['physical_prediction']) or not np.array_equal(tt,arrays['physical_truth']):raise ValueError('dataset inverse exactly once')


def merge_caches(row,plan,parts):
    """Place original values without arithmetic; full keys and exact disjoint union."""
    ids=plan['dataset_indices'];position={i:n for n,i in enumerate(ids)}
    if len(position)!=len(ids):raise ValueError('duplicate full key')
    seen=set();out=None
    for a in parts:
        part=a['indices'].tolist();validate_cache(row,a,part)
        if len(set(part))!=len(part) or seen.intersection(part) or not set(part)<=set(position):raise ValueError('overlap/foreign subset')
        seen.update(part);loc=[position[i] for i in part]
        if out is None:out={k:np.empty((len(ids),*v.shape[1:]),dtype=v.dtype) for k,v in a.items()}
        if set(out)!=set(a):raise ValueError('cache fields')
        for k,v in a.items():
            if v.dtype!=out[k].dtype or v.shape[1:]!=out[k].shape[1:]:raise ValueError('cache dtype/shape')
            out[k][loc]=v
    if seen!=set(position):raise ValueError('missing samples')
    validate_cache(row,out,ids)
    return out


def metric_pair(pred,truth):
    m=PositionErrors(pred.shape[1]);m.add(pred,truth);return m.result()


def region_rows(H,model,arrays,plan):
    if arrays['prediction'].shape!=(len(plan['nodes'])*168,1,1):raise ValueError('all node-week rows')
    rows=[]
    for i,n in enumerate(plan['nodes']):
        sl=slice(i*168,(i+1)*168);r=dict(H=H,node=n,node_index=plan['node_indices'][i],model=model,split='test',samples=168,target_start=plan['target_times'][0],target_end=plan['target_times'][-1])
        for space,pk,tk in [('standardized','prediction','truth'),('physical','physical_prediction','physical_truth')]:
            z=metric_pair(arrays[pk][sl],arrays[tk][sl])
            for m in ('mse','mae'):r[space+'_'+m]=z['overall_'+m]
        rows.append(r)
    return rows


def explicit_models(names):
    if not names or len(names)!=len(set(names)) or any(n not in MODELS for n in names):raise ValueError('explicit nonempty ordered model list required')
    return list(names)
