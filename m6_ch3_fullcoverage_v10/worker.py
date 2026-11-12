"""Only the explicitly missing frozen test subset is presented to a model."""
from pathlib import Path
import numpy as np
import torch
from utils.ch3_time_marks import batch_parts  # Bind before author utils switches.
from m6_ch3_test_reporting_v10.evaluate import restore,infer,append
from m6_ch3_test_reporting_v10.analysis import sample_keys,physical,compare
from .contract import require,read,sha,write_new,load_ref
from .analysis import validate_cache


def datasets(c,j,row,cap,access):
    require(c,j,row,cap)
    from utils.ch3_data import load
    from utils.ch3_contract import digest
    p,t,protocol=row['profile'],row['task'],row['data_protocol']
    append(access,dict(event='raw_test_read_requested',physical_read=row['test_arithmetic'],job=j,capability=cap))
    sets,meta=load(protocol,t,test_capability=dict(purpose='ch3_formal_test',protocol_sha=digest(protocol),run_id=t['id']),resolved_profile=p)
    if 'test' not in sets or not meta['test_observations_accessed'] or len(sets['test'])!=row['test_arithmetic']['full_test_samples']:raise ValueError('exact original test loader')
    for k in ('scaler','node_order','dataset','fold','input_variant','target'):
        if k in row['data'] and meta.get(k)!=row['data'][k]:raise ValueError('data identity '+k)
    if set(meta['interval_counts'])!={'0 days 01:00:00'}:raise ValueError('hourly continuity')
    append(access,dict(event='raw_test_read_completed',records_end=meta['endpoints_read'],physical_full_test_windows=len(sets['test'])))
    return sets['test'],meta


def run(c,j,folder,cap,stop):
    row=load_ref(c['sources_ref'])['rows'][j['cell']];require(c,j,row,cap)
    out=Path(folder)/j['id'];out.mkdir(exist_ok=False);write_new(out/'source.json',row);access=out/'access.jsonl'
    append(access,dict(event='claim_accepted',capability=cap))
    from m6_ch3_supplement_v10.gpu_identity import check_gpu_identity
    check_gpu_identity(c['gpu_uuid'],out/'gpu-identity','before-load',False)
    data,meta=datasets(c,j,row,cap,access)
    indices=load_ref(j['plan_ref'])['dataset_indices'] if j['phase']=='C' else list(range(len(data)))
    if len(indices)!=j['expected']['samples'] or len(set(indices))!=len(indices):raise ValueError('exact unique granted subset')
    from torch.utils.data import DataLoader,Subset
    loader=DataLoader(Subset(data,indices),batch_size=j['expected']['eval_batch'],shuffle=False,drop_last=False,num_workers=0,generator=torch.Generator().manual_seed(2024))
    append(access,dict(event='checkpoint_load_requested',weights_ref=row['weights_ref']))
    model,predict=restore(row,'cuda:0');append(access,dict(event='checkpoint_restored',weights_ref=row['weights_ref']))
    check_gpu_identity(c['gpu_uuid'],out/'gpu-identity','before-forward',True)
    pred,truth,metrics,calls=infer(model,predict,loader,row['profile']['pred_len'],'cuda:0',stop,lambda event,b,n:append(access,dict(event=event,batch=b,samples=n)))
    if (calls,len(pred),pred.size)!=tuple(j['expected'][k] for k in ('batches','samples','elements')):raise ValueError('evaluation coverage')
    keys=sample_keys(row,indices);pp,tt=physical(row,pred,truth,keys['node_indices'])
    arrays=dict(prediction=pred,truth=truth,physical_prediction=pp,physical_truth=tt,**keys);validate_cache(row,arrays,indices)
    np.savez(out/'predictions.npz',**arrays)
    check_gpu_identity(c['gpu_uuid'],out/'gpu-identity','after-forward',True)
    result=dict(job=j,split='test',purpose=c['purpose'],evaluation_key=cap['evaluation_key'],new_test_access=True,metrics=metrics,calls=calls,presentations=len(pred),elements=pred.size,training=0,backward=0,optimizer=0,best_epoch=row['best_epoch'],producer_commit=row['producer_commit'],producer_identity=row['identity'],weights_ref=row['weights_ref'],test_metadata=meta,metric_space='train-standardized target; FP32 subtract then FP64 accumulate',physical_space='dataset train scaler inverse once; model RevIN already denormalized',cache_ref=dict(path=str(out/'predictions.npz'),sha256=sha(out/'predictions.npz')))
    if j['phase']=='P':result['original_test_comparison']=compare(metrics,load_ref(row['metrics_ref']),c['numeric_policy'])
    write_new(out/'result.json',result)
    if j['phase']=='P' and not all(v['passed'] for v in result['original_test_comparison'].values()):raise ValueError('original test tolerance exceeded; preserve result/cache')
    return result
