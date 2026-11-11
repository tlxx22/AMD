"""New explicit test grant bridge; old validation-only dataset guard is untouched."""
from pathlib import Path
import json
import numpy as np
import torch
from .contract import read, sha, load_ref, write_new, require_evaluation
from .analysis import PositionErrors, sample_keys, physical, compare


def append(path, value):
    with Path(path).open('a') as f:f.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')


def test_data(config, job, capability, access_log):
    row=require_evaluation(config,job,capability)
    from utils.ch3_data import load
    from utils.ch3_contract import digest
    c,t,p=row['data_protocol'],row['task'],row['profile']
    # The original low-level loader capability is issued only after the NEW
    # scope authorization. It does not reuse/reset any original test claim.
    cap=dict(purpose='ch3_formal_test',protocol_sha=digest(c),run_id=t['id'])
    append(access_log,dict(event='test_read_requested',job=job['id'],new_test_access=True,
                          physical_read=row['test_arithmetic'],original_run_id=t['id']))
    sets,meta=load(c,t,test_capability=cap,resolved_profile=p)
    if 'test' not in sets or not meta['test_observations_accessed']:raise ValueError('explicit test loader required')
    for k in ('scaler','node_order','target','dataset','fold','input_variant'):
        if k in row['data'] and meta.get(k)!=row['data'][k]:raise ValueError('frozen data identity: '+k)
    if set(meta['interval_counts'])!={'0 days 01:00:00'}:raise ValueError('test clock must remain hourly')
    if len(sets['test'])!=row['test_arithmetic']['full_test_samples']:raise ValueError('test sample count changed')
    append(access_log,dict(event='test_read_completed',parsed_endpoints=meta['endpoints_read'],test_observations_accessed=True,
                          full_table_test_windows=len(sets['test']),new_test_access=True))
    return sets['test'],meta


def strict_weights(model, state, row):
    if state.get('identity')!=row['identity'] or state.get('epoch')!=row['best_epoch']:
        raise ValueError('producer identity/validation-best epoch before live write')
    expected=model.state_dict();saved=state['model']
    if set(saved)!=set(expected):raise ValueError('strict model keys')
    for k,v in saved.items():
        if v.shape!=expected[k].shape or v.dtype!=expected[k].dtype or not bool(torch.isfinite(v).all()):raise ValueError('strict shape/dtype/finite '+k)
        if k in ('u.knots','u.M') and not torch.equal(v.cpu(),expected[k].cpu()):raise ValueError('fixed basis buffer changed')
    model.load_state_dict(saved,strict=True)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model


def restore(row, device):
    # This function is reachable from an owned authorized worker only.
    if sha(row['weights_ref']['path'])!=row['weights_ref']['sha256']:raise ValueError('strict-best checksum')
    state=torch.load(row['weights_ref']['path'],map_location='cpu')
    if state.get('identity')!=row['identity'] or state.get('epoch')!=row['best_epoch']:raise ValueError('frozen producer/best identity')
    p,t=row['profile'],row['task']
    if row['kind']=='sensitivity':
        from m6_ch3_supplement_v10.model import build, validate_state
        from moderntcn_dev_v10.adapter import prediction
        a=row['identity']['analysis'];model=build(p,a['G'],a['R'],task='MS',fold=t['fold'])
        validate_state(model,state['model'],a)
        predict=lambda x,marks:prediction(model,x,marks)
    elif row['kind']=='baseline':
        from models.ch3_adapter import build, target_prediction
        from utils.ch3_contract import profile
        c=load_ref(row['protocol_ref'])
        if profile(c,t)!=p:raise ValueError('baseline profile identity')
        model=build(c,t);predict=lambda x,marks:target_prediction(model,x,p,marks)[0]
    else:
        cid=row['identity']['candidate_id']
        if cid=='moderntcn-uv-dev-v9':
            from moderntcn_dev_v9 import adapter
        elif cid=='moderntcn-uv-dev-v10':
            from moderntcn_dev_v10 import adapter
        else:raise ValueError('frozen UV version')
        model=adapter.build(p,use_u=True,use_v=True,task='MS',fold=t['fold'])
        from moderntcn_dev_v10.modules import validate_fixed_state
        validate_fixed_state(model,state['model'])
        predict=lambda x,marks:adapter.prediction(model,x,marks)
    strict_weights(model,state,row)
    return model.to(device).eval(),predict


def infer(model, predict, loader, P, device, stop, on_batch):
    """Shared CPU-fixture/GPU inference loop, no optimizer or training methods."""
    from utils.ch3_time_marks import batch_parts
    if model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('frozen eval model required')
    persistent={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone() if str(device).startswith('cuda') else None
    meter=PositionErrors(P);preds=[];truths=[];calls=0
    with torch.inference_mode():
        for batch in loader:
            stop();x,y,marks=batch_parts(batch);xc=x.clone();yc=y.clone();mc=marks.clone() if marks is not None else None
            on_batch('before_forward',calls,len(x))
            pred=predict(x.to(device),marks.to(device) if marks is not None else None).cpu().numpy();truth=y.numpy()
            if not torch.equal(x,xc) or not torch.equal(y,yc) or (marks is not None and not torch.equal(marks,mc)):raise ValueError('input/target/marks mutation')
            meter.add(pred,truth);preds.append(pred);truths.append(truth);calls+=1
            on_batch('after_forward',calls,len(x))
    if not torch.equal(rng,torch.get_rng_state()) or (cuda_rng is not None and not torch.equal(cuda_rng,torch.cuda.get_rng_state())):raise ValueError('inference RNG mutation')
    if any(not torch.equal(v,model.state_dict()[k].cpu()) for k,v in persistent.items()):raise ValueError('persistent state/BN mutation')
    return np.concatenate(preds),np.concatenate(truths),meter.result(),calls


def run(config,job,folder,capability,stop):
    row=require_evaluation(config,job,capability);out=Path(folder)/job['id'];out.mkdir(exist_ok=False)
    access=out/'access.jsonl'
    append(access,dict(event='evaluation_claim_accepted',job=job,capability=capability,training=0))
    from m6_ch3_supplement_v10.gpu_identity import check_gpu_identity
    check_gpu_identity(config['gpu_uuid'],out/'gpu-identity','before-model-load',False)
    data,metadata=test_data(config,job,capability,access)
    plan=read(Path(folder)/'case-plan.json') if job['phase']=='C' else None
    indices=plan['dataset_indices'] if plan else list(range(len(data)))
    if len(indices)!=job['expected']['samples']:raise ValueError('exact granted sample range')
    from torch.utils.data import DataLoader,Subset
    loader=DataLoader(Subset(data,indices),batch_size=job['expected']['eval_batch'],shuffle=False,drop_last=False,num_workers=0,generator=torch.Generator().manual_seed(2024))
    append(access,dict(event='checkpoint_load_requested',weights_ref=row['weights_ref'],device='cpu'))
    model,predict=restore(row,'cuda:0')
    append(access,dict(event='checkpoint_restored',weights_ref=row['weights_ref']))
    check_gpu_identity(config['gpu_uuid'],out/'gpu-identity','before-inference',True)
    pred,truth,metrics,calls=infer(model,predict,loader,row['profile']['pred_len'],'cuda:0',stop,
        lambda event,batch,n:append(access,dict(event=event,batch=batch,samples=n)))
    if calls!=job['expected']['batches'] or metrics['count_per_position']*row['profile']['pred_len']!=job['expected']['elements']:raise ValueError('test coverage')
    keys=sample_keys(row,indices);pp,pt=physical(row,pred,truth,keys['node_indices'])
    np.savez(out/'predictions.npz',prediction=pred,truth=truth,physical_prediction=pp,physical_truth=pt,**keys)
    # Exact common sample/truth identity across models, using existing new cache.
    for prior in config['jobs']:
        if prior['id']==job['id']:break
        cache=Path(folder)/prior['id']/'predictions.npz'
        if prior['phase']==job['phase'] and prior['dataset']==job['dataset'] and cache.exists():
            with np.load(cache,allow_pickle=False) as a:
                for key in ('truth','indices','target_record_indices','origin_record_indices'):
                    expected=truth if key=='truth' else keys[key]
                    if not np.array_equal(a[key],expected):raise ValueError('common test sample/truth mismatch')
    check_gpu_identity(config['gpu_uuid'],out/'gpu-identity','after-inference',True)
    physical_meter=PositionErrors(row['profile']['pred_len']);physical_meter.add(pp,pt)
    result=dict(job=job,split='test',purpose=job['purpose'],new_test_access=True,evaluation_key=capability['evaluation_key'],
                metrics=metrics,physical_metrics=physical_meter.result(),metric_space='train-standardized target',physical_units='original volume' if job['dataset']=='UrbanEV' else 'original target unit',
                calls=calls,presentations=len(pred),elements=job['expected']['elements'],train=0,backward=0,optimizer=0,
                producer_identity=row['identity'],producer_commit=row['producer_commit'],weights_ref=row['weights_ref'],
                best_epoch=row['best_epoch'],source_key=job['source_key'],test_data_metadata=metadata,
                cache_ref={'path':str(out/'predictions.npz'),'sha256':sha(out/'predictions.npz')})
    if job['phase']=='P':result['original_test_comparison']=compare(metrics,load_ref(row['metrics_ref']),config['numeric_policy'])
    write_new(out/'result.json',result)
    if job['phase']=='P' and not all(v['passed'] for v in result['original_test_comparison'].values()):raise ValueError('original test discrepancy exceeds preregistered policy; cache retained')
    return result
