"""Separate train/validation and frozen-test entry paths; no old grant bypass."""
import random
from pathlib import Path
import numpy as np
import torch
# Bind project helper BEFORE any author adapter changes utils.__path__.
from utils.ch3_time_marks import batch_parts
from .contract import SCOPE, read, sha, load_ref, write_new, derived_task, require_test
from m6_ch3_test_reporting_v10.evaluate import infer, restore, append
from m6_ch3_test_reporting_v10.analysis import PositionErrors, sample_keys, physical, compare


def train(config, job, folder, stop):
    if job not in config['jobs'] or job['phase']!='T':raise PermissionError('explicit training job only')
    task,p,expected_data=derived_task(config,job)
    from utils.ch3_data import load
    from m6_ch3_supplement_v10.model import build
    from m6_ch3_supplement_v10.sensitivity import fit
    sets,meta=load(task['data_config'],task,resolved_profile=p)
    if set(sets)!={'train','validation'} or meta.get('test_observations_accessed'):raise PermissionError('training must never read test')
    for k in ('scaler','node_order','prefixes','record_prefix_sha256','window_counts'):
        if k in expected_data and meta.get(k)!=expected_data[k]:raise ValueError('source training data differs: '+k)
    if meta.get('prefixes',{})!=task['baseline_prefix_metadata'] or any(meta.get(k)!=v for k,v in task['baseline_record_prefix'].items()):raise ValueError('training prefix mismatch')
    random.seed(2024);np.random.seed(2024);torch.manual_seed(2024);torch.set_num_threads(p['training']['threads'])
    model=build(p,job['G'],job['R'],task='MS',fold=task['fold']).cuda()
    identity=dict(scope=SCOPE,purpose='validation_only_sensitivity',job=job,analysis=model.analysis_identity,
                  analysis_variant_id=job['analysis_variant_id'],profile=p,data=meta,source_binding=config['source_binding'],
                  base_structural_spec=model.spec.value,execution_device=dict(device='cuda:0',gpu_uuid=config['gpu_uuid']))
    return fit(model,p,task,sets,identity,Path(folder)/job['id'],stop,'cuda:0')


def resolve(config, job, folder):
    """Bind current sealed validation-best via its exact successful train receipt."""
    sources=load_ref(config['sources_ref'])
    if job['phase'] in ('P','C'):return sources['rows'][job['source_key']]
    if job['phase']!='S':raise PermissionError('only frozen evaluation has weights')
    train_job=next(j for j in config['jobs'] if j['id']==job['producer_job'])
    if train_job['phase']!='T' or any(train_job[k]!=job[k] for k in ('dataset','H','G','R','prototype','analysis_variant_id')):raise ValueError('wrong training producer')
    from .queue import checked
    receipt=checked(config,folder,train_job);out=Path(folder)/train_job['id'];identity=read(out/'identity.json');result=receipt['result'];seal=read(out/'seal.json')
    task,p,data=derived_task(config,job)
    if identity['scope']!=SCOPE or identity['job']!=train_job or identity['analysis_variant_id']!=job['analysis_variant_id'] or identity['profile']!=p or identity['source_binding']!=config['source_binding']:raise ValueError('wrong sealed analysis identity before live write')
    if (identity['analysis']['G'],identity['analysis']['R'])!=(job['G'],job['R']):raise ValueError('sealed G/R differs from producer task')
    for k in ('scaler','node_order','dataset','fold','input_variant'):
        if k in data and identity['data'].get(k)!=data[k]:raise ValueError('sealed training data identity: '+k)
    if seal['identity']!=identity or result['test_calls']!=0:raise ValueError('not a validation-only producer')
    history=[__import__('json').loads(s) for s in (out/'history.jsonl').read_text().splitlines()]
    best=min(history,key=lambda h:h['validation']['mse'])
    if best['epoch']!=result['best_epoch'] or best['validation']!=result['validation']:raise ValueError('strict validation-best selection changed')
    checksum=read(out/'best.pt.sha256')['sha256']
    if checksum!=seal['files']['best.pt'] or checksum!=receipt['files'][train_job['id']+'/best.pt']:raise ValueError('sealed best checksum chain')
    def ref(p):return dict(path=str(p),sha256=sha(p))
    return dict(kind='sensitivity',task=task,profile=p,data=identity['data'],data_protocol=task['data_config'],
                identity=identity,analysis_variant_id=job['analysis_variant_id'],best_epoch=result['best_epoch'],
                weights_ref=dict(path=str(out/'best.pt'),sha256=checksum),weights_stat=dict(bytes=(out/'best.pt').stat().st_size),
                producer_commit=read(Path(folder)/'runtime.json')['commit'],identity_ref=ref(out/'identity.json'),seal_ref=ref(out/'seal.json'),
                metrics_ref=ref(out/'result.json'),history_ref=ref(out/'history.jsonl'),runtime_ref=ref(Path(folder)/'runtime.json'),
                receipt_ref=ref(Path(folder)/'receipts'/(train_job['id']+'.json')),test_arithmetic=sources['prototypes'][job['prototype']]['test_arithmetic'])


def test_data(config,job,row,cap,access):
    require_test(config,job,row,cap)
    from utils.ch3_data import load
    from utils.ch3_contract import digest
    protocol,task,p=row['data_protocol'],row['task'],row['profile']
    append(access,dict(event='test_read_requested',job=job['id'],physical_read=row['test_arithmetic'],new_test_access=True))
    sets,meta=load(protocol,task,test_capability=dict(purpose='ch3_formal_test',protocol_sha=digest(protocol),run_id=task['id']),resolved_profile=p)
    if 'test' not in sets or not meta['test_observations_accessed']:raise ValueError('explicit new test read required')
    for k in ('scaler','node_order','target','dataset','fold','input_variant'):
        if k in row['data'] and meta.get(k)!=row['data'][k]:raise ValueError('test data identity changed: '+k)
    if set(meta['interval_counts'])!={'0 days 01:00:00'} or len(sets['test'])!=row['test_arithmetic']['full_test_samples']:raise ValueError('clock/test window coverage')
    append(access,dict(event='test_read_completed',parsed_endpoints=meta['endpoints_read'],full_test_windows=len(sets['test']),new_test_access=True))
    return sets['test'],meta


def compare_cache(path,truth,keys):
    with np.load(path,allow_pickle=False) as a:
        for k in ('truth','indices','target_record_indices','origin_record_indices','node_indices'):
            if not np.array_equal(a[k],truth if k=='truth' else keys[k]):raise ValueError('common cache sample/truth mismatch: '+k)


def evaluate(config,job,row,folder,cap,stop):
    require_test(config,job,row,cap);out=Path(folder)/job['id'];out.mkdir(exist_ok=False);access=out/'access.jsonl'
    write_new(out/'source.json',row);append(access,dict(event='evaluation_claim_accepted',capability=cap,training=0))
    from m6_ch3_supplement_v10.gpu_identity import check_gpu_identity
    check_gpu_identity(config['gpu_uuid'],out/'gpu-identity','before-model-load',False)
    data,meta=test_data(config,job,row,cap,access)
    plan=read(Path(folder)/'case-plan.json') if job['phase']=='C' else None
    indices=plan['dataset_indices'] if plan else list(range(len(data)))
    if len(indices)!=job['expected']['samples']:raise ValueError('granted test samples')
    from torch.utils.data import DataLoader,Subset
    loader=DataLoader(Subset(data,indices),batch_size=job['expected']['eval_batch'],shuffle=False,drop_last=False,num_workers=0,generator=torch.Generator().manual_seed(2024))
    append(access,dict(event='checkpoint_load_requested',weights_ref=row['weights_ref']))
    model,predict=restore(row,'cuda:0');append(access,dict(event='checkpoint_restored',weights_ref=row['weights_ref']))
    check_gpu_identity(config['gpu_uuid'],out/'gpu-identity','before-inference',True)
    pred,truth,metrics,calls=infer(model,predict,loader,row['profile']['pred_len'],'cuda:0',stop,lambda event,b,n:append(access,dict(event=event,batch=b,samples=n)))
    if calls!=job['expected']['batches'] or len(pred)!=job['expected']['samples'] or pred.size!=job['expected']['elements']:raise ValueError('test budget coverage')
    keys=sample_keys(row,indices);pp,pt=physical(row,pred,truth,keys['node_indices'])
    np.savez(out/'predictions.npz',prediction=pred,truth=truth,physical_prediction=pp,physical_truth=pt,**keys)
    sources=load_ref(config['sources_ref'])
    if job['phase'] in ('P','C'):
        oldkey=('PT-'+job['dataset'] if job['phase']=='P' else 'CT-UrbanEV')+'-UV'
        old=sources['inherited'][oldkey]['prediction_ref']
        if sha(old['path'])!=old['sha256']:raise ValueError('old cache changed')
        compare_cache(old['path'],truth,keys)
    for prior in config['jobs']:
        if prior['id']==job['id']:break
        cache=Path(folder)/prior['id']/'predictions.npz'
        if prior['phase']==job['phase'] and prior.get('dataset')==job['dataset'] and prior.get('H')==job['H'] and cache.exists():compare_cache(cache,truth,keys)
    check_gpu_identity(config['gpu_uuid'],out/'gpu-identity','after-inference',True)
    meter=PositionErrors(row['profile']['pred_len']);meter.add(pp,pt)
    result=dict(job=job,split='test',new_test_access=True,evaluation_key=cap['key'],metrics=metrics,physical_metrics=meter.result(),
                calls=calls,presentations=len(pred),elements=pred.size,train=0,backward=0,optimizer=0,best_epoch=row['best_epoch'],
                producer_identity=row['identity'],producer_commit=row['producer_commit'],weights_ref=row['weights_ref'],
                test_data_metadata=meta,metric_space='train-standardized target',cache_ref=dict(path=str(out/'predictions.npz'),sha256=sha(out/'predictions.npz')))
    if job['phase']=='P':result['original_test_comparison']=compare(metrics,load_ref(row['metrics_ref']),config['numeric_policy'])
    write_new(out/'result.json',result)
    if job['phase']=='P' and not all(x['passed'] for x in result['original_test_comparison'].values()):raise ValueError('test comparison beyond frozen policy; preserve cache')
    return result
