"""Small fabricated arrays for exercising the actual offline publication path."""
import copy
import json
from pathlib import Path
import numpy as np
from m6_ch3_scope_amendment_v10 import contract as k
from m6_ch3_test_reporting_v10.analysis import PositionErrors


def create(config,sources,folder):
    folder=Path(folder);folder.mkdir(parents=True);old=folder/'synthetic-old';old.mkdir();c=copy.deepcopy(config);s=copy.deepcopy(sources)
    def dump(p,x):
        p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n');return dict(path=str(p),sha256=k.sha(p))
    runtime=dump(folder/'runtime.json',dict(scope='synthetic',commit='synthetic-new'));oldrt=dump(old/'runtime.json',dict(scope='synthetic',commit='synthetic-old'))
    plan=k.load_ref(s['case_plan_ref'])
    def model_result(root,key,job,case=False):
        out=root/key;out.mkdir();P=1 if case or job['dataset']=='UrbanEV' else 24;n=1680 if case else 5
        truth=np.zeros((n,P,1),np.float32);offset={'ModernTCN':.3,'TimeMixer':.2,'TimeXer':.15,'AMD':.25,'UV':.1}[job['model']];pred=np.full_like(truth,offset)
        ids=np.asarray(plan['dataset_indices']) if case else np.arange(n);nodes=np.repeat(plan['node_indices'],168) if case else np.zeros(n,dtype=np.int64)
        origin=np.arange(n);target=origin[:,None]+np.arange(1,P+1)[None,:]
        np.savez(out/'predictions.npz',prediction=pred,truth=truth,physical_prediction=pred*2,physical_truth=truth*2,indices=ids,node_indices=nodes,origin_record_indices=origin,target_record_indices=target)
        meter=PositionErrors(P);meter.add(pred,truth);metrics=meter.result()
        result=dict(job=job,metrics=metrics,best_epoch=2,producer_commit='synthetic-producer',split='test')
        rr=dump(out/'result.json',result);return out,result,rr
    for key,x in s['inherited'].items():
        j=x['job'];out,result,rr=model_result(old,key,j,key.startswith('CT-'))
        source=dict(best_epoch=2,producer_commit='synthetic-old-producer',metrics_ref=dump(out/'training-result.json',dict(validation=dict(mse=.2,mae=.3))))
        receipt=dump(old/'receipts'/(key+'.json'),dict(job=j,result=result,runtime_sha256=oldrt['sha256'],files={str(p.relative_to(old)):k.sha(p) for p in out.iterdir()}))
        s['inherited'][key]=dict(job=j,source=source,result_ref=rr,receipt_ref=receipt,runtime_ref=oldrt,prediction_ref=dict(path=str(out/'predictions.npz'),sha256=k.sha(out/'predictions.npz')))
    for key in s['centers']:
        root=folder/'centers'/key;root.mkdir(parents=True)
        metrics=dump(root/'metrics.json',dict(mse=.1,mae=.2));h=root/'history.jsonl';h.write_text(json.dumps(dict(epoch=2,validation=dict(mse=.2,mae=.3)))+'\n')
        s['centers'][key]=dict(best_epoch=2,producer_commit='synthetic-center',metrics_ref=metrics,history_ref=dict(path=str(h),sha256=k.sha(h)))
    for j in c['jobs']:
        if j['phase'] not in ('S','P','C'):continue
        out,result,rr=model_result(folder,j['id'],j,j['phase']=='C')
        source=dict(best_epoch=2,producer_commit='synthetic-new-producer',metrics_ref=dump(out/'training-result.json',dict(validation=dict(mse=.2,mae=.3))))
        dump(out/'source.json',source)
        dump(folder/'receipts'/(j['id']+'.json'),dict(job=j,result=result,runtime_sha256=runtime['sha256'],files={str(p.relative_to(folder)):k.sha(p) for p in out.iterdir()}))
    c['sources_ref']=dump(folder/'synthetic-sources.json',s)
    return c,folder
