"""Limited synthetic/control acceptance; never reads a real checkpoint or raw test."""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch,Mock
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from m6_ch3_fullcoverage_v10 import contract as ct,analysis as an,reporting as rp,queue as qu
from m6_ch3_fullcoverage_v10 import worker as wk
from m6_ch3_test_reporting_v10.evaluate import strict_weights,infer

W=ct.ROOT
C=ct.read(W/'configs/m6_ch3_fullcoverage_v10.json')
S=ct.load_ref(C['sources_ref'])
E=Path(os.environ['FULLCOVERAGE_TEST_OUTPUT'])
COUNTS=dict(synthetic_model_constructions=0,synthetic_forward_calls=0,synthetic_presentations=0,backward=0,optimizer=0,real_checkpoint=0,raw_test=0,GPU_query=0)


def ref(p):return dict(path=str(p),sha256=ct.sha(p))


def fabricated(row,ids,delta=.125):
    keys=an.sample_keys(row,ids);shape=(len(ids),row['profile']['pred_len'],1)
    y=np.zeros(shape,dtype=np.float32);v=np.full(shape,delta,dtype=np.float32)
    pp,tt=an.physical(row,v,y,keys['node_indices'])
    return dict(prediction=v,truth=y,physical_prediction=pp,physical_truth=tt,**keys)


class Acceptance(unittest.TestCase):
    def test_01_scope_source_budget_rejection(self):
        self.assertEqual(ct.validate(C),C)
        self.assertEqual([sum(j['expected'][k] for j in C['jobs'][:-1]) for k in ('batches','samples','elements')],[10973,1257300,4866000])
        self.assertEqual(len(C['jobs']),40);self.assertEqual(len(S['rows']),54)
        for change in ({'scope':'m6_ch3_test_reporting_v10_v1'},{'q_max':4},{'jobs':C['jobs'][:-1]},{'budget':{}},{'display_policy':dict(C['display_policy'],auto_select_regions=True)}):
            with self.assertRaises((ValueError,PermissionError)):ct.validate(dict(C,**change))
        for mutate in ('missing','H','G','PatchTST','reuse'):
            s=copy.deepcopy(S)
            if mutate=='missing':s['rows'].pop('P-FR-AMD')
            if mutate=='H':s['rows']['C-H3-UV']['profile']['H']=6
            if mutate=='G':s['rows']['P-NP-UV']['kind']='sensitivity'
            if mutate=='PatchTST':s['rows']['P-PJM-PatchTST']['profile']['structure']['e_layers']=2
            if mutate=='reuse':s['inherited']['C-H3-UV']=s['inherited'].pop('C-H6-UV')
            p=E/(mutate+'.json');ct.write_new(p,s)
            with self.assertRaises((ValueError,KeyError)):ct.validate(dict(C,sources_ref=ref(p)))
        for schema in ('m6-ch3-test-reporting-launch-v1',ct.AUTH_SCHEMA):
            with self.assertRaises(PermissionError):ct.permission(W/'configs/m6_ch3_fullcoverage_v10.json',dict(schema=schema,scope=ct.SCOPE,purpose=ct.PURPOSE,commit=None,review_approved=False))

    def test_02_four_H_metadata_windows(self):
        ids=[]
        for h in an.HS:
            plan=an.urban_plan(S['rows'][f'C-H{h}-UV']);self.assertEqual(len(plan['dataset_indices']),46200)
            self.assertEqual(len(plan['nodes']),275);self.assertEqual(plan['target_times'][0],'2023-02-13 00:00:00')
            k=an.sample_keys(S['rows'][f'C-H{h}-UV'],plan['dataset_indices'])
            self.assertTrue(np.all(k['target_record_indices'][:,0]-k['origin_record_indices']==h));ids.append(plan['dataset_indices'])
        self.assertEqual(len({an.index_sha(x) for x in ids}),4)
        r=S['rows']['C-H6-UV'];part=an.urban_plan(r,True);self.assertEqual(len(part['dataset_indices']),44520)
        self.assertFalse(set(part['nodes'])&set(an.TEN))
        with self.assertRaises(ValueError):an.urban_plan(S['rows']['C-H3-UV'],True)

    def test_03_partial_cache_exact_union(self):
        row=S['rows']['C-H6-UV'];plan=an.urban_plan(row);missing=an.urban_plan(row,True)['dataset_indices'];missing_set=set(missing);old=[x for x in plan['dataset_indices'] if x not in missing_set]
        a=fabricated(row,old,.125);b=fabricated(row,missing,.25);out=an.merge_caches(row,plan,[a,b])
        loc={v:i for i,v in enumerate(plan['dataset_indices'])}
        self.assertEqual(out['prediction'][[loc[i] for i in old]].tobytes(),a['prediction'].tobytes())
        self.assertEqual(out['prediction'][[loc[i] for i in missing]].tobytes(),b['prediction'].tobytes())
        for parts in ([a,a,b],[a],[dict(a,target_times=a['target_times']+1),b]):
            with self.assertRaises(ValueError):an.merge_caches(row,plan,parts)
        with self.assertRaises(ValueError):an.merge_caches(S['rows']['C-H3-UV'],plan,[a,b])

    def test_04_FP32_metrics_inverse_tail(self):
        row=S['rows']['P-NP-UV'];a=fabricated(row,range(5))
        a['prediction'][:]=np.float32(.10000003);a['truth'][:]=np.float32(.09999998)
        a['physical_prediction'],a['physical_truth']=an.physical(row,a['prediction'],a['truth'],a['node_indices'])
        an.validate_cache(row,a,range(5));m=rp.cache_metrics(a,2)
        err=(a['prediction']-a['truth']).astype(np.float64)
        self.assertEqual(m['count_per_position'],5);self.assertEqual(m['overall_mse'],float(np.square(err).mean()))
        self.assertEqual(sum(m['sse'])/(5*24),m['overall_mse']);self.assertEqual(sum(m['sae'])/(5*24),m['overall_mae'])
        wrong=dict(a,physical_prediction=a['physical_prediction']+1)
        with self.assertRaises(ValueError):an.validate_cache(row,wrong,range(5))

    def test_05_PatchTST_CPU_frozen_import_isolation(self):
        from models.ch3_adapter import target_prediction
        class Tiny(torch.nn.Module):
            def __init__(self):super().__init__();self.bn=torch.nn.BatchNorm1d(3);self.linear=torch.nn.Linear(3,3)
            def forward(self,x):
                COUNTS['synthetic_forward_calls']+=1;COUNTS['synthetic_presentations']+=len(x)
                return self.linear(self.bn(x.transpose(1,2)).transpose(1,2))[:,-24:,:]
        model=Tiny();COUNTS['synthetic_model_constructions']+=1
        identity=dict(synthetic=True);state=dict(identity=identity,epoch=2,model=copy.deepcopy(model.state_dict()))
        strict_weights(model,state,dict(identity=identity,best_epoch=2))
        x=torch.arange(5*96*3,dtype=torch.float32).reshape(5,96,3)/1000;y=torch.zeros(5,24,1)
        loader=DataLoader(TensorDataset(x,y),batch_size=2,generator=torch.Generator().manual_seed(2024))
        p=S['rows']['P-NP-PatchTST']['profile'];rng=torch.get_rng_state().clone();before=copy.deepcopy(model.state_dict())
        pred,truth,m,n=infer(model,lambda x,marks:target_prediction(model,x,p,marks)[0],loader,24,'cpu',lambda:None,lambda *a:None)
        self.assertEqual((n,len(pred)),(3,5));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
        bad=dict(state,epoch=3)
        with self.assertRaises(ValueError):strict_weights(model,bad,dict(identity=identity,best_epoch=2))
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
        code="import sys,types,torch;from m6_ch3_fullcoverage_v10.worker import batch_parts;sys.modules['utils']=types.ModuleType('utils');a=torch.zeros(1,2,3);b=torch.zeros(1,1,1);assert batch_parts((a,b))[0] is a;assert not torch.cuda.is_initialized();print('bound-project-helper survives author utils switch')"
        r=subprocess.run([sys.executable,'-B','-c',code],cwd=W,text=True,capture_output=True)
        ct.write_new(E/'import-isolation.json',dict(command=code,exit=r.returncode,stdout=r.stdout,stderr=r.stderr));self.assertEqual(r.returncode,0,r.stderr)
        self.assertFalse(torch.cuda.is_initialized())

    def test_06_claim_before_read_and_strict_capability(self):
        j=C['jobs'][0];row=S['rows'][j['cell']];cap=ct.capability(C,j,row)
        with patch('utils.ch3_data.load',side_effect=AssertionError('raw load forbidden in fixture')) as loader:
            for altered in (dict(cap,scope='old'),dict(cap,evaluation_key='changed')):
                with self.assertRaises(PermissionError):wk.datasets(C,j,row,altered,E/'never-access.jsonl')
            loader.assert_not_called()
        with self.assertRaises(PermissionError):ct.capability(C,dict(j,subset='other'),row)
        with self.assertRaises(PermissionError):ct.capability(C,C['jobs'][-1],row)
        folder=E/'claim-fixture';folder.mkdir();runtime=folder/'runtime.json';ct.write_new(runtime,dict(synthetic=True))
        class Claimed(Exception):pass
        def observe(*args):
            claims=list((folder/'claims').glob('*.json'));self.assertEqual(len(claims),1);self.assertEqual(ct.read(claims[0])['capability'],cap);raise Claimed()
        with patch.object(qu,'context',return_value=(C,folder)),patch.object(wk,'run',side_effect=observe):
            with self.assertRaises(Claimed):qu.worker('synthetic-config',runtime,j['id'])
        self.assertFalse((folder/'receipts').exists())
        with self.assertRaises(FileExistsError):ct.write_new(next((folder/'claims').glob('*.json')),dict())

    def test_07_full54_offline_merge_explicit_plots(self):
        folder=E/'offline-fixture';folder.mkdir();cache=folder/'synthetic-caches';cache.mkdir();s=copy.deepcopy(S);inputs={}
        for cell,row in s['rows'].items():
            urban=cell.startswith('C-');h=row['profile']['H'];parts=[]
            if urban:
                plan=ct.load_ref(s['plans'][f'H{h}-all275']);ids=plan['dataset_indices']
                if cell in s['inherited']:
                    missing=ct.load_ref(s['plans']['H6-remaining265'])['dataset_indices'];missing_set=set(missing);groups=[([i for i in ids if i not in missing_set],True),(missing,False)]
                else:groups=[(ids,False)]
            else:
                row['test_arithmetic']['full_test_samples']=5;groups=[(list(range(5)),cell in s['inherited'])]
            for z,(ids,inherited) in enumerate(groups):
                a=fabricated(row,ids);p=cache/(cell+f'-{z}.npz');np.savez_compressed(p,**a)
                result=cache/(cell+f'-{z}.json');ct.write_new(result,dict(metrics=rp.cache_metrics(a,row['profile']['training']['eval_batch'])))
                parts.append(dict(inherited=inherited,prediction_ref=ref(p),result_ref=ref(result),source=row,synthetic=True))
            inputs[cell]=parts
        sp=folder/'sources.json';ct.write_new(sp,s);c=dict(C,sources_ref=ref(sp))
        with patch.object(rp,'resolve_inputs',return_value=inputs):out=rp.completed(c,folder)
        self.assertEqual((out['logical_cells'],out['region_metric_cells'],out['regions_selected']),(54,6600,[]))
        policy=ct.read(folder/'data/display-policy.json');self.assertFalse(policy['automatic_ranking'])
        self.assertEqual(len(ct.read(folder/'data/source-ledger.json')),54)
        with self.assertRaises(ValueError):rp.region_plot(folder/'data',folder/'no-selection',H=3,node_ids=['105'],models=None)
        with self.assertRaises(ValueError):rp.epf_plot(folder/'data',folder/'no-models',market='NP',models=[])
        node=ct.load_ref(s['plans']['H3-all275'])['nodes'][0]
        rp.region_plot(folder/'data',folder/'explicit-region',H=3,node_ids=[node],models=['UV','ModernTCN','TimeXer'])
        rp.epf_plot(folder/'data',folder/'explicit-epf',market='NP',models=list(an.MODELS))
        self.assertEqual(len(list((folder/'explicit-region').glob('*.svg'))),1)
        self.assertEqual(len(list((folder/'explicit-epf').glob('*.svg'))),2)
        with self.assertRaises(FileExistsError):rp.completed(c,folder)

    def test_08_owned_STOP_first_failure_serial_control(self):
        root=E/'control-fixture';root.mkdir();cp=root/'config.json';ap=root/'AUTH.fixture.json'
        c=dict(C,result_root=str(root),attempt='synthetic-owned',global_lock=str(root/'lock'));ct.write_new(cp,c);ct.write_new(ap,dict(commit='synthetic'))
        child=Mock(pid=987654,wait=Mock(return_value=1),poll=Mock(return_value=1))
        with patch.object(qu,'preflight',return_value=c),patch.object(qu,'proc_start',return_value=123),patch.object(qu,'gpu_startup',return_value={'synthetic':True}),patch('m6_ch3_supplement_v10.gpu_identity.check_gpu_identity'),patch.object(qu.subprocess,'Popen',return_value=child) as spawn,patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
            with self.assertRaisesRegex(RuntimeError,'later waves stopped'):qu.supervise(cp,ap)
            self.assertEqual(spawn.call_count,1)
        folder=root/'synthetic-owned';self.assertTrue((folder/'failure.json').exists());self.assertFalse((folder/'complete.json').exists())
        with patch.object(qu,'proc_start',return_value=124):
            with self.assertRaises(PermissionError):qu.owned_stop(folder)
        with patch.object(qu,'proc_start',return_value=123):qu.owned_stop(folder)
        self.assertTrue((folder/'STOP').exists())
        with patch.object(qu,'proc_start',return_value=None):self.assertTrue(qu.status(folder)['failure'])
        r=subprocess.run(['bash','-n',str(W/'scripts/ch3/start_m6_ch3_fullcoverage_v10.sh')],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':
    E.mkdir(exist_ok=False,parents=True)
    try:unittest.main()
    finally:ct.write_new(E/'consumption.json',dict(COUNTS,cuda_initialized=torch.cuda.is_initialized()))
