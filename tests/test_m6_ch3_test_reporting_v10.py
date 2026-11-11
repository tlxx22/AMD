"""CPU fixtures only; no real checkpoint, real test observations, GPU or optimizer."""
import ast
import copy
import json
import tempfile
import unittest
import os
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from m6_ch3_test_reporting_v10 import contract as c
from m6_ch3_test_reporting_v10 import analysis as a
from m6_ch3_test_reporting_v10.evaluate import strict_weights, infer, test_data
from m6_ch3_test_reporting_v10.queue import claim, checked_receipt, owned_stop, proc_start

ROOT=c.ROOT
CONFIG=ROOT/'configs/m6_ch3_test_reporting_v10.json'
E=ROOT.parent/'amd-execution-evidence/m6/M6-ModernTCN-UV-v10-test-reporting-preparation-r1'
COST=dict(fixture_models=0,fixture_forward_calls=0,fixture_presentations=0,synthetic_checkpoint_writes=0,synthetic_checkpoint_reads=0,synthetic_dataset_loads=0,real_checkpoint=0,real_test=0,GPU=0,backward=0,Adam=0)


def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')
    return dict(path=str(p),sha256=c.sha(p))


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.bn=torch.nn.BatchNorm1d(3);self.head=torch.nn.Linear(3,24);COST['fixture_models']+=1
    def forward(self,x):return self.head(self.bn(x.mean(1))).unsqueeze(-1)


class Reporting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config=c.read(CONFIG)
        cls.tmp=tempfile.TemporaryDirectory(prefix='test-reporting-fixture-',dir=E/'tests')
        cls.folder=Path(cls.tmp.name)
    @classmethod
    def tearDownClass(cls):
        # Keep fixtures in the evidence directory (do not erase failure output).
        cls.tmp._finalizer.detach()
        dump(E/'tests/consumption.json',COST)
        assert not torch.cuda.is_initialized()

    def test_01_scope_and_old_authorization(self):
        self.assertEqual(len(c.validate(self.config)['jobs']),20)
        template=c.read(E/'non-execution-preparation/launch-binding.template.json')
        with self.assertRaises(PermissionError):c.permission(CONFIG,template)
        old=c.read(ROOT.parent/'amd-execution-evidence/m6/M6-ModernTCN-UV-v10-ch3-supplement-retry2-launch-authorization-r2/AUTH.json')
        with self.assertRaises(PermissionError):c.permission(CONFIG,old)
        from m6_ch3_supplement_v10.contract import validate
        with self.assertRaises(ValueError):validate(self.config)
        for key,val in [('split','validation'),('purpose','ch3_formal_test'),('model','AMD')]:
            bad=copy.deepcopy(self.config);bad['jobs'][0][key]=val
            with self.assertRaises(ValueError):c.validate(bad)
        bad=copy.deepcopy(self.config);bad['jobs'][1]=bad['jobs'][0]
        with self.assertRaises(ValueError):c.validate(bad)
        bad=copy.deepcopy(self.config);bad['jobs'][0]['G']=5
        with self.assertRaises(ValueError):c.validate(bad)
        for key,value in [('features',['wrong']*11),('input_variant','F1'),('target_idx',2)]:
            ledger=copy.deepcopy(c.load_ref(self.config['sources_ref']));ledger['rows']['ST-UrbanEV-G3-R4']['profile'][key]=value
            bad=copy.deepcopy(self.config);bad['sources_ref']=dump(self.folder/('wrong-source-'+key+'.json'),ledger)
            with self.assertRaises(ValueError):c.validate(bad)

    def test_02_frozen_synthetic_best_inference(self):
        torch.manual_seed(2024);m=Tiny();identity=dict(analysis=dict(G=3,R=4),purpose='synthetic fixture only')
        state=dict(identity=identity,epoch=2,model=copy.deepcopy(m.state_dict()))
        path=self.folder/'synthetic-best.pt';torch.save(state,path);COST['synthetic_checkpoint_writes']+=1
        loaded=torch.load(path,map_location='cpu');COST['synthetic_checkpoint_reads']+=1
        row=dict(identity=identity,best_epoch=2)
        bad=copy.deepcopy(loaded);bad['identity']['analysis']['R']=6
        before=copy.deepcopy(m.state_dict())
        with self.assertRaises(ValueError):strict_weights(m,bad,row)
        for k in before:self.assertTrue(torch.equal(before[k],m.state_dict()[k]))
        strict_weights(m,loaded,row)
        x=torch.arange(60,dtype=torch.float32).reshape(5,4,3)/100;y=torch.zeros(5,24,1)
        loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y),batch_size=2,shuffle=False,generator=torch.Generator().manual_seed(2024))
        events=[];p,t,r,calls=infer(m,lambda x,marks:m(x),loader,24,'cpu',lambda:None,lambda *v:events.append(v))
        COST['fixture_forward_calls']+=calls;COST['fixture_presentations']+=len(p)
        self.assertEqual(calls,3);self.assertEqual(len(p),5);self.assertEqual(len(events),6)
        self.assertAlmostEqual(sum(r['sse'])/(5*24),r['overall_mse'],places=15)
        self.assertFalse(m.training);self.assertTrue(all(not q.requires_grad for q in m.parameters()))

    def test_03_prepare_denies_test_and_synthetic_positive_loader(self):
        job=self.config['jobs'][0]
        with self.assertRaises(PermissionError):test_data(self.config,job,None,self.folder/'denied.jsonl')
        self.assertFalse((self.folder/'denied.jsonl').exists())
        # Exercise the actual original capability loader on generated CSV only.
        source=copy.deepcopy(c.load_ref(self.config['sources_ref'])['rows']['ST-PJM-G3-R4'])
        protocol=source['data_protocol'];d=protocol['datasets']['PJM'];path=self.folder/'synthetic-PJM.csv'
        n=260;vals=np.arange(n*3,dtype=float).reshape(n,3)/100
        frame=pd.DataFrame(vals,columns=d['features']);frame.insert(0,'date',pd.date_range('2020-01-01',periods=n,freq='h'));frame.to_csv(path,index=False)
        d['path']=str(path);d['endpoints']=[160,200,260];d.pop('source_admission',None)
        from utils.ch3_data import load
        sets,meta=load(protocol,source['task'],resolved_profile=source['profile']);COST['synthetic_dataset_loads']+=1
        source['data']=meta;source['test_arithmetic'].update(full_test_samples=37,endpoints=[160,200,260],records_read=260)
        j=copy.deepcopy(self.config['jobs'][4]);j['expected']=dict(samples=37,elements=888,batches=1,eval_batch=128)
        cfg=copy.deepcopy(self.config);cfg['jobs']=[j];cfg['sources_ref']=dump(self.folder/'synthetic-sources.json',{'rows':{j['id']:source}})
        cap=dict(scope=c.SCOPE,purpose=c.PURPOSE,job=j['id'],evaluation_key=c.evaluation_key(cfg,j),mode='authorized_frozen_test')
        data,meta=test_data(cfg,j,cap,self.folder/'synthetic-access.jsonl');COST['synthetic_dataset_loads']+=1
        self.assertEqual(len(data),37);self.assertTrue(meta['test_observations_accessed'])
        self.assertEqual(meta['endpoints_read'],[160,260])
        with self.assertRaises(PermissionError):load(protocol,source['task'],test_capability={'purpose':'wrong'},resolved_profile=source['profile'])

    def test_04_preselection_and_case_clock(self):
        pre=c.load_ref(self.config['preselection_ref']);old=c.load_ref(pre['source_ref'])
        self.assertEqual(len(old['eligible']),137)
        self.assertEqual(a.preselect(old,pre['node_order']),pre['draw_indices'])
        self.assertEqual(len(set(pre['draw_indices'])),10)
        row=c.load_ref(self.config['sources_ref'])['rows']['CT-UrbanEV-UV'];plan=a.case_plan(row,pre)
        self.assertEqual(len(plan['dataset_indices']),1680)
        self.assertEqual(pd.Timestamp(plan['target_times'][0]).dayofweek,0)
        keys=a.sample_keys(row,plan['dataset_indices']);self.assertTrue(np.all(keys['target_record_indices'][:,0]-keys['origin_record_indices']==6))
        self.assertTrue(np.all(keys['target_record_indices']>=3909));self.assertTrue(np.all(keys['target_record_indices']<4344))
        short=pd.date_range('2020-01-01',periods=167,freq='h')
        with self.assertRaises(ValueError):a.case_times(short)
        fallback=pd.date_range('2020-01-01',periods=168,freq='h');self.assertIn('earliest',a.case_times(fallback)['reason'])
        bad=fallback.delete(3)
        with self.assertRaises(ValueError):a.case_times(bad)

    def test_05_rank_ties_zero_and_four_even_without_wins(self):
        rows=[dict(node_index=i,node=str(i),ModernTCN_mse=1.,ModernTCN_mae=1.,TimeMixer_mse=1.,TimeMixer_mae=1.,UV_mse=2.,UV_mae=2.) for i in range(10)]
        ranked=a.rank_cases(rows);self.assertEqual([r['node_index'] for r in ranked[:4]],[0,1,2,3]);self.assertEqual(sum(r['selected_for_plot'] for r in ranked),4)
        rows[9].update(UV_mse=.5,UV_mae=.5);self.assertEqual(a.rank_cases(rows)[0]['node_index'],9)
        rows[0].update(ModernTCN_mse=0.,ModernTCN_mae=0.,TimeMixer_mse=0.,TimeMixer_mae=0.,UV_mse=0.,UV_mae=0.)
        z=next(r for r in a.rank_cases(rows) if r['node_index']==0);self.assertEqual(z['NA_count'],4);self.assertIsNone(z['relative_mean']);self.assertEqual(z['strict_wins'],0)
        with self.assertRaises(ValueError):a.rank_cases(rows[:9])

    def test_06_positions_scaler_and_tail(self):
        rng=np.random.default_rng(2024);p=rng.normal(size=(5,24,1)).astype('float32');t=np.zeros_like(p)
        m=a.PositionErrors(24);m.add(p[:2],t[:2]);m.add(p[2:],t[2:]);r=m.result()
        self.assertEqual(r['count_per_position'],5);self.assertAlmostEqual(r['overall_mse'],float(np.square(p.astype('float64')).mean()),places=15)
        row={'profile':{'dataset':'PJM','target_idx':2},'data':{'scaler':{'mean':[0,0,10],'scale':[1,1,2]}}}
        pp,tt=a.physical(row,p,t,np.zeros(5,dtype=int));np.testing.assert_array_equal(pp,p*2+10);np.testing.assert_array_equal(tt,t*2+10)
        with self.assertRaises(ValueError):m.add(p,t[:,0])

    def test_07_claim_receipt_stop_boundaries(self):
        folder=self.folder/'control';folder.mkdir();runtime=folder/'runtime.json';dump(runtime,{'synthetic':True})
        j=self.config['jobs'][0];cap=claim(folder,self.config,j,runtime)
        with self.assertRaises(FileExistsError):claim(folder,self.config,j,runtime)
        out=folder/j['id'];out.mkdir();result=dict(job=j,evaluation_key=cap['evaluation_key'],**dict(presentations=j['expected']['samples'],calls=j['expected']['batches'],elements=j['expected']['elements'],new_test_access=True,split='test',train=0,backward=0,optimizer=0))
        dump(out/'result.json',result);dump(folder/'receipts'/(j['id']+'.json'),dict(job=j,result=result,files={j['id']+'/result.json':c.sha(out/'result.json')},runtime_sha256=c.sha(runtime)))
        checked_receipt(folder,j,runtime)
        claim_path=folder/'claims'/(cap['evaluation_key']+'.json');original=c.read(claim_path);wrong=copy.deepcopy(original);wrong['runtime_sha256']='0'*64;dump(claim_path,wrong)
        with self.assertRaises(ValueError):checked_receipt(folder,j,runtime)
        dump(claim_path,original)
        (out/'result.json').write_text('{}')
        with self.assertRaises(ValueError):checked_receipt(folder,j,runtime)
        # Assert production entry has no training action and queue stops on failure.
        text=(ROOT/'m6_ch3_test_reporting_v10/queue.py').read_text();self.assertIn('technical failure; no later dispatch',text);self.assertIn("proc_start(owner['pid'])!=owner['start']",text)
        tree=ast.parse((ROOT/'m6_ch3_test_reporting_v10/evaluate.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertFalse(any(n.func.attr in ('backward','step','train') for n in calls))

    def test_09_owned_stop_and_dispatch_failure(self):
        # Actual STOP writes on CPU-only synthetic runtime; no signal is sent.
        folder=self.folder/'owned';folder.mkdir()
        cfg_ref=dump(self.folder/'owned-config.json',dict(result_root=str(self.folder),attempt='owned'))
        runtime=folder/'runtime.json';dump(runtime,dict(scope=c.SCOPE,config_ref=cfg_ref))
        owner=dict(pid=os.getpid(),start=proc_start(os.getpid()),runtime_sha256=c.sha(runtime));dump(folder/'owner.json',owner)
        wrong=dict(owner,start=str(int(owner['start'])+1));dump(folder/'owner.json',wrong)
        with self.assertRaises(PermissionError):owned_stop(folder)
        self.assertFalse((folder/'STOP').exists());dump(folder/'owner.json',owner);owned_stop(folder)
        self.assertTrue((folder/'STOP').exists());owned_stop(folder)
        # Run the real dispatch loop with control-only substitutes. First child
        # fails; no model/GPU/preflight executes and the second job is not sent.
        from m6_ch3_test_reporting_v10 import queue
        cfg=copy.deepcopy(self.config);cfg.update(result_root=str(self.folder),attempt='dispatch-failure',global_lock=str(self.folder/'lock'))
        auth_path=self.folder/'synthetic-auth.json';dump(auth_path,{'commit':'synthetic'})
        config_path=self.folder/'synthetic-dispatch-config.json';dump(config_path,cfg)
        class Child:
            pid=os.getpid();returncode=7
            def poll(self):return 7
        with patch.object(queue,'preflight',return_value=cfg),patch.object(queue,'gpu_startup',return_value={'synthetic':True}),patch('m6_ch3_supplement_v10.gpu_identity.check_gpu_identity',return_value=None),patch.object(queue.subprocess,'Popen',return_value=Child()) as spawn,patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0'):
            with self.assertRaisesRegex(RuntimeError,'technical failure'):queue.supervise(config_path,auth_path)
            self.assertEqual(spawn.call_count,1)
        self.assertTrue((self.folder/'dispatch-failure/failure.json').exists())
        self.assertFalse((self.folder/'dispatch-failure/complete.json').exists())

    def test_08_cache_only_complete_plot_path(self):
        from m6_ch3_test_reporting_v10.plots import completed
        f=self.folder/'SYNTHETIC_PLOTS_NOT_SCIENTIFIC';f.mkdir();runtime=f/'runtime.json';dump(runtime,{'synthetic':True})
        cfg=copy.deepcopy(self.config);ledger=copy.deepcopy(c.load_ref(cfg['sources_ref']));plan=a.case_plan(ledger['rows']['CT-UrbanEV-UV'],c.load_ref(cfg['preselection_ref']));dump(f/'case-plan.json',plan)
        for d in ('UrbanEV','PJM'):
            center=ledger['centers'][d];center['best_epoch']=1;center['metrics_ref']=dump(f/('center-'+d+'.json'),dict(mse=1.,mae=.5,sse=10.,sae=5.,elements=10));h=f/('center-'+d+'.jsonl');h.write_text(json.dumps({'epoch':1,'validation':{'mse':1.,'mae':.5}})+'\n');center['history_ref']=dict(path=str(h),sha256=c.sha(h))
        for job in cfg['jobs']:
            out=f/job['id'];out.mkdir();row=ledger['rows'][job['id']];P=row['profile']['pred_len'];n=1680 if job['phase']=='C' else 3
            p=np.full((n,P,1),.2,dtype='float32');t=np.zeros_like(p);meter=a.PositionErrors(P);meter.add(p,t)
            if job['phase']=='C':np.savez(out/'predictions.npz',prediction=p,truth=t,physical_prediction=p*2,physical_truth=t)
            result=dict(producer_commit='synthetic-only',weights_ref={'path':'synthetic-only','sha256':'0'*64},best_epoch=1,metrics=meter.result(),elements=n*P)
            dump(out/'result.json',result)
            if job['phase']=='S':row['metrics_ref']=dump(f/(job['id']+'-val.json'),{'validation':{'mse':.04,'mae':.2}})
            files={str(p.relative_to(f)):c.sha(p) for p in out.iterdir()};dump(f/'receipts'/(job['id']+'.json'),dict(job=job,result=result,files=files,runtime_sha256=c.sha(runtime)))
        cfg['sources_ref']=dump(f/'synthetic-source-ledger.json',ledger);before=COST['fixture_forward_calls'];r=completed(cfg,f)
        self.assertEqual(r['case_plots'],4);self.assertEqual(COST['fixture_forward_calls'],before)
        self.assertEqual(len(list((f/'plots').glob('case-*-test.svg'))),4)
        self.assertEqual(len(c.read(f/'plots/ten-region-comparison.json')),10)
        self.assertEqual(len(c.read(f/'plots/sensitivity-all-points.json')),10)
        self.assertTrue((f/'plots/source-ledger.json').exists())


if __name__=='__main__':unittest.main(verbosity=2)
