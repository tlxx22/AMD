"""Narrow CPU controls/synthetic caches. Never load real weights/raw test or GPU."""
import ast
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from m6_ch3_scope_amendment_v10 import contract as k,queue,worker
from m6_ch3_scope_amendment_v10.analysis import sensitivity_means,case_rows

W=k.ROOT;R=W.parent/'amd-execution-evidence/m6'
E=R/'M6-ModernTCN-UV-v10-urbanev4h-np-baseline-extension-preparation-r1'
CONFIG=W/'configs/m6_ch3_scope_amendment_v10.json'


def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')
    return dict(path=str(p),sha256=k.sha(p))


class Amendment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=k.read(CONFIG);cls.s=k.load_ref(cls.c['sources_ref']);cls.root=Path(tempfile.mkdtemp(prefix='synthetic-',dir=E/'tests'));torch.set_num_threads(1)

    def test_01_scope_profile_and_arithmetic(self):
        c=k.validate(self.c);self.assertEqual(len(c['jobs']),41);self.assertEqual(len(c['waves']),29)
        self.assertEqual([len(x) for x in c['waves']],[1]*8+[4]*4+[1]*17)
        for cell,steps in zip(k.CELLS,k.STEPS):
            j=next(x for x in c['jobs'] if x.get('prototype')==cell);t,p,data=k.derived_task(c,j)
            self.assertEqual(t['arithmetic']['train_batches'],steps);self.assertEqual(p['training']['scheduler']['steps_per_epoch'],steps)
            self.assertIn(cell.split('-H')[0],t['analysis_variant_id']);self.assertEqual(t['h'],p['H'])
        for field,value in [('waves',c['waves'][1:]),('q_max',1),('attempt','other'),('budget',dict(c['budget'],max_updates=1804641))]:
            bad=copy.deepcopy(c);bad[field]=value
            with self.assertRaises((ValueError,PermissionError)):k.validate(bad)
        s=copy.deepcopy(self.s);s['prototypes']['NP-H24']['task']['dataset']='PJM';bad=dict(c,sources_ref=dump(self.root/'wrong-NP.json',s))
        with self.assertRaises(ValueError):k.validate(bad)

    def test_02_four_H_equal_means_and_missing_rejected(self):
        rows=[]
        for d,hs in [('UrbanEV',(3,6,9,12)),('NP',(24,))]:
            for h in hs:
                for G,Rv in ((3,4),(5,4),(7,4),(5,2),(5,6)):
                    rows.append(dict(dataset=d,H=h,G=G,R=Rv,fold=6 if d=='UrbanEV' else 1,P=1 if d=='UrbanEV' else 24,seed=2024,validation_mse=float(h),validation_mae=float(h)/2,test_mse=float(h)+1,test_mae=float(h)/3,elements=h*100))
        out=sensitivity_means(rows);self.assertEqual(len(out),10);self.assertEqual(out[0]['validation_mse'],7.5)
        for bad in (rows[:-1],rows+[rows[0]],rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError):sensitivity_means(bad)
        bad=copy.deepcopy(rows);bad[-1]['dataset']='PJM'
        with self.assertRaises(ValueError):sensitivity_means(bad)

    def test_03_train_boundary_and_original_fit_dispatch(self):
        j=next(j for j in self.c['jobs'] if j['phase']=='T');task,p,meta=k.derived_task(self.c,j)
        from m6_ch3_supplement_v10 import sensitivity
        original_fit=sensitivity.fit
        fake=SimpleNamespace(cuda=lambda:fake,analysis_identity=dict(G=j['G'],R=j['R']),spec=SimpleNamespace(value={'CPU_control':True}))
        with patch('utils.ch3_data.load',return_value=({'train':[], 'validation':[], 'test':[]},meta)),patch('m6_ch3_supplement_v10.model.build') as build:
            with self.assertRaises(PermissionError):worker.train(self.c,j,self.root,lambda:None)
            build.assert_not_called()
        with patch('utils.ch3_data.load',return_value=({'train':[], 'validation':[]},meta)) as load,patch('m6_ch3_supplement_v10.model.build',return_value=fake) as build,patch.object(sensitivity,'fit',return_value={'CPU_control':True}) as fit:
            worker.train(self.c,j,self.root,lambda:None)
            self.assertNotIn('test_capability',load.call_args.kwargs);self.assertEqual(build.call_args.args,(p,j['G'],j['R']))
            identity=fit.call_args.args[4];self.assertEqual(identity['analysis_variant_id'],j['analysis_variant_id']);self.assertEqual(identity['profile']['H'],3)
        self.assertIs(sensitivity.fit,original_fit)
        with self.assertRaises(PermissionError):worker.train(self.c,self.c['jobs'][0],self.root,lambda:None)

    def test_04_auth_test_boundaries(self):
        with self.assertRaises(PermissionError):k.permission(CONFIG,dict(schema='m6-ch3-test-reporting-launch-v1',review_approved=True,commit='old'))
        with self.assertRaises(PermissionError):k.permission(CONFIG,k.read(E/'non-execution-preparation/launch-binding.template.json'))
        j=self.c['jobs'][0];row=self.s['rows'][j['source_key']]
        with patch('utils.ch3_data.load') as load:
            with self.assertRaises(PermissionError):worker.test_data(self.c,j,row,{},self.root/'never.jsonl')
            load.assert_not_called()
        train=next(j for j in self.c['jobs'] if j['phase']=='T')
        with self.assertRaises(PermissionError):k.test_key(self.c,train,row)

    def test_05_sealed_producer_and_wrong_GH_rejected(self):
        folder=self.root/'sealed-CPU-only';folder.mkdir();dump(folder/'runtime.json',dict(commit='synthetic'))
        j=next(j for j in self.c['jobs'] if j['phase']=='S');tj=next(t for t in self.c['jobs'] if t['id']==j['producer_job']);out=folder/tj['id'];out.mkdir()
        t,p,data=k.derived_task(self.c,j);identity=dict(scope=k.SCOPE,job=tj,analysis_variant_id=j['analysis_variant_id'],analysis=dict(G=j['G'],R=j['R']),profile=p,data=data,source_binding=self.c['source_binding'])
        dump(out/'identity.json',identity);(out/'best.pt').write_bytes(b'SYNTHETIC CONTROL BYTES, NOT A MODEL CHECKPOINT')
        digest=k.sha(out/'best.pt');dump(out/'best.pt.sha256',dict(sha256=digest))
        result=dict(job=tj,epochs=1,updates=t['arithmetic']['train_batches'],test_calls=0,status='validation_only_completed',routes=dict(train=t['arithmetic']['train_batches'],validation=t['arithmetic']['eval_batches']['validation']),best_epoch=1,validation=dict(mse=.1,mae=.2))
        dump(out/'result.json',result);(out/'history.jsonl').write_text(json.dumps(dict(epoch=1,validation=result['validation']))+'\n')
        dump(out/'seal.json',dict(identity=identity,files={'best.pt':digest}))
        def receipt():dump(folder/'receipts'/(tj['id']+'.json'),dict(job=tj,result=result,runtime_sha256=k.sha(folder/'runtime.json'),files={str(p.relative_to(folder)):k.sha(p) for p in out.iterdir()}))
        receipt();row=worker.resolve(self.c,j,folder);self.assertEqual(row['best_epoch'],1);self.assertEqual(row['weights_ref']['sha256'],digest)
        bad=dict(j,producer_job=next(x['id'] for x in self.c['jobs'] if x['phase']=='T' and x['H']==9))
        with self.assertRaises(ValueError):worker.resolve(self.c,bad,folder)
        identity['analysis']['G']=7;dump(out/'identity.json',identity);dump(out/'seal.json',dict(identity=identity,files={'best.pt':digest}));receipt()
        with self.assertRaises(ValueError):worker.resolve(self.c,j,folder)

    def test_06_synthetic_strict_state_only(self):
        from m6_ch3_test_reporting_v10.evaluate import strict_weights
        m=torch.nn.Linear(2,1);identity={'scope':'synthetic','H':9};state=dict(identity=identity,epoch=2,model=copy.deepcopy(m.state_dict()))
        p=self.root/'synthetic-state.pt';torch.save(state,p);loaded=torch.load(p,map_location='cpu');row=dict(identity=identity,best_epoch=2)
        strict_weights(m,loaded,row);self.assertFalse(m.training);self.assertTrue(all(not p.requires_grad for p in m.parameters()))
        before=copy.deepcopy(m.state_dict());bad=copy.deepcopy(loaded);bad['identity']['H']=6
        with self.assertRaises(ValueError):strict_weights(m,bad,row)
        for key in before:self.assertTrue(torch.equal(before[key],m.state_dict()[key]))
        self.assertFalse(torch.cuda.is_initialized())

    def test_07_actual_marks_helper_import_and_tail_cpu_children(self):
        code=r'''
import sys,json,torch,numpy as np
from pathlib import Path
from m6_ch3_scope_amendment_v10 import worker
from models.ch3_adapter import target_prediction
torch.set_num_threads(1)
c=json.loads(Path('configs/m6_ch3_scope_amendment_v10.json').read_text());s=json.loads(Path(c['sources_ref']['path']).read_text());name=sys.argv[1];row=s['rows']['P-NP-'+name];p=row['profile']
class Tiny(torch.nn.Module):
 def __init__(self):super().__init__();self.bn=torch.nn.BatchNorm1d(3);self.weight=torch.nn.Parameter(torch.tensor(.5))
 def forward(self,x,*marks):
  y=self.bn(x.transpose(1,2)).transpose(1,2)[:,-24:,:]*self.weight
  return (y[:,:,:1],None) if name=='AMD' else y[:,:,-1:]
m=Tiny().eval()
for q in m.parameters():q.requires_grad_(False)
x=torch.arange(5*96*3,dtype=torch.float32).reshape(5,96,3)/1000;y=torch.zeros(5,24,1)
mark=torch.zeros(5,96,p['time_mark']['K']) if p.get('time_mark') else None
data=torch.utils.data.TensorDataset(*(x,y,mark) if mark is not None else (x,y))
loader=torch.utils.data.DataLoader(data,batch_size=2,generator=torch.Generator().manual_seed(2024))
import utils
protocol=json.loads(Path(row['protocol_ref']['path']).read_text())
if name=='TimeXer':utils.__path__=[str(Path(protocol['sources'][name]['root'])/'utils')]
else:assert name=='AMD' and name not in protocol['sources']  # AMD uses the native local path.
pred,truth,metrics,calls=worker.infer(m,lambda x,marks:target_prediction(m,x,p,marks)[0],loader,24,'cpu',lambda:None,lambda *a:None)
assert calls==3 and pred.shape==(5,24,1) and not torch.cuda.is_initialized()
assert abs(metrics['overall_mse']-np.square(pred.astype(float)-truth).mean())<1e-12
print(json.dumps(dict(model=name,synthetic_models=1,forwards=3,presentations=5,CUDA_initialized=False,real_weights=0,raw_test=0)))
'''
        for m in ('AMD','TimeXer'):
            r=subprocess.run([sys.executable,'-B','-c',code,m],cwd=W,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),text=True,capture_output=True,timeout=60)
            n=len(list((E/'tests').glob('import-'+m+'-*.json')))+1;dump(E/'tests'/f'import-{m}-{n}.json',dict(exit=r.returncode,stdout=r.stdout,stderr=r.stderr,code=code))
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)

    def test_08_clock_scaler_position_metadata(self):
        from m6_ch3_test_reporting_v10.analysis import case_plan,sample_keys,physical,PositionErrors
        pre=k.load_ref(self.s['old_preselection_ref']);plan=k.load_ref(self.s['case_plan_ref'])
        for m in ('AMD','TimeXer'):self.assertEqual(case_plan(self.s['rows']['C-UrbanEV-'+m],pre),plan)
        for cell,x in self.s['prototypes'].items():
            row=x['source'];ids=[0,row['test_arithmetic']['full_test_samples']-1];keys=sample_keys(row,ids)
            self.assertTrue(np.all(keys['target_record_indices']>=row['test_arithmetic']['endpoints'][1]))
            a=np.zeros((2,row['profile']['pred_len'],1));pp,tt=physical(row,a,a,keys['node_indices']);self.assertTrue(np.array_equal(pp,tt))
        a=np.arange(5*24,dtype=np.float32).reshape(5,24,1)/100;meter=PositionErrors(24)
        meter.add(a[:4],np.zeros_like(a[:4]));meter.add(a[4:],np.zeros_like(a[4:]));r=meter.result()
        self.assertAlmostEqual(r['overall_mse'],float(np.square(a.astype(float)).mean()))

    def test_09_actual_cache_only_O(self):
        from tests.m6_ch3_amendment_plot_fixture import create
        from m6_ch3_scope_amendment_v10.plots import completed
        c,folder=create(self.c,self.s,self.root/'SYNTHETIC-PLOTS-ONLY')
        r=completed(c,folder);self.assertEqual((r['sensitivity_cells'],r['position_cells'],r['case_models'],r['case_plots']),(25,15,5,4))
        self.assertEqual(len(k.read(folder/'plots/sensitivity-25-cells.json')),25)
        self.assertEqual(k.read(folder/'plots/display-selection.json')['selected_nodes'],['1162','1083','1120','969'])
        for node in ('1162','1083','1120','969'):
            text=(folder/f'plots/case-{node}-five-model-test.svg').read_text()
            for name in ('truth',*k.MODELS):self.assertIn('>'+name+'</text>',text)
        self.assertFalse(k.read(folder/'plots/display-selection.json')['reranked'])
        with self.assertRaises(FileExistsError):completed(c,folder)

    def test_12_status_train_and_eval_separate(self):
        folder=self.root/'status';train=folder/'TR-synthetic';train.mkdir(parents=True);ev=folder/'P-synthetic';ev.mkdir()
        (train/'updates.jsonl').write_text(json.dumps(dict(epoch=1,step=3))+'\n')
        (ev/'access.jsonl').write_text(json.dumps(dict(event='after_forward',batch=1,samples=2))+'\n'+json.dumps(dict(event='after_forward',batch=2,samples=1))+'\n')
        state=queue.status(folder)
        self.assertEqual(state['training_progress']['TR-synthetic'],dict(current_epoch=1,current_step=3))
        self.assertEqual(state['evaluation_progress']['P-synthetic']['presentations'],3)
        self.assertEqual(state['evaluation_progress']['P-synthetic']['completed_batches'],2)
        self.assertFalse(state['complete'])

    def test_11_typed_source_identity_and_historical_metadata(self):
        k.verify_source_refs(self.s)
        bad=copy.deepcopy(self.s);bad['rows']['P-NP-AMD']['metrics_ref']['sha256']='0'*64
        with self.assertRaises(ValueError):k.verify_source_refs(bad)
        bad=copy.deepcopy(self.s);bad['rows']['P-NP-TimeXer']['data_protocol']['native_time_mark']['data_binding_artifact']['sha256']='0'*64
        with self.assertRaises(ValueError):k.verify_source_refs(bad)
        self.assertFalse(torch.cuda.is_initialized())

    def test_10_owned_stop_failure_and_no_later_dispatch(self):
        c=copy.deepcopy(self.c);c.update(result_root=str(self.root/'dispatch'),global_lock=str(self.root/'lock'))
        cp=self.root/'dispatch-config.json';dump(cp,c);ap=self.root/'dispatch-auth.json';dump(ap,{'commit':'synthetic'})
        class Child:
            pid=os.getpid();returncode=1
            def poll(self):return 1
        with patch.object(queue,'preflight',return_value=c),patch.object(queue,'gpu_startup',return_value={}),patch('m6_ch3_supplement_v10.gpu_identity.check_gpu_identity',return_value=None),patch.object(queue.subprocess,'Popen',return_value=Child()) as spawn,patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0'):
            with self.assertRaises(RuntimeError):queue.supervise(cp,ap)
            self.assertEqual(spawn.call_count,1);self.assertEqual(spawn.call_args.args[0][-1],'P-NP-AMD')
        folder=Path(c['result_root'])/c['attempt'];self.assertTrue((folder/'failure.json').exists());self.assertFalse((folder/'complete.json').exists())
        queue.owned_stop(folder);self.assertTrue((folder/'STOP').exists())
        old=k.read(folder/'owner.json');dump(folder/'owner.json',dict(old,start='wrong'))
        with self.assertRaises(PermissionError):queue.owned_stop(folder)


if __name__=='__main__':unittest.main(verbosity=2)
