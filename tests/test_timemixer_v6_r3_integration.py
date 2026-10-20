"""Bounded r3 original-path, registered-interface and synthetic-worker checks."""
import copy,json,random,shutil,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np,torch
from tests import timemixer_v6_r3_fixtures as f
from timemixer_dev_v6_r3 import adapter,worker,launcher
from timemixer_dev_v6_r3.worker import run_worker
from timemixer_dev_v6_r3.training_state import read,restore,validate as validate_state
from timemixer_dev_v6_r3.artifacts import audit
from utils.ch3_contract import BestState
from utils.ch3_onecycle import Schedule,configuration


def common(model):return {k:v for k,v in model.state_dict().items() if not k.startswith(('u.','v.'))}
def batch(p,b=2):return torch.randn(b,p['T'],p['C']),torch.randn(b,p['T'],p['time_mark']['K'])
def tree(folder):return {str(p.relative_to(folder)):worker.file_sha(p) for p in Path(folder).rglob('*') if p.is_file()}


class Backbone(unittest.TestCase):
    def test_common_init_rng_module_substreams_initial_four_arm_outputs(self):
        p=f.profile();reference=adapter.build_original(p);rng=torch.get_rng_state().clone();weights=copy.deepcopy(reference.state_dict());models={}
        for name,u,v in [('B',False,False),('U',True,False),('V',False,True),('UV',True,True)]:
            m=adapter.build(p,use_u=u,use_v=v,fold=1);f.nested_equal(self,common(m),weights);self.assertTrue(torch.equal(rng,torch.get_rng_state()));models[name]=m
        f.nested_equal(self,models['U'].u.state_dict(),models['UV'].u.state_dict());f.nested_equal(self,models['V'].v.state_dict(),models['UV'].v.state_dict())
        x,marks=batch(p)
        for training in (False,True):
            reference.train(training);state=torch.get_rng_state();expected=reference(x,marks,None,None);end=torch.get_rng_state()
            for model in models.values():
                model.train(training);torch.set_rng_state(state);actual=model(x,marks,None,None);self.assertTrue(torch.equal(actual,expected));self.assertTrue(torch.equal(torch.get_rng_state(),end))
    def test_off_original_loss_grad_three_updates_and_rng_exact(self):
        p=f.profile();a=adapter.build_original(p);b=adapter.build(p,use_u=False,use_v=False,fold=1);x,marks=batch(p);target=torch.randn(2,1,1)
        opta=torch.optim.Adam(a.parameters(),lr=.01,weight_decay=1e-7);optb=torch.optim.Adam(b.parameters(),lr=.01,weight_decay=1e-7);sa=Schedule(opta,configuration(3,3));sb=Schedule(optb,configuration(3,3));a.train();b.train()
        for _ in range(3):
            state=torch.get_rng_state();opta.zero_grad();ya=a(x,marks,None,None)[:,:,:1];la=(ya-target).square().mean();la.backward();end=torch.get_rng_state();optb.zero_grad();torch.set_rng_state(state);yb=adapter.prediction(b,x,marks);lb=(yb-target).square().mean();lb.backward()
            self.assertTrue(torch.equal(ya,yb));self.assertTrue(torch.equal(la,lb));self.assertTrue(torch.equal(torch.get_rng_state(),end))
            for (ka,pa),(kb,pb) in zip(a.named_parameters(),b.named_parameters()):
                self.assertEqual(ka,kb)
                if pa.grad is None:self.assertIsNone(pb.grad)
                else:self.assertTrue(torch.equal(pa.grad,pb.grad),ka)
            opta.step();sa.after_successful_update();optb.step();sb.after_successful_update();f.nested_equal(self,a.state_dict(),b.state_dict());f.nested_equal(self,opta.state_dict(),optb.state_dict());f.nested_equal(self,sa.state_dict(),sb.state_dict())
    def test_actual_insertions_native_operators_cache_and_supervision(self):
        p=f.profile();m=adapter.build(p,use_u=True,use_v=True,fold=1).eval();x,marks=batch(p);original=x.clone();markcopy=marks.clone();before=copy.deepcopy(m.state_dict())
        with torch.no_grad():m.u.output.weight.fill_(.08);m.v.a.fill_(.4)
        state=copy.deepcopy(m.state_dict())
        def observed(use_u,use_v):
            m.use_u,m.use_v=use_u,use_v;data={};calls={'embedding':0,'projection':0,'norm':[]};handles=[]
            for i,n in enumerate(m.normalize_layers):
                def norm(mod,args,out,i=i):
                    calls['norm'].append((i,args[1]));data['norm'+str(i)+args[1]]=args[0].clone()
                    if args[1]=='norm':
                        data['norm'+str(i)+'cached_mean']=mod.mean.clone();data['norm'+str(i)+'cached_stdev']=mod.stdev.clone()
                handles.append(n.register_forward_hook(norm))
            def emb(mod,args,out):i=calls['embedding'];calls['embedding']+=1;data['embedding'+str(i)]=out.clone()
            def proj(mod,args,out):calls['projection']+=1
            handles.append(m.enc_embedding.register_forward_hook(emb));handles.append(m.projection_layer.register_forward_hook(proj))
            for i,mod in enumerate(m.pdm_blocks):
                def pdm(mod,args,out,i=i):data['pdm'+str(i)]=[v.clone() for v in out]
                handles.append(mod.register_forward_hook(pdm))
            for i,mod in enumerate(m.regression_layers):
                def trend(mod,args,out,i=i):data['trend'+str(i)]=out.clone()
                handles.append(mod.register_forward_hook(trend))
            try:
                with torch.no_grad():y,d=m.forecast(x,marks,None,None,return_diagnostics=True)
            finally:
                for h in handles:h.remove()
                m.use_u=m.use_v=True
            return y,d,data,calls
        y,d,a,c=observed(True,True);offu,du,b,cu=observed(False,True);offv,dv,v,cv=observed(True,False)
        for k in a:
            if k=='norm0denorm':
                self.assertFalse(torch.equal(a[k],b[k]));self.assertFalse(torch.equal(a[k],v[k]))
            elif k.startswith(('norm','embedding','pdm','trend')):f.nested_equal(self,a[k],b[k]);f.nested_equal(self,a[k],v[k])
        for details,capture in [(d,a),(du,b),(dv,v)]:
            self.assertTrue(torch.equal(torch.stack(details['scale_predictions'],dim=-1).sum(-1),capture['norm0denorm']))
        self.assertEqual(c['embedding'],3);self.assertEqual(c['projection'],3);self.assertEqual(c['norm'].count((0,'denorm')),1)
        self.assertTrue(torch.equal(d['U']['gate'],dv['U']['gate']));self.assertTrue(torch.equal(d['V']['redistributed'],du['V']['redistributed']));self.assertTrue(torch.equal(x,original));self.assertTrue(torch.equal(marks,markcopy));f.nested_equal(self,m.state_dict(),state)
        self.assertFalse(torch.equal(y,offu));self.assertFalse(torch.equal(y,offv));self.assertEqual(adapter.prediction(m,x,marks).shape,(2,1,1))
    def test_every_registered_shape_and_actual_parameter_counts(self):
        unique={}
        for t in f.STORED['tasks']:
            p=t['structural_source_profile'];s=p['structure'];key=(p['T'],p['pred_len'],p['C'],s['d_model'],s['down_sampling_layers'],p['time_mark']['K'],p['time_mark']['freq'],t['task'],p['target_idx'],tuple(p['aux_idx']));unique.setdefault(key,t)
        rows=[]
        for key,t in unique.items():
            p=t['structural_source_profile'];m=adapter.build(p,use_u=True,use_v=True,fold=t['fold'],task=t['task']).eval();x,marks=batch(p,b=1)
            with torch.inference_mode():native=m(x,marks,None,None);pred=adapter.prediction(m,x,marks)
            self.assertEqual(native.shape,(1,p['pred_len'],p['C']));self.assertEqual(pred.shape,(1,p['pred_len'],p['C'] if t['task']=='M' else 1));self.assertTrue(torch.isfinite(pred).all())
            nu=sum(v.numel() for v in m.u.parameters());nv=sum(v.numel() for v in m.v.parameters());D=p['structure']['d_model'];slots=len(p['aux_idx']) if t['task']=='MS' else p['C'];self.assertEqual(nu,D*(p['T']+1)+8*slots+41);self.assertEqual(nv,64*D+1)
            rows.append(dict(dataset=t['dataset'],variant=t['input_variant'],L=p['T'],P=p['pred_len'],C=p['C'],D=D,K=p['time_mark']['K'],task=t['task'],target_idx=p['target_idx'],U_parameters=nu,V_parameters=nv,total_parameters=sum(v.numel() for v in m.parameters()),actual_batch=1,finite=True))
        (f.EVIDENCE/'registered-shape-checks.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
        urban=next(x for x in rows if x['dataset']=='UrbanEV' and x['C']==11);self.assertEqual(urban['U_parameters'],953);self.assertEqual(urban['total_parameters'],70885+953+4097)
        self.assertTrue(any(x['C']==21 and x['P']==720 and x['D']==16 for x in rows));self.assertTrue(any(x['target_idx']==2 and x['P']==24 for x in rows))


class Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder=f.case('pipeline');cls.results={};cls.configs={}
        for arm in ('B','U','V','UV'):
            c,task,path=f.fixture(cls.folder,arm=arm);cls.configs[arm]=(c,task,path);cls.results[arm]=run_worker(c,task,cls.folder/'results','four-arms',path_override=path)
    def test_actual_four_arm_worker_and_native_original_equivalence(self):
        for arm,r in self.results.items():
            self.assertEqual(r['manifest']['steps'],4);self.assertEqual(r['metrics']['test_calls'],1);self.assertTrue(all(r['manifest']['routes'][p]>0 for p in ['train','validation','test']));self.assertEqual(r['manifest']['identity']['structural_spec']['structural_revision'],f.REVISION);self.assertEqual(audit(r['path'])['metrics'],r['metrics'])
        c,t,path=self.configs['B'];ref=run_worker(c,t,self.folder/'reference','one',path_override=path,reference_original=True);a=read(Path(self.results['B']['path'])/'last.pt');b=read(Path(ref['path'])/'last.pt')
        for k in ['model','optimizer','scheduler','rng','best','epoch','steps']:f.nested_equal(self,a[k],b[k])
        for metric in ['mse','mae','sse','sae','elements']:self.assertEqual(ref['metrics'][metric],self.results['B']['metrics'][metric])
    def test_one_epoch_resume_exact_without_reselecting_test(self):
        c,t,path=self.configs['UV'];full=run_worker(c,t,self.folder/'full','one',path_override=path);pause=run_worker(c,t,self.folder/'resume','one',path_override=path,pause_after_epoch=1);self.assertEqual(pause['steps'],2)
        continued=run_worker(c,t,self.folder/'resume','one',path_override=path,resume=True)
        f.nested_equal(self,read(Path(full['path'])/'last.pt'),read(Path(continued['path'])/'last.pt'));self.assertEqual(full['metrics'],continued['metrics']);self.assertEqual((Path(full['path'])/'updates.jsonl').read_text(),(Path(continued['path'])/'updates.jsonl').read_text());self.assertEqual(continued['metrics']['test_calls'],1)
    def test_synthetic_state_restore_and_old_same_candidate_revision_refused_before_live_write(self):
        c,t,path=self.configs['UV'];r=self.results['UV'];checkpoint=Path(r['path'])/'last.pt';state=read(checkpoint);p=c['tasks'][0]['structural_source_profile'];m=adapter.build(p,use_u=True,use_v=True,fold=1);tr=p['training'];opt=torch.optim.Adam(m.parameters(),lr=tr['lr'],betas=tuple(tr['betas']),eps=tr['eps'],weight_decay=tr['weight_decay']);schedule=Schedule(opt,tr['scheduler']);generator=torch.Generator().manual_seed(23)
        identity=state['identity'];before=f.live(m,opt,schedule,generator)
        old_revision=dict(u='crossformer_router_core_target_conditioned_v6_r2',v='moderntcn_temporal_operator_only_serial_v6_r2',fusion='two_internal_finest_representation_residuals')
        for field in ['old_revision','provided_old_revision','source','data','arm','optimizer_activity']:
            bad=copy.deepcopy(state);expected=copy.deepcopy(identity)
            if field=='old_revision':bad['identity']['structural_spec']['structural_revision']=old_revision
            elif field=='provided_old_revision':expected['structural_spec']['structural_revision']=old_revision;bad['identity']=copy.deepcopy(expected)
            elif field=='source':bad['identity']['source']['foreign']='sha'
            elif field=='data':bad['identity']['data_sha']='wrong'
            elif field=='arm':bad['identity']['task']['arm']='U'
            else:bad['optimizer']['state']={}
            with self.subTest(field=field),self.assertRaises(ValueError):validate_state(bad,m,opt,schedule,expected)
            f.nested_equal(self,before,f.live(m,opt,schedule,generator))
        best,epoch,steps=restore(checkpoint,m,opt,schedule,generator,identity);self.assertEqual((epoch,steps),(2,4));f.nested_equal(self,m.state_dict(),state['model']);f.nested_equal(self,opt.state_dict(),state['optimizer']);f.nested_equal(self,schedule.state_dict(),state['scheduler']);self.assertTrue(torch.equal(torch.get_rng_state(),state['rng']['torch']));self.assertTrue(torch.equal(generator.get_state(),state['rng']['generator']))
    def test_unknown_claim_saved_test_immutable_success_and_receipt_namespace(self):
        c,t,path=self.configs['UV'];good=self.results['UV'];before=tree(good['path'])
        with self.assertRaises(FileExistsError):run_worker(c,t,self.folder/'results','duplicate',path_override=path)
        self.assertEqual(tree(good['path']),before)
        root=self.folder/'publish-fault'
        with self.assertRaisesRegex(RuntimeError,'sealing failure'):run_worker(c,t,root,'one',path_override=path,fault='before_publish')
        with patch.object(worker,'evaluate',side_effect=AssertionError('saved synthetic test must not repeat')):saved=run_worker(c,t,root,'one',path_override=path,resume=True)
        self.assertEqual(saved['metrics']['test_calls'],1)
        rp=self.folder/'receipt.json';rp.write_text(json.dumps(saved));runtime=dict(schema='timemixer-uv-owned-runtime-v2',candidate_id=c['candidate_id'],structural_revision=c['structural_revision'],source_binding=worker.source_binding(),commit=saved['manifest']['identity']['producer_commit'],synthetic=True,gate_policy=c['gate_policy']);self.assertEqual(launcher.verify_child_receipt(dict(receipt=rp,expected_task_id=t),c,root/'one',runtime,None),t)
        bad=copy.deepcopy(runtime);bad['structural_revision']=dict(u='r2',v='r2',fusion='old')
        with self.assertRaises(ValueError):launcher.verify_child_receipt(dict(receipt=rp,expected_task_id=t),c,root/'one',bad,None)
        # Derive a negative from the new synthetic artifact, preserving the original.
        negative_root=self.folder/'unknown-claim';staged=negative_root/'one/staging'/Path(good['path']).name
        staged.parent.mkdir(parents=True)
        shutil.copytree(good['path'],staged,ignore=shutil.ignore_patterns('test-result.json','test-result.sha256','checksums.sha256','seal.json'))
        original_staging=tree(staged)
        with patch.object(worker,'evaluate',side_effect=AssertionError('ambiguous test claim must never evaluate')):
            with self.assertRaisesRegex(ValueError,'cannot repeat'):
                run_worker(c,t,negative_root,'one',path_override=path,resume=True)
        self.assertEqual(tree(staged),original_staging)
    def test_strict_best_ties_and_no_effect_retry(self):
        b=BestState(2);self.assertTrue(b.update(1.,1));self.assertFalse(b.update(1.,2));self.assertFalse(b.update(2.,3));self.assertTrue(b.stopped);self.assertEqual(b.epoch,1)

if __name__=='__main__':unittest.main(verbosity=2)
