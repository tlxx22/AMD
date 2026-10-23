"""Focused v2 acceptance; exact identity and declared reduction tolerances."""
import copy,json,unittest,math
from unittest.mock import patch
import torch
from torch.nn import functional as F
from tests import moderntcn_v2_fixtures as f
from moderntcn_dev_v2 import adapter,worker,launcher,training_state
from moderntcn_dev_v1 import adapter as old
from moderntcn_dev_v2.modules import ConditionalTemporalFilter as V,WindowStateModulation as U
from moderntcn_dev_v2.spec import CANDIDATE_ID,REVISION
from moderntcn_dev_v2.training_config import validate
from moderntcn_dev_v2.scope import expected_tasks,verified_u_records,expected_waves
from moderntcn_dev_v2.summary import aggregate
from utils.ch3_onecycle import Schedule,configuration

def common(m):return {k:v for k,v in m.state_dict().items() if not k.startswith(('u.','v.'))}
def grads(m):return {k:p.grad for k,p in m.named_parameters()}
def nonzero(m):
    with torch.no_grad():
        if m.use_u:m.u.Wo.weight.fill_(.04);m.u.Wb.weight.fill_(.03)
        if m.use_v:m.v.A.fill_(.05)

class Parallel(unittest.TestCase):
    def test_nonzero_actual_same_h_once_and_Z(self):
        m=adapter.build(f.profile(),use_u=True,use_v=True).train();nonzero(m);x=torch.randn(2,12,11);before=x.clone();seen={};counts={}
        b=m.model.stages[0].blocks[0]
        def observe(name):
            def hook(mod,inp,out):seen[name]=(inp,out);counts[name]=counts.get(name,0)+1
            return hook
        handles=[mod.register_forward_hook(observe(name)) for name,mod in [('drop2',b.ffn1drop2),('U',m.u),('V',m.v),('dw',b.dw),('stem',m.model.downsample_layers[0]),('revin',m.model.revin_layer)]]
        rv=m.model.revin_layer.forward;calls=[]
        def revin(a,mode):
            calls.append(mode);out=rv(a,mode)
            if mode=='norm':seen['Z']=out.permute(0,2,1).detach().clone()
            return out
        with patch.object(m.model.revin_layer,'forward',side_effect=revin):m(x)
        for h in handles:h.remove()
        h=seen['drop2'][1].reshape(2,11,64,12);uh=seen['U'][0][0];vh=seen['V'][0][0]
        self.assertEqual(uh.data_ptr(),h.data_ptr());self.assertEqual(vh.data_ptr(),h.data_ptr());self.assertTrue(torch.equal(uh,vh))
        self.assertFalse(torch.equal(h,seen['stem'][1].reshape_as(h)));self.assertFalse(torch.equal(vh,seen['U'][1]));self.assertFalse(torch.equal(uh,h+seen['V'][1]))
        self.assertTrue(torch.equal(seen['V'][0][1],seen['Z']));self.assertTrue(torch.equal(seen['U'][0][1],x));self.assertEqual(counts['drop2'],1);self.assertEqual(counts['dw'],1);self.assertEqual(calls,['norm','denorm']);self.assertTrue(torch.equal(x,before))
    def test_local_independence_and_both_gradients_to_h(self):
        u=U(3,64,2,[0,1]);v=V(96,64,24,51)
        with torch.no_grad():u.Wo.weight.normal_(0,.1);u.Wb.weight.normal_(0,.1);v.A.normal_(0,.1)
        h=torch.randn(2,3,64,24,requires_grad=True);x=torch.randn(2,96,3);z=torch.randn(2,3,96)
        du=u(h,x);dv=v(h,z);gv=torch.autograd.grad(dv.square().sum(),h,retain_graph=True)[0];gu=torch.autograd.grad((du-h).square().sum(),h)[0]
        self.assertGreater(float(gu.abs().sum()),0);self.assertGreater(float(gv.abs().sum()),0)
        with torch.no_grad():u.Wo.weight.add_(.1)
        self.assertTrue(torch.equal(dv,v(h,z)))
        du=u(h,x)
        with torch.no_grad():v.A.add_(.1)
        self.assertTrue(torch.equal(du,u(h,x)))
    def test_v_explicit_new_h_forward_and_gradients(self):
        for L,N,K in [(12,12,7),(96,24,51)]:
            torch.manual_seed(42);v=V(L,8,N,K).double()
            with torch.no_grad():v.A.normal_(0,.2)
            h=torch.randn(2,3,8,N,dtype=torch.float64,requires_grad=True);z=torch.randn(2,3,L,dtype=torch.float64,requires_grad=True)
            out=v(h,z);a=v.route(z);kernel=torch.einsum('bcr,dr,rk->bcdk',a,v.A,v.T)
            ref=F.conv1d(h.reshape(1,48,N),kernel.reshape(48,1,v.Kv),padding=v.Kv//2,groups=48).reshape_as(h)
            torch.testing.assert_close(out,ref,atol=1e-10,rtol=1e-9)
            weight=torch.randn_like(out);pars=(h,z,*v.parameters());aa=torch.autograd.grad((out*weight).sum(),pars,retain_graph=True);bb=torch.autograd.grad((ref*weight).sum(),pars)
            for a,b in zip(aa,bb):torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-9)
    def test_neutral_native_state_rng_bn_and_grad_exact(self):
        for ds,H in [('UrbanEV',3),('PJM',24)]:
            p=f.profile(ds,H=H);b=adapter.build_original(p);rng=torch.get_rng_state().clone();models=[]
            for u,v in [(False,False),(True,False),(False,True),(True,True)]:
                m=adapter.build(p,use_u=u,use_v=v);models.append(m);f.equal(self,common(m),b.state_dict());self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                self.assertEqual(hasattr(m,'u'),u);self.assertEqual(hasattr(m,'v'),v)
            f.equal(self,models[1].u.state_dict(),models[3].u.state_dict());f.equal(self,models[2].v.state_dict(),models[3].v.state_dict())
            x=torch.randn(2,p['T'],p['C']);start=torch.get_rng_state().clone()
            for train in (False,True):
                b.train(train);b.zero_grad();torch.set_rng_state(start);y=b(x);end=torch.get_rng_state().clone();y.square().mean().backward();g=copy.deepcopy(grads(b))
                for m in models:
                    m.train(train);m.zero_grad();torch.set_rng_state(start);z=m(x);self.assertTrue(torch.equal(y,z));self.assertTrue(torch.equal(end,torch.get_rng_state()));z.square().mean().backward();f.equal(self,{k:v for k,v in grads(m).items() if k in g},g);f.equal(self,common(m),b.state_dict())
    def test_u_only_v1_exact_nonzero_train_eval_updates(self):
        rows=[]
        for ds,H in [('UrbanEV',3),('PJM',24)]:
            p=f.profile(ds,H=H);a=old.build(p,use_u=True,use_v=False,fold=6 if ds=='UrbanEV' else 1);r=torch.get_rng_state().clone();b=adapter.build(p,use_u=True,use_v=False,fold=6 if ds=='UrbanEV' else 1)
            self.assertTrue(torch.equal(r,torch.get_rng_state()));f.equal(self,a.state_dict(),b.state_dict());nonzero(a);nonzero(b)
            opts=[torch.optim.Adam(m.parameters(),lr=.01,weight_decay=1e-7) for m in (a,b)];ss=[Schedule(o,configuration(2,4)) for o in opts]
            x=torch.randn(2,p['T'],p['C'])
            for train in (False,True):
                a.train(train);b.train(train);start=torch.get_rng_state().clone();torch.set_rng_state(start);y=a(x);end=torch.get_rng_state().clone();torch.set_rng_state(start);z=b(x);self.assertTrue(torch.equal(y,z));self.assertTrue(torch.equal(end,torch.get_rng_state()))
            for step in range(2):
                start=torch.get_rng_state().clone();saved=[]
                for m,o,s in zip((a,b),opts,ss):
                    torch.set_rng_state(start);o.zero_grad();y=m(x);y[:,:,p['target_idx']].square().mean().backward();g=copy.deepcopy(grads(m));o.step();s.after_successful_update();saved.append((copy.deepcopy(m.state_dict()),copy.deepcopy(o.state_dict()),g,torch.get_rng_state().clone()))
                f.equal(self,saved[0],saved[1])
            rows.append(dict(dataset=ds,output_state_gradient_Adam_RNG='exact',nonzero_fixture=True))
        f.output('U-only-equivalence',rows)
    def test_delayed_gradients_on_new_interface(self):
        m=adapter.build(f.profile(),use_u=True,use_v=True);o=torch.optim.Adam(m.parameters(),lr=.001,weight_decay=1e-7);rows=[]
        for step in range(3):
            o.zero_grad();m(torch.randn(2,12,11))[:,:,0].square().mean().backward();g={n:float(p.grad.abs().sum()) if p.grad is not None else None for n,p in m.named_parameters() if n.startswith(('u.','v.'))};rows.append(g)
            self.assertTrue(all(v is None or math.isfinite(v) for v in g.values()))
            if step==0:
                for n in ('u.A','u.Wh.weight','v.T','v.W1.weight','v.W2.weight'):self.assertEqual(g[n],0)
                for n in ('u.Wo.weight','u.Wb.weight','v.A'):self.assertGreater(g[n],0)
            o.step()
        for n in ('u.A','u.Wh.weight','v.T','v.W1.weight','v.W2.weight'):self.assertGreater(rows[-1][n],0)
        f.output('gradient-opening',rows)
    def test_shapes_counts_and_target_loss(self):
        rows=[]
        for ds,variant,H in [('UrbanEV',v,3) for v in ('F1','F2','F3','F4')]+[('PJM','MS',24)]:
            p=f.profile(ds,variant,H);m=adapter.build(p,use_u=True,use_v=True).eval();x=torch.randn(2,p['T'],p['C']);y=adapter.prediction(m,x)
            self.assertEqual(y.shape,(2,p['pred_len'],1));self.assertEqual(m.u.target,p['target_idx']);self.assertEqual(list(m.u.aux),p['aux_idx']);self.assertGreater(len(m.u.aux),0)
            U=sum(p.numel() for p in m.u.parameters());Vn=sum(p.numel() for p in m.v.parameters());self.assertEqual(U,1536+24*p['C']+p['C']-1);self.assertEqual(Vn,16*p['T']+64+4*(64+m.v.Kv))
            self.assertRaises(ValueError,adapter.loss,y,torch.zeros(2,p['pred_len'],p['C']))
            rows.append(dict(dataset=ds,variant=variant,C=p['C'],N=m.spec.value['N'],Kv=m.v.Kv,B=sum(v.numel() for n,v in m.named_parameters() if not n.startswith(('u.','v.'))),U=U,V=Vn))
        f.output('parameter-counts',rows)
    def test_observer_purity_and_no_context_leak(self):
        m=adapter.build(f.profile(),use_u=True,use_v=True).eval();nonzero(m);x=torch.randn(2,12,11);before=x.clone();state=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone()
        with torch.inference_mode():
            y=m(x);z,d=m(x,return_diagnostics=True);self.assertTrue(torch.equal(y,z));self.assertTrue(torch.equal(x,before));self.assertTrue(torch.equal(rng,torch.get_rng_state()));f.equal(self,state,m.state_dict());self.assertNotIn('native_BN',d);self.assertNotIn('post_BN_delta',d['V']);self.assertIn('parallel_post_ffn1',d)
            m(torch.randn(1,12,11));self.assertTrue(torch.equal(y,m(x)))
        f.output('diagnostic-example',d)

class Delivery(unittest.TestCase):
    def test_launch_template_exact_scope_profiles_budget_and_offline_rejection(self):
        from moderntcn_dev_v2.scope import validate_launch_scope
        c=f.STORED;ids=[t['id'] for t in c['tasks']]
        a=json.loads((f.EVIDENCE/'non-execution-preparation/launch-binding.template.json').read_text())
        validate_launch_scope(c,ids,a)
        self.assertIsNone(a['commit']);self.assertFalse(a['review_approved'])
        with patch('moderntcn_dev_v2.launcher.git',side_effect=AssertionError('nonexecuting template must reject before network')):
            self.assertRaises(PermissionError,launcher.validate_authorization,c,'unused',ids,a)
        for mutate in (lambda x:x.update(max_optimizer_updates=1867620),lambda x:x['task_profile_bindings'].clear(),lambda x:x.update(task_scope='development_extension_f6_epf_3arm_v1'),lambda x:x['U_reuse'].clear()):
            bad=copy.deepcopy(a);mutate(bad);self.assertRaises(ValueError,validate_launch_scope,c,ids,bad)
        self.assertRaises(ValueError,validate_launch_scope,c,ids[:-1],a)

    def test_scope_reuse_budget_and_profile_science(self):
        c=validate(f.STORED);self.assertEqual(len(c['tasks']),18);self.assertEqual(len(c['U_reuse']),9);self.assertEqual(len(c['future_input_scope']),12)
        self.assertEqual(c['max_optimizer_steps'],1245080);self.assertEqual(c['max_run_epochs'],360);self.assertEqual([len(w) for w in launcher.batch(c,[t['id'] for t in c['tasks']])],[4,4,1,4,4,1]);self.assertEqual(len(verified_u_records(c)),9)
        for mutate in (lambda x:x['tasks'].reverse(),lambda x:x['tasks'].append(x['tasks'][0]),lambda x:x['tasks'][0].update(fold=1),lambda x:x.update(B_ref='paper_C1_r8_v1'),lambda x:x['future_input_scope'].clear()):
            bad=copy.deepcopy(c);mutate(bad);self.assertRaises(ValueError,validate,bad)
        from moderntcn_dev_v1.training_config import validate as original_validate
        original=json.loads((f.ROOT/'configs/moderntcn_uv_extension_f6_epf_3arm_v1.json').read_text());original_validate(original);self.assertRaises(ValueError,validate,original)
    def test_summary_no_B_chain_no_per_H_gate_and_missing(self):
        c=f.STORED;reuse=verified_u_records(c);records=[]
        for t in c['tasks']:
            b=c['existing_B'][t['baseline_pair_cell_id']];records.append((t,dict(task_id=t['id'],mse=b['mse']*.7,mae=b['mae']*.7)))
        for i,(t,m) in enumerate(records):
            if t['arm']=='UV' and t['dataset'] in ('PJM','NP','BE'):m['mae']=c['existing_B'][t['baseline_pair_cell_id']]['mae']
        a=aggregate(c,records,reuse_records=reuse);self.assertEqual(a['goals']['goal2']['winning_cells'],7);self.assertEqual(a['goals']['goal2']['numerical_status'],'Met');self.assertEqual(a['goals']['goal3']['numerical_status'],'Not evaluated');self.assertTrue(a['goals']['goal1']['coverage_complete']);self.assertEqual(a['completed'],18)
        self.assertEqual(aggregate(c,records[:-1],reuse_records=reuse)['goals']['goal2']['numerical_status'],'Not evaluated');self.assertRaises(ValueError,aggregate,c,records+[records[0]],reuse_records=reuse)
    def test_cross_version_state_and_AUTH_rejected_without_mutation(self):
        p=f.profile();m=adapter.build(p,use_u=True,use_v=True);v1=old.build(p,use_u=True,use_v=True);before=copy.deepcopy(m.state_dict());ck=old.synthetic_checkpoint(v1)
        self.assertRaises(ValueError,adapter.load_synthetic_checkpoint,m,ck);f.equal(self,before,m.state_dict())
        ck=adapter.synthetic_checkpoint(m);adapter.load_synthetic_checkpoint(m,ck)
        for key,val in [('candidate_id','moderntcn-uv-dev-v1'),('target_idx',2),('aux_idx',[2,1])]:
            bad=copy.deepcopy(ck);bad['spec'][key]=val;self.assertRaises(ValueError,adapter.load_synthetic_checkpoint,m,bad);f.equal(self,before,m.state_dict())
        with patch('moderntcn_dev_v2.launcher.git',side_effect=AssertionError('must reject before Git/network')):
            self.assertRaises((ValueError,PermissionError),launcher.validate_authorization,f.STORED,'unused',[t['id'] for t in f.STORED['tasks']],{'review_approved':True,'candidate_id':'moderntcn-uv-dev-v1'})
    def test_actual_synthetic_worker_restore_seal_and_protection(self):
        checkpoints=[]
        for mode in ('continuous','resume'):
            folder=f.case(mode);c,tid,path=f.fixture(folder)
            if mode=='resume':worker.run_worker(c,tid,folder/'runs','synthetic',path_override=path,pause_after_epoch=1)
            receipt=worker.run_worker(c,tid,folder/'runs','synthetic',path_override=path,resume=mode=='resume');final=__import__('pathlib').Path(receipt['path'])
            checkpoints.append(worker.read(final/'last.pt'));self.assertEqual(receipt['metrics']['test_calls'],1);self.assertEqual(receipt['metrics']['steps'],8)
            saved={p.name:worker.file_sha(p) for p in final.iterdir()};self.assertRaises(FileExistsError,worker.run_worker,c,tid,folder/'runs','synthetic',path_override=path)
            self.assertEqual(saved,{p.name:worker.file_sha(p) for p in final.iterdir()})
        for k in ('model','optimizer','scheduler','rng','best','epoch','steps'):f.equal(self,checkpoints[0][k],checkpoints[1][k])

if __name__=='__main__':unittest.main()
