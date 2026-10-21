"""Targeted frozen-contract CPU tests, not real-model performance experiments."""
import copy,json,math,unittest
from unittest.mock import patch
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from tests import moderntcn_v1_fixtures as f
from moderntcn_dev_v1 import adapter,worker,launcher
from moderntcn_dev_v1.modules import WindowStateModulation as U,ConditionalTemporalFilter as V
from moderntcn_dev_v1.source import load_author
from moderntcn_dev_v1.spec import REVISION,StructuralSpec
from moderntcn_dev_v1.training_config import validate,derive
from moderntcn_dev_v1.summary import aggregate
from utils.ch3_onecycle import Schedule,configuration

def common(m):return {k:v for k,v in m.state_dict().items() if not k.startswith(('u.','v.'))}
def grads(m):return {k:p.grad for k,p in m.named_parameters() if not k.startswith(('u.','v.'))}
def tree(p):return {str(q.relative_to(p)):worker.file_sha(q) for q in Path(p).rglob('*') if q.is_file()}

class Formula(unittest.TestCase):
    def test_u_independent_formula_and_target_slots(self):
        for C,y in [(3,2),(6,0)]:
            torch.manual_seed(12);u=U(C,64,y,[j for j in range(C) if j!=y]);H=torch.randn(2,C,64,4);X=torch.randn(2,12,C)
            with torch.no_grad():u.Wo.weight.normal_(0,.2);u.Wb.weight.normal_(0,.2)
            out=u(H,X);mu=X.mean(1);var=((X-mu[:,None])**2).mean(1);s=torch.stack([mu,torch.log1p(var)],-1)
            h=torch.stack([torch.tanh(F.linear(s[:,c],u.A[c],u.e[c])) for c in range(C)],1)
            gates=torch.stack([torch.sigmoid((h[:,y]*h[:,j]).sum(-1)/math.sqrt(8)+u.b[k]) for k,j in enumerate(u.aux)],1)
            q=sum(gates[:,k,None]*h[:,j] for k,j in enumerate(u.aux))/(1+gates.sum(1,keepdim=True))
            expected=H.clone()
            for n in range(4):expected[:,y,:,n]+=F.linear(q*F.linear(H[:,y,:,n],u.Wh.weight),u.Wo.weight)+F.linear(q,u.Wb.weight)
            torch.testing.assert_close(out,expected,atol=2e-6,rtol=2e-5)
            for c in u.aux:self.assertTrue(torch.equal(out[:,c],H[:,c]))
            changed=X.clone();changed[:,:,u.aux[0]]+=4;self.assertFalse(torch.equal(out,u(H,changed)))
    def test_u_constants_no_aux_and_nonfinite(self):
        u=U(1,64,0,[]);H=torch.randn(2,1,64,3)
        with torch.no_grad():u.Wo.weight.fill_(1);u.Wb.weight.fill_(1)
        self.assertTrue(torch.equal(u(H,torch.ones(2,12,1)),H))
        u=U(3,64,2,[0,1]);q,g=u.controller(torch.full((2,12,3),3.));self.assertTrue(torch.isfinite(q).all());self.assertTrue(torch.isfinite(g).all())
        with self.assertRaises(ValueError):u.controller(torch.full((2,12,3),float('nan')))
        with self.assertRaises(ValueError):u.controller(torch.full((2,12,3),1e38)*torch.tensor([1.,-1.,1.])[None,None,:].expand(2,12,3).clone().index_fill(1,torch.tensor([0]),-1))
    def test_v_explicit_kernel_forward_gradients_all_profiles(self):
        for L,N,K in [(12,12,7),(96,24,51),(12,3,51)]:
            torch.manual_seed(42);v=V(L,8,N,K).double()
            with torch.no_grad():v.A.normal_(0,.2)
            H=torch.randn(2,3,8,N,dtype=torch.float64,requires_grad=True);Z=torch.randn(2,3,L,dtype=torch.float64,requires_grad=True)
            out=v(H,Z);a=v.route(Z);kern=torch.einsum('bcr,dr,rk->bcdk',a,v.A,v.T)
            ref=F.conv1d(H.reshape(1,48,N),kern.reshape(48,1,v.Kv),padding=v.Kv//2,groups=48).reshape_as(H)
            torch.testing.assert_close(out,ref,atol=1e-10,rtol=1e-9)
            pars=(H,Z,*v.parameters());weight=torch.randn_like(out)
            aa=torch.autograd.grad((out*weight).sum(),pars,retain_graph=True);bb=torch.autograd.grad((ref*weight).sum(),pars)
            for x,y in zip(aa,bb):torch.testing.assert_close(x,y,atol=1e-10,rtol=1e-9)
            self.assertTrue(torch.equal(v(H,torch.zeros_like(Z)),torch.zeros_like(H)))
    def test_v_router_variable_independence_and_kernel_orientation(self):
        v=V(12,4,12,7);H=torch.randn(2,3,4,12);Z=torch.randn(2,3,12)
        with torch.no_grad():v.A.fill_(.5)
        out=v(H,Z);z=Z.clone();z[:,1]+=3;changed=v(H,z)
        self.assertTrue(torch.equal(out[:,0],changed[:,0]));self.assertFalse(torch.equal(out[:,1],changed[:,1]))

class Native(unittest.TestCase):
    def test_initial_native_common_rng_output_gradient_bn_identity(self):
        p=f.profile();original=adapter.build_original(p);rng=torch.get_rng_state().clone();models=[adapter.build(p,use_u=u,use_v=v) for u,v in [(False,False),(True,False),(False,True),(True,True)]]
        for m in models:f.equal(self,common(m),original.state_dict());self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        f.equal(self,models[1].u.state_dict(),models[3].u.state_dict());f.equal(self,models[2].v.state_dict(),models[3].v.state_dict())
        X=torch.randn(2,12,11);start=torch.get_rng_state().clone()
        for training in (False,True):
            original.train(training);torch.set_rng_state(start);y=original(X);end=torch.get_rng_state();y.square().mean().backward();og=copy.deepcopy(grads(original))
            for m in models:
                m.train(training);torch.set_rng_state(start);z=m(X);self.assertTrue(torch.equal(y,z));self.assertTrue(torch.equal(end,torch.get_rng_state()));z.square().mean().backward();f.equal(self,grads(m),og);f.equal(self,common(m),original.state_dict());m.zero_grad(set_to_none=True)
            original.zero_grad(set_to_none=True)
    def test_off_adam_onecycle_two_updates_exact(self):
        p=f.profile();a=adapter.build_original(p);b=adapter.build(p,use_u=False,use_v=False)
        opts=[torch.optim.Adam(m.parameters(),lr=.01,weight_decay=1e-7) for m in (a,b)];ss=[Schedule(o,configuration(2,4)) for o in opts]
        for step in range(2):
            X=torch.randn(2,12,11);rng=torch.get_rng_state().clone();states=[]
            for m,o,s in zip((a,b),opts,ss):
                torch.set_rng_state(rng);o.zero_grad();out=m(X);out[:,:,0].square().mean().backward();o.step();s.after_successful_update();states.append((copy.deepcopy(m.state_dict()),copy.deepcopy(o.state_dict()),torch.get_rng_state().clone()))
            f.equal(self,states[0],states[1])
    def test_exact_two_insertions_one_revin_and_input_unchanged(self):
        m=adapter.build(f.profile(),use_u=True,use_v=True).eval();X=torch.randn(2,12,11);before=X.clone();calls=[];rv=m.model.revin_layer.forward
        def revin(x,mode):calls.append(mode);return rv(x,mode)
        with patch.object(m.model.revin_layer,'forward',side_effect=revin),patch.object(m.u,'forward',wraps=m.u.forward) as u,patch.object(m.v,'forward',wraps=m.v.forward) as v:
            y=m(X)
            self.assertEqual(calls,['norm','denorm']);self.assertEqual(u.call_count,1);self.assertEqual(v.call_count,1)
            self.assertTrue(torch.equal(u.call_args.args[1],X));self.assertEqual(tuple(v.call_args.args[0].shape),(2,11,64,12))
        self.assertTrue(torch.equal(X,before));self.assertEqual(y.shape,(2,1,11))
    def test_shapes_parameters_and_paper_pending(self):
        rows=[]
        for dataset,variant,H in [('UrbanEV',v,3) for v in ('F1','F2','F3','F4')]+[('PJM','MS',24)]:
            p=f.profile(dataset,variant,H);m=adapter.build(p,use_u=True,use_v=True).eval();C=p['C'];L=p['T'];N=m.spec.value['N'];D=64;kv=m.spec.value['Kv']
            self.assertEqual(sum(x.numel() for x in m.u.parameters()),3*D*8+3*C*8+C-1)
            self.assertEqual(sum(x.numel() for x in m.v.parameters()),16*L+64+4*(D+kv))
            y=adapter.prediction(m,torch.randn(2,L,C));self.assertEqual(y.shape,(2,p['pred_len'],1))
            rows.append(dict(dataset=dataset,variant=variant,B=sum(x.numel() for n,x in m.named_parameters() if not n.startswith(('u.','v.'))),U=sum(x.numel() for x in m.u.parameters()),V=sum(x.numel() for x in m.v.parameters()),N=N,Kv=kv))
        for d,var,h in [('UrbanEV','F4',3),('PJM','MS',24)]:
            p=f.profile(d,var,h);p['backbone_variant']='paper_C1_r8_v1';p['structure'].update(patch_size=8,patch_stride=4,large_size=[51],small_size=[5],ffn_ratio=8)
            m=adapter.build(p,use_u=True,use_v=True).eval();self.assertEqual(m.spec.value['N'],3 if d=='UrbanEV' else 24);self.assertEqual(m.spec.value['Kv'],5 if d=='UrbanEV' else 47)
            self.assertEqual(m(torch.randn(2,p['T'],p['C'])).shape,(2,p['pred_len'],p['C']))
        (f.EVIDENCE/'parameter-counts.json').write_text(json.dumps(rows,indent=2))
    def test_delayed_gradients_no_zero_shortcut(self):
        m=adapter.build(f.profile(),use_u=True,use_v=True);opt=torch.optim.Adam(m.parameters(),lr=.001,weight_decay=1e-7)
        logs=[]
        for step in range(3):
            opt.zero_grad();m(torch.randn(2,12,11))[:,:,0].square().mean().backward()
            g={n:float(p.grad.abs().sum()) if p.grad is not None else None for n,p in m.named_parameters() if n.startswith(('u.','v.'))};logs.append(g)
            self.assertTrue(all(v is None or math.isfinite(v) for v in g.values()))
            if step==0:
                for n in ('u.A','u.Wh.weight','v.T','v.W1.weight','v.W2.weight'):self.assertEqual(g[n],0)
                for n in ('u.Wo.weight','u.Wb.weight','v.A'):self.assertGreater(g[n],0)
            opt.step()
        for n in ('u.A','u.Wh.weight','v.T','v.W1.weight','v.W2.weight'):self.assertGreater(logs[-1][n],0)
        (f.EVIDENCE/'gradient-opening.json').write_text(json.dumps(logs,indent=2))
    def test_observer_no_pollution_and_continuous_batches(self):
        m=adapter.build(f.profile(),use_u=True,use_v=True).eval()
        with torch.no_grad():m.u.Wo.weight.fill_(.02);m.v.A.fill_(.03)
        X=torch.randn(2,12,11);state=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone()
        y=m(X);z,d=m(X,return_diagnostics=True);self.assertTrue(torch.equal(y,z));self.assertTrue(torch.equal(rng,torch.get_rng_state()));f.equal(self,state,m.state_dict())
        self.assertIn('post_BN_delta',d['V']);m(torch.randn(1,12,11));self.assertTrue(torch.equal(y,m(X)))
        (f.EVIDENCE/'diagnostic-example.json').write_text(json.dumps(d,indent=2))

class Delivery(unittest.TestCase):
    def test_config_53_13_8_order_and_not_evaluated(self):
        c=validate(f.STORED);self.assertEqual(len(c['tasks']),53);self.assertEqual(len(c['existing_B']),13);self.assertEqual(len(c['F4_reuse']),8)
        self.assertTrue(all(t['task']=='MS' for t in c['tasks']));result=aggregate(c,[])
        self.assertTrue(all(v['numerical_status']=='Not evaluated' for v in result['goals'].values()))
        ids=[t['id'] for arm in ('U','V','UV') for fold in (1,4) for t in sorted(c['tasks'],key=lambda t:t['h']) if t['goal']==1 and t['arm']==arm and t['fold']==fold]
        waves=launcher.batch(c,ids);self.assertEqual(len(waves),6);self.assertTrue(all(len(w)==4 for w in waves))
        with self.assertRaises(PermissionError):launcher.validate_authorization(c,'unused',ids,{'review_approved':False})
        bad=copy.deepcopy(c);bad['structural_revision']['u']='old';self.assertRaises(ValueError,validate,bad)
    def test_schema_checkpoint_wrong_identity_no_mutation(self):
        m=adapter.build(f.profile(),use_u=True,use_v=True);ck=adapter.synthetic_checkpoint(m);before=copy.deepcopy(m.state_dict())
        for key,value in [('candidate_id','timemixer-uv-dev-v6'),('target_idx',2),('features',list(reversed(ck['spec']['features']))),('aux_idx',[2,1])]:
            bad=copy.deepcopy(ck);bad['spec'][key]=value
            with self.assertRaises(ValueError):adapter.load_synthetic_checkpoint(m,bad)
            f.equal(self,before,m.state_dict())
        adapter.load_synthetic_checkpoint(m,ck);f.equal(self,before,m.state_dict())
    def test_real_synthetic_worker_four_arms_and_epoch_resume(self):
        continuous=None
        for arm in ('B','U','V','UV'):
            folder=f.case('worker-'+arm);c,tid,path=f.fixture(folder,arm);r=worker.run_worker(c,tid,folder/'runs','synthetic',path_override=path)
            files=list((folder/'runs'/'synthetic'/'runs').iterdir());self.assertEqual(len(files),1)
            metrics=json.loads((files[0]/'metrics.json').read_text());self.assertEqual(metrics['test_calls'],1)
            saved=tree(files[0])
            with self.assertRaises(FileExistsError):worker.run_worker(c,tid,folder/'runs','synthetic',path_override=path)
            self.assertEqual(saved,tree(files[0]))
            if arm=='UV':continuous=worker.read(files[0]/'last.pt')
        folder=f.case('resume');c,tid,path=f.fixture(folder,'UV');r=worker.run_worker(c,tid,folder/'runs','synthetic',path_override=path,pause_after_epoch=1)
        worker.run_worker(c,tid,folder/'runs','synthetic',path_override=path,resume=True)
        final=next((folder/'runs'/'synthetic'/'runs').iterdir());resumed=worker.read(final/'last.pt')
        for k in ('model','optimizer','scheduler','rng','best','epoch','steps'):
            f.equal(self,continuous[k],resumed[k])

    def test_goals_strict_coverage_and_scientific_source(self):
        c=f.STORED;records=[]
        for t in c['tasks']:
            b=c['existing_B'][t['baseline_cell_id']]
            factor={'U':.8,'V':.85,'UV':.7}[t['arm']] if t['goal']==1 else {'F1':1.0,'F2':.9,'F3':.85}.get(t['input_variant'],.7)
            metrics=dict(task_id=t['id'],mse=b['mse']*factor,mae=b['mae']*factor)
            if t['goal']==2:metrics.update(mse=b['mse']*.9,mae=b['mae']*.9)
            records.append((t,metrics))
        epf=[i for i,(t,m) in enumerate(records) if t['goal']==2]
        for i in epf[:3]:records[i][1]['mae']=c['existing_B'][records[i][0]['baseline_cell_id']]['mae']
        a=aggregate(c,records);self.assertTrue(a['goals']['goal1']['numerical_conditions_met']);self.assertTrue(a['goals']['goal3']['numerical_conditions_met'])
        self.assertEqual(a['goals']['goal2']['winning_cells'],7);self.assertTrue(a['goals']['goal2']['numerical_conditions_met'])
        reduced=[row for i,row in enumerate(records) if i!=epf[-1]];self.assertEqual(aggregate(c,reduced)['goals']['goal2']['numerical_status'],'Not evaluated')
        self.assertRaises(ValueError,aggregate,c,records+[records[0]])
        fresh=derive();self.assertEqual(fresh,c);self.assertEqual(c['global_lock'],json.loads((f.ROOT/'configs/timemixer_uv_round2_resolved_v6_r4.json').read_text())['global_lock'])
        self.assertTrue(all(t['structural_source_profile']['model']=='ModernTCN' and t['structural_source_profile'].get('time_mark') is None for t in c['tasks']))
        self.assertEqual(sum(t['arithmetic']['max_optimizer_steps'] for t in c['tasks'] if t['goal']==1),1485300)
    def test_initial_u_v_registration_and_orthogonality(self):
        u=U(11,64,0,list(range(1,11)));v=V(12,64,12,7)
        for x in (u.Wo.weight,u.Wb.weight,u.e,u.b,v.A):self.assertTrue(torch.equal(x,torch.zeros_like(x)))
        for x in (u.A,u.Wh.weight,v.W1.weight,v.W2.weight,v.T):self.assertGreater(float(x.abs().sum()),0)
        torch.testing.assert_close(v.T@v.T.T,torch.eye(4),atol=2e-6,rtol=2e-5)

if __name__=='__main__':unittest.main()
