"""v3 formula, unchanged real-author backbone and output-coordinate acceptance."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from timemixer_dev_v3 import adapter
from timemixer_dev_v3.model import AuthorModel
from timemixer_dev_v3.modules import (SelectiveOutputResidual as U,
    PrototypeOutputCalibration as V, window_statistics, receiver_indices)
from timemixer_dev_v3.spec import StructuralSpec
from tests.timemixer_v3_fixtures import ROOT, EVIDENCE, profile, small_profile, nested_equal
from utils.ch3_onecycle import Schedule, configuration


class Identity(unittest.TestCase):
    def test_off_and_common_B_keys_initialization_RNG_and_substreams(self):
        p=small_profile();original=adapter.build_original(p);rng=torch.get_rng_state().clone();models={}
        for arm in ('B','U','V','UV'):
            m=adapter.build(p,use_u=arm in ('U','UV'),use_v=arm in ('V','UV'));models[arm]=m
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            nested_equal(self,original.state_dict(),{k:m.state_dict()[k] for k in original.state_dict()})
        self.assertEqual(set(models['B'].state_dict()),set(original.state_dict()))
        nested_equal(self,models['U'].u.state_dict(),models['UV'].u.state_dict())
        nested_equal(self,models['V'].v.state_dict(),models['UV'].v.state_dict())
        x=torch.randn(3,12,11);mark=torch.randn(3,12,4)
        original.eval();expected=original(x,mark,None,None)
        for m in models.values():
            m.eval();self.assertTrue(torch.equal(expected,m(x,mark,None,None)))

    def test_off_train_grad_loss_Adam_OneCycle_exact(self):
        p=small_profile();a=adapter.build_original(p);b=adapter.build(p,use_u=False,use_v=False)
        a.train();b.train();x=torch.randn(3,12,11);marks=torch.randn(3,12,4);target=torch.randn(3,1,1)
        oa=torch.optim.Adam(a.parameters(),lr=.01,weight_decay=1e-7);ob=torch.optim.Adam(b.parameters(),lr=.01,weight_decay=1e-7)
        sa=Schedule(oa,configuration(3,2));sb=Schedule(ob,configuration(3,2))
        for _ in range(2):
            rng=torch.get_rng_state().clone();oa.zero_grad(set_to_none=True);ob.zero_grad(set_to_none=True)
            ya=a(x,marks,None,None)[:,:,:1];la=adapter.loss(ya,target);la.backward();after=torch.get_rng_state().clone()
            torch.set_rng_state(rng);yb=adapter.prediction(b,x,marks);lb=adapter.loss(yb,target);lb.backward()
            self.assertTrue(torch.equal(ya,yb));self.assertTrue(torch.equal(la,lb));self.assertTrue(torch.equal(after,torch.get_rng_state()))
            for k,pa in a.named_parameters():
                pb=dict(b.named_parameters())[k]
                if pa.grad is None:self.assertIsNone(pb.grad)
                else:self.assertTrue(torch.equal(pa.grad,pb.grad),k)
            oa.step();sa.after_successful_update();ob.step();sb.after_successful_update()
            nested_equal(self,a.state_dict(),b.state_dict());nested_equal(self,oa.state_dict(),ob.state_dict());nested_equal(self,sa.state_dict(),sb.state_dict())

    def test_author_once_denorm_once_branch_does_not_mutate_cache_input_or_dropout_RNG(self):
        p=small_profile();a=adapter.build_original(p);m=adapter.build(p,use_u=True,use_v=True)
        x=torch.randn(2,12,11);marks=torch.randn(2,12,4);xb=x.clone();mb=marks.clone()
        a.train();m.train();rng=torch.get_rng_state().clone();expected=a(x,marks,None,None);after=torch.get_rng_state().clone()
        norm=a.normalize_layers[0];cache={k:v.clone() for k,v in vars(norm).items() if isinstance(v,torch.Tensor)}
        torch.set_rng_state(rng);calls=[];denorm=[];original_forecast=AuthorModel.forecast
        def forecast(instance,*args,**kw):calls.append(1);return original_forecast(instance,*args,**kw)
        handle=m.normalize_layers[0].register_forward_hook(lambda mod,inp,out:denorm.append(inp[1]))
        try:
            with patch.object(AuthorModel,'forecast',forecast):out=m(x,marks,None,None)
        finally:handle.remove()
        self.assertEqual(len(calls),1);self.assertEqual(denorm.count('denorm'),1)
        self.assertTrue(torch.equal(after,torch.get_rng_state()));self.assertTrue(torch.equal(expected,out))
        self.assertTrue(torch.equal(x,xb));self.assertTrue(torch.equal(marks,mb))
        for k,v in cache.items():self.assertTrue(torch.equal(v,getattr(m.normalize_layers[0],k)),k)

    def test_output_bounds_after_author_denorm_and_independent_V(self):
        m=adapter.build(small_profile(),use_u=True,use_v=True).eval()
        with torch.no_grad():
            m.u.out.weight.fill_(.7);m.v.final.weight.fill_(.3);m.v.final.bias.fill_(.4)
            x=torch.randn(3,12,11);marks=torch.randn(3,12,4)
            both,d=m.forecast(x,marks,None,None,return_diagnostics=True)
            m.use_u=False
            try:only_v,dv=m.forecast(x,marks,None,None,return_diagnostics=True)
            finally:m.use_u=True
        self.assertTrue(torch.equal(d['V']['Delta_V'],dv['V']['Delta_V']))
        self.assertTrue(torch.equal(both[:,:,1:],d['Y0'][:,:,1:]))
        torch.testing.assert_close(both[:,:,:1],d['Y0'][:,:,:1]+d['U']['Delta_U'].transpose(1,2)+d['V']['Delta_V'].transpose(1,2),rtol=0,atol=0)
        torch.testing.assert_close(both[:,:,:1]-only_v[:,:,:1],d['U']['Delta_U'].transpose(1,2),rtol=0,atol=2e-7)
        for key,delta in [('U','Delta_U'),('V','Delta_V')]:self.assertTrue(bool((d[key][delta].abs()<=1).all()))

    def test_profile_shapes_native_marks_all_horizons_and_parameter_counts(self):
        cases=[('UrbanEV',v,h) for v in ('F1','F2','F3','F4') for h in (3,6,9,12)]
        cases += [('PJM','MS',24)]+[(d,'M',h) for d in ('ETTh1','Exchange','Weather') for h in (96,192,336,720)]
        with torch.no_grad():
            for dataset,variant,H in cases:
                p=small_profile(dataset,variant,H);m=adapter.build(p,use_u=True,use_v=True,task=p.get('task','MS')).eval()
                x=torch.randn(2,p['T'],p['C']);mark=torch.randn(2,p['T'],p['time_mark']['K']);out=adapter.prediction(m,x,mark)
                self.assertEqual(tuple(out.shape),(2,p['pred_len'],p['C'] if p.get('task')=='M' else 1))
        counts={}
        for arm in ('B','U','V','UV'):
            m=adapter.build(profile(),use_u=arm in ('U','UV'),use_v=arm in ('V','UV'))
            counts[arm]=sum(v.numel() for v in m.parameters())
        self.assertEqual(counts,dict(B=70885,U=75718,V=71575,UV=76408))
        (EVIDENCE/'parameter-counts.json').write_text(json.dumps(counts,indent=2)+'\n')

    def test_old_versions_rejected_without_writes_and_all_arms_strict_synthetic(self):
        for arm in ('B','U','V','UV'):
            a=adapter.build(small_profile(),use_u=arm in ('U','UV'),use_v=arm in ('V','UV'))
            state=adapter.synthetic_checkpoint(a);b=adapter.build(small_profile(),use_u=a.use_u,use_v=a.use_v)
            adapter.load_synthetic_checkpoint(b,state);nested_equal(self,a.state_dict(),b.state_dict())
            before=copy.deepcopy(b.state_dict())
            for candidate in ('timemixer-uv-dev-v1','timemixer-uv-dev-v2'):
                bad=copy.deepcopy(state);bad['spec']['candidate_id']=candidate
                with self.assertRaises(ValueError):adapter.load_synthetic_checkpoint(b,bad)
                nested_equal(self,before,b.state_dict())
        snapshot=json.loads((EVIDENCE/'before/snapshot.json').read_text())
        from timemixer_dev_v3.training_state import file_sha
        for name,ref in snapshot['files'].items():
            if not name.startswith('docs/'):
                self.assertEqual(file_sha(ROOT/name),ref['sha256'],name)


class UFormula(unittest.TestCase):
    def test_float64_independent_reference_and_source_axis(self):
        torch.manual_seed(17);u=U(12,5,5,'MS',3,[4,0,2]).double()
        with torch.no_grad():u.out.weight.normal_(0,.2)
        x=torch.randn(3,12,5,dtype=torch.float64);actual,d=u(x,return_diagnostics=True)
        mu=x.mean(1);sigma=((x-mu[:,None]).square().mean(1)+1e-5).sqrt();z=(x-mu[:,None])/sigma[:,None]
        e=F.layer_norm(F.linear(z.transpose(1,2),u.linear_z.weight,u.linear_z.bias),(32,),u.norm.weight,u.norm.bias,1e-5)+F.linear(torch.stack((mu,sigma.log1p()),-1),u.linear_a.weight,u.linear_a.bias)
        values=F.linear(F.gelu(F.linear(e,u.v1.weight,u.v1.bias)),u.v2.weight,u.v2.bias)
        logs=[]
        for j in range(5):
            v=F.linear(F.gelu(F.linear(e[:,3],u.aq.weight)+F.linear(e[:,j],u.ak.weight)+u.b_cond),u.ao.weight,u.ao.bias)
            logs.append(v if j in [4,0,2] else torch.full_like(v,-torch.inf))
        logs.append(torch.zeros_like(logs[0]));alpha=torch.stack(logs,1).softmax(1)
        core=(alpha[:,:5]*values).sum(1);gate=F.linear(torch.cat((e[:,3],core),-1),u.gate.weight,u.gate.bias).sigmoid()
        expected=(gate*F.linear(core,u.out.weight)).tanh()
        torch.testing.assert_close(actual[:,0],expected,rtol=1e-12,atol=1e-12)
        self.assertTrue(torch.equal(d['alpha'][:,:,1],torch.zeros_like(d['alpha'][:,:,1])))
        self.assertTrue(bool((d['alpha'][:,:,-1]>0).all()));self.assertEqual(u.sources,(4,0,2))

    def test_M_simultaneous_diagonal_and_per_receiver_reference(self):
        u=U(12,3,4,'M',0,[3,1,2]).double()
        with torch.no_grad():u.out.weight.normal_(0,.1)
        x=torch.randn(3,12,4,dtype=torch.float64);out,d=u(x,return_diagnostics=True)
        for r in range(4):
            self.assertTrue(torch.equal(d['alpha'][:,r,r],torch.zeros_like(d['alpha'][:,r,r])))
            ref=U(12,3,4,'MS',r,[j for j in range(4) if j!=r]).double();ref.load_state_dict(u.state_dict(),strict=True)
            torch.testing.assert_close(out[:,r],ref(x)[:,0],rtol=1e-12,atol=1e-12)
        for aux in ([0],[1,1],[-1],[4],[True]):
            with self.assertRaises(ValueError):receiver_indices(4,'MS',0,aux)

    def test_empty_zero_core_constant_near_constant_large_and_bounds(self):
        for x in (torch.ones(2,12,3),torch.ones(2,12,3)+torch.arange(12)[None,:,None]*1e-7,torch.randn(2,12,3)*1e6):
            u=U(12,2,3,'MS',1,[2,0]);self.assertTrue(torch.equal(u(x),torch.zeros(2,1,2)))
            with torch.no_grad():u.out.weight.fill_(1e6)
            d=u(x);self.assertTrue(bool(torch.isfinite(d).all()));self.assertTrue(bool((d.abs()<=1).all()))
            with torch.no_grad():u.v2.weight.zero_();u.v2.bias.zero_()
            self.assertTrue(torch.equal(u(x),torch.zeros_like(d)))
        for C,task,aux in [(3,'MS',[]),(1,'M',[])]:
            u=U(12,2,C,task,0,aux);self.assertTrue(torch.equal(u(torch.randn(2,12,C)),torch.zeros(2,1,2)))


class VFormula(unittest.TestCase):
    def test_float64_formula_marks_state_prototypes_and_split_order(self):
        torch.manual_seed(19);v=V(5,3,4,5,'MS',2,[3,0,1]).double()
        with torch.no_grad():v.final.weight.normal_(0,.2);v.final.bias.normal_(0,.2)
        x=torch.randn(3,5,4,dtype=torch.float64);marks=torch.randn(3,5,5,dtype=torch.float64);y=torch.randn(3,3,4,dtype=torch.float64)
        out,d=v(x,marks,y,return_diagnostics=True);mu=x.mean(1);sigma=((x-mu[:,None]).square().mean(1)+1e-5).sqrt()
        state=torch.stack((mu,sigma.log1p(),x[:,-1],x.diff(dim=1).abs().mean(1)),-1)[:,2:3]
        m=torch.cat((marks[:,0],marks[:,-1]),-1)
        q=F.layer_norm(F.gelu(F.linear(m,v.time.weight,v.time.bias)[:,None]+F.linear(state,v.state.weight,v.state.bias)),(16,),v.norm.weight,v.norm.bias,1e-5)
        pi=(q@v.prototypes.T/4).softmax(-1);ctx=pi@v.prototypes;ab=F.linear(F.gelu(F.linear(ctx,v.hidden.weight,v.hidden.bias)),v.final.weight,v.final.bias)
        av,bv=ab[:,:,:3],ab[:,:,3:];expected=(av*(y[:,:,2].unsqueeze(1)-mu[:,2,None,None])+bv).tanh()
        torch.testing.assert_close(out,expected,rtol=1e-12,atol=1e-12);nested_equal(self,d['marks'],m);nested_equal(self,d['state'],state)
        torch.testing.assert_close(pi.sum(-1),torch.ones(3,1,dtype=torch.float64),rtol=1e-12,atol=1e-12)
        self.assertTrue(bool((out.abs()<=1).all()))

    def test_marks_refusal_L1_M_receivers_and_shared_parameters(self):
        v=V(1,24,3,4,'M',2,[1,0]);x=torch.randn(2,1,3);marks=torch.randn(2,1,4);y=torch.randn(2,24,3)
        with torch.no_grad():v.final.weight.normal_(0,.1)
        out,d=v(x,marks,y,return_diagnostics=True);self.assertTrue(torch.equal(d['state'][:,:,-1],torch.zeros(2,3)))
        for r in range(3):
            ref=V(1,24,3,4,'MS',r,[j for j in range(3) if j!=r]);ref.load_state_dict(v.state_dict(),strict=True)
            torch.testing.assert_close(out[:,r],ref(x,marks,y)[:,0],rtol=1e-6,atol=1e-7)
        for m in (None,torch.zeros(2,1,5),torch.zeros(2,2,4)):
            with self.assertRaises(ValueError):v(x,m,y)
        self.assertTrue(bool(torch.isfinite(v(x*1e6,marks,y*1e6)).all()))
        p=small_profile();p['time_mark']=None
        with self.assertRaises(ValueError):StructuralSpec.from_profile(p,use_u=False,use_v=True)

    def test_history_only_and_U_does_not_enter_V(self):
        v=V(12,1,11,4,'MS',0,list(range(1,11)));x=torch.randn(3,12,11);marks=torch.randn(3,12,4);y=torch.randn(3,1,11)
        with torch.no_grad():v.final.weight.normal_(0,.1)
        before=v(x,marks,y);interior=marks.clone();interior[:,1:-1]=torch.randn_like(interior[:,1:-1])*1e3
        self.assertTrue(torch.equal(before,v(x,interior,y)))
        # V's API has no U/core/forecast-end marks input.
        u=U(12,1,11,'MS',0,list(range(1,11)))
        with torch.no_grad():u.out.weight.fill_(4.)
        u(x);self.assertTrue(torch.equal(before,v(x,marks,y)))


class Gradients(unittest.TestCase):
    def test_zero_final_heads_then_all_branch_parameter_groups_connect(self):
        torch.manual_seed(101);m=adapter.build(small_profile(),use_u=True,use_v=True)
        opt=torch.optim.Adam(m.parameters(),lr=.001);x=torch.randn(4,12,11);mark=torch.randn(4,12,4);target=torch.randn(4,1,1)
        m.eval();_,diag=m.forecast(x,mark,None,None,return_diagnostics=True)
        self.assertTrue(torch.equal(diag['U']['Delta_U'],torch.zeros(4,1,1)));self.assertTrue(torch.equal(diag['V']['Delta_V'],torch.zeros(4,1,1)))
        seen=set()
        for step in range(3):
            opt.zero_grad(set_to_none=True);loss=adapter.loss(adapter.prediction(m,x,mark),target);loss.backward()
            for name,p in m.named_parameters():
                if name.startswith(('u.','v.')):
                    self.assertIsNotNone(p.grad,name);self.assertTrue(bool(torch.isfinite(p.grad).all()),name)
                    if torch.count_nonzero(p.grad):seen.add(name)
            if step==0:
                self.assertTrue(torch.equal(m.u.linear_z.weight.grad,torch.zeros_like(m.u.linear_z.weight)))
                self.assertTrue(torch.equal(m.v.prototypes.grad,torch.zeros_like(m.v.prototypes)))
                self.assertTrue(bool(torch.count_nonzero(m.u.out.weight.grad)));self.assertTrue(bool(torch.count_nonzero(m.v.final.weight.grad)))
            opt.step()
        expected={name for name,_ in m.named_parameters() if name.startswith(('u.','v.'))}
        self.assertEqual(seen,expected)


if __name__=='__main__':unittest.main(verbosity=2)
