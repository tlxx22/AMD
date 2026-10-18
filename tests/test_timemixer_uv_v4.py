"""v4 exact author identity, efficient adoption formulas and competitive fusion."""
import copy
import json
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from timemixer_dev_v4 import adapter
from timemixer_dev_v4.model import AuthorModel
from timemixer_dev_v4.modules import (SelectiveOutputResidual as U,
    PrototypeOutputCalibration as V, window_statistics, adoption_state, competitive_fusion)
from timemixer_dev_v4.spec import StructuralSpec
from tests.timemixer_v4_fixtures import ROOT, EVIDENCE, profile, small_profile, nested_equal
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

    def test_competition_changes_adoption_but_never_V_proposal_or_score(self):
        m=adapter.build(small_profile(),use_u=True,use_v=True).eval()
        with torch.no_grad():
            m.u.out.weight.fill_(.7);m.v.final.weight.fill_(.3);m.v.final.bias.fill_(.4)
            m.u.adoption.bias.fill_(1.2);m.v.adoption.bias.fill_(-.2)
            x=torch.randn(3,12,11);marks=torch.randn(3,12,4)
            both,d=m.forecast(x,marks,None,None,return_diagnostics=True)
            m.use_u=False
            try:only_v,dv=m.forecast(x,marks,None,None,return_diagnostics=True)
            finally:m.use_u=True
        self.assertTrue(torch.equal(d['V']['Delta_V'],dv['V']['Delta_V']))
        self.assertTrue(torch.equal(d['ell_V'],dv['ell_V']))
        self.assertFalse(torch.equal(d['adoption_alpha'][...,2],dv['adoption_alpha'][...,2]))
        self.assertTrue(torch.equal(both[:,:,1:],d['Y0'][:,:,1:]))
        # Fusion adds the combined correction once (the approved operation order).
        correction=d['adoption_alpha'][...,1]*d['U']['Delta_U']+d['adoption_alpha'][...,2]*d['V']['Delta_V']
        self.assertTrue(torch.equal(both[:,:,:1],(d['Y0'][:,:,:1].transpose(1,2)+correction).transpose(1,2)))
        self.assertTrue(bool((d['correction'].abs()<=1).all()))

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
        self.assertEqual(counts,dict(B=70885,U=75773,V=71598,UV=76486))
        (EVIDENCE/'parameter-counts.json').write_text(json.dumps(counts,indent=2)+'\n')

    def test_old_versions_rejected_without_writes_and_all_arms_strict_synthetic(self):
        for arm in ('B','U','V','UV'):
            a=adapter.build(small_profile(),use_u=arm in ('U','UV'),use_v=arm in ('V','UV'))
            state=adapter.synthetic_checkpoint(a);b=adapter.build(small_profile(),use_u=a.use_u,use_v=a.use_v)
            adapter.load_synthetic_checkpoint(b,state);nested_equal(self,a.state_dict(),b.state_dict())
            before=copy.deepcopy(b.state_dict())
            for candidate in ('timemixer-uv-dev-v1','timemixer-uv-dev-v2','timemixer-uv-dev-v3'):
                bad=copy.deepcopy(state);bad['spec']['candidate_id']=candidate
                with self.assertRaises(ValueError):adapter.load_synthetic_checkpoint(b,bad)
                nested_equal(self,before,b.state_dict())
        snapshot=json.loads((EVIDENCE/'before/snapshot.json').read_text())
        from timemixer_dev_v4.training_state import file_sha
        for name,ref in snapshot['files'].items():
            if not name.startswith('docs/'):
                self.assertEqual(file_sha(ROOT/name),ref['sha256'],name)


class AdoptionFormula(unittest.TestCase):
    def test_float64_U54_V22_reference_position_broadcast_and_gradients(self):
        torch.manual_seed(39)
        for task,target,aux in [('MS',2,[3,0,1]),('M',2,[3,0,1])]:
            u=U(5,7,4,task,target,aux).double();v=V(5,7,4,5,task,target,aux).double()
            with torch.no_grad():
                u.out.weight.normal_(0,.1);v.final.weight.normal_(0,.1)
                for mod in (u,v):mod.adoption.weight.normal_(0,.2);mod.adoption.bias.fill_(.3)
            x=torch.randn(3,5,4,dtype=torch.float64);marks=torch.randn(3,5,5,dtype=torch.float64);y=torch.randn(3,7,4,dtype=torch.float64)
            for mod,args,static_dim in [(u,(x,),48),(v,(x,marks,y),16)]:
                proposal,d=mod(*args,return_diagnostics=True)
                actual=mod.adoption_logit(x,y,proposal,d)
                mu=x.mean(1);sigma=((x-mu[:,None]).square().mean(1)+1e-5).sqrt()
                hist=torch.stack((mu,sigma.log1p(),x[:,-1]-mu,x.diff(dim=1).abs().mean(1)),-1)[:,mod.receivers]
                scalar=torch.cat((hist[:,:,None,:].expand(-1,-1,7,-1),
                    (y[:,:,mod.receivers].transpose(1,2)-mu[:,mod.receivers,None])[...,None]),-1)
                static=torch.cat((d['receiver_embedding'],d['core']),-1) if mod is u else d['context']
                # Deliberate large concatenation only in this independent reference.
                q=torch.cat((static[:,:,None,:].expand(-1,-1,7,-1),scalar,proposal[...,None]),-1)
                self.assertEqual(q.shape[-1],static_dim+6)
                expected=F.linear(q,mod.adoption.weight,mod.adoption.bias).squeeze(-1)
                torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
                ga=torch.autograd.grad(actual.sum(),(mod.adoption.weight,mod.adoption.bias),retain_graph=True)
                gb=torch.autograd.grad(expected.sum(),(mod.adoption.weight,mod.adoption.bias))
                for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)

    def test_long_P_only_static_Linear_inputs_no_large_feature_concatenation(self):
        u=U(96,720,21,'M',0,list(range(1,21)));v=V(96,720,21,5,'M',0,list(range(1,21)))
        x=torch.randn(2,96,21);marks=torch.randn(2,96,5);y=torch.randn(2,720,21)
        with torch.no_grad():
            for mod,args,dim in [(u,(x,),52),(v,(x,marks,y),20)]:
                proposal,d=mod(*args,return_diagnostics=True);seen=[];linear=F.linear
                def capture(inp,*a,**kw):seen.append(tuple(inp.shape));return linear(inp,*a,**kw)
                with patch.object(F,'linear',side_effect=capture):out=mod.adoption_logit(x,y,proposal,d)
                self.assertEqual(seen,[(2,21,dim)]);self.assertEqual(tuple(out.shape),(2,21,720))
                self.assertEqual(tuple(mod.adoption.weight.shape),(1,54 if mod is u else 22))

    def test_v3_proposals_initialization_formula_and_params_retained(self):
        from timemixer_dev_v3.modules import SelectiveOutputResidual as OldU, PrototypeOutputCalibration as OldV
        x=torch.randn(3,12,11);marks=torch.randn(3,12,4);y=torch.randn(3,1,11)
        for new,old,args,inputs in [(U,OldU,(12,1,11,'MS',0,list(range(1,11))),(x,)),
                                   (V,OldV,(12,1,11,4,'MS',0,list(range(1,11))),(x,marks,y))]:
            torch.manual_seed(2024);a=old(*args);torch.manual_seed(2024);b=new(*args)
            nested_equal(self,a.state_dict(),{k:v for k,v in b.state_dict().items() if not k.startswith('adoption.')})
            with torch.no_grad():
                head_a=a.out if new is U else a.final;head_b=b.out if new is U else b.final
                head_a.weight.normal_(0,.2);head_b.weight.copy_(head_a.weight)
                if head_a.bias is not None:head_a.bias.normal_(0,.1);head_b.bias.copy_(head_a.bias)
            self.assertTrue(torch.equal(a(*inputs),b(*inputs)))

    def test_L1_marks_state_constant_large_and_U_source_semantics(self):
        x=torch.ones(2,1,3);marks=torch.randn(2,1,4);y=torch.randn(2,24,3)
        for task,aux in [('MS',[1,0]),('M',[1,0])]:
            u=U(1,24,3,task,2,aux);v=V(1,24,3,4,task,2,aux)
            with torch.no_grad():u.out.weight.fill_(1e3);v.final.weight.fill_(1e3)
            du,ud=u(x,return_diagnostics=True);dv,vd=v(x,marks,y,return_diagnostics=True)
            hist,_=adoption_state(x,y,u.receivers,ud['mu'],ud['sigma'])
            self.assertTrue(torch.equal(hist[...,-1],torch.zeros_like(hist[...,-1])))
            for mod,d,p in [(u,ud,du),(v,vd,dv)]:
                self.assertTrue(bool(torch.isfinite(p).all()));self.assertTrue(bool((p.abs()<=1).all()))
                self.assertTrue(bool(torch.isfinite(mod.adoption_logit(x,y,p,d)).all()))
            if task=='M':
                for r in range(3):self.assertTrue(torch.equal(ud['alpha'][:,r,r],torch.zeros_like(ud['alpha'][:,r,r])))
            else:
                self.assertEqual(u.sources,(1,0));self.assertTrue(bool((ud['alpha'][:,:,-1]>0).all()))
            for invalid in (None,torch.zeros(2,1,5),torch.zeros(2,2,4)):
                with self.assertRaises(ValueError):v(x,invalid,y)
        u=U(12,1,3,'MS',0,[1,2])
        with torch.no_grad():u.v2.weight.zero_();u.v2.bias.zero_();u.out.weight.fill_(100.)
        self.assertTrue(torch.equal(u(torch.randn(2,12,3)),torch.zeros(2,1,1)))
        for bad in ([0],[1,1],[-1],[3],[True]):
            with self.assertRaises(ValueError):U(12,1,3,'MS',0,bad)


class Choices(unittest.TestCase):
    def test_initial_single_half_double_thirds_and_invalid_U_exclusion(self):
        z=torch.zeros(2,3,5);u=torch.ones_like(z)*.8;v=-torch.ones_like(z)*.6
        valid=torch.tensor([True,False,True])
        out,a,c=competitive_fusion(z,u=u,v=v,ell_u=z,ell_v=z,u_valid=valid)
        self.assertTrue(torch.equal(a[:,0],torch.full_like(a[:,0],1/3)))
        self.assertTrue(torch.equal(a[:,1, :,1],torch.zeros_like(z[:,1])))
        self.assertTrue(torch.equal(a[:,1,:,2],torch.full_like(z[:,1],.5)))
        self.assertTrue(torch.equal(c,a[...,1]*u+a[...,2]*v))
        for kw,index in [(dict(u=u,ell_u=z,u_valid=torch.ones(3,dtype=torch.bool)),1),(dict(v=v,ell_v=z),2)]:
            _,a,_=competitive_fusion(z,**kw);self.assertTrue(torch.equal(a[...,index],torch.full_like(z,.5)))
        out,a,c=competitive_fusion(z);self.assertTrue(torch.equal(out,z));self.assertTrue(torch.equal(a[...,0],torch.ones_like(z)))

    def test_extreme_logits_stable_axis_convex_bound_and_shape_refusal(self):
        torch.manual_seed(43);z=torch.randn(3,4,24,dtype=torch.float64);u=torch.randn_like(z).tanh();v=torch.randn_like(z).tanh()
        lu=torch.full_like(z,1e30);lv=-lu
        out,a,c=competitive_fusion(z,u=u,v=v,ell_u=lu,ell_v=lv,u_valid=torch.ones(4,dtype=torch.bool))
        self.assertTrue(bool(torch.isfinite(out).all()));self.assertTrue(torch.equal(a[...,1],torch.ones_like(z)))
        self.assertTrue(bool((c.abs()<=1).all()));self.assertTrue(torch.equal(c,u))
        lu=torch.randn_like(z)*5;lv=torch.randn_like(z)*5
        _,a,c=competitive_fusion(z,u=u,v=v,ell_u=lu,ell_v=lv,u_valid=torch.ones(4,dtype=torch.bool))
        torch.testing.assert_close(a.sum(-1),torch.ones_like(z),rtol=1e-15,atol=1e-15)
        self.assertTrue(bool((c.abs()<=a[...,1]*u.abs()+a[...,2]*v.abs()).all()))
        self.assertTrue(bool((c.abs()<=1).all()))
        with self.assertRaises(ValueError):competitive_fusion(z,u=u[...,:1],ell_u=lu,u_valid=torch.ones(4,dtype=torch.bool))

    def test_model_no_U_sources_does_not_dilute_V_and_nonzero_MS_target(self):
        p=small_profile('PJM','MS',24);self.assertEqual(p['target_idx'],2);p['aux_idx']=[]
        m=adapter.build(p,use_u=True,use_v=True).eval()
        x=torch.randn(2,96,3);marks=torch.randn(2,96,p['time_mark']['K'])
        with torch.no_grad():
            m.v.final.bias.fill_(.2)
            both,d=m.forecast(x,marks,None,None,return_diagnostics=True);m.use_u=False
            try:only,dv=m.forecast(x,marks,None,None,return_diagnostics=True)
            finally:m.use_u=True
        self.assertTrue(torch.equal(both,only));self.assertTrue(torch.equal(d['adoption_alpha'][...,1],torch.zeros(2,1,24)))
        self.assertTrue(torch.equal(both[:,:,:2],d['Y0'][:,:,:2]));self.assertTrue(torch.equal(d['ell_V'],dv['ell_V']))


class Gradients(unittest.TestCase):
    def test_zero_proposals_scores_and_delayed_all_new_parameter_paths(self):
        m=adapter.build(small_profile(),use_u=True,use_v=True).eval()
        x=torch.randn(4,12,11);marks=torch.randn(4,12,4);target=torch.randn(4,1,1)
        out,d=m.forecast(x,marks,None,None,return_diagnostics=True)
        self.assertTrue(torch.equal(out,d['Y0']))
        for key in ('ell_U','ell_V'):self.assertTrue(torch.equal(d[key],torch.zeros(4,1,1)))
        for key,field in [('U','Delta_U'),('V','Delta_V')]:self.assertTrue(torch.equal(d[key][field],torch.zeros(4,1,1)))
        opt=torch.optim.Adam(m.parameters(),lr=.001);seen=set()
        for step in range(4):
            opt.zero_grad(set_to_none=True);adapter.loss(adapter.prediction(m,x,marks),target).backward()
            for name,p in m.named_parameters():
                if name.startswith(('u.','v.')):
                    self.assertIsNotNone(p.grad,name);self.assertTrue(bool(torch.isfinite(p.grad).all()),name)
                    if torch.count_nonzero(p.grad):seen.add(name)
            if step==0:
                for p in (m.u.adoption.weight,m.u.adoption.bias,m.v.adoption.weight,m.v.adoption.bias,m.u.linear_z.weight,m.v.prototypes):
                    self.assertTrue(torch.equal(p.grad,torch.zeros_like(p)))
                self.assertTrue(bool(torch.count_nonzero(m.u.out.weight.grad)));self.assertTrue(bool(torch.count_nonzero(m.v.final.weight.grad)))
            opt.step()
        self.assertEqual(seen,{n for n,_ in m.named_parameters() if n.startswith(('u.','v.'))})


if __name__=='__main__':unittest.main(verbosity=2)
