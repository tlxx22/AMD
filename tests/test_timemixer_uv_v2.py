"""Targeted v2 mathematical and real-backbone acceptance; no real data/weights."""
import copy
import json
import unittest
from pathlib import Path
import torch
from timemixer_dev_v2 import adapter
from timemixer_dev_v2.modules import SelectiveVariableInteraction as U, HistoryConditionedScaleAdjustment as V
from timemixer_dev_v2.spec import StructuralSpec
from timemixer_dev_v2.source import implementation_binding
from tests.timemixer_v2_fixtures import ROOT, EVIDENCE, small_profile, nested_equal
from utils.ch3_onecycle import configuration, Schedule


class Identity(unittest.TestCase):
    def test_exact_backbone_initialization_RNG_and_module_substreams(self):
        p=small_profile();original=adapter.build_original(p);reference_rng=torch.get_rng_state().clone()
        models={}
        for arm,u,v in [('B',False,False),('U',True,False),('V',False,True),('UV',True,True)]:
            m=adapter.build(p,use_u=u,use_v=v);models[arm]=m
            self.assertTrue(torch.equal(reference_rng,torch.get_rng_state()))
            for k,val in original.state_dict().items():self.assertTrue(torch.equal(val,m.state_dict()[k]),k)
        self.assertEqual(set(models['B'].state_dict()),set(original.state_dict()))
        nested_equal(self,models['U'].u.state_dict(),models['UV'].u.state_dict())
        nested_equal(self,models['V'].v.state_dict(),models['UV'].v.state_dict())
        x=torch.randn(2,12,11);mark=torch.randn(2,12,4)
        original.eval();models['B'].eval()
        self.assertTrue(torch.equal(original(x,mark,None,None),models['B'](x,mark,None,None)))

    def test_off_train_loss_gradient_Adam_OneCycle_RNG_exact(self):
        p=small_profile();a=adapter.build_original(p);b=adapter.build(p,use_u=False,use_v=False)
        a.train();b.train();x=torch.randn(2,12,11);mark=torch.randn(2,12,4);target=torch.randn(2,1,1)
        oa=torch.optim.Adam(a.parameters(),lr=.01,weight_decay=1e-7);ob=torch.optim.Adam(b.parameters(),lr=.01,weight_decay=1e-7)
        sa=Schedule(oa,configuration(3,2));sb=Schedule(ob,configuration(3,2))
        for _ in range(2):
            rng=torch.get_rng_state().clone();oa.zero_grad(set_to_none=True);ob.zero_grad(set_to_none=True)
            ya=a(x,mark,None,None)[:,:,:1];la=adapter.loss(ya,target);la.backward();ra=torch.get_rng_state().clone()
            torch.set_rng_state(rng);yb=adapter.prediction(b,x,mark);lb=adapter.loss(yb,target);lb.backward()
            self.assertTrue(torch.equal(ya,yb));self.assertTrue(torch.equal(la,lb));self.assertTrue(torch.equal(ra,torch.get_rng_state()))
            for pa,pb in zip(a.parameters(),b.parameters()):
                if pa.grad is None:self.assertIsNone(pb.grad)
                else:self.assertTrue(torch.equal(pa.grad,pb.grad))
            oa.step();sa.after_successful_update();ob.step();sb.after_successful_update()
            nested_equal(self,a.state_dict(),b.state_dict());nested_equal(self,oa.state_dict(),ob.state_dict());nested_equal(self,sa.state_dict(),sb.state_dict())

    def test_v1_bytes_binding_and_entry_identity_preserved(self):
        from timemixer_dev import adapter as old
        from timemixer_dev.source import implementation_binding as old_binding
        from timemixer_dev_v2.training_state import file_sha
        before=json.loads((EVIDENCE/'before.json').read_text())
        for name,ref in before['old_files'].items():self.assertEqual(file_sha(ROOT/name),ref['sha256'],name)
        self.assertEqual(old_binding(),{Path(k).name:v['sha256'] for k,v in before['old_files'].items() if Path(k).parent==Path('timemixer_dev') and Path(k).suffix in ('.py','.json')})
        p=small_profile();m=old.build(p,use_u=True,use_v=False)
        self.assertEqual(m.spec.value['candidate_id'],'timemixer-uv-dev-v1')
        newer=adapter.build(p,use_u=True,use_v=False)
        before_state=copy.deepcopy(newer.state_dict())
        with self.assertRaises(ValueError):adapter.load_synthetic_checkpoint(newer,old.synthetic_checkpoint(m))
        nested_equal(self,before_state,newer.state_dict())
        from timemixer_dev_v2.model import TimeMixerUV
        with self.assertRaises(ValueError):TimeMixerUV(m.spec)


class BoundedU(unittest.TestCase):
    def test_float64_independent_formula_axis_and_MS_preservation(self):
        torch.manual_seed(27);u=U(12,5,'MS',3,[4,0,2]).double();u.gamma.data.fill_(-.37)
        z=torch.randn(3,12,5,dtype=torch.float64,requires_grad=True);x=torch.randn_like(z)
        result,d=u(z,x,return_diagnostics=True);zr=z[:,:,3].unsqueeze(1)
        expected_s=((zr-zr.mean(-1,keepdim=True)).square().mean(-1,keepdim=True)+1e-5).sqrt()
        expected_r=expected_s*torch.tanh(u.gamma*d['delta']/expected_s)
        torch.testing.assert_close(d['s'],expected_s,rtol=0,atol=0)
        torch.testing.assert_close(result[:,:,3],z[:,:,3]+expected_r[:,0],rtol=0,atol=0)
        self.assertEqual(tuple(d['s'].shape),(3,1,1));self.assertTrue(torch.equal(result[:,:,[0,1,2,4]],z[:,:,[0,1,2,4]]))
        self.assertTrue(bool((d['r_limited'].abs()<=d['s']).all()))
        result.sum().backward();self.assertTrue(torch.isfinite(z.grad).all())

    def test_empty_zero_core_constant_near_constant_negative_and_saturation(self):
        for dtype in (torch.float32,torch.float64):
            for kind in ('constant','near_constant','random'):
                z=torch.ones(2,12,3,dtype=dtype)
                if kind=='near_constant':z=z+torch.arange(12,dtype=dtype)[None,:,None]*1e-8
                elif kind=='random':z=torch.randn_like(z)
                u=U(12,3,'MS',1,[2,0]).to(dtype=dtype);u.gamma.data.fill_(-1e6)
                result,d=u(z,z,return_diagnostics=True)
                self.assertTrue(torch.isfinite(result).all());self.assertTrue(torch.isfinite(d['s']).all())
                self.assertTrue(bool((d['r_limited'].abs()<=d['s']).all()))
                u.out.weight.data.zero_();self.assertTrue(torch.equal(u(z,z),z))
                u=U(12,3,'MS',1,[]).to(dtype=dtype);self.assertIs(u(z,z),z)
            z=torch.randn(2,12,1,dtype=dtype);self.assertIs(U(12,1,'M',0,[]).to(dtype=dtype)(z,z),z)
        u=U(12,3,'MS',1,[0,2]);u.v2.weight.data.zero_();u.v2.bias.data.zero_()
        z=torch.randn(2,12,3);out,d=u(z,z,return_diagnostics=True)
        self.assertTrue(torch.equal(d['core'],torch.zeros_like(d['core'])));self.assertTrue(torch.equal(out,z))

    def test_M_mask_simultaneous_reference_and_nonsaturated_gradients(self):
        torch.manual_seed(11);u=U(12,4,'M',0,[3,1,2]).double();z=torch.randn(3,12,4,dtype=torch.float64,requires_grad=True);x=torch.randn_like(z)
        out,d=u(z,x,return_diagnostics=True)
        for r in range(4):
            self.assertTrue(torch.equal(d['alpha'][:,r,r],torch.zeros_like(d['alpha'][:,r,r])))
            ref=U(12,4,'MS',r,[j for j in range(4) if j!=r]).double();ref.load_state_dict(u.state_dict(),strict=True)
            torch.testing.assert_close(out[:,:,r],ref(z,x)[:,:,r],rtol=1e-12,atol=1e-12)
        (out*torch.randn_like(out)).square().mean().backward()
        for name,p in u.named_parameters():
            self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
            self.assertTrue(bool(torch.count_nonzero(p.grad)),name)
        self.assertTrue(torch.isfinite(z.grad).all())


class TrendV(unittest.TestCase):
    def test_descriptors_nonzero_target_L1_and_48_weights(self):
        from timemixer_dev.modules import HistoryConditionedScaleAdjustment as OldV
        torch.manual_seed(7);v=V(8,3,24,3,'MS',2,[1,0],0)
        # Constructor CI is explicit; compare parameter counts independently.
        old=OldV(8,3,24,3,'MS',2,[1,0],0)
        self.assertEqual(sum(p.numel() for p in v.parameters())-sum(p.numel() for p in old.parameters()),48)
        trends=[torch.randn(2,l,3,requires_grad=True) for l in (12,6,1)]
        states=[torch.randn(2,l,8,requires_grad=True) for l in (12,6,1)]
        predictions=[torch.randn(2,24,3) for _ in range(3)]
        out,d=v(states,predictions,trends=trends,return_diagnostics=True)
        self.assertEqual(tuple(d['tau'].shape),(2,1,3,3))
        for i,t in enumerate(trends):
            target=t[:,:,2].contiguous();change=(target[:,1:]-target[:,:-1]).abs().mean(-1) if target.shape[1]>1 else torch.zeros(2)
            self.assertTrue(torch.equal(d['tau'][:,0,i],torch.stack((target.mean(-1),target[:,-1],change),-1)))
        out.square().mean().backward()
        grad=v.h.weight.grad[:,-3:];self.assertTrue(torch.isfinite(grad).all());self.assertTrue(bool(torch.count_nonzero(grad)))
        for t in trends:self.assertIsNotNone(t.grad);self.assertTrue(bool(torch.count_nonzero(t.grad[:,:,2])))
        for h in states:self.assertIsNotNone(h.grad);self.assertTrue(bool(torch.count_nonzero(h.grad)))
        for name,p in v.named_parameters():self.assertTrue(torch.isfinite(p.grad).all(),name);self.assertTrue(bool(torch.count_nonzero(p.grad)),name)
        base=torch.stack(predictions,-1).sum(-1);self.assertTrue(torch.equal(out[:,:,:2],base[:,:,:2]))
        with self.assertRaises(ValueError):v(states,predictions)

    def test_actual_pre_enc_trend_with_U_and_single_denorm(self):
        for dataset,variant,H in [('UrbanEV','F4',3),('PJM','MS',24)]:
            p=small_profile(dataset,variant,H);m=adapter.build(p,use_u=True,use_v=True)
            m.eval();saved={};norm=[]
            original=m.pre_enc
            def observed(xs):
                output=original(xs);saved['trends']=output[1];return output
            m.pre_enc=observed
            def capture(module,args,kwargs):saved['passed']=kwargs['trends']
            hook=m.v.register_forward_pre_hook(capture,with_kwargs=True)
            hooks=[layer.register_forward_pre_hook(lambda layer,args,i=i:norm.append((i,args[1]))) for i,layer in enumerate(m.normalize_layers)]
            x=torch.randn(2,p['T'],p['C']);mark=torch.randn(2,p['T'],p['time_mark']['K'])
            with torch.no_grad():
                y=adapter.prediction(m,x,mark);tau=m.v.trend_descriptors(saved['passed'],2).clone()
                for a,b in zip(saved['trends'],saved['passed']):self.assertIs(a,b)
                m.use_u=False;adapter.prediction(m,x,mark);other=m.v.trend_descriptors(saved['passed'],2)
            self.assertFalse(torch.equal(tau,other));self.assertTrue(torch.equal(tau[:,:,1:],other[:,:,1:]))
            self.assertEqual(tuple(y.shape),(2,p['pred_len'],1))
            self.assertEqual(norm,([(i,'norm') for i in range(p['structure']['down_sampling_layers']+1)]+[(0,'denorm')])*2)
            self.assertTrue(all(not hasattr(m,n) for n in ('diagnostics','last_diagnostics')))
            hook.remove()
            for h in hooks:h.remove()
            m.pre_enc=original

    def test_CI1_V_exact_v1_math_parameters_and_gradients(self):
        from timemixer_dev.modules import HistoryConditionedScaleAdjustment as OldV
        torch.manual_seed(2024);a=OldV(8,3,96,7,'M',0,[1,2,3,4,5,6],1)
        torch.manual_seed(2024);b=V(8,3,96,7,'M',0,[1,2,3,4,5,6],1)
        nested_equal(self,a.state_dict(),b.state_dict())
        states=[torch.randn(14,l,8) for l in (96,48,24)];ys=[torch.randn(2,96,7) for _ in range(3)]
        ya,da=a(states,ys,return_diagnostics=True);yb,db=b(states,ys,return_diagnostics=True)
        self.assertTrue(torch.equal(ya,yb));self.assertTrue(torch.equal(da['pi'],db['pi']));self.assertIsNone(db['tau'])
        ya.square().mean().backward();yb.square().mean().backward()
        for pa,pb in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(pa.grad,pb.grad))
        with self.assertRaises(ValueError):b(states,ys,trends=[torch.randn(2,l,7) for l in (96,48,24)])

    def test_uniform_pi_original_sum_and_single_scale(self):
        for S in (1,3):
            v=V(8,S,1,3,'MS',2,[0,1],0);v.p.weight.data.zero_()
            states=[torch.randn(2,4,8) for _ in range(S)];ys=[torch.randn(2,1,3) for _ in range(S)];trends=[torch.randn(2,4,3) for _ in range(S)]
            out,d=v(states,ys,trends=trends,return_diagnostics=True)
            self.assertTrue(torch.equal(out,torch.stack(ys,-1).sum(-1)))
            self.assertTrue(torch.equal(d['correction'],torch.zeros_like(d['correction'])))


if __name__=='__main__':unittest.main(verbosity=2)
