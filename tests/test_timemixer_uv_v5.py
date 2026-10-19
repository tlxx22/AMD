"""Focused synthetic v5 math, author equivalence and insertion acceptance."""
import copy,math,unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from tests.timemixer_v5_fixtures import profile,small_profile,nested_equal
from timemixer_dev_v5 import adapter
from timemixer_dev_v5.modules import SegmentedVariableInteraction as U,LatentTemporalRefinement as V
from utils.ch3_onecycle import configuration,Schedule


def nonzero(module):
    with torch.no_grad():
        layer=module.bridge if isinstance(module,U) else module.project
        layer.weight.copy_(torch.linspace(-.13,.19,layer.weight.numel()).reshape_as(layer.weight))
        if layer.bias is not None:layer.bias.fill_(.07)


def u_reference(m,x):
    b,L,C=x.shape;pad=(-L)%3;z=torch.cat([x[:,:1,:].expand(-1,pad,-1),x],1)if pad else x
    output=[]
    for ib in range(b):
        receivers=[]
        for ri,r in enumerate(m.receivers):
            segments=[]
            for n in range(z.shape[1]//3):
                e=[]
                for j in range(C):
                    raw=z[ib,n*3:n*3+3,j];mu=raw.mean();sigma=torch.sqrt(((raw-mu)**2).mean()+1e-5)
                    a=F.layer_norm(F.linear((raw-mu)/sigma,m.patch_embedding.weight,m.patch_embedding.bias),(32,),m.norm.weight,m.norm.bias,1e-5)
                    e.append(a+F.linear(torch.stack([mu,torch.log1p(sigma)]),m.statistics.weight,m.statistics.bias))
                heads=[]
                q=F.linear(e[r],m.query.weight,m.query.bias).reshape(4,8)
                for hd in range(4):
                    scores=[];values=[]
                    for j in m.sources:
                        if m.task=='M'and j==r:score=x.new_tensor(-torch.inf)
                        else:score=(q[hd]*F.linear(e[j],m.key.weight,m.key.bias).reshape(4,8)[hd]).sum()/math.sqrt(8)
                        scores.append(score);values.append(F.linear(e[j],m.value.weight,m.value.bias).reshape(4,8)[hd])
                    scores.append(x.new_zeros(()));values.append(x.new_zeros(8))
                    weights=torch.softmax(torch.stack(scores),0)
                    heads.append(sum(w*v for w,v in zip(weights,values)))
                context=torch.cat(heads)
                segments.append(F.linear(context,m.bridge.weight).reshape(3,m.d_model))
            receivers.append(torch.cat(segments,0)[pad:])
        output.append(torch.stack(receivers))
    return torch.stack(output)


class AuthorAndIdentity(unittest.TestCase):
    def test_common_initialization_substreams_keys_RNG_and_parameter_counts(self):
        p=profile();base=adapter.build_original(p);rng=torch.get_rng_state().clone();bs=base.state_dict();us=vs=None
        for u,v,number in [(False,False,70885),(True,False,80485),(False,True,88229),(True,True,97829)]:
            m=adapter.build(p,use_u=u,use_v=v,task='MS',fold=1);self.assertEqual(sum(q.numel()for q in m.parameters()),number);self.assertTrue(torch.equal(torch.get_rng_state(),rng))
            self.assertEqual(set(k for k in m.state_dict()if not k.startswith(('u.','v.'))),set(bs))
            for k,value in bs.items():self.assertTrue(torch.equal(m.state_dict()[k],value),k)
            if u:
                if us is None:us=copy.deepcopy(m.u.state_dict())
                else:nested_equal(self,us,m.u.state_dict())
            if v:
                if vs is None:vs=copy.deepcopy(m.v.state_dict())
                else:nested_equal(self,vs,m.v.state_dict())

    def test_off_loss_gradient_Adam_OneCycle_exact(self):
        p=small_profile();a=adapter.build_original(p);b=adapter.build(p,use_u=False,use_v=False,task='MS');x=torch.randn(3,12,11);mark=torch.randn(3,12,4);y=torch.randn(3,1,1)
        oa,ob=[torch.optim.Adam(m.parameters(),lr=.01,weight_decay=1e-7)for m in(a,b)];sa,sb=[Schedule(o,configuration(2,2))for o in(oa,ob)]
        for step in range(2):
            results=[]
            for m,o,s in[(a,oa,sa),(b,ob,sb)]:
                torch.manual_seed(2024+step);m.train();o.zero_grad(set_to_none=True);pred=m(x,mark,None,None)[:,:,:1];loss=F.mse_loss(pred,y);loss.backward();grads={k:None if q.grad is None else q.grad.clone()for k,q in m.named_parameters()};o.step();s.after_successful_update();results.append((pred.detach(),loss.detach(),grads,torch.get_rng_state().clone()))
            nested_equal(self,results[0],results[1]);nested_equal(self,a.state_dict(),b.state_dict());nested_equal(self,oa.state_dict(),ob.state_dict());nested_equal(self,sa.state_dict(),sb.state_dict())

    def test_zero_enabled_author_order_output_and_dropout_RNG(self):
        p=small_profile();base=adapter.build_original(p);x=torch.randn(2,12,11);mark=torch.randn(2,12,4)
        for u,v in[(False,False),(True,False),(False,True),(True,True)]:
            m=adapter.build(p,use_u=u,use_v=v,task='MS');base.train();m.train();before=x.clone();mb=mark.clone();torch.manual_seed(2024);expected=base(x,mark,None,None);rng=torch.get_rng_state().clone();torch.manual_seed(2024);actual=m(x,mark,None,None)
            self.assertTrue(torch.equal(actual,expected));self.assertTrue(torch.equal(torch.get_rng_state(),rng));self.assertTrue(torch.equal(x,before));self.assertTrue(torch.equal(mark,mb))

    def test_U_before_PDM_preserves_preenc_trend_and_normalize_calls(self):
        p=small_profile();base=adapter.build(p,use_u=False,use_v=False,task='MS');m=adapter.build(p,use_u=True,use_v=False,task='MS');nonzero(m.u);x=torch.randn(2,12,11);marks=torch.randn(2,12,4);capt={}
        handles=[]
        for label,model in [('base',base),('on',m)]:
            def before_pdm(mod,args,label=label):capt[label]=[a.detach().clone()for a in args[0]]
            handles.append(model.pdm_blocks[0].register_forward_pre_hook(before_pdm));model.eval()
        originals=[]
        def wrap_pre(*args):
            values=original(*args);originals.append(tuple([t.detach().clone() for t in group] for group in values));return values
        for model in(base,m):
            original=model.pre_enc
            with patch.object(model,'pre_enc',side_effect=wrap_pre)as pre,patch.object(model,'_Model__multi_scale_process_inputs',wraps=model._Model__multi_scale_process_inputs)as scales,patch.object(model.normalize_layers[0],'forward',wraps=model.normalize_layers[0].forward)as norm:
                model(x,marks,None,None);self.assertEqual(pre.call_count,1);self.assertEqual(scales.call_count,1);self.assertEqual([c.args[1]for c in norm.call_args_list],['norm','denorm'])
        nested_equal(self,originals[0],originals[1]);update=m.u(x)[:,0]
        self.assertTrue(torch.equal(capt['base'][0]+update,capt['on'][0]));nested_equal(self,capt['base'][1:],capt['on'][1:])
        for h in handles:h.remove()

    def test_V_only_final_finest_and_native_MS_non_target_changes_allowed(self):
        p=small_profile();m=adapter.build(p,use_u=False,use_v=True,task='MS');nonzero(m.v);m.eval();capt={};x=torch.randn(2,12,11);marks=torch.randn(2,12,4)
        h=m.pdm_blocks[-1].register_forward_hook(lambda mod,args,out:capt.update(last=[v.detach().clone()for v in out]))
        original=m.future_multi_mixing
        def future(b,enc,xlist):capt['future']=[v.detach().clone()for v in enc];return original(b,enc,xlist)
        with patch.object(m,'future_multi_mixing',side_effect=future)as fn,patch.object(m.normalize_layers[0],'forward',wraps=m.normalize_layers[0].forward)as norm:y=m(x,marks,None,None)
        self.assertEqual(fn.call_count,1);nested_equal(self,capt['last'][1:],capt['future'][1:]);self.assertTrue(torch.equal(m.v(capt['last'][0]),capt['future'][0]));self.assertEqual(tuple(y.shape),(2,1,11));self.assertEqual([c.args[1]for c in norm.call_args_list],['norm','denorm']);h.remove()
        self.assertTrue(torch.equal(adapter.prediction(m,x,marks),y[:,:,:1]))

    def test_real_profiles_F1_F4_EPF_target2_M_long_P_and_variable_D(self):
        for variant,C in [('F1',6),('F2',8),('F3',9),('F4',11)]:
            p=small_profile(variant=variant,H=12);m=adapter.build(p,use_u=True,use_v=True,task='MS');m.eval();self.assertEqual(tuple(adapter.prediction(m,torch.randn(2,12,C),torch.randn(2,12,4)).shape),(2,1,1));self.assertEqual(m.u.d_model,8)
        for ds,var,H in [('PJM','MS',24),('ETTh1','M',96),('ETTh1','M',720)]:
            p=small_profile(ds,var,H);m=adapter.build(p,use_u=True,use_v=True,task='M'if var=='M'else'MS');m.eval();y=adapter.prediction(m,torch.randn(2,96,p['C']),torch.randn(2,96,p['time_mark']['K']));self.assertEqual(tuple(y.shape),(2,H,p['C']if var=='M'else 1));self.assertEqual(p['target_idx'],2) if ds=='PJM' else self.assertEqual(len(m.u.receivers),p['C'])

    def test_M_b_major_variable_minor_U_injection_and_historical_marks_order(self):
        p=small_profile('ETTh1','M',96);m=adapter.build(p,use_u=True,use_v=False,task='M');base=adapter.build(p,use_u=False,use_v=False,task='M');nonzero(m.u);m.eval();base.eval();x=torch.randn(2,96,7);marks=torch.randn(2,96,4);capt={};handles=[]
        for label,model in [('on',m),('base',base)]:
            handles.append(model.pdm_blocks[0].register_forward_pre_hook(lambda mod,args,label=label:capt.update({label:[v.clone()for v in args[0]]})))
            handles.append(model.enc_embedding.register_forward_pre_hook(lambda mod,args,label=label:capt.setdefault(label+'_marks',[]).append(args[1].clone())))
        m(x,marks,None,None);base(x,marks,None,None);update=m.u(x).reshape(14,96,8);self.assertTrue(torch.equal(capt['base'][0]+update,capt['on'][0]));self.assertTrue(torch.equal(capt['on_marks'][0],marks.repeat(7,1,1)))
        for h in handles:h.remove()

    def test_previous_revision_strict_refusal_before_state_write(self):
        p=small_profile();m=adapter.build(p,use_u=True,use_v=True,task='MS');c=adapter.synthetic_checkpoint(m);before=copy.deepcopy(m.state_dict())
        for version in(1,2,3,4):
            bad=copy.deepcopy(c);bad['spec']['candidate_id']='timemixer-uv-dev-v'+str(version)
            with self.assertRaises(ValueError):adapter.load_synthetic_checkpoint(m,bad)
            nested_equal(self,before,m.state_dict())


class ModuleFormula(unittest.TestCase):
    def test_U_independent_patch_head_source_null_restore_formula(self):
        for L,mode,C,target,aux in[(1,'MS',4,2,[3,0]),(4,'MS',4,2,[3,0]),(7,'M',3,0,[1,2])]:
            m=U(L,5,C,mode,target,aux).double();nonzero(m);x=torch.randn(2,L,C,dtype=torch.float64);actual,d=m(x,return_diagnostics=True);expected=u_reference(m,x);torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
            self.assertEqual(tuple(actual.shape),(2,1 if mode=='MS'else C,L,5));torch.testing.assert_close(d['alpha'].sum(-1),torch.ones_like(d['alpha'].sum(-1)),rtol=1e-12,atol=1e-12)
            if mode=='M':self.assertTrue(torch.equal(d['alpha'][...,:-1].diagonal(dim1=-2,dim2=-1),torch.zeros_like(d['alpha'][...,:-1].diagonal(dim1=-2,dim2=-1))))
            else:self.assertEqual(d['source_indices'],(3,0))

    def test_U_empty_constant_near_constant_and_large_finite(self):
        for mode,C,aux in[('MS',1,[]),('M',1,[])]:
            m=U(1,7,C,mode,0,aux);nonzero(m);x=torch.ones(2,1,C)*1e6;self.assertTrue(torch.equal(m(x),torch.zeros(2,1,1,7)))
        m=U(5,3,3,'MS',0,[2,1]);nonzero(m)
        for x in(torch.ones(2,5,3),torch.ones(2,5,3)+torch.arange(5).view(1,5,1)*1e-6,torch.ones(2,5,3)*1e6):self.assertTrue(torch.isfinite(m(x)).all())
        d=m(torch.arange(30.).reshape(2,5,3),return_diagnostics=True)[1];self.assertTrue(torch.equal(d['raw_patch'][0,0,:,0],torch.arange(3.)))

    def test_V_independent_depthwise_LN_FFN_formula_short_and_no_batch_mix(self):
        for L in(1,2,12):
            m=V(5).double();nonzero(m);g=torch.randn(3,L,5,dtype=torch.float64);mu=g.mean(-1,keepdim=True);s=torch.sqrt(((g-mu)**2).mean(-1,keepdim=True)+1e-5);a=((g-mu)/s*m.norm.weight+m.norm.bias).transpose(1,2);q=torch.zeros_like(a)
            for layer,kernel in[(m.dw7,7),(m.dw3,3)]:
                padded=F.pad(a,(kernel//2,kernel//2),mode='replicate')
                for i in range(kernel):q=q+padded[:,:,i:i+L]*layer.weight[:,0,i][None,:,None]
            expected=g+F.linear(F.gelu(F.linear(q.transpose(1,2),m.expand.weight,m.expand.bias)),m.project.weight,m.project.bias);torch.testing.assert_close(m(g),expected,rtol=1e-12,atol=1e-12)
            changed=g.clone();changed[0]+=3;self.assertTrue(torch.equal(m(g)[1:],m(changed)[1:]));self.assertEqual(m.dw7.groups,5)

    def test_zero_last_layers_and_delayed_gradients_all_module_parameters(self):
        p=small_profile();x=torch.randn(4,12,11);marks=torch.randn(4,12,4);target=torch.randn(4,1,1)
        for u,v in[(True,False),(False,True),(True,True)]:
            m=adapter.build(p,use_u=u,use_v=v,task='MS');m.train();o=torch.optim.Adam(m.parameters(),lr=.002);seen={n:False for n,q in m.named_parameters()if n.startswith(('u.','v.'))}
            if u:self.assertTrue(torch.equal(m.u(x),torch.zeros(4,1,12,8)))
            if v:
                g=torch.randn(4,12,8);self.assertTrue(torch.equal(m.v(g),g))
            for step in range(4):
                o.zero_grad(set_to_none=True);loss=F.mse_loss(adapter.prediction(m,x,marks),target);loss.backward()
                for n,q in m.named_parameters():
                    if n in seen:
                        self.assertIsNotNone(q.grad);self.assertTrue(torch.isfinite(q.grad).all());seen[n]|=bool(torch.count_nonzero(q.grad))
                        if step==0 and not n.startswith(('u.bridge.','v.project.')):self.assertFalse(bool(torch.count_nonzero(q.grad)),n)
                o.step()
            self.assertTrue(all(seen.values()),[n for n,v in seen.items()if not v])


if __name__=='__main__':unittest.main(verbosity=2)
