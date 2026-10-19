"""v6 author-path and internal insertion regression, exact comparisons retained."""
import copy,unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from tests.timemixer_v6_fixtures import profile,small_profile,nested_equal
from timemixer_dev_v6 import adapter
from timemixer_dev_v6.modules import SegmentedVariableInteraction as U,LatentTemporalRefinement as V
from utils.ch3_onecycle import configuration,Schedule
def nonzero(m):
 with torch.no_grad():
  layer=m.bridge if isinstance(m,U)else m.operator.expanded.bn
  layer.weight.copy_(torch.linspace(-.13,.19,layer.weight.numel()).reshape_as(layer.weight))
  if layer.bias is not None:layer.bias.fill_(.07)
class AuthorAndIdentity(unittest.TestCase):
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
            old=copy.deepcopy(c);old['spec']['structural_revision']=dict(u='crossformer_full_TSA_receiver_conditioned_router_v6',v='moderntcn_Block_serial_nonlinear_local_dilated_v6',fusion='two_internal_finest_representation_residuals')
            with self.assertRaises(ValueError):adapter.load_synthetic_checkpoint(m,old)
            nested_equal(self,before,m.state_dict())
            for version in(1,2,3,4,5):
                bad=copy.deepcopy(c);bad['spec']['candidate_id']='timemixer-uv-dev-v'+str(version)
                with self.assertRaises(ValueError):adapter.load_synthetic_checkpoint(m,bad)
                nested_equal(self,before,m.state_dict())
