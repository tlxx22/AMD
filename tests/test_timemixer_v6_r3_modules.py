"""Independent synthetic equations and submodule boundaries, no real data."""
import copy,json,math,unittest
from unittest.mock import patch
import torch
from torch import nn
from timemixer_dev_v6_r3.modules import TargetConditionedAuxBasisGate as U,TemporalQKRawValueOperator as V,gated_native_projection
from tests.timemixer_v6_r3_fixtures import EVIDENCE


def active_u(length=5,D=8,C=4,mode='MS',target=2,aux=(3,0,1)):
    torch.manual_seed(17);u=U(length,D,C,mode,target,aux).double()
    with torch.no_grad():u.output.weight.fill_(.12);u.output.bias.fill_(.03)
    return u


class Equations(unittest.TestCase):
    def close(self,a,b):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
    def test_u_independent_scalar_formula_ordered_MS_nonzero_target(self):
        u=active_u();x=torch.randn(2,5,4,dtype=torch.float64);m,d=u(x,return_diagnostics=True);E=u.history(x.transpose(1,2));expected=torch.empty_like(m)
        for b in range(2):
            for dd in range(8):
                q=E[b,2,dd];logit=u.output.bias[0].clone()
                for k in range(8):
                    S=sum(u.mix_weight[k,j]*E[b,orig,dd] for j,orig in enumerate((3,0,1)))/3
                    kap=2*torch.sigmoid(u.kappa.weight[k,0]*q+u.kappa.bias[k]);h=torch.relu(kap*S+u.target_term.weight[k,0]*q+u.target_term.bias[k]);logit=logit+u.output.weight[0,k]*h
                expected[b,0,dd]=2*torch.sigmoid(logit)
        self.close(m,expected);self.assertEqual(u.source_slot_indices.tolist(),[3,0,1]);self.assertEqual(u.receiver_order.tolist(),[2]);self.assertEqual(d['hidden'].shape,(2,1,8,8));self.assertTrue(torch.all((m>0)&(m<2)))
    def test_u_M_exclusive_prefix_suffix_formula_and_no_self_value(self):
        u=active_u(mode='M');E=torch.randn(2,4,8,dtype=torch.float64,requires_grad=True);actual=u.auxiliary_bases(E)
        for receiver in range(4):
            expected=sum(E[:,j,:,None]*u.mix_weight[:,j] for j in range(4) if j!=receiver)/3
            self.close(actual[:,receiver],expected)
            grad=torch.autograd.grad(actual[:,receiver].sum(),E,retain_graph=True)[0];self.assertTrue(torch.equal(grad[:,receiver],torch.zeros_like(grad[:,receiver])))
        self.assertEqual(actual.shape,(2,4,8,8))
    def test_u_target_modulates_mixture_and_aux_changes_gate(self):
        u=active_u();x=torch.arange(40,dtype=torch.float64).reshape(2,5,4)/10
        with torch.no_grad():u.history.weight.fill_(.1);u.history.bias.fill_(.2);u.mix_weight.fill_(.3);u.kappa.weight.fill_(.4);u.kappa.bias.zero_();u.target_term.weight.zero_();u.target_term.bias.fill_(.5)
        m,d=u(x,return_diagnostics=True);changed=x.clone();changed[:,:,2]+=2;mt,dt=u(changed,return_diagnostics=True)
        self.assertTrue(torch.equal(d['bases'],dt['bases']));self.assertFalse(torch.equal(d['kappa'],dt['kappa']));self.assertFalse(torch.equal(d['kappa']*d['bases'],dt['kappa']*dt['bases']));self.assertFalse(torch.equal(m,mt))
        changed=x.clone();changed[:,:,3]+=2;ma,da=u(changed,return_diagnostics=True);self.assertTrue(torch.equal(d['target'],da['target']));self.assertFalse(torch.equal(d['bases'],da['bases']));self.assertFalse(torch.equal(m,ma))
    def test_u_no_aux_skips_history_and_target_gate(self):
        for u,x in [(U(1,8,1,'M',0,[]),torch.randn(2,1,1)),(U(3,8,3,'MS',1,[]),torch.randn(2,3,3))]:
            with patch.object(u.history,'forward',side_effect=AssertionError('no-aux controller must be skipped')):m=u(x)
            self.assertTrue(torch.equal(m,torch.ones_like(m)))
        with self.assertRaises(ValueError):U(3,8,3,'MS',1,[0,1])
        with self.assertRaises(ValueError):U(3,8,3,'MS',1,[0,0])
    def test_u_native_CI0_and_CI1_reference_bias_P_sharing(self):
        torch.manual_seed(8);B,C,D,P=2,4,8,7;z=torch.randn(B,P,D,dtype=torch.float64);native=nn.Linear(D,C).double();gate=.2+torch.rand(B,1,D,dtype=torch.float64)
        result=gated_native_projection(z,native,gate,(2,),batch_size=B,channels=C,channel_independence=0);expected=native(z).clone()
        expected[:,:,2]=torch.stack([sum(native.weight[2,d]*gate[b,0,d]*z[b,:,d] for d in range(D))+native.bias[2] for b in range(B)])
        self.close(result,expected)
        for c in (0,1,3):self.assertTrue(torch.equal(result[:,:,c],native(z)[:,:,c]))
        shared=nn.Linear(D,1).double();gate=.3+torch.rand(B,C,D,dtype=torch.float64);carrier=torch.randn(B*C,P,D,dtype=torch.float64)
        actual=gated_native_projection(carrier,shared,gate,tuple(range(C)),batch_size=B,channels=C,channel_independence=1)
        for b in range(B):
            for c in range(C):
                ref=sum(shared.weight[0,d]*gate[b,c,d]*carrier[b*C+c,:,d] for d in range(D))+shared.bias[0];self.close(actual[b*C+c,:,0],ref)
        self.assertTrue(torch.equal(gated_native_projection(z,native,torch.ones(B,1,D,dtype=torch.float64),(2,),batch_size=B,channels=C,channel_independence=0),native(z)))
    def test_u_shape_memory_and_parameter_formula(self):
        for mode,C,aux in [('MS',11,list(range(1,11))),('M',21,list(range(1,21))),('MS',6,list(range(1,6)))]:
            u=U(12,64,C,mode,0,aux);slots=len(aux) if mode=='MS' else C
            self.assertEqual(sum(p.numel() for p in u.parameters()),64*13+8*slots+41)
            m,d=u(torch.randn(2,12,C),return_diagnostics=True);self.assertEqual(m.shape,(2,1 if mode=='MS' else C,64));self.assertEqual(d['hidden'].ndim,4)
            self.assertFalse(any(isinstance(q,(nn.LayerNorm,nn.BatchNorm1d,nn.Dropout)) for q in u.modules()))
    def test_v_true_L_relation_independent_heads_raw_value_formula(self):
        torch.manual_seed(11);v=V(8).double();v.a.data.fill_(.6);g=torch.randn(2,5,8,dtype=torch.float64);out,d=v(g,return_diagnostics=True)
        padded=torch.cat([g[:,:1].expand(-1,2,-1),g],1);ma=torch.stack([padded[:,i:i+3].mean(1) for i in range(5)],1);R=(g-ma).detach();self.close(d['relation'],R);self.assertFalse(d['relation'].requires_grad)
        q=v.query(R).reshape(2,5,4,8);k=v.key(R).reshape(2,5,4,8);expected=torch.empty_like(g)
        for b in range(2):
            for h in range(4):
                A=torch.softmax(q[b,:,h]@k[b,:,h].T/math.sqrt(8),dim=-1);expected[b,:,h*2:(h+1)*2]=A@g[b,:,h*2:(h+1)*2]
        self.close(d['redistributed'],expected);self.close(out,g+.5*torch.tanh(v.a)*(expected-g));self.close(d['attention'].sum(-1),torch.ones(2,4,5,dtype=torch.float64))
    def test_v_identity_signed_bound_L1_constant_and_component_tree(self):
        torch.manual_seed(4);v=V(16).double();g=torch.randn(2,7,16,dtype=torch.float64);self.assertTrue(torch.equal(v(g),g));self.assertTrue(v.query.weight.count_nonzero()>0)
        self.assertEqual(sum(p.numel() for p in v.parameters()),64*16+1);self.assertEqual(set(dict(v.named_children())),{'query','key'})
        for a in (-4.,0.,4.):
            v.a.data.fill_(a);y=v(g);self.assertLessEqual(y.abs().max().item(),2*g.abs().max().item());self.assertTrue(torch.equal(v(g[:,:1]),g[:,:1]))
        constant=torch.arange(16,dtype=torch.float64).reshape(1,1,16).expand(2,8,-1);self.assertTrue(torch.equal(v(constant),constant))
        nonbinary=constant[:,:7];result=v(nonbinary);self.close(result,nonbinary)
        (EVIDENCE/'repair-r1/constant-rounding.json').write_text(json.dumps(dict(dtype='float64',length=7,max_abs_difference=float((result-nonbinary).abs().max()),policy='pre-registered independent double equations atol=rtol=1e-12; exact alpha0/L1/length8 checks retained'))+'\n')
        v.a.data.fill_(-4.);r=torch.tensor([-1.,1.],dtype=torch.float64).reshape(1,2,1).expand(1,2,16);self.assertTrue((v(r).abs()>1).any())
        with self.assertRaises(ValueError):V(6)
    def test_v_detach_blocks_only_relation_input_not_QK_or_values(self):
        torch.manual_seed(18);v=V(8).double();v.a.data.fill_(.5);g=torch.randn(2,5,8,dtype=torch.float64,requires_grad=True);y,d=v(g,return_diagnostics=True);weights=torch.randn_like(y)
        actual=torch.autograd.grad((y*weights).sum(),(g,v.query.weight,v.key.weight),retain_graph=True);self.assertTrue(actual[1].abs().sum()>0);self.assertTrue(actual[2].abs().sum()>0)
        raw=g.reshape(2,5,4,2).transpose(1,2);ref=(d['attention'].detach()@raw).transpose(1,2).reshape_as(g);expected=torch.autograd.grad(((g+.5*torch.tanh(v.a)*(ref-g))*weights).sum(),g)[0];self.close(actual[0],expected)
    def test_delayed_gradient_channels_legal_then_finite_active(self):
        torch.manual_seed(21);u=U(5,8,4,'MS',2,[3,0,1]);x=torch.randn(3,5,4);opt=torch.optim.Adam(u.parameters(),lr=.005)
        for step in range(3):
            opt.zero_grad();loss=(u(x)-.65).square().mean();loss.backward()
            if step==0:self.assertTrue(torch.equal(u.history.weight.grad,torch.zeros_like(u.history.weight.grad)))
            if step==2:
                for name in ('history.weight','mix_weight','kappa.weight','target_term.weight','output.weight'):
                    grad=dict(u.named_parameters())[name].grad;self.assertTrue(torch.isfinite(grad).all());self.assertTrue(grad.abs().sum()>0,name)
            opt.step()
        v=V(8);g=torch.randn(3,7,8,requires_grad=True);target=torch.randn_like(g);opt=torch.optim.Adam(v.parameters(),lr=.005)
        for step in range(3):
            opt.zero_grad();loss=(v(g)-target).square().mean();loss.backward()
            if step==0:self.assertTrue(torch.equal(v.query.weight.grad,torch.zeros_like(v.query.weight.grad)))
            if step==2:
                for parameter in (v.query.weight,v.key.weight,v.a):self.assertTrue(torch.isfinite(parameter.grad).all());self.assertTrue(parameter.grad.abs().sum()>0)
            opt.step()

if __name__=='__main__':unittest.main(verbosity=2)
