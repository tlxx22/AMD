"""v6 literal source comparison, independent formulas, BN and delayed gradients."""
import ast,copy,hashlib,math,unittest
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from tests.timemixer_v6_fixtures import profile,small_profile,nested_equal
from timemixer_dev_v6 import adapter
from timemixer_dev_v6 import author_components as ac
from timemixer_dev_v6.axis_ops import rearrange,repeat
from timemixer_dev_v6.source import binding
from timemixer_dev_v6.modules import SegmentedVariableInteraction as U,LatentTemporalRefinement as V

def source_classes(path,names):
    text=Path(path).read_text();tree=ast.parse(text);nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))and n.name in names]
    ns=dict(torch=torch,nn=nn,F=F,sqrt=math.sqrt,rearrange=rearrange,repeat=repeat)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns,nodes

def attention_ref(m,q,k,v):
    H=m.n_heads;B,L,D=q.shape;S=k.shape[1]
    Q=F.linear(q,m.query_projection.weight,m.query_projection.bias).reshape(B,L,H,-1)
    K=F.linear(k,m.key_projection.weight,m.key_projection.bias).reshape(B,S,H,-1)
    Vv=F.linear(v,m.value_projection.weight,m.value_projection.bias).reshape(B,S,H,-1)
    rows=[]
    for b in range(B):
        hs=[]
        for h in range(H):
            weights=torch.softmax(Q[b,:,h]@K[b,:,h].T/math.sqrt(Q.shape[-1]),-1)
            hs.append(weights@Vv[b,:,h])
        rows.append(torch.stack(hs,1).reshape(L,-1))
    return F.linear(torch.stack(rows),m.out_projection.weight,m.out_projection.bias)

def nonzero_v(m):
    with torch.no_grad():
        m.operator.expanded.bn.weight.copy_(torch.linspace(-.1,.2,m.d_model))
        m.operator.expanded.bn.bias.fill_(.03)

class Mechanisms(unittest.TestCase):
    def test_leaf_source_bodies_and_actual_module_boundary(self):
        b=binding();root=Path(b['Crossformer']['root'])
        am={n.name:n for n in ast.parse(Path(ac.__file__).read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
        expected={'DSW_embedding','FullAttention','AttentionLayer','get_conv1d','get_bn','conv_bn'}
        self.assertEqual(set(am),expected)
        for path,names in [(root/'cross_models/attn.py',{'FullAttention','AttentionLayer'}),(root/'cross_models/cross_embed.py',{'DSW_embedding'}),(Path(b['ModernTCN']['root'])/'ModernTCN-Long-term-forecasting/models/ModernTCN.py',{'get_conv1d','get_bn','conv_bn'})]:
            _,nodes=source_classes(path,names)
            for n in nodes:self.assertEqual(ast.dump(n,include_attributes=False),ast.dump(am[n.name],include_attributes=False))
        u=U(7,5,3,'MS',1,[2,0]);v=V(5)
        self.assertEqual(set(u._modules),{'enc_value_embedding','pre_norm','dim_sender','dim_receiver','bridge'})
        self.assertEqual(set(v._modules),{'operator'});self.assertEqual(set(v.operator._modules),{'local','activation','expanded'})
        forbidden=('TwoStageAttentionLayer','Block','SerialTemporalBlock','ReparamLargeKernelConv')
        self.assertFalse(any(type(m).__name__ in forbidden for m in list(u.modules())+list(v.modules())))
        for key in list(u.state_dict())+list(v.state_dict()):
            self.assertFalse(any(x in key for x in ('time_attention','norm1','norm2','norm3','norm4','MLP','ffn','block.','statistics','null')))
        calls=[]
        handles=[m.register_forward_hook(lambda m,a,o,n=n:calls.append(n))for n,m in u.named_modules()if n in ('enc_value_embedding','pre_norm','dim_sender','dim_receiver','bridge')]
        u(torch.randn(2,7,3))
        for h in handles:h.remove()
        self.assertEqual(calls,['enc_value_embedding','pre_norm','dim_sender','dim_receiver','bridge'])

    def test_DSW_left_pad_pre_norm_and_router_independent_formula(self):
        for mode,C,target,aux in [('MS',4,2,[3,0]),('M',3,1,[0,2])]:
            m=U(4,5,C,mode,target,aux).double();x=torch.randn(2,4,C,dtype=torch.float64)
            with torch.no_grad():m.bridge.weight.normal_(0,.03)
            actual,d=m(x,return_diagnostics=True);pad=torch.cat([x[:,:1].expand(-1,2,-1),x],1)
            embedded=torch.stack([torch.stack([torch.stack([F.linear(pad[i,n*3:n*3+3,c],m.enc_value_embedding.linear.weight,m.enc_value_embedding.linear.bias)for n in range(2)])for c in range(C)])for i in range(2)])
            expected=F.layer_norm(embedded+m.enc_pos_embedding,(32,),m.pre_norm.weight,m.pre_norm.bias,1e-5)
            torch.testing.assert_close(d['embedding'],expected,rtol=1e-12,atol=1e-12)
            E=expected.permute(0,2,1,3);outputs=[]
            for j,r in enumerate(m.receivers):
                legal=list(m.sources[j]);self.assertNotIn(r,legal)
                q=(m.router[None]+E[:,:,r,None,:]).reshape(4,2,32);kv=E[:,:,legal,:].reshape(4,len(legal),32)
                buf=attention_ref(m.dim_sender,q,kv,kv)
                torch.testing.assert_close(buf.reshape(2,2,2,32),d['buffer'][:,:,j],rtol=1e-12,atol=1e-12)
                msg=attention_ref(m.dim_receiver,E[:,:,r,:].reshape(4,1,32),buf,buf)
                torch.testing.assert_close(msg.reshape(2,2,32),d['message'][:,j],rtol=1e-12,atol=1e-12)
                outputs.append(F.linear(msg,m.bridge.weight).reshape(2,6,5)[:,2:])
            torch.testing.assert_close(actual,torch.stack(outputs,1),rtol=1e-12,atol=1e-12)
        # Target conditioning changes the sender's buffer before receiver attention.
        m=U(6,4,3,'MS',1,[2,0]).double();x=torch.randn(2,6,3,dtype=torch.float64);xx=x.clone();xx[:,:,1]+=torch.linspace(-2,3,6)
        _,a=m(x,return_diagnostics=True);_,b=m(xx,return_diagnostics=True)
        self.assertTrue(torch.equal(a['router_state'][:,:,[2,0]],b['router_state'][:,:,[2,0]]))
        self.assertFalse(torch.equal(a['conditioned_router'],b['conditioned_router']));self.assertFalse(torch.equal(a['buffer'],b['buffer']))

    def test_U_empty_sources_short_left_pad_constant_finite(self):
        for mode in ['MS','M']:
            m=U(1,4,1,mode,0,[])
            with torch.no_grad():m.bridge.weight.fill_(1)
            calls=[];h=m.dim_sender.register_forward_hook(lambda *a:calls.append(1))
            self.assertTrue(torch.equal(m(torch.ones(2,1,1)),torch.zeros(2,1,1,4)));h.remove();self.assertEqual(calls,[])
        for L in [1,2,4,7]:
            m=U(L,4,3,'MS',2,[0,1])
            with torch.no_grad():m.bridge.weight.fill_(.01)
            for value in [0.,1.,1e4]:self.assertTrue(torch.isfinite(m(torch.full((2,L,3),value))).all())

    def test_serial_operator_and_skip_independent_formula(self):
        m=V(5).double().eval();nonzero_v(m);x=torch.randn(2,4,5,dtype=torch.float64)
        def cb(v,c):return F.batch_norm(F.conv1d(v,c.conv.weight,None,stride=1,padding=c.conv.padding,dilation=c.conv.dilation,groups=5),c.bn.running_mean,c.bn.running_var,c.bn.weight,c.bn.bias,training=False,momentum=.1,eps=1e-5)
        z=x.transpose(1,2);q=cb(F.gelu(cb(z,m.operator.local)),m.operator.expanded)
        torch.testing.assert_close(q,m.operator(z),rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(m(x),x+q.transpose(1,2),rtol=1e-12,atol=1e-12)
        for conv,dilation,padding in [(m.operator.local.conv,1,1),(m.operator.expanded.conv,3,3)]:
            self.assertEqual(conv.groups,5);self.assertIsNone(conv.bias);self.assertEqual(conv.kernel_size,(3,));self.assertEqual(conv.dilation,(dilation,));self.assertEqual(conv.padding,(padding,));self.assertEqual(conv.padding_mode,'zeros')
        linear=cb(cb(z,m.operator.local),m.operator.expanded);self.assertFalse(torch.equal(q,linear));self.assertFalse(hasattr(m.operator,'merge_kernel'))

    def test_two_BN_train_eval_identity_running_buffers_and_small_boundary(self):
        m=V(5).double();g=torch.randn(3,4,5,dtype=torch.float64);self.assertTrue(torch.equal(m(g),g))
        bns=[a for a in m.modules()if isinstance(a,nn.BatchNorm1d)];self.assertEqual(len(bns),2)
        for bn in bns:self.assertEqual(bn.num_batches_tracked.item(),1);self.assertEqual((bn.eps,bn.momentum),(1e-5,.1));self.assertTrue(bn.affine and bn.track_running_stats)
        self.assertTrue(torch.equal(m.operator.local.bn.weight,torch.ones(5,dtype=torch.float64)))
        self.assertEqual(float(m.operator.expanded.bn.weight.abs().sum()),0);self.assertEqual(float(m.operator.expanded.bn.bias.abs().sum()),0)
        self.assertGreater(float(m.operator.local.conv.weight.abs().sum()),0);self.assertGreater(float(m.operator.expanded.conv.weight.abs().sum()),0)
        m.eval();nonzero_v(m);before=copy.deepcopy(m.state_dict());allout=m(g);one=m(g[:1]);torch.testing.assert_close(allout[:1],one,rtol=1e-12,atol=1e-12);nested_equal(self,before,m.state_dict())
        for L in [1,2]:self.assertTrue(torch.isfinite(m(torch.randn(2,L,5,dtype=torch.float64))).all())
        m.train()
        with self.assertRaises(ValueError):m(torch.ones(1,1,5,dtype=torch.float64))

    def test_common_B_rng_substreams_parameter_and_buffer_accounting(self):
        p=profile();base=adapter.build_original(p);rng=torch.get_rng_state().clone();us=vs=None
        for u,v in [(False,False),(True,False),(False,True),(True,True)]:
            m=adapter.build(p,use_u=u,use_v=v,task='MS');self.assertTrue(torch.equal(rng,torch.get_rng_state()));nested_equal(self,base.state_dict(),{k:v for k,v in m.state_dict().items()if not k.startswith(('u.','v.'))})
            expected=70885+u*(8640+64*4+32*11*4+96*64)+v*10*64;self.assertEqual(sum(a.numel()for a in m.parameters()),expected)
            if u:
                if us is None:us=copy.deepcopy(m.u.state_dict())
                else:nested_equal(self,us,m.u.state_dict())
            if v:
                if vs is None:vs=copy.deepcopy(m.v.state_dict())
                else:nested_equal(self,vs,m.v.state_dict())
                self.assertEqual(sum(b.numel()for b in m.v.buffers()),4*64+2)
                self.assertEqual(sum(b.numel()*b.element_size()for b in m.v.buffers()),4*64*4+2*8)

    def test_delayed_identifiable_gradients_finite_no_bias_nonzero_requirement(self):
        p=small_profile()
        for u,v in [(True,False),(False,True),(True,True)]:
            m=adapter.build(p,use_u=u,use_v=v,task='MS');opt=torch.optim.Adam(m.parameters(),lr=.003);x=torch.randn(3,12,11);marks=torch.randn(3,12,4);target=torch.randn(3,1,1)
            if u:self.assertEqual(float(m.u.bridge.weight.abs().sum()),0)
            if v:self.assertEqual(float(m.v.operator.expanded.bn.weight.abs().sum()),0)
            active=set()
            for step in range(4):
                m.train();opt.zero_grad(set_to_none=True);F.mse_loss(adapter.prediction(m,x,marks),target).backward()
                for name,par in m.named_parameters():
                    if par.grad is not None:
                        self.assertTrue(torch.isfinite(par.grad).all(),name)
                        if par.grad.abs().sum()>0:active.add(name)
                opt.step()
            required=[]
            if u:required+=['u.enc_value_embedding.linear.weight','u.enc_pos_embedding','u.router','u.dim_sender.query_projection.weight','u.dim_receiver.value_projection.weight','u.bridge.weight']
            if v:required+=['v.operator.local.conv.weight','v.operator.expanded.conv.weight','v.operator.local.bn.weight','v.operator.expanded.bn.weight']
            self.assertTrue(set(required)<=active,set(required)-active)

if __name__=='__main__':unittest.main(verbosity=2)
