"""Explicit CPU synthetic acceptance; never discovers data or checkpoints."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from torch.nn import functional as F

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from timemixer_dev.adapter import build, build_original, prediction, loss, synthetic_checkpoint, load_synthetic_checkpoint
from timemixer_dev.modules import SelectiveVariableInteraction as U, HistoryConditionedScaleAdjustment as V
from timemixer_dev.source import load_author, binding, implementation_binding
from timemixer_dev.spec import StructuralSpec
# Existing dedicated exact nested-state assertion, without concurrency policies.
from tests.test_runner import _assert_nested_equal

ROOT = Path(__file__).resolve().parents[1]
ARMS = ((False, False), (True, False), (False, True), (True, True))


def profiles():
    result = {}
    for name in ('ch3_type1_urban6_four_H_v1.json', 'ch3_type1_m_all_patchtst_enc2_dm4_selected_v1.json', 'ch3_type1_epf_all_v3.json'):
        config = json.loads((ROOT / 'configs' / name).read_text())
        for p in config['resolved_profiles'].values():
            if p['model'] == 'TimeMixer':
                result.setdefault(p['dataset'], p)
    return result


PROFILES = profiles()


def fixture_profile(dataset='UrbanEV', *, channels=None, length=None, pred_len=None, horizon=3, marks=True):
    p = copy.deepcopy(PROFILES[dataset])
    if dataset == 'UrbanEV':
        from utils.feature_schema import get_feature_schema
        c = channels or 11
        p['input_variant'] = {6: 'F1', 8: 'F2', 9: 'F3', 11: 'F4'}[c]
        p['features'] = list(get_feature_schema(p['input_variant']).feature_names)
        p.update(C=c, target_idx=0, aux_idx=list(range(1,c)), H=horizon)
    if length is not None:
        p['T'] = length
    if pred_len is not None:
        p['pred_len'] = pred_len
        if dataset != 'UrbanEV':
            p['H'] = pred_len
    if not marks:
        p.pop('time_mark', None)
    return p


def inputs(p):
    torch.random.default_generator.manual_seed(391)
    x = torch.randn(2, p['T'], p['C']) + torch.arange(p['C'])[None,None,:] / 3
    mark = torch.randn(2, p['T'], p['time_mark']['K']) if p.get('time_mark') else None
    return x, mark


class AIdentity(unittest.TestCase):
    def test_off_exact_eval_train_gradient_adam_and_rng(self):
        for dataset in ('UrbanEV', 'ETTh1', 'PJM'):
            with self.subTest(dataset=dataset):
                p = fixture_profile(dataset)
                original = build_original(p)
                rng_original = torch.get_rng_state().clone()
                off = build(p, use_u=False, use_v=False)
                self.assertTrue(torch.equal(rng_original, torch.get_rng_state()))
                self.assertFalse(hasattr(off, 'u') or hasattr(off, 'v'))
                self.assertEqual([(k,tuple(t.shape)) for k,t in original.state_dict().items()],
                                 [(k,tuple(t.shape)) for k,t in off.state_dict().items()])
                _assert_nested_equal(self, original.state_dict(), off.state_dict())
                x, marks = inputs(p)
                original.eval(); off.eval()
                _assert_nested_equal(self, original(x,marks,None,None), off(x,marks,None,None))
                original.train(); off.train()
                o1, o2 = torch.optim.Adam(original.parameters(), lr=1e-3), torch.optim.Adam(off.parameters(), lr=1e-3)
                rng = torch.get_rng_state().clone()
                y1 = original(x,marks,None,None)
                l1 = y1.square().mean(); l1.backward(); after1 = torch.get_rng_state().clone()
                torch.set_rng_state(rng)
                y2 = off(x,marks,None,None)
                l2 = y2.square().mean(); l2.backward()
                _assert_nested_equal(self, (y1,l1,after1), (y2,l2,torch.get_rng_state()))
                _assert_nested_equal(self, {k:p.grad for k,p in original.named_parameters()}, {k:p.grad for k,p in off.named_parameters()})
                o1.step(); o2.step()
                _assert_nested_equal(self, original.state_dict(), off.state_dict())
                _assert_nested_equal(self, o1.state_dict(), o2.state_dict())

    def test_module_substreams_match_and_leave_backbone_rng(self):
        p = fixture_profile()
        models = [build(p,use_u=u,use_v=v) for u,v in ARMS]
        states = [m.state_dict() for m in models]
        for state in states[1:]:
            _assert_nested_equal(self, states[0], {k:v for k,v in state.items() if not k.startswith(('u.','v.'))})
        _assert_nested_equal(self, models[1].u.state_dict(), models[3].u.state_dict())
        _assert_nested_equal(self, models[2].v.state_dict(), models[3].v.state_dict())
        torch.random.default_generator.manual_seed(2024)
        build_original(p); expected = torch.get_rng_state().clone()
        for u,v in ARMS:
            build(p,use_u=u,use_v=v)
            _assert_nested_equal(self, expected, torch.get_rng_state())

    def test_enabled_forecast_noop_exact_author_marks_and_cache(self):
        # No-op insertion proves the copied enabled forecast retains all B math.
        for dataset in ('UrbanEV','ETTh1'):
            for marks in (False,True):
                p=fixture_profile(dataset,marks=marks); b=build_original(p).eval()
                m=build(p,use_u=True,use_v=True).eval();x,mark=inputs(p)
                with torch.no_grad():
                    m.u.gamma.zero_();m.v.p.weight.zero_()
                    _assert_nested_equal(self,b(x,mark,None,None),m(x,mark,None,None))
                for i in range(len(b.normalize_layers)):
                    _assert_nested_equal(self,(b.normalize_layers[i].mean,b.normalize_layers[i].stdev),
                                             (m.normalize_layers[i].mean,m.normalize_layers[i].stdev))

    def test_existing_adapter_original_path_exact(self):
        # Separate process confines the legacy adapter's intentional aliases.
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='.uv-synthetic-') as temp:
            out=Path(temp)/'original.pt'
            body="""import json,torch,sys
from pathlib import Path
from models.ch3_adapter import build
c=json.loads(Path('configs/ch3_type1_urban6_four_H_v1.json').read_text())
t=next(t for t in c['tasks'] if t['model']=='TimeMixer' and t['fold']==1 and t['h']==3)
m=build(c,t)
torch.save(m.state_dict(),sys.argv[1])
"""
            r=subprocess.run([sys.executable,'-B','-c',body,str(out)],cwd=ROOT,env=os.environ.copy(),capture_output=True,text=True,timeout=30)
            self.assertEqual(r.returncode,0,r.stderr)
            _assert_nested_equal(self,torch.load(out,weights_only=True),build(fixture_profile(),use_u=False,use_v=False).state_dict())


class BVariable(unittest.TestCase):
    def setUp(self):
        torch.random.default_generator.manual_seed(2024)

    def test_ms_target_zero_nonzero_untouched_and_order(self):
        for target,aux in ((0,[3,1]),(2,[4,0])):
            u=U(12,5,'MS',target,aux);x=torch.randn(2,12,5);z=torch.randn_like(x); before=z.clone()
            y,d=u(z,x,return_diagnostics=True)
            others=[j for j in range(5) if j!=target]
            self.assertTrue(torch.equal(y[:,:,others],z[:,:,others]))
            self.assertTrue(torch.equal(z,before));self.assertFalse(torch.equal(y[:,:,target],z[:,:,target]))
            self.assertEqual(u.sources,tuple(aux))
            self.assertEqual(d['alpha'].shape,(2,1,6,16))
            for j in set(range(5))-set(aux):self.assertTrue((d['alpha'][:,:,j]==0).all())

    def test_illegal_indices_rejected(self):
        for target,aux in ((-1,[1]),(5,[1]),(0,[0]),(0,[1,1]),(0,[5]),(0,[-1]),(0,[True]),(True,[1])):
            with self.subTest(target=target,aux=aux),self.assertRaises(ValueError):U(12,5,'MS',target,aux)

    def test_m_diagonal_simultaneous_float64_receiver_reference(self):
        u=U(5,4,'M',1,[0,2,3]).double();z=torch.randn(2,5,4,dtype=torch.float64);x=torch.randn_like(z)
        y,d=u(z,x,return_diagnostics=True);e=d['embeddings']; values=u.v2(F.gelu(u.v1(e)))
        reference=z.clone()
        for r in range(4):
            js=[j for j in range(4) if j!=r]
            logits=u.ao(F.gelu(u.aq(e[:,r:r+1,:])+u.ak(e[:,js,:])+u.ba))
            alpha=torch.cat((logits,logits.new_zeros(2,1,16)),dim=1).softmax(dim=1)
            core=(alpha[:,:-1]*values[:,js,:]).sum(dim=1)
            delta=torch.sigmoid(u.gate(torch.cat((e[:,r],core),dim=-1)))*u.out(core)
            reference[:,:,r]=z[:,:,r]+u.gamma*delta
            self.assertTrue((d['alpha'][:,r,r,:]==0).all())
        torch.testing.assert_close(y,reference,rtol=1e-12,atol=1e-12)

    def test_empty_sources_exact_bypass(self):
        for task,c,target,aux in [('MS',3,2,[]),('M',1,0,[])]:
            u=U(12,c,task,target,aux);x=torch.randn(2,12,c)
            self.assertIs(u(x,x),x)

    def test_constant_and_nearzero_histories_finite(self):
        for task in ('M','MS'):
            u=U(12,3,task,1,[2,0]);z=torch.zeros(2,12,3)
            for scale in (0.,1e-4):
                x=torch.full_like(z,17.)+scale*torch.randn_like(z)
                y=u(z,x);self.assertTrue(torch.isfinite(y).all())
                y.square().mean().backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in u.parameters()))

    def test_null_available_and_zero_core_exact_delta(self):
        u=U(12,4,'MS',2,[3,0]);z=torch.randn(2,12,4)
        _,d=u(z,z,return_diagnostics=True)
        self.assertTrue((d['alpha'][:,:,-1]>0).all())
        with torch.no_grad():u.v2.weight.zero_();u.v2.bias.zero_()
        y,d=u(z,z,return_diagnostics=True)
        self.assertTrue((d['core']==0).all());self.assertTrue((d['delta']==0).all());self.assertTrue(torch.equal(y,z))
        self.assertIsNone(u.out.bias)

    def test_receiver_and_source_change_selection(self):
        u=U(12,4,'MS',2,[3,0]);z=torch.randn(2,12,4);x=torch.randn_like(z)
        _,d=u(z,x,return_diagnostics=True)
        for j in (2,3):
            changed=z.clone();changed[:,:,j]+=torch.linspace(-2,4,12)
            _,e=u(changed,x,return_diagnostics=True)
            self.assertFalse(torch.equal(d['alpha'],e['alpha']))
        # Identical normalized shape but distinct level/volatility still matters.
        _,e=u(z,x*2+4,return_diagnostics=True)
        self.assertFalse(torch.equal(d['alpha'],e['alpha']))

    def test_stats_projection_outside_ln_float64(self):
        u=U(12,4,'MS',2,[0,3]).double();z=torch.randn(2,12,4,dtype=torch.float64);x=torch.randn_like(z)
        _,d=u(z,x,return_diagnostics=True)
        stats=torch.stack((x.mean(1),torch.log1p(torch.sqrt(x.var(1,unbiased=False)+1e-5))),dim=-1)
        ref=u.norm(u.linear_z(z.transpose(1,2)))+u.linear_a(stats)
        self.assertTrue(torch.equal(d['embeddings'],ref))


class CScale(unittest.TestCase):
    def make(self,task='M',scales=3,pred_len=5,c=4,d=6):
        torch.random.default_generator.manual_seed(2024)
        return V(d,scales,pred_len,c,task,2 if c>2 else 0,[j for j in range(c) if j!=(2 if c>2 else 0)],int(task=='M'))

    def states(self,v,dtype=torch.float32):
        return [torch.randn(2*(v.channels if v.ci else 1),l,v.d_model,dtype=dtype,requires_grad=True) for l in (8,3,1)[:v.scales]]

    def test_ci_views_and_descriptor_axes_float64(self):
        for task in ('MS','M'):
            v=self.make(task).double();states=self.states(v,torch.float64);actual=v.descriptors(states,2)
            for s,h in enumerate(states):
                for b in range(2):
                    for r in range(len(v.receivers)):
                        g=h[b*v.channels+r] if v.ci else h[b]
                        q=(g[1:]-g[:-1]).abs().mean(0) if len(g)>1 else torch.zeros_like(g[0])
                        ref=torch.cat((g.mean(0),g[-1],q))
                        self.assertTrue(torch.equal(actual[b,r,s],ref))

    def test_softmax_axis_weights_sum_boundaries_and_state_response(self):
        for task in ('MS','M'):
            v=self.make(task);states=self.states(v);pi=v.probabilities(states,2)
            torch.testing.assert_close(pi.sum(2),torch.ones_like(pi.sum(2)),atol=1e-7,rtol=1e-7)
            w=1-v.rho+v.rho*v.scales*pi
            torch.testing.assert_close(w.sum(2),torch.full_like(w.sum(2),v.scales),atol=5e-7,rtol=1e-7)
            self.assertTrue((w>=0.5).all() and (w<=0.5+0.5*v.scales).all())
            changed=[s.clone() for s in states];changed[1]=changed[1]*3+2
            self.assertFalse(torch.equal(pi,v.probabilities(changed,2)))
            self.assertEqual(v.rho,0.5);self.assertIsNone(v.p.bias)

    def test_uniform_recovers_sum_and_single_scale_zero(self):
        for s in (1,3):
            v=self.make(scales=s);states=self.states(v);ys=[torch.randn(2,5,4) for _ in range(s)]
            if s > 1:
                with torch.no_grad():v.p.weight.zero_()
            y,d=v(states,ys,return_diagnostics=True)
            self.assertTrue(torch.equal(y,torch.stack(ys,-1).sum(-1)))
            self.assertTrue((d['correction']==0).all())

    def test_ms_target_and_m_channel_prediction_axes_reference(self):
        for task in ('MS','M'):
            v=self.make(task).double();states=self.states(v,torch.float64);ys=[torch.randn(2,5,4,dtype=torch.float64) for _ in range(3)]
            y,d=v(states,ys,return_diagnostics=True);base=torch.stack(ys,-1).sum(-1);ref=base.clone()
            for ri,r in enumerate(v.receivers):
                for b in range(2):
                    for p in range(5):
                        ref[b,p,r]+=0.5*sum((3*d['pi'][b,ri,s,p]-1)*ys[s][b,p,r] for s in range(3))
            torch.testing.assert_close(y,ref,atol=1e-12,rtol=1e-12)
            if task=='MS':self.assertTrue(torch.equal(y[:,:,[0,1,3]],base[:,:,[0,1,3]]))

    def test_history_gradients_not_detached(self):
        v=self.make();states=self.states(v);ys=[torch.randn(2,5,4,requires_grad=True) for _ in range(3)]
        v(states,ys).square().mean().backward()
        for h in states:self.assertTrue(torch.isfinite(h.grad).all() and h.grad.abs().sum()>0)

    def test_no_persistent_graph_diagnostics(self):
        v=self.make();states=self.states(v);ys=[torch.randn(2,5,4) for _ in range(3)]
        names=set(vars(v));v(states,ys);self.assertEqual(names,set(vars(v)))
        self.assertFalse(any(torch.is_tensor(t) and t.grad_fn is not None for t in vars(v).values()))


class DIntegration(unittest.TestCase):
    def test_gradient_groups_and_three_adam_steps(self):
        for dataset in ('UrbanEV','ETTh1','PJM'):
            for u,v in ARMS[1:]:
                with self.subTest(dataset=dataset,u=u,v=v):
                    p=fixture_profile(dataset,pred_len=None if dataset=='UrbanEV' else 24)
                    m=build(p,use_u=u,use_v=v);x,marks=inputs(p);opt=torch.optim.Adam(m.parameters(),lr=1e-3)
                    before={n:t.detach().clone() for n,t in m.named_parameters() if n.startswith(('u.','v.'))}
                    for step in range(3):
                        opt.zero_grad(set_to_none=True);y=prediction(m,x,marks);target=torch.randn_like(y)
                        value=loss(y,target);self.assertTrue(torch.isfinite(value));value.backward()
                        for n,t in m.named_parameters():
                            if n.startswith(('u.','v.')):
                                self.assertIsNotNone(t.grad,n);self.assertTrue(torch.isfinite(t.grad).all(),n)
                                self.assertGreater(t.grad.abs().sum().item(),0,n)
                        opt.step()
                    for n,t in m.named_parameters():
                        if n in before:self.assertFalse(torch.equal(t,before[n]),n)

    def test_norm_once_and_original_input_statistics_with_insertions(self):
        for dataset in ('UrbanEV','ETTh1'):
            p=fixture_profile(dataset);m=build(p,use_u=True,use_v=True);x,marks=inputs(p)
            raw,_=m._Model__multi_scale_process_inputs(x,marks)
            calls=[];handles=[];pdm=[];fmm=[]
            for i,norm in enumerate(m.normalize_layers):
                handles.append(norm.register_forward_pre_hook(lambda module,args,i=i:calls.append((i,args[1]))))
            handles.append(m.pdm_blocks[-1].register_forward_hook(lambda mod,args,out:pdm.append(out)))
            handles.append(m.v.register_forward_pre_hook(lambda mod,args:fmm.append(args)))
            try:prediction(m,x,marks)
            finally:
                for h in handles:h.remove()
            self.assertEqual(calls,[(i,'norm') for i in range(len(raw))]+[(0,'denorm')])
            self.assertIs(fmm[0][0],pdm[0])
            for norm,window in zip(m.normalize_layers,raw):
                self.assertTrue(torch.equal(norm.mean,window.mean(1,keepdim=True)))
                self.assertTrue(torch.equal(norm.stdev,torch.sqrt(window.var(1,keepdim=True,unbiased=False)+1e-5)))

    def test_ms_nonzero_target_prediction_and_loss_broadcast_rejection(self):
        p=fixture_profile('PJM');m=build(p,use_u=True,use_v=True);x,mark=inputs(p)
        m.eval();y=prediction(m,x,mark);full=m(x,mark,None,None)
        self.assertTrue(torch.equal(y,full[:,:,2:3]))
        with self.assertRaises(ValueError):loss(y,torch.zeros(2,24))
        with self.assertRaises(ValueError):prediction(m,x,None)

    def test_ett_minute_native_marks_profile(self):
        for dataset in ('ETTm1','ETTm2'):
            p=fixture_profile(dataset,pred_len=96);m=build(p,use_u=True,use_v=True);x,marks=inputs(p)
            self.assertEqual(marks.shape[-1],5);self.assertEqual(prediction(m,x,marks).shape,(2,96,7))


def shape_test(dataset,channels,length,pred_len,horizon,marks,arm):
    def test(self):
        p=fixture_profile(dataset,channels=channels,length=length,pred_len=pred_len,horizon=horizon,marks=marks)
        m=build(p,use_u=arm[0],use_v=arm[1]).eval();x,mark=inputs(p)
        with torch.no_grad():y=prediction(m,x,mark)
        self.assertEqual(tuple(y.shape),(2,p['pred_len'],p['C'] if p.get('task')=='M' else 1))
        self.assertTrue(torch.isfinite(y).all())
    return test


for c in (6,8,9,11):
    for h in (3,6,9,12):
        for marks in (False,True):
            for a,arm in enumerate(ARMS):
                setattr(DIntegration,f'test_urban_C{c}_H{h}_marks{int(marks)}_arm{a}',shape_test('UrbanEV',c,12,1,h,marks,arm))
for dataset,c in (('ETTh1',7),('Exchange',8),('Weather',21)):
    for p in (96,192,336,720):
        for marks in (False,True):
            for a,arm in enumerate(ARMS):
                setattr(DIntegration,f'test_m_C{c}_P{p}_marks{int(marks)}_arm{a}',shape_test(dataset,c,96,p,3,marks,arm))
for length in (96,168):
    for marks in (False,True):
        for a,arm in enumerate(ARMS):
            setattr(DIntegration,f'test_epf_L{length}_marks{int(marks)}_arm{a}',shape_test('PJM',3,length,24,3,marks,arm))


class EEngineering(unittest.TestCase):
    def test_existing_task_resolution_preserves_structure_and_fold(self):
        from timemixer_dev.adapter import build_from_task
        from models.ch3_adapter import native_options
        from utils.ch3_contract import profile
        for filename in ('ch3_type1_urban6_four_H_v1.json','ch3_type1_m_all_patchtst_enc2_dm4_selected_v1.json','ch3_type1_epf_all_v3.json'):
            c=json.loads((ROOT/'configs'/filename).read_text())
            t=next(t for t in c['tasks'] if t['model']=='TimeMixer')
            p=profile(c,t);m=build_from_task(c,t,use_u=True,use_v=True)
            self.assertEqual(m.spec.value['native_options'],native_options(p))
            self.assertEqual(m.spec.value['fold'],t['fold'])
            x,marks=inputs(p);self.assertTrue(torch.isfinite(prediction(m,x,marks)).all())
        with self.assertRaises(ValueError):build_from_task(c,dict(t,model='J'),use_u=True,use_v=True)

    def test_inherited_single_scale_pdm_limitation_is_not_silently_fixed(self):
        # V itself has tested S=1 zero correction. Bound B's PDM requires >=2
        # scales (season_list[1]/trend_list_reverse[1]); current profiles use 3/4.
        p=fixture_profile();p['structure']['down_sampling_layers']=0
        x,marks=inputs(p)
        for model in (build_original(p),build(p,use_u=True,use_v=True)):
            with self.assertRaises(IndexError):model(x,marks,None,None)

    def test_strict_disk_checkpoint_roundtrip_all_arms(self):
        p=fixture_profile()
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='.uv-synthetic-') as temp:
            for i,(u,v) in enumerate(ARMS):
                m=build(p,use_u=u,use_v=v);file=Path(temp)/f'synthetic-{i}.pt'
                torch.save(synthetic_checkpoint(m),file)
                n=build(p,use_u=u,use_v=v);loaded=torch.load(file,weights_only=True)
                r=load_synthetic_checkpoint(n,loaded);self.assertEqual(r.missing_keys,[]);self.assertEqual(r.unexpected_keys,[])
                _assert_nested_equal(self,m.state_dict(),n.state_dict())

    def test_incompatible_switch_task_and_source_rejected_before_write(self):
        p=fixture_profile();m=build(p,use_u=True,use_v=True);ck=synthetic_checkpoint(m)
        for key,value in [('use_u',False),('use_v',False),('target_idx',1),('aux_idx',[2,1]),('label_horizon',6),('fold',4),('candidate_id','J'),('L',96),('pred_len',12),('task','M'),('sources',{}),('implementation',{})]:
            bad=copy.deepcopy(ck);bad['spec'][key]=value;before=copy.deepcopy(m.state_dict())
            with self.subTest(key=key),self.assertRaises(ValueError):load_synthetic_checkpoint(m,bad)
            _assert_nested_equal(self,before,m.state_dict())
        other=build(p,use_u=False,use_v=True)
        with self.assertRaises(ValueError):load_synthetic_checkpoint(other,ck)
        bad=copy.deepcopy(ck);key=next(iter(bad['state_dict']));bad['state_dict'][key]=torch.zeros(1)
        with self.assertRaises(ValueError):load_synthetic_checkpoint(m,bad)

    def test_author_identity_and_namespace_preservation(self):
        expected=binding()['TimeMixer'];self.assertEqual(str(Path(load_author().__file__).resolve()),expected['entry'])
        # Reload loader in a subprocess with hostile pre-existing layers namespace.
        body="""import sys,types
old=types.ModuleType('layers');old.__path__=['/not/an/author'];sys.modules['layers']=old
child=types.ModuleType('layers.foreign');sys.modules['layers.foreign']=child
import models,utils
mp=list(models.__path__);up=list(utils.__path__);sp=list(sys.path)
from timemixer_dev.source import load_author
load_author()
assert sys.modules['layers'] is old and sys.modules['layers.foreign'] is child
assert list(models.__path__)==mp and list(utils.__path__)==up and sys.path==sp
"""
        r=subprocess.run([sys.executable,'-B','-c',body],cwd=ROOT,env=os.environ.copy(),capture_output=True,text=True,timeout=30)
        self.assertEqual(r.returncode,0,r.stderr)

    def test_spec_no_training_license_and_implementation_coverage(self):
        spec=StructuralSpec.from_profile(fixture_profile(),use_u=True,use_v=True)
        self.assertEqual(spec.value['implementation'],implementation_binding())
        self.assertIn('model.py',spec.value['implementation']);self.assertIn('modules.py',spec.value['implementation'])
        self.assertFalse(spec.value['execution_authorized']);self.assertEqual(spec.value['training_protocol'],'undecided')
        self.assertNotIn('training',spec.value)
        p=fixture_profile();p['pred_len']=3
        with self.assertRaises(ValueError):StructuralSpec.from_profile(p,use_u=True,use_v=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
