"""Frozen r4 targeted CPU synthetic checks. No real data/checkpoints or effects."""
import copy,json,math,os,random,shutil,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np,torch
from torch import nn
from tests import timemixer_v6_r4_fixtures as f
from timemixer_dev_v6_r4 import adapter,worker,launcher
from timemixer_dev_v6_r4.modules import TargetConditionedAuxBasisGate as U,TemporalQKRawValueOperator as V,norm_preserving_carrier,gated_native_projection
from timemixer_dev_v6_r4.spec import REVISION,StructuralSpec
from timemixer_dev_v6_r4.training_config import validate,derive
from timemixer_dev_v6_r4.training_state import read,restore,validate as validate_state
from timemixer_dev_v6_r4.summary import aggregate
from utils.ch3_onecycle import Schedule,configuration


def batch(p,n=2):return torch.randn(n,p['T'],p['C']),torch.randn(n,p['T'],p['time_mark']['K'])
def common(m):return {k:v for k,v in m.state_dict().items()if not k.startswith(('u.','v.'))}
def tree(p):return {str(q.relative_to(p)):worker.file_sha(q)for q in Path(p).rglob('*')if q.is_file()}
def active_u(L=5,D=8,C=4,target=2,aux=(3,0,1)):
    torch.manual_seed(41);u=U(L,D,C,'MS',target,aux).double()
    with torch.no_grad():u.output.weight.fill_(.13);u.output.bias.fill_(.02)
    return u

class UInterface(unittest.TestCase):
    def test_formula_norm_direction_and_common_positive_gate_scale(self):
        torch.manual_seed(5);z=torch.randn(2,24,8,dtype=torch.float64);m=.2+torch.rand(2,1,8,dtype=torch.float64)
        logits=torch.logit(m/2);out=norm_preserving_carrier(z,logits);v=m*z;expected=z.norm(dim=-1,keepdim=True)*v/v.norm(dim=-1,keepdim=True)
        torch.testing.assert_close(out,expected,atol=1e-11,rtol=1e-11)
        torch.testing.assert_close(out.norm(dim=-1),z.norm(dim=-1),atol=1e-11,rtol=1e-11)
        torch.testing.assert_close(out,norm_preserving_carrier(z,torch.logit(m*.4/2)),atol=1e-11,rtol=1e-11)
        self.assertEqual(out.shape,(2,24,8));self.assertTrue(((out*v).sum(-1)>0).all())
    def test_initial_identity_keeps_logit_gradient_and_norm_gradients(self):
        z=torch.tensor([[[1.,2.,-3.,4.],[2.,0.,1.,-1.]]],requires_grad=True);l=torch.zeros(1,1,4,requires_grad=True)
        out=norm_preserving_carrier(z,l);self.assertTrue(torch.equal(out,z));(out*torch.arange(4)).sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all());self.assertGreater(float(l.grad.abs().sum()),0)
        z=torch.randn(1,2,4,dtype=torch.float64,requires_grad=True);l=torch.randn(1,1,4,dtype=torch.float64,requires_grad=True);weight=torch.randn_like(z)
        value=(norm_preserving_carrier(z,l)*weight).sum();analytic=torch.autograd.grad(value,(z,l))
        for tensor,grad,idx in [(z,analytic[0],(0,1,2)),(l,analytic[1],(0,0,1))]:
            plus=tensor.detach().clone();minus=plus.clone();plus[idx]+=1e-6;minus[idx]-=1e-6
            args=(plus,l.detach())if tensor is z else(z.detach(),plus);vplus=(norm_preserving_carrier(*args)*weight).sum()
            args=(minus,l.detach())if tensor is z else(z.detach(),minus);vminus=(norm_preserving_carrier(*args)*weight).sum()
            self.assertAlmostEqual(float(grad[idx]),float((vplus-vminus)/2e-6),places=6)
    def test_zero_sparse_extreme_common_gate_large_carrier_and_explicit_failure(self):
        for z,l in [(torch.zeros(2,24,8),torch.full((2,1,8),-10000.)),(torch.tensor([[[1.,0.,0.,-2.]]]),torch.full((1,1,4),-10000.)),(torch.randn(2,3,8)*1e20,torch.tensor([[[-1000.,1000.]*4]]).expand(2,-1,-1))]:
            z=z.requires_grad_();l=l.clone().requires_grad_();out=norm_preserving_carrier(z,l);self.assertTrue(torch.isfinite(out).all());out.sum().backward();self.assertTrue(torch.isfinite(z.grad).all());self.assertTrue(torch.isfinite(l.grad).all())
            torch.testing.assert_close(out.double().norm(dim=-1),z.double().norm(dim=-1),atol=2e-6,rtol=2e-5)
        # Required explicit finite-precision zero-denominator refusal, not epsilon fallback.
        with self.assertRaisesRegex(FloatingPointError,'numerical zero'):
            norm_preserving_carrier(torch.tensor([[[1.,0.]]]),torch.tensor([[[-10000.,10000.]]]))
        with self.assertRaises(FloatingPointError):norm_preserving_carrier(torch.tensor([[[float('nan'),0.]]]),torch.zeros(1,1,2))
    def test_native_projection_receiver_nonzero_bias_unscaled_and_P24(self):
        torch.manual_seed(7);B,C,D,P=2,4,8,24;z=torch.randn(B,P,D,dtype=torch.float64);native=nn.Linear(D,C).double();l=torch.randn(B,1,D,dtype=torch.float64)
        out=gated_native_projection(z,native,l,(2,),batch_size=B,channels=C,channel_independence=0)
        expected=native(z).clone();star=norm_preserving_carrier(z,l);expected[:,:,2]=torch.nn.functional.linear(star,native.weight[2:3],native.bias[2:3]).squeeze(-1)
        torch.testing.assert_close(out,expected,atol=1e-11,rtol=1e-11)
        for c in (0,1,3):self.assertTrue(torch.equal(out[:,:,c],native(z)[:,:,c]))
        self.assertTrue(torch.equal(gated_native_projection(z,native,torch.zeros_like(l),(2,),batch_size=B,channels=C,channel_independence=0),native(z)))
    def test_controller_preserved_target_condition_and_aux_response(self):
        u=active_u();x=torch.randn(2,5,4,dtype=torch.float64);m,d=u(x,return_diagnostics=True);E=u.history(x.transpose(1,2));q=E[:,2];S=torch.einsum('bjd,kj->bdk',E[:,[3,0,1]],u.mix_weight)/3
        kap=2*torch.sigmoid(q[:,:,None]*u.kappa.weight[:,0]+u.kappa.bias)
        h=torch.relu(kap*S+q[:,:,None]*u.target_term.weight[:,0]+u.target_term.bias);l=(h*u.output.weight[0]).sum(-1)+u.output.bias[0]
        torch.testing.assert_close(m[:,0],2*torch.sigmoid(l),atol=1e-11,rtol=1e-11)
        xx=x.clone();xx[:,:,2]+=1;_,dt=u(xx,return_diagnostics=True);self.assertTrue(torch.equal(d['bases'],dt['bases']));self.assertFalse(torch.equal(d['kappa']*d['bases'],dt['kappa']*dt['bases']))
        xx=x.clone();xx[:,:,3]+=1;self.assertFalse(torch.equal(u(x),u(xx)));self.assertEqual(u.source_slot_indices.tolist(),[3,0,1])
    def test_no_aux_bypasses_entire_controller_and_native_interface(self):
        u=U(1,8,3,'MS',1,[])
        with patch.object(u.history,'forward',side_effect=AssertionError('no aux must skip')):self.assertTrue(torch.equal(u(torch.randn(2,1,3)),torch.ones(2,1,8)))
        p=f.profile('PJM','MS',24);p['aux_idx']=[];m=adapter.build(p,use_u=True,use_v=False).eval();x,marks=batch(p)
        with patch.object(m.u,'forward',side_effect=AssertionError('no aux native path')):
            y=m(x,marks,None,None)
        m.use_u=False;self.assertTrue(torch.equal(y,m(x,marks,None,None)))
    def test_native_CI1_mapping_lower_component_only_no_M_task_authorization(self):
        B,C,P,D=2,3,5,8;z=torch.randn(B*C,P,D,dtype=torch.float64);l=torch.randn(B,C,D,dtype=torch.float64);native=nn.Linear(D,1).double()
        out=gated_native_projection(z,native,l,tuple(range(C)),batch_size=B,channels=C,channel_independence=1)
        for b in range(B):
            for c in range(C):
                expected=native(norm_preserving_carrier(z[b*C+c],l[b,c][None,:]));torch.testing.assert_close(out[b*C+c],expected,atol=1e-11,rtol=1e-11)

class VOperator(unittest.TestCase):
    def test_stable_time_residual_first_constant_and_L1_exact(self):
        for L in (1,12,96):
            g=torch.randn(2,L,8);r=V.relation_input(g);self.assertTrue(torch.equal(r[:,0],torch.zeros_like(r[:,0])))
            self.assertTrue(torch.equal(V.relation_input(g[:,:1].expand(-1,L,-1)),torch.zeros_like(g)))
            prev=torch.cat([g[:,:1],g[:,:-1]],1);prev2=torch.cat([g[:,:1].expand(-1,2,-1),g[:,:-2]],1)[:,:L]
            self.assertTrue(torch.equal(r,((g-prev)+(g-prev2))/3))
    def test_independent_double_formula_and_content_response(self):
        torch.manual_seed(17);v=V(8).double();v.a.data.fill_(.6);g=torch.randn(2,12,8,dtype=torch.float64);out,d=v(g,return_diagnostics=True)
        r=V.relation_input(g);q=torch.nn.functional.linear(r,v.query.weight).reshape(2,12,4,8).transpose(1,2);k=torch.nn.functional.linear(r,v.key.weight).reshape(2,12,4,8).transpose(1,2)
        q=q/torch.sqrt(q.square().mean(-1,keepdim=True)+1e-6);k=k/torch.sqrt(k.square().mean(-1,keepdim=True)+1e-6)
        A=(q@k.transpose(-1,-2)/math.sqrt(8)).softmax(-1);Y=(A@g.reshape(2,12,4,2).transpose(1,2)).transpose(1,2).reshape(2,12,8)
        torch.testing.assert_close(out,g+.5*v.a.tanh()*(Y-g),atol=1e-11,rtol=1e-11)
        torch.testing.assert_close(d['attention'],A,atol=1e-11,rtol=1e-11);self.assertTrue(torch.equal(A[:,:,0],torch.full_like(A[:,:,0],1/12)))
        other=g.clone();other[:,5]+=3;_,dd=v(other,return_diagnostics=True);self.assertFalse(torch.equal(A,dd['attention']))
    def test_attention_bounds_head_axis_and_large_QK_finite(self):
        for L in (12,96):
            v=V(8);v.query.weight.data.mul_(1e19);v.key.weight.data.mul_(1e19);g=torch.randn(2,L,8);v.a.data.fill_(.4);out,d=v(g,return_diagnostics=True)
            bound=math.exp(2*math.sqrt(8))/(math.exp(2*math.sqrt(8))+L-1)
            self.assertLessEqual(float(d['attention'].max()),bound+2e-6);self.assertTrue(torch.isfinite(out).all());self.assertEqual(d['attention'].shape,(2,4,L,L));self.assertTrue(torch.equal(d['attention'][:,:,0],torch.full_like(d['attention'][:,:,0],1/L)))
            self.assertLessEqual(float(d['normalized_query'].double().norm(dim=-1).max()),math.sqrt(8)+2e-6)
    def test_initial_identity_detach_and_value_parameter_gradient_paths(self):
        v=V(8);g=torch.randn(2,12,8,requires_grad=True);out,d=v(g,return_diagnostics=True);self.assertTrue(torch.equal(out,g));self.assertFalse(d['relation'].requires_grad)
        out.square().mean().backward();self.assertGreater(abs(float(v.a.grad)),0);self.assertEqual(float(v.query.weight.grad.abs().sum()),0)
        v.zero_grad();g.grad=None;v.a.data.fill_(.4);out,d=v(g,return_diagnostics=True);out.square().mean().backward();self.assertGreater(float(v.query.weight.grad.abs().sum()),0);self.assertGreater(float(v.key.weight.grad.abs().sum()),0);self.assertGreater(float(g.grad.abs().sum()),0)
        for t in (g.grad,v.query.weight.grad,v.key.weight.grad):self.assertTrue(torch.isfinite(t).all())
    def test_local_signed_alpha_bound_and_no_extra_parameters(self):
        v=V(8);self.assertEqual(sum(p.numel()for p in v.parameters()),513);self.assertEqual(set(dict(v.named_modules())),{'','query','key'})
        for a in (-9.,9.):
            v.a.data.fill_(a);g=torch.randn(2,12,8);out=v(g);self.assertLessEqual(float(out.abs().max()),2*float(g.abs().max())+1e-6)
        with self.assertRaises(ValueError):V(7)

class Integration(unittest.TestCase):
    def test_four_arms_common_B_initial_keys_outputs_rng_dropout(self):
        p=f.profile();original=adapter.build_original(p);state=torch.get_rng_state().clone();models={}
        for arm,u,v in [('B',False,False),('U',True,False),('V',False,True),('UV',True,True)]:
            m=adapter.build(p,use_u=u,use_v=v,fold=1);f.nested_equal(self,common(m),original.state_dict());self.assertTrue(torch.equal(state,torch.get_rng_state()));models[arm]=m
        f.nested_equal(self,models['U'].u.state_dict(),models['UV'].u.state_dict());f.nested_equal(self,models['V'].v.state_dict(),models['UV'].v.state_dict())
        x,marks=batch(p)
        for train in (False,True):
            original.train(train);start=torch.get_rng_state();expected=original(x,marks,None,None);end=torch.get_rng_state()
            for model in models.values():
                model.train(train);torch.set_rng_state(start);self.assertTrue(torch.equal(model(x,marks,None,None),expected));self.assertTrue(torch.equal(torch.get_rng_state(),end))
    def test_observation_preserves_inputs_buffers_rng_and_uses_single_denorm(self):
        p=f.profile();m=adapter.build(p,use_u=True,use_v=True).eval();m.u.output.weight.data.fill_(.1);m.v.a.data.fill_(.4);x,marks=batch(p,10);original=x.clone();marks0=marks.clone();state=copy.deepcopy(m.state_dict());rng=torch.get_rng_state();count=[]
        handle=m.normalize_layers[0].register_forward_hook(lambda mod,args,out:count.append(args[1]))
        with torch.inference_mode():a=m(x,marks,None,None);b,d=m.forecast(x,marks,None,None,return_diagnostics=True)
        handle.remove();self.assertTrue(torch.equal(a,b));self.assertEqual(count.count('denorm'),2);self.assertEqual(count.count('norm'),2);self.assertTrue(torch.equal(x,original));self.assertTrue(torch.equal(marks,marks0));self.assertTrue(torch.equal(torch.get_rng_state(),rng));f.nested_equal(self,m.state_dict(),state)
        self.assertEqual(d['U']['gate'].shape[0],8);self.assertEqual(d['V']['attention'].shape[0],8);self.assertEqual(d['U_interface'][0]['carrier_norm'].shape[0],8)
    def test_actual_insertions_keep_preenc_PDM_trend_and_coarse_predictions(self):
        p=f.profile();m=adapter.build(p,use_u=True,use_v=True).eval();m.u.output.weight.data.fill_(.1);m.v.a.data.fill_(.4);x,marks=batch(p)
        def run(u,v):
            m.use_u,m.use_v=u,v;captured={};handles=[]
            for i,n in enumerate(m.pdm_blocks):
                def hook(mod,args,out,i=i):captured['PDM'+str(i)]=[z.clone()for z in out]
                handles.append(n.register_forward_hook(hook))
            for i,n in enumerate(m.regression_layers):
                def hook(mod,args,out,i=i):captured['trend'+str(i)]=out.clone()
                handles.append(n.register_forward_hook(hook))
            try:
                with torch.inference_mode():y,d=m.forecast(x,marks,None,None,return_diagnostics=True)
            finally:
                for h in handles:h.remove()
            return y,d,captured
        y,d,a=run(True,True);yy,dd,b=run(False,False)
        for k in a:f.nested_equal(self,a[k],b[k])
        # Coarse head snapshots from U-on and U-off with V unchanged are exact.
        _,du,_=run(False,True)
        for i in (1,2):self.assertTrue(torch.equal(d['scale_predictions'][i],du['scale_predictions'][i]))
        self.assertFalse(torch.equal(y,yy));m.use_u,m.use_v=True,True
    def test_MS_profiles_F1_F4_EPF_target2_parameter_counts(self):
        rows=[]
        for ds,variant,h in [('UrbanEV',v,3)for v in ['F1','F2','F3','F4']]+[('PJM','MS',24)]:
            p=f.profile(ds,variant,h);m=adapter.build(p,use_u=True,use_v=True).eval();x,marks=batch(p,1)
            with torch.inference_mode():y=adapter.prediction(m,x,marks)
            self.assertEqual(y.shape,(1,p['pred_len'],1));D=p['structure']['d_model'];nu=sum(q.numel()for q in m.u.parameters());nv=sum(q.numel()for q in m.v.parameters());self.assertEqual(nu,D*(p['T']+1)+8*len(p['aux_idx'])+41);self.assertEqual(nv,64*D+1)
            rows.append(dict(dataset=ds,variant=variant,L=p['T'],P=p['pred_len'],D=D,C=p['C'],target=p['target_idx'],U_parameters=nu,V_parameters=nv,total=sum(q.numel()for q in m.parameters())))
            if ds=='PJM':self.assertEqual(p['target_idx'],2)
        (f.EVIDENCE/'parameter-shape-checks.json').write_text(json.dumps(rows,indent=2)+'\n');r=next(r for r in rows if r['variant']=='F4');self.assertEqual(r['total'],75935)
    def test_three_arms_delayed_gradients_and_effective_new_paths(self):
        for arm in ['U','V','UV']:
            p=f.profile();p['structure'].update(d_model=8,d_ff=16,e_layers=1);m=adapter.build(p,use_u='U'in arm,use_v='V'in arm);x,marks=batch(p,4);target=torch.randn(4,1,1);opt=torch.optim.Adam(m.parameters(),lr=.003)
            active={k:False for k,v in m.named_parameters()if k.startswith(('u.','v.'))}
            for step in range(4):
                opt.zero_grad();loss=(adapter.prediction(m,x,marks)-target).square().mean();loss.backward()
                for k,v in m.named_parameters():
                    if v.grad is not None:self.assertTrue(torch.isfinite(v.grad).all());active[k]=active.get(k,False)or bool(v.grad.abs().sum()>0)
                opt.step()
            for key in active:
                if key.startswith(('u.','v.')):self.assertTrue(active[key],(arm,key))

class PolicyIdentity(unittest.TestCase):
    def test_goal1_fold_independence_goal3_eight_cells_and_same_revision(self):
        c=copy.deepcopy(f.STORED);records=[]
        for t in c['tasks']:
            if t['goal']==1:
                b=c['existing_B'][t['baseline_pair_cell_id']]
                factor={'U':.8,'V':.9,'UV':.7}[t['arm']]
                value={k:b[k]*factor for k in ('mse','mae')}
            elif t['goal']==3:
                value=dict(mse={'F1':4.,'F2':3.,'F3':2.}[t['input_variant']],mae={'F1':4.,'F2':3.,'F3':2.}[t['input_variant']])
            else:
                b=c['existing_B'][t['baseline_pair_cell_id']];value={k:b[k]*.9 for k in ('mse','mae')}
            records.append((t,dict(task_id=t['id'],**value)))
        result=aggregate(c,records);self.assertTrue(result['goals']['goal1']['numerical_conditions_met']);self.assertTrue(result['goals']['goal3']['numerical_conditions_met'])
        for t,v in records:
            if t['goal']==1 and t['arm']=='UV':
                b=c['existing_B'][t['baseline_pair_cell_id']]
                for k in ('mse','mae'):v[k]=b[k]*(.001 if t['fold']==1 else .95)
        self.assertFalse(aggregate(c,records)['goals']['goal1']['numerical_conditions_met'])
        missing=[r for r in records if not(r[0]['goal']==3 and r[0]['fold']==1 and r[0]['h']==3 and r[0]['input_variant']=='F2')]
        self.assertEqual(aggregate(c,missing)['goals']['goal3']['numerical_status'],'Not evaluated')
        foreign=copy.deepcopy(records);foreign[0][0]['id']=foreign[0][0]['id'].replace('|r4|','|r3|')
        with self.assertRaises(ValueError):aggregate(c,foreign)
    def test_53MS_scientific_fields_B_mapping_and_24_six_waves(self):
        c=f.STORED;validate(c);old=json.loads((f.ROOT/'configs/timemixer_uv_round2_resolved_v6_r3.json').read_text());oldtasks={t['id']:t for t in old['tasks']if t['task']=='MS'}
        self.assertEqual(len(c['tasks']),53);self.assertEqual(len(c['existing_B']),13);self.assertEqual(sum(t['arithmetic']['max_epochs']for t in c['tasks']),1060)
        self.assertEqual(c['max_run_epochs'],1060);self.assertEqual(c['max_optimizer_steps'],sum(t['arithmetic']['max_optimizer_steps']for t in c['tasks']))
        for t in c['tasks']:
            o=oldtasks[t['id'].replace('|r4|','|r3|')]
            for k in set(t)-{'id','model','candidate_id','structural_revision'}:self.assertEqual(t[k],o[k],k)
        ids=[next(t['id']for t in c['tasks']if(t['fold'],t['h'],t['arm'],t['goal'])==(fold,h,arm,1))for arm in ('U','V','UV')for fold in (1,4)for h in (3,6,9,12)]
        waves=launcher.batch(c,ids);self.assertEqual(len(waves),6);self.assertEqual([v for wave in waves for v in wave],ids);tasks={t['id']:t for t in c['tasks']};self.assertEqual(sum(tasks[i]['arithmetic']['max_optimizer_steps']for i in ids),1485300);self.assertEqual(len(c['F4_reuse']),8)
    def test_new_goal2_seven_of_ten_not_two_separate_thresholds_missing_tie_duplicate(self):
        c=copy.deepcopy(f.STORED);records=[]
        for i,t in enumerate(t for t in c['tasks']if t['goal']==2):
            b=c['existing_B'][t['baseline_pair_cell_id']];records.append((t,dict(task_id=t['id'],mse=b['mse']*.9,mae=b['mae']*(.9 if i<2 else 1.1))))
        result=aggregate(c,records)['goals']['goal2'];self.assertEqual(result['winning_cells'],7);self.assertTrue(result['numerical_conditions_met']);self.assertEqual(result['metrics']['mae']['winner_count'],2)
        self.assertEqual(aggregate(c,records[:-1])['goals']['goal2']['numerical_status'],'Not evaluated')
        records[0][1]['mae']=c['existing_B'][records[0][0]['baseline_pair_cell_id']]['mae'];self.assertFalse(aggregate(c,records)['goals']['goal2']['numerical_conditions_met'])
        with self.assertRaises(ValueError):aggregate(c,records+[records[0]])
    def test_old_revision_grant_config_synthetic_checkpoint_refused(self):
        p=f.profile();m=adapter.build(p,use_u=True,use_v=True);ck=adapter.synthetic_checkpoint(m);before=copy.deepcopy(m.state_dict());bad=copy.deepcopy(ck);bad['spec']['structural_revision']['u']='xlinear_vgm_target_conditioned_aux_basis_v6_r3'
        with self.assertRaises(ValueError):adapter.load_synthetic_checkpoint(m,bad)
        f.nested_equal(self,before,m.state_dict());old=json.loads((f.ROOT/'configs/timemixer_uv_round2_resolved_v6_r3.json').read_text())
        with self.assertRaises(ValueError):validate(old)
        ids=[f.STORED['tasks'][0]['id']]
        for grant in [dict(schema='timemixer-uv-launch-v2',review_approved=False),dict(schema='timemixer-uv-launch-v2',review_approved=True,candidate_id=f.STORED['candidate_id'],structural_revision=bad['spec']['structural_revision'])]:
            with self.assertRaises(PermissionError):launcher.validate_authorization(f.STORED,'unused',ids,grant)

class Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reuse=os.environ.get('M6_R4_REUSE_PIPELINE')
        if reuse:
            cls.folder=Path(reuse).resolve();cls.results={};cls.configs={}
            assert cls.folder.is_relative_to(f.EVIDENCE/'synthetic-fixtures')
            from timemixer_dev_v6_r4.artifacts import audit
            for path in (cls.folder/'results/first/runs').iterdir():
                receipt=audit(path);c=json.loads((path/'config.json').read_text());validate(c)
                t=c['tasks'][0];assert c['purpose']=='synthetic'
                cls.results[t['arm']]=receipt;cls.configs[t['arm']]=(c,t['id'],Path(t['data_config']['datasets'][t['dataset']]['path']))
            assert set(cls.results)=={'B','U','V','UV'}
            return
        cls.folder=f.case('pipeline');cls.results={};cls.configs={}
        for arm in ('B','U','V','UV'):
            c,tid,path=f.fixture(cls.folder,arm=arm);cls.configs[arm]=(c,tid,path);cls.results[arm]=worker.run_worker(c,tid,cls.folder/'results','first',path_override=path)
    def test_actual_worker_four_arms_seal_test_once_and_off_original_exact(self):
        for arm,r in self.results.items():self.assertEqual(r['metrics']['test_calls'],1);self.assertEqual(r['manifest']['steps'],4)
        c,tid,path=self.configs['B'];original=worker.run_worker(c,tid,self.folder/'original','first',path_override=path,reference_original=True)
        a=read(Path(original['path'])/'last.pt');b=read(Path(self.results['B']['path'])/'last.pt')
        for key in ['model','optimizer','scheduler','rng','best','epoch','steps']:f.nested_equal(self,a[key],b[key])
    def test_legal_epoch_restore_uninterrupted_exact_and_old_state_no_live_write(self):
        c,tid,path=self.configs['UV']
        if os.environ.get('M6_R4_REUSE_PIPELINE'):
            from timemixer_dev_v6_r4.artifacts import audit
            resumed=audit(next((self.folder/'resume/first/runs').iterdir()))
        else:
            paused=worker.run_worker(c,tid,self.folder/'resume','first',path_override=path,pause_after_epoch=1);self.assertEqual(paused['steps'],2)
            resumed=worker.run_worker(c,tid,self.folder/'resume','first',path_override=path,resume=True)
        a=read(Path(resumed['path'])/'last.pt');b=read(Path(self.results['UV']['path'])/'last.pt')
        for key in ['model','optimizer','scheduler','rng','best','epoch','steps']:f.nested_equal(self,a[key],b[key])
        p=c['tasks'][0]['structural_source_profile'];m=adapter.build(p,use_u=True,use_v=True,fold=c['tasks'][0]['fold']);tr=p['training'];opt=torch.optim.Adam(m.parameters(),lr=tr['lr'],betas=tuple(tr['betas']),eps=tr['eps'],weight_decay=tr['weight_decay']);sched=Schedule(opt,tr['scheduler']);g=torch.Generator().manual_seed(23);live=f.live(m,opt,sched,g)
        bad=copy.deepcopy(a);bad['identity']['structural_spec']['structural_revision']['v']='timebridge_temporal_qk_raw_value_operator_v6_r3'
        with self.assertRaises(ValueError):validate_state(bad,m,opt,sched,a['identity'])
        f.nested_equal(self,live,f.live(m,opt,sched,g));restore(Path(resumed['path'])/'last.pt',m,opt,sched,g,a['identity']);f.nested_equal(self,m.state_dict(),a['model'])
    def test_duplicate_and_unknown_test_claim_preserve_original(self):
        c,tid,path=self.configs['UV'];good=self.results['UV'];before=tree(Path(good['path']))
        with self.assertRaises(FileExistsError):worker.run_worker(c,tid,self.folder/'results','duplicate',path_override=path)
        self.assertEqual(before,tree(Path(good['path'])))
        root=self.folder/'unknown';dest=root/'first/staging'/Path(good['path']).name;dest.parent.mkdir(parents=True);shutil.copytree(good['path'],dest,ignore=shutil.ignore_patterns('test-result.json','test-result.sha256','checksums.sha256','seal.json'));start=tree(dest)
        with patch.object(worker,'evaluate',side_effect=AssertionError('uncertain test cannot repeat')):
            with self.assertRaisesRegex(ValueError,'cannot repeat'):worker.run_worker(c,tid,root,'first',path_override=path,resume=True)
        self.assertEqual(start,tree(dest))
    def test_EPF_P24_real_loader_worker_target_only(self):
        folder=f.case('epf');c,tid,path=f.fixture(folder,arm='UV',dataset='PJM',variant='MS',H=24);r=worker.run_worker(c,tid,folder/'results','first',path_override=path);self.assertEqual(r['metrics']['test_calls'],1);self.assertEqual(r['metrics']['metric_scope'],'target_only');self.assertEqual(c['tasks'][0]['structural_source_profile']['target_idx'],2)

if __name__=='__main__':unittest.main(verbosity=2)
