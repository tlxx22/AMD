"""Directed v7 synthetic acceptance: no real data/checkpoints or training CLI."""
import copy,json,math,os,unittest,ast
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from tests import moderntcn_v7_fixtures as f
from moderntcn_dev_v7 import adapter,training_state,launcher
from moderntcn_dev_v7.modules import TiedHistoryComponent
from moderntcn_dev_v3 import adapter as previous
from moderntcn_dev_v7.guards import *
def gradients(m):return {n:p.grad for n,p in m.named_parameters()}
def common(m):return {n:p for n,p in m.state_dict().items() if not n.startswith(('u.','v.'))}
def close(a,b):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
class Differential(unittest.TestCase):
    def setUp(self):torch.manual_seed(401)

    def test_bases_shared_coefficients_and_loader_offsets(self):
        from utils.ch3_data import Windows
        rows=[]
        for ds,Hn in [('UrbanEV',h) for h in [3,6,9,12]]+[('PJM',24)]:
            p=f.profile(ds,H=Hn);L,C,P=p['T'],p['C'],p['pred_len'];u=TiedHistoryComponent(L,C,P,p['target_idx'],p['aux_idx'],Hn,ds=='UrbanEV')
            tau_b=torch.tensor([(i-(L-1))/(L-1) for i in range(L)]);tau_f=torch.tensor([Hn/(L-1)] if ds=='UrbanEV' else [i/(L-1) for i in range(1,25)])
            close(u.Phi_b,torch.stack([torch.ones_like(tau_b),tau_b,tau_b**2],-1));close(u.Phi_f,torch.stack([torch.ones_like(tau_f),tau_f,tau_f**2],-1))
            theta=torch.eye(3);X=torch.randn(3,L,C);Z=X.permute(0,2,1)
            with patch.object(u,'encode',return_value=theta) as enc:back,fore=u(X,Z)
            self.assertEqual(enc.call_count,1);f.equal(self,back,u.Phi_b.T);f.equal(self,fore,u.Phi_f.T)
            if ds=='UrbanEV':
                xx=np.zeros((48,2,C),dtype='float32');yy=np.arange(48,dtype='float32')[:,None].repeat(2,1);win=Windows(xx,yy,0,48,L,1,urban=True,label_horizon=Hn);self.assertEqual(float(win[0][1][0,0])-(L-1),Hn)
            else:
                win=Windows(np.zeros((150,C),dtype='float32'),np.arange(150,dtype='float32')[:,None],0,150,L,P);f.equal(self,win[0][1].flatten(),torch.arange(96,120,dtype=torch.float32))
            self.assertEqual(set(u.state_dict()),{'W_enc','b_enc','W_coef','Phi_b','Phi_f'});self.assertEqual(tuple(u.Phi_b.shape),(L,3));rows.append(dict(dataset=ds,H=Hn,P=P,Phi_f=u.Phi_f.tolist(),offset_matches_loader=True))
        self.assertEqual(len({tuple(tuple(r) for r in x['Phi_f']) for x in rows[:4]}),4)
        f.output('bases',rows)

    def test_encoding_population_variance_and_history_order(self):
        rows=[]
        for ds,Hn in [('UrbanEV',3),('PJM',24)]:
            p=f.profile(ds,H=Hn);L,C,P=p['T'],p['C'],p['pred_len'];u=TiedHistoryComponent(L,C,P,p['target_idx'],p['aux_idx'],Hn,ds=='UrbanEV');X=torch.arange(L,dtype=torch.float32)[None,:,None].expand(2,L,C).clone()/L;Z=X.permute(0,2,1).clone()
            with torch.no_grad():u.W_enc.zero_();u.b_enc.fill_(1);u.W_enc[0,0]=1;u.W_coef.zero_();u.W_coef[0,0]=1
            mu=X.mean(1);var=((X-mu[:,None,:])**2).mean(1);s=torch.stack([mu,torch.log1p(var)],-1);r=torch.cat([Z.reshape(2,-1),s.reshape(2,-1)],-1);expected=F.linear(F.relu(F.linear(r,u.W_enc,u.b_enc)),u.W_coef)
            f.equal(self,u.encode(X,Z),expected)
            Xrev=X.flip(1);Zrev=Z.flip(2);f.equal(self,X.mean(1),Xrev.mean(1));f.equal(self,((X-X.mean(1)[:,None,:])**2).mean(1),((Xrev-Xrev.mean(1)[:,None,:])**2).mean(1))
            a=u(X,Z);b=u(Xrev,Zrev);self.assertFalse(torch.equal(a[0],b[0]));self.assertFalse(torch.equal(a[1],b[1]));f.equal(self,a,u(X,Z))
            self.assertTrue(torch.isfinite(u(torch.ones_like(X),torch.zeros_like(Z))[0]).all())
            bad=X.clone();bad[0,0,0]=float('nan');self.assertRaises(ValueError,u,bad,Z)
            rows.append(dict(dataset=ds,formula='squared centered population variance confirmed by user',C_major_time_minor=True,same_moments_different_order_response=True,no_cross_batch_context=True))
        f.output('encoding',rows)

    def test_nonzero_residual_forecast_single_ReVIN_wiring(self):
        rows=[]
        for ds,Hn in [('UrbanEV',6),('PJM',24)]:
            p=f.profile(ds,H=Hn);m=adapter.build(p,use_u=True,use_v=True);f.open_branches(m);m.eval();X=torch.randn(2,p['T'],p['C']);original=X.clone();state=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone();a=m(X);seen={};calls={};hooks=[];b=m.model.stages[0].blocks[0]
            def out(name):
                def hook(mod,args,result):calls[name]=calls.get(name,0)+1;seen[name]=(args,result)
                return hook
            def rev(mod,args,result):
                name='revin_'+args[1];calls[name]=calls.get(name,0)+1;seen[name]=(args,result.clone())
            hooks.append(m.model.revin_layer.register_forward_hook(rev))
            for name,mod in [('u',m.u),('stem',m.model.downsample_layers[0]),('h',b.ffn1drop2),('v',m.v),('head',m.model.head),('ffn1drop1',b.ffn1drop1),('ffn2drop1',b.ffn2drop1),('ffn2drop2',b.ffn2drop2),('block_BN',b.norm)]:hooks.append(mod.register_forward_hook(out(name)))
            actual,diag=m(X,return_diagnostics=True)
            for h in hooks:h.remove()
            f.equal(self,a,actual);f.equal(self,state,m.state_dict());f.equal(self,X,original);f.equal(self,rng,torch.get_rng_state());self.assertTrue(all(c==1 for c in calls.values()));self.assertEqual(calls['revin_norm'],1);self.assertEqual(calls['revin_denorm'],1)
            Z0=seen['revin_norm'][1].permute(0,2,1);back,fore=seen['u'][1];self.assertGreater(float(back.abs().sum()),0);self.assertGreater(float(fore.abs().sum()),0);f.equal(self,seen['u'][0][0],X);f.equal(self,seen['u'][0][1],Z0)
            Zres=Z0.clone();Zres[:,p['target_idx']]-=back;stem=Zres.reshape(2*p['C'],1,p['T'])
            if ds!='UrbanEV':stem=torch.cat([stem,stem[:,:,-1:].repeat(1,1,4)],-1)
            f.equal(self,seen['stem'][0][0],stem);f.equal(self,Zres[:,p['aux_idx']],Z0[:,p['aux_idx']]);f.equal(self,seen['v'][0][0],seen['h'][1].reshape(2,p['C'],64,-1));f.equal(self,seen['v'][0][1],Z0);f.equal(self,seen['v'][0][2],X)
            combined=seen['head'][1].permute(0,2,1).clone();combined[:,:,p['target_idx']]+=fore;f.equal(self,seen['revin_denorm'][0][0],combined)
            self.assertIn('U_tied_component',diag);self.assertNotIn('U_shared_rotation',diag)
            # U-only still performs both paths; offU must not call new U.
            m.use_v=False;self.assertEqual(m(X).shape,(2,p['pred_len'],p['C']));m.use_u=False
            with patch.object(m.u,'forward',side_effect=AssertionError('disabled U must not execute')):m(X)
            rows.append(dict(dataset=ds,norm_denorm_calls=[1,1],nonzero=True,V_router='original Z0/X',filter='residual-history h',target_history_subtract=True,target_forecast_before_denorm=True,observer='exact',native_call_counts=calls))
        f.output('wiring',rows)

    def test_shared_theta_backcast_forecast_local_derivatives(self):
        rows=[]
        for L,P,Hn,C in [(12,1,9,11),(96,24,24,3)]:
            u=TiedHistoryComponent(L,C,P,0 if C==11 else 2,list(range(1,C)) if C==11 else [0,1],Hn,C==11);theta=torch.tensor([[.2,-.1,.3]],dtype=torch.float64,requires_grad=True);pb=u.Phi_b.double();pf=u.Phi_f.double();Z=torch.linspace(-1,1,L,dtype=torch.float64)[None];W=torch.arange(P*L,dtype=torch.float64).reshape(P,L)/(P*L);b=theta@pb.T;fut=theta@pf.T;historical=F.linear(Z-b,W);pred=historical+fut;weight=torch.linspace(.5,1.5,P,dtype=torch.float64)[None]
            gb=torch.autograd.grad((historical*weight).sum(),theta,retain_graph=True)[0];gf=torch.autograd.grad((fut*weight).sum(),theta,retain_graph=True)[0];g=torch.autograd.grad((pred*weight).sum(),theta)[0];expected=weight@(pf-W@pb)
            torch.testing.assert_close(g,expected,atol=1e-12,rtol=1e-12);torch.testing.assert_close(g,gb+gf,atol=1e-12,rtol=1e-12);self.assertGreater(float(gb.abs().sum()),0);self.assertGreater(float(gf.abs().sum()),0);rows.append(dict(L=L,P=P,both_nonzero=True,max_abs=float((g-expected).abs().max())))
        f.output('shared-gradient',rows)

    def test_coefficient_then_encoder_gradient_opening(self):
        rows=[]
        for ds,Hn in [('UrbanEV',3),('PJM',24)]:
            p=f.profile(ds,H=Hn);m=adapter.build(p,use_u=True,use_v=True);opt=torch.optim.Adam(m.parameters(),lr=.01,weight_decay=1e-7);X=torch.randn(2,p['T'],p['C']);target=torch.randn(2,p['pred_len'],1);history=[]
            for step in range(3):
                opt.zero_grad();adapter.loss(adapter.prediction(m,X),target).backward();g={n:float(p.grad.abs().sum()) for n,p in m.named_parameters() if n.startswith(('u.','v.'))};history.append(g)
                if step==0:
                    self.assertGreater(g['u.W_coef'],0);self.assertEqual(g['u.W_enc'],0);self.assertEqual(g['u.b_enc'],0);self.assertGreater(g['v.A'],0)
                self.assertGreater(float(m.model.stages[0].blocks[0].ffn1pw1.weight.grad.abs().sum()),0);self.assertGreater(float(m.model.stages[0].blocks[0].ffn2pw1.weight.grad.abs().sum()),0);opt.step()
            for n in ['u.W_enc','u.b_enc','v.P_s','v.W1.weight']:self.assertGreater(history[-1][n],0)
            rows.append(dict(dataset=ds,loss_gradients_before_Adam=history))
        f.output('gradient-opening',rows)

    def test_profiles_parameters_target_loss(self):
        rows=[]
        for ds,var,Hn,U in [('UrbanEV','F1',3,2816),('UrbanEV','F2',6,3712),('UrbanEV','F3',9,4160),('UrbanEV','F4',12,5056),('PJM','F4',24,9536)]:
            p=f.profile(ds,var,Hn);m=adapter.build(p,use_u=True,use_v=True);x=torch.randn(2,p['T'],p['C']);out=adapter.prediction(m,x);self.assertEqual(out.shape,(2,p['pred_len'],1));self.assertEqual(m.u.target,p['target_idx']);self.assertEqual(list(m.u.aux),p['aux_idx']);self.assertGreater(len(m.u.aux),0)
            u=sum(p.numel() for p in m.u.parameters());v=sum(p.numel() for p in m.v.parameters());self.assertEqual(u,U);self.assertEqual(v,572 if ds=='UrbanEV' else 2076);self.assertRaises(ValueError,adapter.loss,out,torch.zeros(2,p['pred_len'],p['C']))
            rows.append(dict(dataset=ds,variant=var,C=p['C'],L=p['T'],P=p['pred_len'],H=Hn,N=m.spec.value['N'],aux=p['aux_idx'],B=sum(p.numel() for n,p in m.named_parameters() if not n.startswith(('u.','v.'))),U=u,V=v,UV=u+v,fixed_basis_elements=m.u.Phi_b.numel()+m.u.Phi_f.numel()))
        f.output('parameter-counts',rows)

    def test_neutral_native_state_gradient_BN_RNG(self):
            for ds,Hn in [('UrbanEV',3),('PJM',24)]:
                p=f.profile(ds,H=Hn);native=adapter.build_original(p);base_rng=torch.get_rng_state().clone();mods=[adapter.build(p,use_u=u,use_v=v) for u,v in [(False,False),(True,False),(False,True),(True,True)]]
                for m in mods:f.equal(self,native.state_dict(),common(m));f.equal(self,base_rng,torch.get_rng_state());self.assertEqual(hasattr(m,'u'),m.use_u);self.assertEqual(hasattr(m,'v'),m.use_v)
                f.equal(self,mods[1].u.state_dict(),mods[3].u.state_dict());f.equal(self,mods[2].v.state_dict(),mods[3].v.state_dict())
                X=torch.randn(2,p['T'],p['C'])
                for train in [False,True]:
                    start=torch.get_rng_state().clone();native.train(train);native.zero_grad();y=native(X);end=torch.get_rng_state().clone()
                    if train:y.square().mean().backward()
                    for m in mods:
                        m.train(train);m.zero_grad();torch.set_rng_state(start);out=m(X);f.equal(self,y,out);f.equal(self,end,torch.get_rng_state());f.equal(self,native.state_dict(),common(m))
                        if train:
                            out.square().mean().backward();g=gradients(native);f.equal(self,g,{k:v for k,v in gradients(m).items() if k in g})
            f.output('neutral',dict(native_common_state_output_gradient_BN_RNG='exact',dropout_order='unchanged',zero_coefficient_is_not_bypass=True))

    def test_V_only_v3_exact_nonzero_train_eval_updates(self):
            from utils.ch3_onecycle import Schedule,configuration
            rows=[]
            for ds,Hn in [('UrbanEV',3),('PJM',24)]:
                p=f.profile(ds,H=Hn);a=previous.build(p,use_u=False,use_v=True);rng=torch.get_rng_state().clone();b=adapter.build(p,use_u=False,use_v=True);f.equal(self,rng,torch.get_rng_state());f.equal(self,a.state_dict(),b.state_dict())
                for m in [a,b]:f.open_branches(m)
                opts=[torch.optim.Adam(m.parameters(),lr=.01,weight_decay=1e-7) for m in [a,b]];sched=[Schedule(o,configuration(2,4)) for o in opts];X=torch.randn(2,p['T'],p['C'])
                for train in [False,True]:
                    start=torch.get_rng_state().clone();obs=[]
                    for m in [a,b]:
                        m.train(train);m.zero_grad();torch.set_rng_state(start);out=m(X);out.square().mean().backward();obs.append((out.detach(),copy.deepcopy(m.state_dict()),copy.deepcopy(gradients(m)),torch.get_rng_state().clone()))
                    f.equal(self,*obs)
                for step in range(2):
                    start=torch.get_rng_state().clone();obs=[]
                    for m,o,s in zip([a,b],opts,sched):
                        torch.set_rng_state(start);o.zero_grad();out=m(X);out[:,:,p['target_idx']].square().mean().backward();gr=copy.deepcopy(gradients(m));o.step();s.after_successful_update();obs.append((copy.deepcopy(m.state_dict()),copy.deepcopy(o.state_dict()),s.state_dict(),gr,torch.get_rng_state().clone()))
                    f.equal(self,*obs)
                rows.append(dict(dataset=ds,nonzero_A_Ps=True,train_eval_state_gradient_Adam_scheduler_BN_RNG='exact',updates_per_model=2))
            f.output('V-only-equivalence',rows)

    def test_same_v7_guard_exact_and_fault_boundaries(self):
            from utils.ch3_onecycle import Schedule,configuration
            p=f.profile();mods=[adapter.build(p,use_u=True,use_v=True) for _ in range(2)]
            for m in mods:f.open_branches(m)
            opts=[torch.optim.Adam(m.parameters(),lr=.01,weight_decay=1e-7) for m in mods];sched=[Schedule(o,configuration(2,4)) for o in opts];X=torch.randn(2,12,11);target=torch.randn(2,1,1)
            for step in range(2):
                rng=torch.get_rng_state().clone();obs=[]
                for m,o,s,g,pguard in zip(mods,opts,sched,[gradients_original,gradients_collected],[parameters_original,parameters_collected]):
                    torch.set_rng_state(rng);o.zero_grad(set_to_none=True);out=adapter.prediction(m,X);loss=adapter.loss(out,target);loss_guard(loss);loss.backward();named=list(m.named_parameters());g(named);grad=copy.deepcopy(gradients(m));o.step();pguard(named);s.after_successful_update();obs.append((out.detach(),loss.detach(),grad,copy.deepcopy(m.state_dict()),copy.deepcopy(o.state_dict()),s.state_dict(),torch.get_rng_state().clone()))
                f.equal(self,*obs)
            # Control spies deliberately do not perform an extra Adam update on faults.
            rows=[]
            for bad in [float('nan'),float('inf'),-float('inf'),0.,-0.,None]:
                outs=[]
                for checker in [gradients_original,gradients_collected]:
                    p1=torch.nn.Parameter(torch.ones(1));p2=torch.nn.Parameter(torch.ones(1));p1.grad=None if bad is None else torch.tensor([bad]);p2.grad=None;order=[]
                    try:checker([('first',p1),('second',p2)]);order.append('Adam')
                    except GuardError as e:order.append((e.boundary,e.name))
                    outs.append(order)
                f.equal(self,*outs);self.assertNotIn('Adam',outs[0]);rows.append(str(bad))
            for fn in [parameters_original,parameters_collected]:
                p1=torch.nn.Parameter(torch.tensor([float('inf')]));p2=torch.nn.Parameter(torch.tensor([float('nan')]))
                with self.assertRaises(GuardError) as cm:fn([('first',p1),('second',p2)])
                self.assertEqual((cm.exception.boundary,cm.exception.name),('after_Adam_before_scheduler','first'))
            with self.assertRaises(GuardError) as cm:loss_guard(torch.tensor(float('nan')))
            self.assertEqual(cm.exception.boundary,'before_backward')
            f.output('guard-integration',dict(full_v7_updates_per_path=2,all_outputs_gradients_states_RNG='exact',fault_boundaries='same order, no extra update',cases=rows,previous_GPU_evidence_reused=True))

    def test_version_rejection_before_write_and_synthetic_IO(self):
            m=adapter.build(f.profile(),use_u=True,use_v=True);state=copy.deepcopy(m.state_dict());ck=adapter.synthetic_checkpoint(m)
            for version in ['v1','v2','v3','v4','v5','v6']:
                bad=copy.deepcopy(ck);bad['format']='moderntcn-uv-synthetic-'+version;bad['spec']['candidate_id']='moderntcn-uv-dev-'+version
                self.assertRaises(ValueError,adapter.load_synthetic_checkpoint,m,bad);f.equal(self,state,m.state_dict())
                with patch('timemixer_dev.training_state.restore',side_effect=AssertionError('must reject before read/write')):self.assertRaises(ValueError,training_state.restore,'forbidden-real-path',m,None,None,None,dict(candidate_id=bad['spec']['candidate_id'],structural_spec=bad['spec']))
            path=f.EVIDENCE/'tests'/('synthetic-state-'+os.environ['MTCN_TEST_RUN']+'.pt');torch.save(ck,path);loaded=torch.load(path,map_location='cpu',weights_only=False);adapter.load_synthetic_checkpoint(m,loaded);f.equal(self,state,m.state_dict())
            bad=copy.deepcopy(ck);bad['spec']['target_idx']=2;self.assertRaises(ValueError,adapter.load_synthetic_checkpoint,m,bad);f.equal(self,state,m.state_dict())

class Metadata(unittest.TestCase):
    def test_scope_summary_duplicates_budget_and_old_config(self):
            from moderntcn_dev_v7.training_config import validate
            from moderntcn_dev_v7.scope import verified_v_records,expected_waves
            from moderntcn_dev_v7.summary import aggregate
            from moderntcn_dev_v6.training_config import validate as old_validate
            c=validate(f.STORED);old=json.loads((f.ROOT/'configs/moderntcn_uv_fold6_epf_orthogonal_variable_v6.json').read_text());old_validate(old);refs=verified_v_records(c)
            self.assertEqual(len(refs),9);self.assertEqual([len(w) for w in expected_waves()],[4,4,1,4,4,1]);self.assertEqual([t['arm'] for t in c['tasks']],['U']*9+['UV']*9)
            self.assertEqual(sum(t['arithmetic']['max_optimizer_steps'] for t in c['tasks']),1245080);self.assertEqual(sum(t['arithmetic']['max_epochs'] for t in c['tasks']),360)
            for a,b in zip(c['tasks'],old['tasks']):
                for k in ['structural_source_profile','data_config','data_stat','profile_sha','arithmetic','baseline_training']:self.assertEqual(a[k],b[k])
            for mutate in [lambda c:c['tasks'].reverse(),lambda c:c['tasks'].append(c['tasks'][0]),lambda c:c['tasks'][0].update(fold=1),lambda c:c.update(B_ref='paper_C1_r8_v1')]:
                bad=copy.deepcopy(c);mutate(bad);self.assertRaises(ValueError,validate,bad)
            empty=aggregate(c,[],reuse_records=refs);self.assertEqual(empty['goals']['goal1']['numerical_status'],'Not evaluated');self.assertEqual(empty['completed'],0);self.assertEqual(empty['V_equivalent_reuse'],9)
            records=[]
            for t in c['tasks']:
                b=c['existing_B'][t['baseline_pair_cell_id']];records.append((t,dict(task_id=t['id'],mse=b['mse']*.8,mae=b['mae']*(1 if t['dataset'] in ['PJM','NP','BE'] else .8))))
            result=aggregate(c,records,reuse_records=refs);self.assertEqual(result['goals']['goal2']['winning_cells'],7);self.assertEqual(result['goals']['goal3']['numerical_status'],'Not evaluated')
            self.assertEqual(aggregate(c,records[:-1],reuse_records=refs)['goals']['goal1']['numerical_status'],'Not evaluated');self.assertRaises(ValueError,aggregate,c,records+[records[0]],reuse_records=refs)
            self.assertRaises(ValueError,aggregate,c,records,reuse_records=refs+[refs[0]])
            f.output('scope',dict(fresh=18,referenced_V=9,old_config_valid=True,profile_data_training_equal=True,budget=[360,1245080],waves=[4,4,1,4,4,1],duplicate_missing_rejection=True))

    def test_template_rejection_and_guard_AST_worker_boundary(self):
            from moderntcn_dev_v7.scope import validate_launch_scope
            a=json.loads((f.EVIDENCE/'non-execution-preparation/launch-binding.template.json').read_text());c=f.STORED;ids=[t['id'] for t in c['tasks']];validate_launch_scope(c,ids,a)
            self.assertIsNone(a['commit']);self.assertFalse(a['review_approved']);self.assertFalse(c['execution_authorized'])
            with patch('moderntcn_dev_v7.launcher.git',side_effect=AssertionError('reject before Git/network')):
                self.assertRaises(PermissionError,launcher.validate_authorization,c,'unused',ids,a)
                for version in ['v1','v2','v3','v4','v5','v6']:
                    bad=copy.deepcopy(a);bad.update(review_approved=True,candidate_id='moderntcn-uv-dev-'+version);self.assertRaises(PermissionError,launcher.validate_authorization,c,'unused',ids,bad)
            old=(f.EVIDENCE.parent/'M6-ModernTCN-UV-v4-postrun-review-diagnosis-r1/performance/guards.py').read_text();new=(f.ROOT/'moderntcn_dev_v7/guards.py').read_text()
            def functions(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
            self.assertEqual(functions(old),functions(new))
            s=(f.ROOT/'moderntcn_dev_v7/worker.py').read_text();seq=['loss_guard(loss)','loss.backward()','gradients_collected(named)','opt.step();steps+=1','parameters_collected(named)','scheduler.after_successful_update()'];pos=[s.index(k) for k in seq];self.assertEqual(pos,sorted(pos));self.assertNotIn('torch.cuda.synchronize',s)
            for mutate in [lambda a:a.update(max_optimizer_updates=1),lambda a:a['task_profile_bindings'].clear(),lambda a:a['V_reuse'].clear()]:
                bad=copy.deepcopy(a);mutate(bad);self.assertRaises(ValueError,validate_launch_scope,c,ids,bad)
            f.output('guard-AST',dict(all_functions_and_error_class='exact AST',worker_boundaries=seq,extra_CUDA_sync=False,legacy_AUTH_rejected=True))
