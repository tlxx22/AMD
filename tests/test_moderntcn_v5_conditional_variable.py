"""Frozen v5 differential CPU tests; no real data or numerical search."""
import copy,json,math,os,unittest,ast
from unittest.mock import patch
import torch
from torch.nn import functional as F
from tests import moderntcn_v5_fixtures as f
from moderntcn_dev_v5 import adapter,training_state,launcher
from moderntcn_dev_v5.modules import ConditionalVariableOperator
from moderntcn_dev_v3 import adapter as previous
from moderntcn_dev_v1.modules import WindowStateModulation
from moderntcn_dev_v5.guards import *

def gradients(m):return {n:p.grad for n,p in m.named_parameters()}
def common(m):return {n:p for n,p in m.state_dict().items() if not n.startswith(('u.','v.'))}
def close(a,b):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)

class Differential(unittest.TestCase):
    def setUp(self):torch.manual_seed(401)
    def test_controller_exact_order_target_batch_and_finite(self):
        for L,C,y in [(12,11,0),(96,3,2)]:
            aux=[j for j in range(C) if j!=y];u=ConditionalVariableOperator(L,C,3,y,aux);old=WindowStateModulation(C,3,y,aux)
            with torch.no_grad():old.A.copy_(u.A);old.e.copy_(u.e);old.b.copy_(u.b)
            X=torch.randn(2,L,C);q,g=u.controller(X);f.equal(self,(q,g),old.controller(X))
            mu=X.mean(1);var=(X-mu[:,None]).square().mean(1);h=torch.tanh(torch.einsum('crs,bcs->bcr',u.A,torch.stack([mu,torch.log1p(var)],-1))+u.e)
            gr=torch.sigmoid((h[:,y,None]*h[:,aux]).sum(-1)/math.sqrt(8)+u.b);qr=(gr[:,:,None]*h[:,aux]).sum(1)/(1+gr.sum(1,keepdim=True));f.equal(self,q,qr)
            # Batch independence is checked at the SAME shape/algorithm. B=1
            # versus B=2 may select a different einsum kernel; it is not an
            # original-order bitwise identity contract. Keep old-formula exact
            # at both shapes and exact re-entry after an intervening batch.
            other=X.clone();other[1]+=3
            f.equal(self,u.controller(other)[0][:1],q[:1])
            f.equal(self,u.controller(X[:1]),old.controller(X[:1]))
            u.controller(X[:1]+2);f.equal(self,u.controller(X)[0],q)
            z=torch.ones_like(X);self.assertTrue(torch.isfinite(u.controller(z)[0]).all());bad=X.clone();bad[0,0,y]=float('nan');self.assertRaises(ValueError,u.controller,bad)
            noaux=ConditionalVariableOperator(L,C,3,y,[]);f.equal(self,noaux.controller(X)[0],torch.zeros(2,8))
            # Explicitly prescribed random draw order and no legacy projection registrations.
            torch.manual_seed(700);made=ConditionalVariableOperator(L,C,3,y,aux);end=torch.get_rng_state();torch.manual_seed(700)
            aa=torch.empty(C,8,2);pp=torch.empty(3,8,C)
            for v in aa:torch.nn.init.xavier_uniform_(v)
            for v in pp:torch.nn.init.xavier_uniform_(v)
            f.equal(self,made.A,aa);f.equal(self,made.P_weight.view(3,8,C),pp);f.equal(self,end,torch.get_rng_state());self.assertEqual(set(dict(made.named_parameters())),{'A','e','b','P_weight','Q_weight'})
        f.output('controller',dict(old_formula='exact with copied A/e/b',targets=[0,2],init_order='A per variable then P per group exact RNG',no_aux='q exact zero'))

    def test_grouped_C_explicit_effective_weight_and_gradients(self):
        rows=[]
        for L,C,N,y in [(12,11,12,0),(96,3,24,2)]:
            D=3;u=ConditionalVariableOperator(L,C,D,y,[j for j in range(C) if j!=y])
            with torch.no_grad():u.Q_weight.copy_(torch.randn_like(u.Q_weight)*.04)
            X=torch.randn(2,L,C,requires_grad=True);x=torch.randn(2,D*C,N,requires_grad=True);q,_=u.controller(X);actual=u(x,X)
            P=u.P_weight.reshape(D,8,C);Q=u.Q_weight.reshape(D,C,8);xx=x.reshape(2,D,C,N)
            ref=torch.stack([torch.stack([(Q[d]*q[b][None])@P[d]@xx[b,d] for d in range(D)]) for b in range(2)]).reshape_as(actual)
            close(actual,ref);weight=torch.randn_like(actual);pars=(x,X,*u.parameters());ga=torch.autograd.grad((actual*weight).sum(),pars,retain_graph=True);gb=torch.autograd.grad((ref*weight).sum(),pars,retain_graph=True)
            for a,b in zip(ga,gb):close(a,b)
            nativeW=torch.randn(D,C,C)*.02;bias=torch.randn(D*C)*.03
            a0=F.conv1d(x,nativeW.reshape(D*C,C,1),bias,groups=D)
            weff=torch.stack([torch.stack([(nativeW[d]+(Q[d]*q[b][None])@P[d])@xx[b,d]+bias.reshape(D,C)[d,:,None] for d in range(D)]) for b in range(2)]).reshape_as(actual)
            close(a0+actual,weff)
            one=torch.zeros_like(x);one[:,:C]=x[:,:C];isolated=u(one,X).reshape(2,D,C,N);f.equal(self,isolated[:,1:],torch.zeros_like(isolated[:,1:]))
            changed=x.clone();changed[:,:,0]+=1;out=u(changed,X);f.equal(self,out[:,:,1:],actual[:,:,1:]);self.assertGreater(float(actual.abs().max()),0)
            rows.append(dict(C=C,N=N,axes='per D, C variable mixing, no temporal mixing',reference='explicit W_eff float32 2e-6/2e-5 forward and gradients',non_target_delta=float(actual.reshape(2,D,C,N)[:,:,(y+1)%C].abs().max())))
        f.output('operator-reference',rows)

    def test_nonzero_insertion_observer_and_branch_inputs(self):
        rows=[]
        for ds,Hn in [('UrbanEV',3),('PJM',24)]:
            p=f.profile(ds,H=Hn);m=adapter.build(p,use_u=True,use_v=True).eval();f.open_branches(m);block=m.model.stages[0].blocks[0];X=torch.randn(2,p['T'],p['C']);saved=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone();before=X.clone();seen={};counts={}
            def hook(n):
                def h(mod,args,out):seen[n]=(args,out);counts[n]=counts.get(n,0)+1
                return h
            handles=[mod.register_forward_hook(hook(n)) for n,mod in [('ffn1',block.ffn1drop2),('V',m.v),('U',m.u),('native',block.ffn2pw1),('drop1',block.ffn2drop1),('act',block.ffn2act),('drop2',block.ffn2drop2)]]
            y=m(X);B=2;C=p['C'];D=64;N=m.spec.value['N'];h=seen['ffn1'][1].reshape(B,C,D,N);dv=seen['V'][1];flat=(h+dv).permute(0,2,1,3).reshape(B,D*C,N)
            f.equal(self,seen['V'][0][0],h);f.equal(self,seen['U'][0][0],flat);f.equal(self,seen['native'][0][0],flat);f.equal(self,seen['drop1'][0][0],seen['native'][1]+seen['U'][1]);f.equal(self,seen['act'][0][0],seen['drop1'][1]);self.assertTrue(all(n==1 for n in counts.values()));self.assertGreater(float(seen['U'][1].abs().max()),0)
            for handle in handles:handle.remove()
            torch.set_rng_state(rng);yo,diag=m(X,return_diagnostics=True);f.equal(self,y,yo);f.equal(self,saved,m.state_dict());f.equal(self,before,X);f.equal(self,rng,torch.get_rng_state());self.assertIn('U_variable_operator',diag)
            prev=dv.detach().clone();hand=m.v.register_forward_hook(hook('V'))
            with torch.no_grad():m.u.Q_weight.mul_(2)
            m(X);hand.remove();f.equal(self,prev,seen['V'][1]);m.use_v=False;hand=m.u.register_forward_hook(hook('U'));m(X);hand.remove();self.assertGreater(float(seen['U'][1].abs().max()),0)
            rows.append(dict(dataset=ds,insertion='native a0 -> a0+delta -> original dropout/GELU exact',FFN1_once=True,U_does_not_change_direct_V=True,observer='exact input/output/state/RNG'))
        f.output('wiring',rows)

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
        f.output('neutral',dict(native_common_state_output_gradient_BN_RNG='exact',dropout_order='unchanged',zero_Q_is_not_bypass=True))

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

    def test_same_v5_guard_exact_and_fault_boundaries(self):
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
        f.output('guard-integration',dict(full_v5_updates_per_path=2,all_outputs_gradients_states_RNG='exact',fault_boundaries='same order, no extra update',cases=rows,previous_GPU_evidence_reused=True))

    def test_Q_then_P_controller_loss_gradient_opening(self):
        rows=[]
        for ds,Hn in [('UrbanEV',3),('PJM',24)]:
            p=f.profile(ds,H=Hn);m=adapter.build(p,use_u=True,use_v=True);o=torch.optim.Adam(m.parameters(),lr=.01,weight_decay=1e-7);X=torch.randn(2,p['T'],p['C']);target=torch.randn(2,p['pred_len'],1);history=[]
            for step in range(3):
                o.zero_grad();y=adapter.prediction(m,X);adapter.loss(y,target).backward();gs={n:float(p.grad.abs().sum()) for n,p in m.named_parameters() if n.startswith(('u.','v.'))};history.append(gs)
                if step==0:
                    self.assertGreater(gs['u.Q_weight'],0);self.assertGreater(gs['v.A'],0)
                    for name in ['u.A','u.e','u.b','u.P_weight','v.P_s']:self.assertEqual(gs[name],0)
                self.assertGreater(float(m.model.stages[0].blocks[0].ffn1pw1.weight.grad.abs().sum()),0);self.assertGreater(float(m.model.stages[0].blocks[0].ffn2pw1.weight.grad.abs().sum()),0);o.step()
            for name in ['u.A','u.e','u.b','u.P_weight','v.P_s','v.W1.weight']:self.assertGreater(history[-1][name],0)
            rows.append(dict(dataset=ds,loss_gradients_before_Adam=history))
        f.output('gradient-opening',rows)

    def test_profiles_parameters_target_loss(self):
        rows=[]
        for ds,var,Hn,U in [('UrbanEV','F1',3,6293),('UrbanEV','F2',6,8391),('UrbanEV','F3',9,9440),('UrbanEV','F4',12,11538),('PJM','F4',24,3146)]:
            p=f.profile(ds,var,Hn);m=adapter.build(p,use_u=True,use_v=True);x=torch.randn(2,p['T'],p['C']);out=adapter.prediction(m,x);self.assertEqual(out.shape,(2,p['pred_len'],1));self.assertEqual(m.u.target,p['target_idx']);self.assertEqual(list(m.u.aux),p['aux_idx']);self.assertGreater(len(m.u.aux),0)
            u=sum(p.numel() for p in m.u.parameters());v=sum(p.numel() for p in m.v.parameters());self.assertEqual(u,U);self.assertEqual(v,572 if ds=='UrbanEV' else 2076);self.assertRaises(ValueError,adapter.loss,out,torch.zeros(2,p['pred_len'],p['C']))
            rows.append(dict(dataset=ds,variant=var,C=p['C'],L=p['T'],P=p['pred_len'],H=Hn,N=m.spec.value['N'],aux=p['aux_idx'],B=sum(p.numel() for n,p in m.named_parameters() if not n.startswith(('u.','v.'))),U=u,V=v,UV=u+v))
        f.output('parameter-counts',rows)

    def test_version_rejection_before_write_and_synthetic_IO(self):
        m=adapter.build(f.profile(),use_u=True,use_v=True);state=copy.deepcopy(m.state_dict());ck=adapter.synthetic_checkpoint(m)
        for version in ['v1','v2','v3','v4']:
            bad=copy.deepcopy(ck);bad['format']='moderntcn-uv-synthetic-'+version;bad['spec']['candidate_id']='moderntcn-uv-dev-'+version
            self.assertRaises(ValueError,adapter.load_synthetic_checkpoint,m,bad);f.equal(self,state,m.state_dict())
            with patch('timemixer_dev.training_state.restore',side_effect=AssertionError('must reject before read/write')):self.assertRaises(ValueError,training_state.restore,'forbidden-real-path',m,None,None,None,dict(candidate_id=bad['spec']['candidate_id'],structural_spec=bad['spec']))
        path=f.EVIDENCE/'tests'/('synthetic-state-'+os.environ['MTCN_TEST_RUN']+'.pt');torch.save(ck,path);loaded=torch.load(path,map_location='cpu',weights_only=False);adapter.load_synthetic_checkpoint(m,loaded);f.equal(self,state,m.state_dict())
        bad=copy.deepcopy(ck);bad['spec']['target_idx']=2;self.assertRaises(ValueError,adapter.load_synthetic_checkpoint,m,bad);f.equal(self,state,m.state_dict())

class Metadata(unittest.TestCase):
    def test_goal1_same_market_AND_missing_ties(self):
        from moderntcn_dev_v5.summary import goal1
        from moderntcn_dev_v5.scope import MARKETS
        def row(u=2.,v=2.,uv=1.):return {a:dict(mse=z,mae=z) for a,z in [('U',u),('V',v),('UV',uv)]}
        urban={str(h):row() for h in [3,6,9,12]};epf={d:row(2,2,3) for d in MARKETS};cases=[]
        a=goal1(urban,epf);self.assertTrue(a['Urban_OK']);self.assertFalse(a['EPF_OK']);self.assertEqual(a['numerical_status'],'Not met');cases.append(a)
        # Different markets supply MSE, MAE, or different deletion comparisons: none qualify.
        epf['PJM']['UV']=dict(mse=1.,mae=3.);epf['NP']['UV']=dict(mse=3.,mae=1.)
        epf['BE']=dict(U=dict(mse=2.,mae=.5),V=dict(mse=.5,mae=2.),UV=dict(mse=1.,mae=1.))
        a=goal1(urban,epf);self.assertEqual(a['qualifying_markets'],[]);self.assertEqual(a['numerical_status'],'Not met');cases.append(a)
        epf['DE']=row();a=goal1(urban,epf);self.assertEqual(a['qualifying_markets'],['DE']);self.assertEqual(a['numerical_status'],'Met');cases.append(a)
        epf['DE']['UV']['mse']=2.;self.assertEqual(goal1(urban,epf)['numerical_status'],'Not met')
        epf['DE']=row();epf['FR']['U']=None;a=goal1(urban,epf);self.assertEqual(a['numerical_status'],'Not evaluated');self.assertFalse(a['coverage_complete']);cases.append(a)
        epf['FR']=row(2,2,3);urban['3']['UV']=dict(mse=4.,mae=4.)
        # One H loses but mean=(4+1+1+1)/4 <2: no every-H condition.
        self.assertTrue(goal1(urban,epf)['Urban_OK']);urban['6']['UV']=dict(mse=4.,mae=4.);self.assertEqual(goal1(urban,epf)['numerical_status'],'Not met')
        f.output('goal1-policy-fixtures',cases)

    def test_scope_summary_duplicates_budget_and_old_config(self):
        from moderntcn_dev_v5.training_config import validate
        from moderntcn_dev_v5.scope import verified_v_records,expected_waves
        from moderntcn_dev_v5.summary import aggregate
        from moderntcn_dev_v4.training_config import validate as old_validate
        c=validate(f.STORED);old=json.loads((f.ROOT/'configs/moderntcn_uv_fold6_epf_aux_content_v4.json').read_text());old_validate(old);refs=verified_v_records(c)
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
        from moderntcn_dev_v5.scope import validate_launch_scope
        a=json.loads((f.EVIDENCE/'non-execution-preparation/launch-binding.template.json').read_text());c=f.STORED;ids=[t['id'] for t in c['tasks']];validate_launch_scope(c,ids,a)
        self.assertIsNone(a['commit']);self.assertFalse(a['review_approved']);self.assertFalse(c['execution_authorized'])
        with patch('moderntcn_dev_v5.launcher.git',side_effect=AssertionError('reject before Git/network')):
            self.assertRaises(PermissionError,launcher.validate_authorization,c,'unused',ids,a)
            for version in ['v1','v2','v3','v4']:
                bad=copy.deepcopy(a);bad.update(review_approved=True,candidate_id='moderntcn-uv-dev-'+version);self.assertRaises(PermissionError,launcher.validate_authorization,c,'unused',ids,bad)
        old=(f.EVIDENCE.parent/'M6-ModernTCN-UV-v4-postrun-review-diagnosis-r1/performance/guards.py').read_text();new=(f.ROOT/'moderntcn_dev_v5/guards.py').read_text()
        def functions(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        self.assertEqual(functions(old),functions(new))
        s=(f.ROOT/'moderntcn_dev_v5/worker.py').read_text();seq=['loss_guard(loss)','loss.backward()','gradients_collected(named)','opt.step();steps+=1','parameters_collected(named)','scheduler.after_successful_update()'];pos=[s.index(k) for k in seq];self.assertEqual(pos,sorted(pos));self.assertNotIn('torch.cuda.synchronize',s)
        for mutate in [lambda a:a.update(max_optimizer_updates=1),lambda a:a['task_profile_bindings'].clear(),lambda a:a['V_reuse'].clear()]:
            bad=copy.deepcopy(a);mutate(bad);self.assertRaises(ValueError,validate_launch_scope,c,ids,bad)
        f.output('guard-AST',dict(all_functions_and_error_class='exact AST',worker_boundaries=seq,extra_CUDA_sync=False,legacy_AUTH_rejected=True))

if __name__=='__main__':unittest.main()
