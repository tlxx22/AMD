"""Frozen v6 differential CPU tests; no real data or numerical search."""
import copy,json,math,os,unittest,ast
from unittest.mock import patch
import torch
from torch.nn import functional as F
from tests import moderntcn_v6_fixtures as f
from moderntcn_dev_v6 import adapter,training_state,launcher
from moderntcn_dev_v6.modules import ConditionalVariableRotation, cayley
from moderntcn_dev_v3 import adapter as previous
from moderntcn_dev_v1.modules import WindowStateModulation
from moderntcn_dev_v6.guards import *

def gradients(m):return {n:p.grad for n,p in m.named_parameters()}
def common(m):return {n:p for n,p in m.state_dict().items() if not n.startswith(('u.','v.'))}
def close(a,b):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)

class Differential(unittest.TestCase):

    def test_controller_nonzero_rotation_schema_and_rng(self):
        rows=[]
        for L,C,y in [(12,11,0),(96,3,2)]:
            aux=[j for j in range(C) if j!=y]
            u=ConditionalVariableRotation(L,C,3,y,aux);old=WindowStateModulation(C,3,y,aux)
            with torch.no_grad():
                old.A.copy_(u.A);old.e.copy_(u.e);old.b.copy_(u.b)
                u.Theta.copy_(torch.randn_like(u.Theta)*.3)
            X=torch.randn(2,L,C);q,g=u.controller(X)
            f.equal(self,(q,g),old.controller(X))
            changed=X.clone();changed[1,:,y]+=2
            f.equal(self,u.controller(changed)[0][:1],q[:1])
            r,s=u.rotation(q);rr,_=u.rotation(u.controller(changed)[0])
            self.assertGreater(float((rr[1]-r[1]).abs().max()),0)
            f.equal(self,r[0],rr[0]);self.assertGreater(float((r-u.identity).abs().max()),0)
            f.equal(self,s,-s.transpose(-1,-2));f.equal(self,s.diagonal(dim1=-2,dim2=-1),torch.zeros(2,C))
            u.controller(X[:1]+3);f.equal(self,u.controller(X),old.controller(X))
            self.assertTrue(torch.isfinite(u.controller(torch.ones_like(X))[0]).all())
            for bad in [float('nan'),float('inf')]:
                xx=X.clone();xx[0,0,y]=bad;self.assertRaises(ValueError,u.controller,xx)
            noaux=ConditionalVariableRotation(L,C,3,y,[]);f.equal(self,noaux.controller(X)[0],torch.zeros(2,8))
            torch.manual_seed(700);made=ConditionalVariableRotation(L,C,3,y,aux);end=torch.get_rng_state();torch.manual_seed(700)
            aa=torch.empty(C,8,2)
            for a in aa:torch.nn.init.xavier_uniform_(a)
            f.equal(self,made.A,aa);f.equal(self,end,torch.get_rng_state())
            self.assertEqual(set(dict(made.named_parameters())),{'A','e','b','Theta'})
            self.assertEqual(made.upper_indices.t().tolist(),[[i,j] for i in range(C) for j in range(i+1,C)])
            rows.append(dict(C=C,target=y,controller='old formula exact',changed_history_changes_R=True,init_RNG='A only exact'))
        f.output('controller',rows)

    def test_cayley_sign_geometry_axes_and_reference_gradient(self):
        t=torch.tensor(.75);s=torch.tensor([[0.,.75],[-.75,0.]])
        ref=torch.stack([torch.stack([1-t*t,2*t]),torch.stack([-2*t,1-t*t])])/(1+t*t)
        close(cayley(s,torch.eye(2)),ref)
        rows=[]
        for C in [3,6,8,9,11]:
            u=ConditionalVariableRotation(12,C,3,0,list(range(1,C)))
            # Bounded substantial rotation fixture; not a parameter search.
            with torch.no_grad():u.Theta.copy_(torch.linspace(-1.5,1.5,u.Theta.numel()).reshape_as(u.Theta))
            X=torch.randn(2,12,C);q,_=u.controller(X);r,s=u.rotation(q)
            z=torch.randn(2,C,3,4,requires_grad=True);actual=u(z,X)
            expected=torch.stack([torch.stack([r[b]@z[b,:,d,:] for d in range(3)],1) for b in range(2)])
            close(actual,expected)
            eye=torch.eye(C).expand(2,C,C)
            gram=r.transpose(-1,-2)@r;close(gram,eye);close(torch.linalg.det(r),torch.ones(2))
            close(actual.square().sum(1),z.square().sum(1))
            close(torch.bmm(r.transpose(-1,-2),actual.reshape(2,C,-1)),z.reshape(2,C,-1))
            w=torch.randn(3,C,C)*.2;wr=w[None]@r[:,None]
            close(wr@wr.transpose(-1,-2),(w@w.transpose(-1,-2))[None].expand(2,-1,-1,-1))
            isolated=torch.zeros_like(z);isolated[:,:,1,2]=z[:,:,1,2]
            isolated_out=u(isolated,X);mask=torch.ones_like(z,dtype=torch.bool);mask[:,:,1,2]=False
            f.equal(self,isolated_out[mask],torch.zeros_like(isolated_out[mask]))
            changed=z.detach().clone();changed[1]+=2;f.equal(self,u(changed,X)[0],actual[0])
            coeff=torch.randn_like(actual)
            ga=torch.autograd.grad((actual*coeff).sum(),(z,u.Theta),retain_graph=True)
            gb=torch.autograd.grad((expected*coeff).sum(),(z,u.Theta),retain_graph=True)
            for a,b in zip(ga,gb):close(a,b)
            self.assertGreater(float((actual[:,1:]-z[:,1:]).abs().max()),0)
            rows.append(dict(C=C,orthogonal_max_abs=float((gram-eye).abs().max()),norm_max_abs=float((actual.square().sum(1)-z.square().sum(1)).abs().max()),rotation_from_identity=float((r-eye).abs().max()),same_R_all_D_N=True))
        f.output('geometry',rows)

    def test_local_double_cayley_derivative(self):
        q=torch.tensor([[.2,-.4,.1,.3,.2,.1,-.2,.4]],dtype=torch.double)
        theta=(torch.arange(24,dtype=torch.double).reshape(8,3)-12)*.01;theta.requires_grad_()
        indices=torch.triu_indices(3,3,1);eye=torch.eye(3,dtype=torch.double)
        def operation(th):
            values=q@th;upper=values.new_zeros((1,3,3));upper[:,indices[0],indices[1]]=values
            return cayley(upper-upper.transpose(-1,-2),eye)
        self.assertTrue(torch.autograd.gradcheck(operation,(theta,),eps=1e-6,atol=1e-5,rtol=1e-3))
        f.output('local-derivative',dict(fixed_q=True,double_local_only=True,gradcheck=True,production_dtype='float32 unchanged'))

    def test_nonzero_insertion_and_observer(self):
        for ds,Hn in [('UrbanEV',3),('PJM',24)]:
            p=f.profile(ds,H=Hn);m=adapter.build(p,use_u=True,use_v=True).eval();f.open_branches(m)
            block=m.model.stages[0].blocks[0];X=torch.randn(2,p['T'],p['C']);before=X.clone();state=copy.deepcopy(m.state_dict());rng=torch.get_rng_state();seen={};counts={}
            def hook(n):
                def h(mod,args,out):seen[n]=(args,out);counts[n]=counts.get(n,0)+1
                return h
            handles=[mod.register_forward_hook(hook(n)) for n,mod in [('ffn1drop1',block.ffn1drop1),('ffn1',block.ffn1drop2),('V',m.v),('U',m.u),('native',block.ffn2pw1),('drop1',block.ffn2drop1),('act',block.ffn2act),('drop2',block.ffn2drop2)]]
            y=m(X);C=p['C'];N=m.spec.value['N'];h=seen['ffn1'][1].reshape(2,C,64,N);z=h+seen['V'][1]
            f.equal(self,seen['V'][0][0],h);f.equal(self,seen['U'][0][0],z)
            r,_=m.u.rotation(m.u.controller(X)[0]);expected=torch.bmm(r,z.reshape(2,C,-1)).reshape_as(z)
            f.equal(self,seen['U'][1],expected);self.assertGreater(float((expected-z).abs().max()),0)
            f.equal(self,seen['native'][0][0],expected.permute(0,2,1,3).reshape(2,64*C,N))
            f.equal(self,seen['native'][1],F.conv1d(seen['native'][0][0],block.ffn2pw1.weight,block.ffn2pw1.bias,groups=64))
            f.equal(self,seen['drop1'][0][0],seen['native'][1]);f.equal(self,seen['act'][0][0],seen['drop1'][1]);self.assertTrue(all(n==1 for n in counts.values()))
            for handle in handles:handle.remove()
            yo,diag=m(X,return_diagnostics=True);f.equal(self,y,yo);f.equal(self,state,m.state_dict());f.equal(self,X,before);f.equal(self,rng,torch.get_rng_state());self.assertIn('U_shared_rotation',diag)
            prev=seen['V'][1].clone();hand=m.v.register_forward_hook(hook('V'))
            with torch.no_grad():m.u.Theta.mul_(2)
            m(X);hand.remove();f.equal(self,prev,seen['V'][1])
            m.use_u=False
            with patch('moderntcn_dev_v6.modules.cayley',side_effect=AssertionError('off U must not solve')):m(X)
        f.output('wiring',dict(nonzero_Theta=True,insertion='R(h+V) before native ffn2pw1',V_input='unrotated h',native_bias_once=True,FFN1_dropout_once=True,observer='exact input/output/state/RNG'))

    def test_Theta_then_controller_gradient_opening(self):
        rows=[]
        for ds,Hn in [('UrbanEV',3),('PJM',24)]:
            p=f.profile(ds,H=Hn);m=adapter.build(p,use_u=True,use_v=True);o=torch.optim.Adam(m.parameters(),lr=.01,weight_decay=1e-7);X=torch.randn(2,p['T'],p['C']);target=torch.randn(2,p['pred_len'],1);history=[]
            for step in range(3):
                o.zero_grad();adapter.loss(adapter.prediction(m,X),target).backward()
                gs={n:float(p.grad.abs().sum()) for n,p in m.named_parameters() if n.startswith(('u.','v.'))};history.append(gs)
                if step==0:
                    self.assertGreater(gs['u.Theta'],0);self.assertGreater(gs['v.A'],0)
                    for name in ['u.A','u.e','u.b','v.P_s']:self.assertEqual(gs[name],0)
                self.assertGreater(float(m.model.stages[0].blocks[0].ffn1pw1.weight.grad.abs().sum()),0)
                self.assertGreater(float(m.model.stages[0].blocks[0].ffn2pw1.weight.grad.abs().sum()),0)
                o.step()
            for name in ['u.A','u.e','u.b','v.P_s','v.W1.weight']:self.assertGreater(history[-1][name],0)
            rows.append(dict(dataset=ds,loss_gradients_before_Adam=history))
        f.output('gradient-opening',rows)

    def setUp(self):torch.manual_seed(401)

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
            f.output('neutral',dict(native_common_state_output_gradient_BN_RNG='exact',dropout_order='unchanged',zero_Theta_is_not_bypass=True))

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

    def test_same_v6_guard_exact_and_fault_boundaries(self):
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
            f.output('guard-integration',dict(full_v6_updates_per_path=2,all_outputs_gradients_states_RNG='exact',fault_boundaries='same order, no extra update',cases=rows,previous_GPU_evidence_reused=True))

    def test_profiles_parameters_target_loss(self):
            rows=[]
            for ds,var,Hn,U in [('UrbanEV','F1',3,269),('UrbanEV','F2',6,423),('UrbanEV','F3',9,512),('UrbanEV','F4',12,714),('PJM','F4',24,98)]:
                p=f.profile(ds,var,Hn);m=adapter.build(p,use_u=True,use_v=True);x=torch.randn(2,p['T'],p['C']);out=adapter.prediction(m,x);self.assertEqual(out.shape,(2,p['pred_len'],1));self.assertEqual(m.u.target,p['target_idx']);self.assertEqual(list(m.u.aux),p['aux_idx']);self.assertGreater(len(m.u.aux),0)
                u=sum(p.numel() for p in m.u.parameters());v=sum(p.numel() for p in m.v.parameters());self.assertEqual(u,U);self.assertEqual(v,572 if ds=='UrbanEV' else 2076);self.assertRaises(ValueError,adapter.loss,out,torch.zeros(2,p['pred_len'],p['C']))
                rows.append(dict(dataset=ds,variant=var,C=p['C'],L=p['T'],P=p['pred_len'],H=Hn,N=m.spec.value['N'],aux=p['aux_idx'],B=sum(p.numel() for n,p in m.named_parameters() if not n.startswith(('u.','v.'))),U=u,V=v,UV=u+v))
            f.output('parameter-counts',rows)

    def test_version_rejection_before_write_and_synthetic_IO(self):
            m=adapter.build(f.profile(),use_u=True,use_v=True);state=copy.deepcopy(m.state_dict());ck=adapter.synthetic_checkpoint(m)
            for version in ['v1','v2','v3','v4','v5']:
                bad=copy.deepcopy(ck);bad['format']='moderntcn-uv-synthetic-'+version;bad['spec']['candidate_id']='moderntcn-uv-dev-'+version
                self.assertRaises(ValueError,adapter.load_synthetic_checkpoint,m,bad);f.equal(self,state,m.state_dict())
                with patch('timemixer_dev.training_state.restore',side_effect=AssertionError('must reject before read/write')):self.assertRaises(ValueError,training_state.restore,'forbidden-real-path',m,None,None,None,dict(candidate_id=bad['spec']['candidate_id'],structural_spec=bad['spec']))
            path=f.EVIDENCE/'tests'/('synthetic-state-'+os.environ['MTCN_TEST_RUN']+'.pt');torch.save(ck,path);loaded=torch.load(path,map_location='cpu',weights_only=False);adapter.load_synthetic_checkpoint(m,loaded);f.equal(self,state,m.state_dict())
            bad=copy.deepcopy(ck);bad['spec']['target_idx']=2;self.assertRaises(ValueError,adapter.load_synthetic_checkpoint,m,bad);f.equal(self,state,m.state_dict())

class Metadata(unittest.TestCase):
    def test_scope_summary_duplicates_budget_and_old_config(self):
            from moderntcn_dev_v6.training_config import validate
            from moderntcn_dev_v6.scope import verified_v_records,expected_waves
            from moderntcn_dev_v6.summary import aggregate
            from moderntcn_dev_v5.training_config import validate as old_validate
            c=validate(f.STORED);old=json.loads((f.ROOT/'configs/moderntcn_uv_fold6_epf_conditional_variable_v5.json').read_text());old_validate(old);refs=verified_v_records(c)
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
            from moderntcn_dev_v6.scope import validate_launch_scope
            a=json.loads((f.EVIDENCE/'non-execution-preparation/launch-binding.template.json').read_text());c=f.STORED;ids=[t['id'] for t in c['tasks']];validate_launch_scope(c,ids,a)
            self.assertIsNone(a['commit']);self.assertFalse(a['review_approved']);self.assertFalse(c['execution_authorized'])
            with patch('moderntcn_dev_v6.launcher.git',side_effect=AssertionError('reject before Git/network')):
                self.assertRaises(PermissionError,launcher.validate_authorization,c,'unused',ids,a)
                for version in ['v1','v2','v3','v4','v5']:
                    bad=copy.deepcopy(a);bad.update(review_approved=True,candidate_id='moderntcn-uv-dev-'+version);self.assertRaises(PermissionError,launcher.validate_authorization,c,'unused',ids,bad)
            old=(f.EVIDENCE.parent/'M6-ModernTCN-UV-v4-postrun-review-diagnosis-r1/performance/guards.py').read_text();new=(f.ROOT/'moderntcn_dev_v6/guards.py').read_text()
            def functions(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
            self.assertEqual(functions(old),functions(new))
            s=(f.ROOT/'moderntcn_dev_v6/worker.py').read_text();seq=['loss_guard(loss)','loss.backward()','gradients_collected(named)','opt.step();steps+=1','parameters_collected(named)','scheduler.after_successful_update()'];pos=[s.index(k) for k in seq];self.assertEqual(pos,sorted(pos));self.assertNotIn('torch.cuda.synchronize',s)
            for mutate in [lambda a:a.update(max_optimizer_updates=1),lambda a:a['task_profile_bindings'].clear(),lambda a:a['V_reuse'].clear()]:
                bad=copy.deepcopy(a);mutate(bad);self.assertRaises(ValueError,validate_launch_scope,c,ids,bad)
            f.output('guard-AST',dict(all_functions_and_error_class='exact AST',worker_boundaries=seq,extra_CUDA_sync=False,legacy_AUTH_rejected=True))
