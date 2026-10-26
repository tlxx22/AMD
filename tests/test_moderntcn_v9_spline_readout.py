"""Directed synthetic CPU acceptance, independently specified spline oracle."""
import ast,copy,json,math,os,unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from tests import moderntcn_v9_fixtures as f
from moderntcn_dev_v9 import adapter,training_state,launcher
from moderntcn_dev_v9.modules import SplineTargetReadout,fixed_knots
from moderntcn_dev_v3 import adapter as previous
from moderntcn_dev_v9.guards import *
def gradients(m):return {n:p.grad for n,p in m.named_parameters()}
def common(m):return {n:p for n,p in m.state_dict().items() if not n.startswith(('u.','v.'))}
def close(a,b):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)

def scalar_basis(u,k,j,d):
    if d==0:return ((u>=k[j]) & (u<k[j+1])).to(u.dtype)
    return (u-k[j])/(k[j+d]-k[j])*scalar_basis(u,k,j,d-1)+(k[j+d+1]-u)/(k[j+d+1]-k[j+1])*scalar_basis(u,k,j+1,d-1)
def reference(a,coef,k):
    # Deliberately scalar per feature/base/output, independently recursive.
    b=torch.stack([torch.stack([torch.stack([scalar_basis(torch.tanh(x),k,j,3) for j in range(8)]) for x in row]) for row in a])
    y=torch.stack([torch.stack([sum(coef[p,i,j]*b[n,i,j] for i in range(a.shape[1]) for j in range(8))/a.shape[1] for p in range(coef.shape[0])]) for n in range(a.shape[0])])
    return y,b

class Differential(unittest.TestCase):
    def setUp(self):torch.manual_seed(401)

    def test_scalar_reference_nodes_endpoints_and_saturation(self):
        coords=torch.tensor([-1.,-.999999,-.6,-.2,0.,.2,.6,.999999,1.])
        # Atanh only for interior; finite a=+/-30 intentionally reaches tanh +/-1.
        a=torch.cat([torch.tensor([-30.]),torch.atanh(coords[1:-1]),torch.tensor([30.])])[None]
        u=SplineTargetReadout(a.shape[1],2);u.coef.data.copy_(torch.linspace(-.7,.9,u.coef.numel()).reshape_as(u.coef))
        actual=u(a);expected,b=reference(a,u.coef,u.knots);close(actual,expected);close(u.basis(a),b)
        # Uniform cubic cardinal basis at mathematical zero: [1,23,23,1]/48.
        midpoint=u.basis(torch.zeros_like(a))[0,0];hand=torch.zeros(8);hand[2:6]=torch.tensor([1.,23.,23.,1.])/48
        close(midpoint,hand)
        f.output('scalar-reference',dict(max_abs_forward=float((actual-expected).abs().max()),max_abs_basis=float((u.basis(a)-b).abs().max()),midpoint_error=float((midpoint-hand).abs().max()),atol=2e-6,rtol=2e-5,finite_saturation=True))

    def test_basis_properties_bound_and_rejections(self):
        a=torch.linspace(-30,30,241).reshape(1,-1);u=SplineTargetReadout(a.shape[1],3);u.coef.data.copy_(torch.linspace(-2,3,u.coef.numel()).reshape_as(u.coef));b=u.basis(a)
        self.assertTrue(bool((b>=0).all()));close(b.sum(-1),torch.ones_like(a));self.assertTrue(bool(((b>0).sum(-1)<=4).all()))
        bound=u.coef.abs().max(-1).values.mean(-1);self.assertTrue(bool((u(a).abs()<=bound+2e-6).all()));f.equal(self,u.knots,fixed_knots())
        # Near interior knots, function and first derivative converge continuously.
        continuity=[]
        for c in [-.6,-.2,.2,.6]:
            x=torch.atanh(torch.tensor(c));ab=torch.full_like(a,float(x));minus=ab-1e-5;plus=ab+1e-5
            jump=float((u.basis(plus)-u.basis(minus)).abs().max());self.assertLess(jump,5e-5);continuity.append(jump)
        for bad in [a.double(),a[:,:-1],a.reshape(-1)]:self.assertRaises(ValueError,u,bad)
        for value in [float('nan'),float('inf'),-float('inf')]:
            bad=a.clone();bad[0,0]=value;self.assertRaises(ValueError,u,bad)
        u.coef.data[0,0,0]=float('nan');self.assertRaises(ValueError,u,a)
        u.coef.data.zero_();u.knots[1]=u.knots[0];self.assertRaises(ValueError,u,a)
        f.output('basis-properties',dict(nonnegative=True,partition_residual=float((b.sum(-1)-1).abs().max()),max_active=int((b>0).sum(-1).max()),continuity_absolute=continuity,bound='abs(delta_p)<=mean_i max_k abs(coef_pik)',invalid_rejected=True))

    def test_nonzero_reference_gradients_and_local_double(self):
        a=torch.tensor([[-.73,.11,.47],[.23,-.31,.67]],requires_grad=True);u=SplineTargetReadout(3,2);u.coef.data.copy_(torch.linspace(-.5,.6,u.coef.numel()).reshape_as(u.coef))
        y=u(a);r,b=reference(a,u.coef,u.knots);close(y,r);ga=torch.autograd.grad(y.square().sum(),[a,u.coef],retain_graph=True);gr=torch.autograd.grad(r.square().sum(),[a,u.coef]);errors=[]
        for x,z in zip(ga,gr):close(x,z);errors.append(float((x-z).abs().max()))
        # Pure local double oracle: finite differences of independent recurrence.
        ad=torch.tensor([[-.71,.13]],dtype=torch.double,requires_grad=True);co=torch.linspace(-.4,.5,16,dtype=torch.double).reshape(1,2,8).requires_grad_();kd=fixed_knots().double()
        self.assertTrue(torch.autograd.gradcheck(lambda x,c:reference(x,c,kd)[0],(ad,co),eps=1e-6,atol=2e-6,rtol=2e-5))
        f.output('local-gradients',dict(max_abs=errors,double_local_gradcheck=True,production='float32',tolerance=[2e-6,2e-5]))

    def test_nonzero_final_head_wiring_and_observer(self):
        rows=[]
        for ds,Hn in [('UrbanEV',6),('PJM',24)]:
            p=f.profile(ds,H=Hn);m=adapter.build(p,use_u=True,use_v=True);f.open_branches(m);m.eval();X=torch.randn(2,p['T'],p['C']);original=X.clone();state=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone();out=m(X);seen={};calls={};order=[];hooks=[];block=m.model.stages[0].blocks[0]
            def hook(name):
                def observe(mod,args,result):seen[name]=(args,result);calls[name]=calls.get(name,0)+1;order.append(name)
                return observe
            def rev(mod,args,result):
                name='revin_'+args[1];seen[name]=(args,result.clone());calls[name]=calls.get(name,0)+1;order.append(name)
            hooks.append(m.model.revin_layer.register_forward_hook(rev))
            for name,mod in [('u',m.u),('v',m.v),('stem',m.model.downsample_layers[0]),('h',block.ffn1drop2),('ffn1drop1',block.ffn1drop1),('ffn2pw1',block.ffn2pw1),('ffn2drop1',block.ffn2drop1),('ffn2act',block.ffn2act),('ffn2pw2',block.ffn2pw2),('ffn2drop2',block.ffn2drop2),('BN',block.norm),('head',m.model.head)]:hooks.append(mod.register_forward_hook(hook(name)))
            actual,diag=m(X,return_diagnostics=True)
            for h in hooks:h.remove()
            f.equal(self,out,actual);f.equal(self,state,m.state_dict());f.equal(self,original,X);f.equal(self,rng,torch.get_rng_state());self.assertTrue(all(n==1 for n in calls.values()))
            final=seen['head'][0][0];B,C,D,N=final.shape;initial=seen['stem'][1].reshape(B,C,D,N);post=seen['ffn2drop2'][1].reshape(B,D,C,N).permute(0,2,1,3)
            f.equal(self,final,initial+post);f.equal(self,seen['u'][0][0],final[:,p['target_idx']].reshape(B,D*N))
            h=seen['h'][1].reshape(B,C,D,N);f.equal(self,seen['v'][0][0],h);f.equal(self,seen['v'][0][1],seen['revin_norm'][1].permute(0,2,1));f.equal(self,seen['v'][0][2],X)
            f.equal(self,seen['ffn2pw1'][0][0],(h+seen['v'][1]).permute(0,2,1,3).reshape(B,D*C,N))
            y0=seen['head'][1].permute(0,2,1);combined=seen['revin_denorm'][0][0];t=p['target_idx'];f.equal(self,combined[:,:,p['aux_idx']],y0[:,:,p['aux_idx']]);f.equal(self,combined[:,:,t],y0[:,:,t]+seen['u'][1]);self.assertGreater(float(seen['u'][1].abs().sum()),0)
            self.assertLess(order.index('head'),order.index('u'));self.assertLess(order.index('u'),order.index('revin_denorm'));self.assertIn('U_spline_readout',diag)
            m.use_v=False;self.assertEqual(m(X).shape,(2,p['pred_len'],p['C']));m.use_u=False
            with patch.object(m.u,'forward',side_effect=AssertionError('disabled U')):m(X)
            rows.append(dict(dataset=ds,calls=calls,order=order,post_skip=True,flatten='D-major/N-minor',aux_exact=True,observer='exact'))
        f.output('wiring',rows)

    def test_loss_gradient_learning_and_zero_path_execution(self):
        rows=[]
        for ds,Hn in [('UrbanEV',3),('PJM',24)]:
            p=f.profile(ds,H=Hn);m=adapter.build(p,use_u=True,use_v=True);opt=torch.optim.Adam(m.parameters(),lr=.01,weight_decay=1e-7);X=torch.randn(2,p['T'],p['C']);target=torch.randn(2,p['pred_len'],1);history=[];calls=[]
            hook=m.u.register_forward_hook(lambda module,args,result:calls.append(result.detach().clone()))
            for step in range(3):
                opt.zero_grad();adapter.loss(adapter.prediction(m,X),target).backward();g={n:float(p.grad.abs().sum()) for n,p in m.named_parameters() if n.startswith(('u.','v.'))};history.append(g)
                self.assertGreater(g['u.coef'],0);self.assertGreater(float(m.model.head.linear.weight.grad.abs().sum()),0)
                self.assertGreater(float(m.model.stages[0].blocks[0].ffn1pw1.weight.grad.abs().sum()),0);self.assertGreater(float(m.model.stages[0].blocks[0].ffn2pw1.weight.grad.abs().sum()),0);opt.step()
            hook.remove();self.assertEqual(len(calls),3);self.assertEqual(float(calls[0].abs().sum()),0);self.assertGreater(float(calls[1].abs().sum()),0);self.assertGreater(history[-1]['v.P_s'],0)
            rows.append(dict(dataset=ds,loss_gradients=history,zero_exit_called=True,uncovered_coefficients_allowed_zero=True))
        f.output('gradient-opening',rows)

    def test_profiles_parameters_target_loss(self):
        rows=[]
        for ds,var,Hn,U in [('UrbanEV','F1',3,6144),('UrbanEV','F2',6,6144),('UrbanEV','F3',9,6144),('UrbanEV','F4',12,6144),('PJM','F4',24,294912)]:
            p=f.profile(ds,var,Hn);m=adapter.build(p,use_u=True,use_v=True);out=adapter.prediction(m,torch.randn(2,p['T'],p['C']));self.assertEqual(out.shape,(2,p['pred_len'],1));self.assertEqual(m.spec.value['target_idx'],p['target_idx']);self.assertEqual(m.spec.value['label_horizon'],Hn)
            u=sum(p.numel() for p in m.u.parameters());v=sum(p.numel() for p in m.v.parameters());self.assertEqual(u,U);self.assertEqual(v,572 if ds=='UrbanEV' else 2076);self.assertRaises(ValueError,adapter.loss,out,torch.zeros(2,p['pred_len'],p['C']));self.assertEqual(set(dict(m.u.named_parameters())),{'coef'});self.assertEqual(tuple(m.u.coef.shape),(p['pred_len'],64*m.spec.value['N'],8));self.assertEqual(m.u.knots.numel()*4,48)
            rows.append(dict(dataset=ds,variant=var,C=p['C'],L=p['T'],P=p['pred_len'],H=Hn,N=m.spec.value['N'],B=sum(p.numel() for n,p in m.named_parameters() if not n.startswith(('u.','v.'))),U=u,V=v,UV=u+v,buffer_bytes=48))
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

    def test_same_v9_guard_exact_and_fault_boundaries(self):
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
            f.output('guard-integration',dict(full_v9_updates_per_path=2,all_outputs_gradients_states_RNG='exact',fault_boundaries='same order, no extra update',cases=rows,previous_GPU_evidence_reused=True))

    def test_version_rejection_before_write_and_synthetic_IO(self):
            m=adapter.build(f.profile(),use_u=True,use_v=True);state=copy.deepcopy(m.state_dict());ck=adapter.synthetic_checkpoint(m)
            for version in ['v1','v2','v3','v4','v5','v6','v7','v8']:
                bad=copy.deepcopy(ck);bad['format']='moderntcn-uv-synthetic-'+version;bad['spec']['candidate_id']='moderntcn-uv-dev-'+version
                self.assertRaises(ValueError,adapter.load_synthetic_checkpoint,m,bad);f.equal(self,state,m.state_dict())
                with patch('timemixer_dev.training_state.restore',side_effect=AssertionError('must reject before read/write')):self.assertRaises(ValueError,training_state.restore,'forbidden-real-path',m,None,None,None,dict(candidate_id=bad['spec']['candidate_id'],structural_spec=bad['spec']))
            path=f.EVIDENCE/'tests'/('synthetic-state-'+os.environ['MTCN_TEST_RUN']+'.pt');torch.save(ck,path);loaded=torch.load(path,map_location='cpu',weights_only=False);adapter.load_synthetic_checkpoint(m,loaded);f.equal(self,state,m.state_dict())
            bad=copy.deepcopy(ck);bad['spec']['target_idx']=2;self.assertRaises(ValueError,adapter.load_synthetic_checkpoint,m,bad);f.equal(self,state,m.state_dict())
            bad=copy.deepcopy(ck);bad['state_dict']['u.knots'][0]+=0.01;self.assertRaises(ValueError,adapter.load_synthetic_checkpoint,m,bad);f.equal(self,state,m.state_dict())

class Metadata(unittest.TestCase):
    def test_scope_summary_duplicates_budget_and_old_config(self):
            from moderntcn_dev_v9.training_config import validate
            from moderntcn_dev_v9.scope import verified_v_records,expected_waves
            from moderntcn_dev_v9.summary import aggregate
            from moderntcn_dev_v8.training_config import validate as old_validate
            c=validate(f.STORED);old=json.loads((f.ROOT/'configs/moderntcn_uv_fold6_epf_lagged_auxiliary_v8.json').read_text());old_validate(old);refs=verified_v_records(c)
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
            from moderntcn_dev_v9.scope import validate_launch_scope
            a=json.loads((f.EVIDENCE/'non-execution-preparation/launch-binding.template.json').read_text());c=f.STORED;ids=[t['id'] for t in c['tasks']];validate_launch_scope(c,ids,a)
            self.assertIsNone(a['commit']);self.assertFalse(a['review_approved']);self.assertFalse(c['execution_authorized'])
            with patch('moderntcn_dev_v9.launcher.git',side_effect=AssertionError('reject before Git/network')):
                self.assertRaises(PermissionError,launcher.validate_authorization,c,'unused',ids,a)
                for version in ['v1','v2','v3','v4','v5','v6','v7','v8']:
                    bad=copy.deepcopy(a);bad.update(review_approved=True,candidate_id='moderntcn-uv-dev-'+version);self.assertRaises(PermissionError,launcher.validate_authorization,c,'unused',ids,bad)
            old=(f.EVIDENCE.parent/'M6-ModernTCN-UV-v4-postrun-review-diagnosis-r1/performance/guards.py').read_text();new=(f.ROOT/'moderntcn_dev_v9/guards.py').read_text()
            def functions(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
            self.assertEqual(functions(old),functions(new))
            s=(f.ROOT/'moderntcn_dev_v9/worker.py').read_text();seq=['loss_guard(loss)','loss.backward()','gradients_collected(named)','opt.step();steps+=1','parameters_collected(named)','scheduler.after_successful_update()'];pos=[s.index(k) for k in seq];self.assertEqual(pos,sorted(pos));self.assertNotIn('torch.cuda.synchronize',s)
            for mutate in [lambda a:a.update(max_optimizer_updates=1),lambda a:a['task_profile_bindings'].clear(),lambda a:a['V_reuse'].clear()]:
                bad=copy.deepcopy(a);mutate(bad);self.assertRaises(ValueError,validate_launch_scope,c,ids,bad)
            f.output('guard-AST',dict(all_functions_and_error_class='exact AST',worker_boundaries=seq,extra_CUDA_sync=False,legacy_AUTH_rejected=True))
