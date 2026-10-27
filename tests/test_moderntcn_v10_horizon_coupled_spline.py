"""Bounded CPU synthetic acceptance for fixed horizon coupling and two reference classes."""
import ast,copy,json,math,unittest,subprocess
from unittest.mock import patch
import torch
from torch.nn import functional as F
from tests import moderntcn_v10_fixtures as f
from moderntcn_dev_v10 import adapter,training_state,launcher,scope
from moderntcn_dev_v10.modules import *
from moderntcn_dev_v3 import adapter as previous
from moderntcn_dev_v9 import adapter as v9
from moderntcn_dev_v10.guards import *
def gradients(m):return {n:p.grad for n,p in m.named_parameters()}
def common(m):return {n:p for n,p in m.state_dict().items() if not n.startswith(('u.','v.'))}
def close(a,b):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
def config():return json.loads((f.ROOT/'configs/moderntcn_uv_fold6_epf_horizon_coupled_spline_v10.json').read_text())

def scalar_matrix(P):
    K=max(1,math.ceil(P/4));rows=[]
    for p in range(P):
        if K==1:rows.append([1.]);continue
        x=p*(K-1)/(P-1);left=min(math.floor(x),K-2)
        rows.append([1-(x-left) if r==left else x-left if r==left+1 else 0. for r in range(K)])
    return torch.tensor(rows,dtype=torch.float32)

class Differential(unittest.TestCase):
    def setUp(self):torch.manual_seed(401)

    def test_nonzero_final_head_wiring_and_observer(self):
            rows=[]
            for ds,Hn in [('PJM',24)]:
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
            for ds,Hn in [('PJM',24)]:
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
            for ds,var,Hn,U in [('UrbanEV','F1',3,6144),('UrbanEV','F2',6,6144),('UrbanEV','F3',9,6144),('UrbanEV','F4',12,6144),('PJM','F4',24,73728)]:
                p=f.profile(ds,var,Hn);m=adapter.build(p,use_u=True,use_v=True);out=adapter.prediction(m,torch.randn(2,p['T'],p['C']));self.assertEqual(out.shape,(2,p['pred_len'],1));self.assertEqual(m.spec.value['target_idx'],p['target_idx']);self.assertEqual(m.spec.value['label_horizon'],Hn)
                u=sum(p.numel() for p in m.u.parameters());v=sum(p.numel() for p in m.v.parameters());self.assertEqual(u,U);self.assertEqual(v,572 if ds=='UrbanEV' else 2076);self.assertRaises(ValueError,adapter.loss,out,torch.zeros(2,p['pred_len'],p['C']));self.assertEqual(set(dict(m.u.named_parameters())),{'coef'});self.assertEqual(tuple(m.u.coef.shape),(control_count(p['pred_len']),64*m.spec.value['N'],8));self.assertEqual(m.u.knots.numel()*4,48)
                rows.append(dict(dataset=ds,variant=var,C=p['C'],L=p['T'],P=p['pred_len'],H=Hn,N=m.spec.value['N'],B=sum(p.numel() for n,p in m.named_parameters() if not n.startswith(('u.','v.'))),U=u,V=v,UV=u+v,buffer_bytes=sum(b.numel()*b.element_size() for b in m.u.buffers())))
            f.output('parameter-counts',rows)

    def test_neutral_native_state_gradient_BN_RNG(self):
                for ds,Hn in [('PJM',24)]:
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

    def test_interpolation_independent_oracle(self):
        rows=[]
        for P in (1,2,4,5,8,9,24):
            M=interpolation_matrix(P);K=control_count(P);ref=scalar_matrix(P)
            close(M,ref);self.assertEqual(K,math.ceil(P/4));self.assertTrue(bool((M>=0).all()));close(M.sum(1),torch.ones(P));self.assertLessEqual(int((M!=0).sum(1).max()),2)
            close(M@torch.ones(K),torch.ones(P));self.assertEqual(float(M[0,0]),1.);self.assertEqual(float(M[-1,-1]),1.)
            rows.append(dict(P=P,K=K,max_abs=float((M-ref).abs().max())))
        for P in (0,-1,1.5,True):self.assertRaises(ValueError,interpolation_matrix,P)
        f.output('interpolation-oracle',rows)

    def test_effective_coefficients_and_gradients(self):
        m=HorizonCoupledSplineReadout(7,24)
        with torch.no_grad():m.coef.copy_(torch.randn_like(m.coef)*.15)
        a=torch.linspace(-1.2,1.2,14).reshape(2,7).requires_grad_();actual=m(a)
        # Independent scalar interpolation after control outputs, plus expanded coefficients.
        M=scalar_matrix(24);basis=m.basis(a);ce=torch.einsum('pr,rik->pik',M,m.coef)
        expanded=F.linear(basis.reshape(2,-1),ce.reshape(24,-1))/7
        q=F.linear(basis.reshape(2,-1),m.coef.reshape(6,-1))/7
        oracle=torch.stack([sum(q[:,r]*float(M[p,r]) for r in range(6)) for p in range(24)],1)
        close(actual,expanded);close(actual,oracle)
        weights=torch.linspace(-.8,1.1,48).reshape(2,24)
        ga=torch.autograd.grad((actual*weights).sum(),(a,m.coef),retain_graph=True)
        ge=torch.autograd.grad((expanded*weights).sum(),(a,m.coef))
        for x,y in zip(ga,ge):close(x,y)
        self.assertFalse(m.M.requires_grad);self.assertIsNone(m.M.grad)
        # Local double finite difference for interpolation/linear coefficient derivative only.
        qd=torch.randn(2,6,dtype=torch.float64,requires_grad=True);md=M.double()
        self.assertTrue(torch.autograd.gradcheck(lambda q:F.linear(q,md),(qd,),eps=1e-6,atol=1e-7,rtol=1e-5))
        changed=q.detach().clone();changed[:,2]+=1.;diff=F.linear(changed,M)-F.linear(q.detach(),M)
        close(diff,M[:,2].expand(2,-1));self.assertGreater(int((M[:,2]==0).sum()),0)
        f.output('effective-coef',dict(forward_abs=float((actual-expanded).abs().max()),scalar_abs=float((actual-oracle).abs().max()),gradient_abs=[float((x-y).abs().max()) for x,y in zip(ga,ge)],fixed_M=True,local_double_gradcheck=True))

    def test_P1_v9_exact_U_UV(self):
        from utils.ch3_onecycle import Schedule,configuration
        from moderntcn_dev_v9.model import ModernTCNUV as Old
        from moderntcn_dev_v10.model import ModernTCNUV as New
        self.assertIs(New.forward,Old.forward);self.assertIs(New._block,Old._block)
        rows=[]
        for use_v in (False,True):
            p=f.profile(H=9);mods=[v9.build(p,use_u=True,use_v=use_v),adapter.build(p,use_u=True,use_v=use_v)]
            self.assertIs(type(mods[1].u),SplineTargetReadout);self.assertNotIn('u.M',mods[1].state_dict())
            f.equal(self,mods[0].state_dict(),mods[1].state_dict())
            for m in mods:f.open_branches(m)
            opts=[torch.optim.Adam(m.parameters(),lr=.01,weight_decay=1e-7) for m in mods];sch=[Schedule(o,configuration(2,4)) for o in opts];X=torch.randn(2,12,11)
            for train in (False,True):
                rng=torch.get_rng_state().clone();obs=[]
                for m in mods:
                    m.train(train);m.zero_grad();torch.set_rng_state(rng);out=m(X);out[:,:,0].square().mean().backward();obs.append((out.detach(),copy.deepcopy(m.state_dict()),copy.deepcopy(gradients(m)),torch.get_rng_state().clone()))
                f.equal(self,*obs)
            for step in range(2):
                rng=torch.get_rng_state().clone();obs=[]
                for m,o,s in zip(mods,opts,sch):
                    torch.set_rng_state(rng);o.zero_grad();out=m(X);loss=out[:,:,0].square().mean();loss.backward();gr=copy.deepcopy(gradients(m));o.step();s.after_successful_update();obs.append((out.detach(),loss.detach(),gr,copy.deepcopy(m.state_dict()),copy.deepcopy(o.state_dict()),s.state_dict(),torch.get_rng_state().clone()))
                f.equal(self,*obs)
            rows.append(dict(arm='UV' if use_v else 'U',nonzero=True,train_eval_grad_state_BN_RNG_2Adam_OneCycle='exact'))
        # H is metadata only; P1 exact code path independent of the label offset.
        from moderntcn_dev_v10.spec import StructuralSpec
        for H in (3,6,9,12):
            p=f.profile(H=H);s=StructuralSpec.from_profile(p,use_u=True,use_v=True).value
            self.assertEqual((s['pred_len'],s['label_horizon'],s['horizon_coupling']['K']),(1,H,1))
        f.output('P1-v9-equivalence',dict(rows=rows,H=[3,6,9,12],shared_execution=True,no_checkpoint_migration=True))

    def test_guard_identity_and_same_v10_update(self):
        from utils.ch3_onecycle import Schedule,configuration
        self.assertEqual((f.ROOT/'moderntcn_dev_v10/guards.py').read_bytes(),(f.ROOT/'moderntcn_dev_v9/guards.py').read_bytes())
        text=(f.ROOT/'moderntcn_dev_v10/worker.py').read_text();seq=['loss_guard(loss)','loss.backward()','gradients_collected(named)','opt.step();steps+=1','parameters_collected(named)','scheduler.after_successful_update()'];pos=[text.index(s) for s in seq];self.assertEqual(pos,sorted(pos))
        mods=[adapter.build(f.profile('PJM',H=24),use_u=True,use_v=True) for _ in range(2)]
        for m in mods:f.open_branches(m)
        opts=[torch.optim.Adam(m.parameters(),lr=.01,weight_decay=1e-7) for m in mods];sch=[Schedule(o,configuration(2,4)) for o in opts];X=torch.randn(2,96,3)
        for step in range(2):
            rng=torch.get_rng_state().clone();obs=[]
            for m,o,s,g,pg in zip(mods,opts,sch,[gradients_original,gradients_collected],[parameters_original,parameters_collected]):
                torch.set_rng_state(rng);o.zero_grad();out=m(X);loss=out[:,:,2].square().mean();loss_guard(loss);loss.backward();named=list(m.named_parameters());g(named);gr=copy.deepcopy(gradients(m));o.step();pg(named);s.after_successful_update();obs.append((out.detach(),loss.detach(),gr,copy.deepcopy(m.state_dict()),copy.deepcopy(o.state_dict()),s.state_dict(),torch.get_rng_state().clone()))
            f.equal(self,*obs)
        f.output('guard-integration',dict(byte_identity=True,worker_boundaries=seq,same_v10_2updates='exact',old_GPU_evidence_reused=True))

    def test_checkpoint_identity_and_fixed_buffers(self):
        for ds,H in [('UrbanEV',3),('PJM',24)]:
            m=adapter.build(f.profile(ds,H=H),use_u=True,use_v=True);f.open_branches(m);original=copy.deepcopy(m.state_dict());state=adapter.synthetic_checkpoint(m)
            path=f.EVIDENCE/'tests'/(__import__('os').environ['MTCN_TEST_RUN']+'-'+ds+'-synthetic.pt');torch.save(state,path);saved=torch.load(path,map_location='cpu');adapter.load_synthetic_checkpoint(m,saved);f.equal(self,original,m.state_dict())
            for version in range(1,10):
                wrong=copy.deepcopy(saved);wrong['format']='moderntcn-uv-synthetic-v'+str(version)
                with self.assertRaises(ValueError):adapter.load_synthetic_checkpoint(m,wrong)
                f.equal(self,original,m.state_dict())
            for key in ['u.knots']+(['u.M'] if ds=='PJM' else []):
                wrong=copy.deepcopy(saved);wrong['state_dict'][key].flatten()[0]+=.125
                with self.assertRaises(ValueError):adapter.load_synthetic_checkpoint(m,wrong)
                f.equal(self,original,m.state_dict())
            wrong=copy.deepcopy(saved);wrong['state_dict']['u.coef']=wrong['state_dict']['u.coef'][:,:,:7]
            with self.assertRaises(ValueError):adapter.load_synthetic_checkpoint(m,wrong)
            f.equal(self,original,m.state_dict())
        # Static restoration sequence: validation including buffers precedes first live write.
        text=(f.ROOT/'moderntcn_dev_v10/training_state.py').read_text();restore=text[text.index('def restore'):]
        self.assertLess(restore.index('validate(read(path)'),restore.index('model.load_state_dict'))
        f.output('checkpoint',dict(synthetic_writes=2,synthetic_reads=2,old_candidate_before_write=True,fixed_M_knots_rejected=True,live_state_unchanged=True))

    def test_references_summary_and_policy(self):
        from moderntcn_dev_v10.summary import aggregate,goal1
        from moderntcn_dev_v10.training_config import validate
        c=config();v=scope.verified_v_records(c);p1=scope.verified_p1_records(c)
        self.assertEqual((len(v),len(p1)),(9,8));empty=aggregate(c,[])
        self.assertEqual(empty['evaluation_cells_available'],17)
        self.assertEqual(empty['goals']['goal1']['numerical_status'],'Not evaluated');self.assertEqual(empty['goals']['goal2']['numerical_status'],'Not evaluated')
        records=[(t,dict(task_id=t['id'],mse=.1 if t['arm']=='UV' else .2,mae=.1 if t['arm']=='UV' else .2)) for t in c['tasks']]
        full=aggregate(c,records);self.assertEqual(full['evaluation_cells_available'],27);self.assertEqual(full['fresh_completed'],10);self.assertEqual(full['goals']['goal1']['numerical_status'],'Met')
        self.assertEqual(aggregate(c,records[:-1])['goals']['goal1']['numerical_status'],'Not evaluated')
        with self.assertRaises(ValueError):aggregate(c,records+[records[0]])
        bad=copy.deepcopy(p1);bad[0][1]['mse']+=1
        with self.assertRaises(ValueError):aggregate(c,[],p1_records=bad)
        for key in ('producer_commit','original_task_id'):
            bad=copy.deepcopy(c);bad['P1_v9_reuse'][0][key]='forged'
            with self.assertRaises(ValueError):validate(bad)
        for key in ('arm','structural_source_profile','data_config'):
            bad=copy.deepcopy(c);bad['P1_v9_reuse'][0]['original_task'][key]='forged'
            with self.assertRaises(ValueError):validate(bad)
        urban={str(h):{a:dict(mse=1 if a=='UV' else 2,mae=1 if a=='UV' else 2) for a in ('U','V','UV')} for h in (3,6,9,12)}
        epf={ds:{a:dict(mse=2,mae=2) for a in ('U','V','UV')} for ds in scope.MARKETS}
        epf['PJM']['UV']['mse']=1;epf['NP']['UV']['mae']=1
        self.assertEqual(goal1(urban,epf)['numerical_status'],'Not met')
        epf['BE']['UV']=dict(mse=1,mae=1);self.assertEqual(goal1(urban,epf)['numerical_status'],'Met')
        epf['BE']['UV']=dict(mse=2,mae=1);self.assertEqual(goal1(urban,epf)['numerical_status'],'Not met')
        f.output('scope-summary',dict(fresh=10,V3=9,P1_v9=8,empty_summary=empty,metadata_fixture_full=full,missing_duplicate_forged_tie_crossmarket='rejected or NE/Not met'))

    def test_nonexecution_budget_and_source(self):
        from moderntcn_dev_v10.worker import source_binding
        from moderntcn_dev_v10.training_config import validate
        c=config();self.assertEqual(c['source_binding'],source_binding());self.assertEqual([len(w) for w in scope.expected_waves()],[4,1,4,1]);self.assertEqual(sum(t['arithmetic']['max_optimizer_steps'] for t in c['tasks']),57000);self.assertEqual(sum(t['arithmetic']['max_epochs'] for t in c['tasks']),200)
        self.assertTrue(all(t['dataset']!='UrbanEV' for t in c['tasks']));self.assertEqual(len(c['future_input_scope']),12)
        with patch.object(launcher,'git',side_effect=AssertionError('must reject before network')):
            with self.assertRaises(PermissionError):launcher.validate_authorization(c,'unused',[t['id'] for t in c['tasks']],dict(commit=None,review_approved=False))
            with self.assertRaises(ValueError):launcher.validate_authorization(c,'unused',[t['id'] for t in c['tasks']],dict(commit='x',review_approved=True,task_scope='old'))
        for version in range(1,10):
            bad=copy.deepcopy(c);bad['candidate_id']='moderntcn-uv-dev-v'+str(version)
            with self.assertRaises(ValueError):validate(bad)
        self.assertIn('moderntcn_dev_v9/model.py',source_binding());self.assertIn('moderntcn_dev_v9/modules.py',source_binding())
        subprocess.run(['bash','-n',str(f.ROOT/'scripts/ch3/start_moderntcn_uv_v10.sh')],check=True)
        f.output('nonexecution',dict(source_count=len(source_binding()),waves=[4,1,4,1],epochs=200,updates=57000,old_AUTH_rejected=True,template_before_network=True,script_syntax=True,preflight_executed=False))
