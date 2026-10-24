"""Fixed v4-only risks, single-thread CPU synthetic acceptance, no effect search."""
import copy, json, math, os, unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from tests import moderntcn_v4_fixtures as f
from moderntcn_dev_v4 import adapter, training_state, launcher
from moderntcn_dev_v4.modules import GRN, AuxiliaryPatchSelection, ConditionalTemporalFilter
from moderntcn_dev_v3 import adapter as previous

def gradients(m): return {n:p.grad for n,p in m.named_parameters()}
def common(m): return {n:p for n,p in m.state_dict().items() if not n.startswith(('u.','v.'))}
def lin(x,l): return F.linear(x,l.weight,l.bias)
def ref_grn(g,a,c=None,manual=False):
    h=lin(a,g.L_in)
    if c is not None: h=h+lin(c,g.L_ctx)
    q=lin(F.elu(h,alpha=1),g.L_hidden)
    z=(lin(a,g.skip) if isinstance(g.skip,torch.nn.Linear) else a)+torch.sigmoid(lin(q,g.L_gate))*lin(q,g.L_value)
    if not manual: return F.layer_norm(z,(8,),g.norm.weight,g.norm.bias,1e-3)
    mu=z.mean(-1,keepdim=True); var=(z-mu).square().mean(-1,keepdim=True)
    return (z-mu)/torch.sqrt(var+1e-3)*g.norm.weight+g.norm.bias

class Differential(unittest.TestCase):
    def setUp(self): torch.random.default_generator.manual_seed(401)

    def test_grn_selector_weighted_values_independent_formula(self):
        u=AuxiliaryPatchSelection(96,3,64,24,8,4,2,[0,1])
        X=torch.randn(2,96,3,requires_grad=True); h=torch.randn(2,3,64,24,requires_grad=True)
        with torch.no_grad():u.W_out.weight.copy_(torch.randn_like(u.W_out.weight)*.04)
        e,z=u.content(X);c=h[:,2].transpose(1,2)
        zr=torch.stack([ref_grn(g,e[:,:,j]) for j,g in enumerate(u.content_grns)],2)
        t=ref_grn(u.selector,e.flatten(-2),c); alpha=F.softmax(lin(t,u.score),dim=-1)
        m=(alpha[:,:,:,None]*zr).sum(2)
        a,actual_m=u.select(e,z,c)
        f.equal(self,z,zr); f.equal(self,a,alpha); f.equal(self,actual_m,m)
        f.equal(self,u.selector(e.flatten(-2),c),t)
        manual=ref_grn(u.selector,e.flatten(-2),c,manual=True)
        torch.testing.assert_close(t,manual,atol=2e-6,rtol=2e-5)
        weight=torch.randn_like(t);pars=(X,h,*u.selector.parameters())
        ga=torch.autograd.grad((t*weight).sum(),pars,retain_graph=True)
        gb=torch.autograd.grad((manual*weight).sum(),pars,retain_graph=True)
        for x,y in zip(ga,gb):torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-5)
        actual=u(h,X);d=F.linear(m,u.W_out.weight).transpose(1,2)
        reference=torch.cat((h[:,:2],(h[:,2]+d).unsqueeze(1)),1)
        f.equal(self,actual,reference)
        weight=torch.randn_like(actual);pars=(X,h,*u.parameters())
        ga=torch.autograd.grad((actual*weight).sum(),pars,retain_graph=True)
        gb=torch.autograd.grad((reference*weight).sum(),pars,retain_graph=True)
        for x,y in zip(ga,gb):torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-5)
        self.assertEqual(u.selector.norm.normalized_shape,(8,));self.assertEqual(u.score.out_features,2)
        f.output('formula-reference',dict(same_order='exact',manual_LN='atol2e-6 rtol2e-5 incl local gradients',selector_LN_width=8))

    def test_patch_alignment_reorder_and_auxiliary_content(self):
        rows=[]
        from moderntcn_dev_v1.modules import WindowStateModulation
        for L,C,y,N,p,s in [(12,6,0,12,1,1),(96,3,2,24,8,4)]:
            aux=[j for j in range(C) if j!=y];u=AuxiliaryPatchSelection(L,C,64,N,p,s,y,aux)
            with torch.no_grad():u.W_out.weight.copy_((torch.arange(512).reshape(64,8)%11-5)*.02)
            X=(torch.arange(2*L*C).reshape(2,L,C)%4).float()/4
            H=torch.randn(2,C,64,N); original=X.clone();patches=u.patches(X)
            expected=torch.stack([torch.stack([X[:,min(n*s+k,L-1),aux] for k in range(p)],-1) for n in range(N)],1)
            f.equal(self,patches,expected)
            changed=X.clone();changed[:,:,aux]=X.flip(1)[:,:,aux]
            f.equal(self,X[:,:,aux].mean(1),changed[:,:,aux].mean(1))
            f.equal(self,X[:,:,aux].var(1,unbiased=False),changed[:,:,aux].var(1,unbiased=False))
            # Old statistical controller depends only on these same moments.
            old=WindowStateModulation(C,64,y,aux)
            f.equal(self,old.controller(X),old.controller(changed))
            old_before=old(H,X);old_after=old(H,changed)
            f.equal(self,old_before,old_after)
            out=u(H,X);out2=u(H,changed)
            self.assertGreater(float((out-out2).abs().max()),1e-5)
            f.equal(self,out[:,aux],H[:,aux]);f.equal(self,X,original)
            e,z=u.content(X);target_changed=X.clone();target_changed[:,:,y]+=31
            ee,zz=u.content(target_changed);f.equal(self,e,ee);f.equal(self,z,zz);f.equal(self,u(H,target_changed),out)
            a,_=u.select(e,z,H[:,y].transpose(1,2));b,_=u.select(e,z,H[:,y].transpose(1,2)+2)
            self.assertGreater(float((a-b).abs().max()),1e-5)
            one=X.clone();one[:,0,aux[0]]+=2
            self.assertGreater(float((u(H,one)-out).abs().max()),1e-5)
            u(H[:1],X[:1]+3);f.equal(self,u(H,X),out)
            rows.append(dict(L=L,target=y,patch_alignment='exact',reorder_delta=float((out-out2).abs().max()),context_alpha_change=float((a-b).abs().max()),target_excluded_from_values=True))
        f.output('nonzero-auxiliary-capability',rows)

    def test_nonzero_actual_post_V_wiring_and_observer(self):
        rows=[]
        for ds,Hn in [('UrbanEV',3),('PJM',24)]:
            p=f.profile(ds,H=Hn);m=adapter.build(p,use_u=True,use_v=True).eval();f.open_branches(m)
            X=torch.randn(2,p['T'],p['C']);input_before=X.clone();state=copy.deepcopy(m.state_dict());start=torch.get_rng_state().clone()
            seen={};counts={}
            def hook(name):
                def collect(module,args,out):
                    seen[name]=(args,out);counts[name]=counts.get(name,0)+1
                return collect
            block=m.model.stages[0].blocks[0]
            hooks=[block.ffn1drop2.register_forward_hook(hook('ffn1')),m.v.register_forward_hook(hook('V')),m.u.register_forward_hook(hook('U')),m.u.selector.register_forward_hook(hook('selector'))]
            y=m(X);hv=seen['U'][0][0];h=seen['ffn1'][1].reshape_as(hv);d=seen['V'][1]
            f.equal(self,seen['V'][0][0],h);f.equal(self,hv,h+d)
            f.equal(self,seen['selector'][0][1],hv[:,p['target_idx']].transpose(1,2))
            self.assertEqual(counts['ffn1'],1);self.assertGreater(float(d.abs().max()),0)
            self.assertGreater(float((seen['U'][1]-hv).abs().max()),0)
            v_before=d.detach().clone()
            for handle in hooks:handle.remove()
            torch.set_rng_state(start);observed,diag=m(X,return_diagnostics=True)
            f.equal(self,y,observed);f.equal(self,m.state_dict(),state);f.equal(self,X,input_before);f.equal(self,torch.get_rng_state(),start)
            self.assertIn('U_auxiliary_content',diag);self.assertIn('v_then_auxiliary_content',diag)
            with torch.no_grad():m.u.W_out.weight.mul_(2)
            hnd=m.v.register_forward_hook(hook('V'));m(X);hnd.remove();f.equal(self,seen['V'][1],v_before)
            m.use_v=False;capt={};handle=m.u.register_forward_hook(lambda mod,args,out:capt.update(delta=(out-args[0]).detach()))
            m(X);handle.remove();self.assertGreater(float(capt['delta'].abs().max()),0);m.use_v=True
            rows.append(dict(dataset=ds,V_input='original h exact',U_input='h+deltaV exact',FFN1_calls=1,observer='output/input/persistent state/RNG exact',U_active_without_V=True))
        f.output('wiring-and-observer',rows)

    def test_native_neutral_initialization_bn_rng_common_gradients(self):
        for ds,H in [('UrbanEV',3),('PJM',24)]:
            p=f.profile(ds,H=H);native=adapter.build_original(p);rng=torch.get_rng_state().clone();models=[]
            for u,v in [(False,False),(True,False),(False,True),(True,True)]:
                m=adapter.build(p,use_u=u,use_v=v);models.append(m)
                f.equal(self,common(m),native.state_dict());f.equal(self,rng,torch.get_rng_state())
                self.assertEqual(hasattr(m,'u'),u);self.assertEqual(hasattr(m,'v'),v)
            f.equal(self,models[1].u.state_dict(),models[3].u.state_dict());f.equal(self,models[2].v.state_dict(),models[3].v.state_dict())
            X=torch.randn(2,p['T'],p['C']);start=torch.get_rng_state().clone()
            for train in (False,True):
                native.train(train);native.zero_grad();torch.set_rng_state(start);y=native(X);end=torch.get_rng_state().clone();y.square().mean().backward();g=copy.deepcopy(gradients(native))
                for m in models:
                    m.train(train);m.zero_grad();torch.set_rng_state(start);out=m(X)
                    f.equal(self,y,out);f.equal(self,end,torch.get_rng_state());out.square().mean().backward()
                    f.equal(self,g,{k:v for k,v in gradients(m).items() if k in g});f.equal(self,native.state_dict(),common(m))

    def test_V_only_v3_exact_nonzero_train_eval_updates(self):
        rows=[]
        from utils.ch3_onecycle import Schedule,configuration
        for ds,H in [('UrbanEV',3),('PJM',24)]:
            p=f.profile(ds,H=H);a=previous.build(p,use_u=False,use_v=True);rng=torch.get_rng_state().clone();b=adapter.build(p,use_u=False,use_v=True)
            f.equal(self,rng,torch.get_rng_state());f.equal(self,a.state_dict(),b.state_dict())
            for m in (a,b):f.open_branches(m)
            opts=[torch.optim.Adam(m.parameters(),lr=.01,weight_decay=1e-7) for m in (a,b)]
            sched=[Schedule(o,configuration(2,4)) for o in opts]
            X=torch.randn(2,p['T'],p['C'])
            for train in (False,True):
                start=torch.get_rng_state().clone();obs=[]
                for m in (a,b):
                    m.train(train);m.zero_grad();torch.set_rng_state(start);out=m(X);out.square().mean().backward()
                    obs.append((out.detach(),copy.deepcopy(m.state_dict()),copy.deepcopy(gradients(m)),torch.get_rng_state().clone()))
                f.equal(self,*obs)
            for step in range(2):
                start=torch.get_rng_state().clone();obs=[]
                for m,o,s in zip((a,b),opts,sched):
                    torch.set_rng_state(start);o.zero_grad();out=m(X);out[:,:,p['target_idx']].square().mean().backward();g=copy.deepcopy(gradients(m));o.step();s.after_successful_update()
                    obs.append((copy.deepcopy(m.state_dict()),copy.deepcopy(o.state_dict()),s.state_dict(),g,torch.get_rng_state().clone()))
                f.equal(self,*obs)
            rows.append(dict(dataset=ds,output_state_gradient_Adam_scheduler_BN_RNG='exact',nonzero_A_Ps=True,updates_per_model=2))
        f.output('V-only-equivalence',rows)

    def test_output_then_internal_loss_gradient_opening(self):
        rows=[]
        for ds,H in [('UrbanEV',3),('PJM',24)]:
            p=f.profile(ds,H=H);m=adapter.build(p,use_u=True,use_v=True);o=torch.optim.Adam(m.parameters(),lr=.01,weight_decay=1e-7)
            X=torch.randn(2,p['T'],p['C']);target=torch.randn(2,p['pred_len'],1);history=[]
            for step in range(3):
                o.zero_grad();y=adapter.prediction(m,X);adapter.loss(y,target).backward()
                gs={n:float(p.grad.abs().sum()) for n,p in m.named_parameters() if n.startswith(('u.','v.'))};history.append(gs)
                self.assertTrue(all(math.isfinite(v) for v in gs.values()))
                if step==0:
                    self.assertGreater(gs['u.W_out.weight'],0);self.assertGreater(gs['v.A'],0)
                    self.assertTrue(all(v==0 for k,v in gs.items() if k.startswith('u.') and k!='u.W_out.weight'))
                    self.assertEqual(gs['v.P_s'],0)
                self.assertGreater(float(m.model.stages[0].blocks[0].ffn1pw1.weight.grad.abs().sum()),0)
                o.step()
            for n in ['u.embeddings.0.weight','u.content_grns.0.L_in.weight','u.selector.L_ctx.weight','u.score.weight','v.P_s','v.W1.weight']:
                self.assertGreater(history[-1][n],0,n)
            rows.append(dict(dataset=ds,loss_gradients_before_step=history))
        f.output('gradient-opening',rows)

    def test_profiles_parameters_loss_and_finite_inputs(self):
        rows=[]
        for ds,var,H,expected in [('UrbanEV','F1',3,3557),('UrbanEV','F2',6,4471),('UrbanEV','F3',9,4928),('UrbanEV','F4',12,5842),('PJM','F4',24,2298)]:
            p=f.profile(ds,var,H);m=adapter.build(p,use_u=True,use_v=True);X=torch.randn(2,p['T'],p['C']);out=adapter.prediction(m,X)
            self.assertEqual(out.shape,(2,p['pred_len'],1));self.assertEqual(m.u.target,p['target_idx']);self.assertEqual(list(m.u.aux),p['aux_idx']);self.assertGreaterEqual(len(m.u.aux),2)
            U=sum(p.numel() for p in m.u.parameters());V=sum(p.numel() for p in m.v.parameters());self.assertEqual(U,expected);self.assertEqual(V,572 if ds=='UrbanEV' else 2076)
            self.assertRaises(ValueError,adapter.loss,out,torch.zeros(2,p['pred_len'],p['C']))
            bad=X.clone();bad[0,0,m.u.aux[0]]=float('nan');self.assertRaises(ValueError,m.u.patches,bad)
            rows.append(dict(dataset=ds,variant=var,C=p['C'],P=p['pred_len'],H=H,N=m.spec.value['N'],patch=m.u.patch_len,aux=list(m.u.aux),B=sum(p.numel() for n,p in m.named_parameters() if not n.startswith(('u.','v.'))),U=U,V=V,UV=U+V))
        f.output('parameter-counts',rows)

    def test_cross_version_rejection_before_live_write(self):
        m=adapter.build(f.profile(),use_u=True,use_v=True);state=copy.deepcopy(m.state_dict());ck=adapter.synthetic_checkpoint(m)
        for version in ('v1','v2','v3'):
            bad=copy.deepcopy(ck);bad['format']='moderntcn-uv-synthetic-'+version;bad['spec']['candidate_id']='moderntcn-uv-dev-'+version
            self.assertRaises(ValueError,adapter.load_synthetic_checkpoint,m,bad);f.equal(self,state,m.state_dict())
            identity=dict(candidate_id='moderntcn-uv-dev-'+version,structural_spec=bad['spec'])
            with patch('timemixer_dev.training_state.restore',side_effect=AssertionError('must reject before read/write')):
                self.assertRaises(ValueError,training_state.restore,'nonexistent',m,None,None,None,identity)
        path=f.EVIDENCE/'tests'/('synthetic-state-'+os.environ['MTCN_TEST_RUN']+'.pt')
        torch.save(ck,path);loaded=torch.load(path,map_location='cpu',weights_only=False);adapter.load_synthetic_checkpoint(m,loaded);f.equal(self,state,m.state_dict())
        for key,value in [('target_idx',2),('aux_idx',[2,1])]:
            bad=copy.deepcopy(ck);bad['spec'][key]=value;self.assertRaises(ValueError,adapter.load_synthetic_checkpoint,m,bad);f.equal(self,state,m.state_dict())

class Metadata(unittest.TestCase):
    def test_exact_scope_reuse_budget_old_configuration(self):
        from moderntcn_dev_v4.training_config import validate
        from moderntcn_dev_v4.scope import verified_v_records,expected_waves
        from moderntcn_dev_v3.training_config import validate as old_validate
        c=validate(f.STORED);old=json.loads((f.ROOT/'configs/moderntcn_uv_fold6_epf_target_state_v3.json').read_text());old_validate(old)
        refs=verified_v_records(c);self.assertEqual(len(refs),9);self.assertTrue(all(t['arm']=='V' for t,m in refs));self.assertNotIn('U_reuse',c)
        self.assertEqual([len(w) for w in expected_waves()],[4,4,1,4,4,1]);self.assertEqual([t['arm'] for t in c['tasks']],['U']*9+['UV']*9)
        self.assertEqual(sum(t['arithmetic']['max_optimizer_steps'] for t in c['tasks']),1245080);self.assertEqual(sum(t['arithmetic']['max_epochs'] for t in c['tasks']),360)
        for a,b in zip(c['tasks'],old['tasks']):
            for k in ['structural_source_profile','data_config','data_stat','profile_sha','arithmetic','baseline_training']:self.assertEqual(a[k],b[k])
        for mutate in [lambda c:c['tasks'].reverse(),lambda c:c['tasks'].append(c['tasks'][0]),lambda c:c['tasks'][0].update(fold=1),lambda c:c.update(B_ref='paper_C1_r8_v1')]:
            bad=copy.deepcopy(c);mutate(bad);self.assertRaises(ValueError,validate,bad)
        self.assertRaises(ValueError,validate,old)

    def test_summary_missing_ties_and_reference_attribution(self):
        from moderntcn_dev_v4.summary import aggregate
        from moderntcn_dev_v4.scope import verified_v_records
        c=f.STORED;refs=verified_v_records(c);empty=aggregate(c,[],reuse_records=refs)
        self.assertEqual(empty['completed'],0);self.assertEqual(empty['V_equivalent_reuse'],9);self.assertEqual(empty['goals']['goal1']['numerical_status'],'Not evaluated');self.assertEqual(empty['goals']['goal2']['numerical_status'],'Not evaluated')
        records=[]
        for t in c['tasks']:
            b=c['existing_B'][t['baseline_pair_cell_id']];records.append((t,dict(task_id=t['id'],mse=b['mse']*.8,mae=b['mae']*(1 if t['dataset'] in ('PJM','NP','BE') else .8))))
        result=aggregate(c,records,reuse_records=refs);self.assertEqual(result['goals']['goal2']['winning_cells'],7);self.assertEqual(result['goals']['goal3']['numerical_status'],'Not evaluated')
        self.assertEqual(aggregate(c,records[:-1],reuse_records=refs)['goals']['goal2']['numerical_status'],'Not evaluated');self.assertRaises(ValueError,aggregate,c,records+[records[0]],reuse_records=refs)

    def test_nonexecuting_template_and_foreign_authorizations(self):
        from moderntcn_dev_v4.scope import validate_launch_scope
        a=json.loads((f.EVIDENCE/'non-execution-preparation/launch-binding.template.json').read_text());c=f.STORED;ids=[t['id'] for t in c['tasks']]
        validate_launch_scope(c,ids,a);self.assertIsNone(a['commit']);self.assertFalse(a['review_approved'])
        with patch('moderntcn_dev_v4.launcher.git',side_effect=AssertionError('reject before Git/network')):
            self.assertRaises(PermissionError,launcher.validate_authorization,c,'unused',ids,a)
            for version in ('v1','v2','v3'):
                bad=copy.deepcopy(a);bad.update(review_approved=True,candidate_id='moderntcn-uv-dev-'+version)
                self.assertRaises(PermissionError,launcher.validate_authorization,c,'unused',ids,bad)
        for mutate in [lambda a:a.update(max_optimizer_updates=1867620),lambda a:a['task_profile_bindings'].clear(),lambda a:a['V_reuse'].clear()]:
            bad=copy.deepcopy(a);mutate(bad);self.assertRaises(ValueError,validate_launch_scope,c,ids,bad)

if __name__=='__main__':unittest.main()
