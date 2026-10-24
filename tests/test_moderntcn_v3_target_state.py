"""Only v3 differential risks; frozen tolerances and exact identities retained."""
import copy, json, math, unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from tests import moderntcn_v3_fixtures as f
from moderntcn_dev_v3 import adapter, training_state, launcher
from moderntcn_dev_v3.modules import ConditionalTemporalFilter as V
from moderntcn_dev_v2 import adapter as v2
from moderntcn_dev_v1 import adapter as v1

def common(m):return {k:v for k,v in m.state_dict().items() if not k.startswith(('u.','v.'))}
def gradients(m):return {n:p.grad for n,p in m.named_parameters()}
def opened(m, state=False):
    with torch.no_grad():
        if m.use_u:m.u.Wo.weight.fill_(.04);m.u.Wb.weight.fill_(.03)
        if m.use_v:
            m.v.A.fill_(.05)
            if state and hasattr(m.v,'P_s'):m.v.P_s.fill_(.08)

class Differential(unittest.TestCase):
    def setUp(self):torch.random.default_generator.manual_seed(17)

    def test_nonzero_target_condition_changes_route_kernel_delta(self):
        rows=[]
        for L,C,y,N,K in [(12,11,0,12,7),(96,3,2,24,51)]:
            v=V(L,8,N,K,C,y)
            with torch.no_grad():
                v.P_s[:,0].fill_(.08);v.P_s[:,1].fill_(.03)
                v.W1.weight.fill_(.01);v.W2.weight.fill_(.04);v.A.fill_(.2)
            X=torch.randn(2,L,C)*.3;Z=torch.randn(2,C,L)*.2;h=torch.randn(2,C,8,N)
            a=v.route(Z,X);d=v(h,Z,X);kernel=torch.einsum('bcr,dr,rk->bcdk',a,v.A,v.T)
            changed=X.clone();changed[:,:,y]+=2
            a2=v.route(Z,changed);d2=v(h,Z,changed);k2=torch.einsum('bcr,dr,rk->bcdk',a2,v.A,v.T)
            for before,after in [(a,a2),(d,d2),(kernel,k2)]:self.assertGreater(float((before-after).abs().max()),1e-5)
            other=X.clone();other[:,:,(y+1)%C]+=100
            self.assertTrue(torch.equal(a,v.route(Z,other)));self.assertTrue(torch.equal(d,v(h,Z,other)))
            rows.append(dict(L=L,target=y,route_change=float((a-a2).abs().max()),kernel_change=float((kernel-k2).abs().max()),delta_change=float((d-d2).abs().max()),non_target_fixed_Z='exact unchanged'))
        f.output('nonzero-condition-response',rows)

    def test_state_axes_constants_nonfinite_and_no_context(self):
        v=V(12,8,12,7,3,0);X=torch.zeros(2,12,3);X[0,:,0]=2;X[1,:,0]=-3
        self.assertTrue(torch.equal(v.target_state(X),torch.tensor([[2.,0.],[-3.,0.]])))
        ramp=torch.arange(12,dtype=torch.float32);X[0,:,0]=ramp
        expected=torch.stack((ramp.mean(),torch.log1p(((ramp-ramp.mean())**2).mean())))
        self.assertTrue(torch.equal(v.target_state(X)[0],expected))
        h=torch.randn(2,3,8,12);z=torch.randn(2,3,12)
        with torch.no_grad():v.P_s.fill_(.02);v.A.fill_(.05)
        original=v(h,z,X);v(h[:1],z[:1],X[:1]+3);self.assertTrue(torch.equal(original,v(h,z,X)))
        for val in (float('inf'),float('nan'),3e38):
            bad=X.clone();bad[0,0,0]=val;self.assertRaises(ValueError,v.target_state,bad)
        self.assertRaises(ValueError,v.target_state,X.transpose(1,2))
        self.assertEqual(v.target_state(X.double()).dtype,torch.float32)

    def test_zero_projection_nonzero_v2_common_graph_exact(self):
        rows=[]
        for ds,H in [('UrbanEV',3),('PJM',24)]:
            p=f.profile(ds,H=H);a=v2.build(p,use_u=True,use_v=True);rng=torch.get_rng_state().clone();b=adapter.build(p,use_u=True,use_v=True)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(set(b.state_dict())-set(a.state_dict()),{'v.P_s'})
            f.equal(self,a.state_dict(),{k:v for k,v in b.state_dict().items() if k!='v.P_s'})
            opened(a);opened(b);self.assertEqual(float(b.v.P_s.abs().sum()),0)
            X=torch.randn(2,p['T'],p['C']);start=torch.get_rng_state().clone()
            for train in (False,True):
                observed=[]
                for m in (a,b):
                    m.train(train);m.zero_grad();torch.set_rng_state(start)
                    out=m(X);out.square().mean().backward()
                    observed.append((out.detach(),copy.deepcopy(m.state_dict()),copy.deepcopy(gradients(m)),torch.get_rng_state().clone()))
                aa,bb=observed;f.equal(self,aa[0],bb[0]);f.equal(self,aa[1],{k:v for k,v in bb[1].items() if k!='v.P_s'});f.equal(self,aa[2],{k:v for k,v in bb[2].items() if k!='v.P_s'});f.equal(self,aa[3],bb[3])
                self.assertTrue(torch.isfinite(b.v.P_s.grad).all());self.assertGreater(float(b.v.P_s.grad.abs().sum()),0)
            rows.append(dict(dataset=ds,common_output_state_grad_BN_RNG='exact',nonzero_old_branches=True,extra_P_s_gradient='finite nonzero; new parameter only'))
        f.output('zero-Ps-v2-equivalence',rows)

    def test_neutral_native_common_initialization_rng_and_grad(self):
        for ds,H in [('UrbanEV',3),('PJM',24)]:
            p=f.profile(ds,H=H);native=adapter.build_original(p);rng=torch.get_rng_state().clone();models=[]
            for u,v in [(False,False),(True,False),(False,True),(True,True)]:
                m=adapter.build(p,use_u=u,use_v=v);models.append(m)
                f.equal(self,common(m),native.state_dict());self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                self.assertEqual(hasattr(m,'u'),u);self.assertEqual(hasattr(m,'v'),v)
            f.equal(self,models[1].u.state_dict(),models[3].u.state_dict());f.equal(self,models[2].v.state_dict(),models[3].v.state_dict())
            X=torch.randn(2,p['T'],p['C']);start=torch.get_rng_state().clone()
            for train in (False,True):
                native.train(train);native.zero_grad();torch.set_rng_state(start);y=native(X);end=torch.get_rng_state().clone();y.square().mean().backward();g=copy.deepcopy(gradients(native))
                for m in models:
                    m.train(train);m.zero_grad();torch.set_rng_state(start);z=m(X)
                    self.assertTrue(torch.equal(y,z));self.assertTrue(torch.equal(end,torch.get_rng_state()));z.square().mean().backward()
                    f.equal(self,g,{k:v for k,v in gradients(m).items() if k in g});f.equal(self,native.state_dict(),common(m))

    def test_u_only_v1_v2_v3_exact_updates(self):
        rows=[]
        for ds,H in [('UrbanEV',3),('PJM',24)]:
            p=f.profile(ds,H=H);models=[];rngs=[]
            for a in (v1,v2,adapter):
                m=a.build(p,use_u=True,use_v=False,fold=6 if ds=='UrbanEV' else 1);models.append(m);rngs.append(torch.get_rng_state().clone());opened(m)
            for m,r in zip(models[1:],rngs[1:]):f.equal(self,models[0].state_dict(),m.state_dict());f.equal(self,rngs[0],r)
            opts=[torch.optim.Adam(m.parameters(),lr=.01,weight_decay=1e-7) for m in models]
            from utils.ch3_onecycle import Schedule,configuration
            sched=[Schedule(o,configuration(2,4)) for o in opts]
            X=torch.randn(2,p['T'],p['C'])
            for train in (False,True):
                start=torch.get_rng_state().clone();ys=[]
                for m in models:
                    m.train(train);torch.set_rng_state(start);ys.append((m(X),torch.get_rng_state().clone()))
                for value in ys[1:]:f.equal(self,ys[0],value)
            for step in range(2):
                start=torch.get_rng_state().clone();rows_step=[]
                for m,o,s in zip(models,opts,sched):
                    torch.set_rng_state(start);o.zero_grad();out=m(X);out[:,:,p['target_idx']].square().mean().backward()
                    g=copy.deepcopy(gradients(m));o.step();s.after_successful_update()
                    rows_step.append((copy.deepcopy(m.state_dict()),copy.deepcopy(o.state_dict()),g,torch.get_rng_state().clone(),s.state_dict()))
                for value in rows_step[1:]:f.equal(self,rows_step[0],value)
            rows.append(dict(dataset=ds,versions=['v1','v2','v3'],nonzero_U=True,output_state_gradient_Adam_scheduler_RNG='exact',updates_per_model=2))
        f.output('U-only-equivalence',rows)

    def test_A_then_target_projection_and_router_loss_gradients(self):
        rows=[]
        for ds,H in [('UrbanEV',3),('PJM',24)]:
            p=f.profile(ds,H=H);m=adapter.build(p,use_u=True,use_v=True);o=torch.optim.Adam(m.parameters(),lr=.01,weight_decay=1e-7)
            X=torch.randn(2,p['T'],p['C']);target=torch.randn(2,p['pred_len'],1)
            gs=[]
            for step in range(3):
                o.zero_grad();out=adapter.prediction(m,X);adapter.loss(out,target).backward()
                g={n:float(p.grad.abs().sum()) for n,p in m.named_parameters() if n.startswith(('u.','v.'))};gs.append(g)
                self.assertTrue(all(math.isfinite(v) for v in g.values()))
                if step==0:
                    for n in ['v.P_s','v.W1.weight','v.W2.weight','v.T','u.A','u.Wh.weight']:self.assertEqual(g[n],0)
                    for n in ['v.A','u.Wo.weight','u.Wb.weight']:self.assertGreater(g[n],0)
                o.step()
            for n in ['v.P_s','v.W1.weight','v.W2.weight','v.T','u.A','u.Wh.weight']:self.assertGreater(gs[-1][n],0)
            rows.append(dict(dataset=ds,loss_gradients_before_optimizer_step=gs))
        f.output('gradient-opening',rows)

    def test_conditioned_shared_basis_explicit_kernel_forward_gradient(self):
        for L,C,y,N,K in [(12,3,0,12,7),(96,3,2,24,51)]:
            v=V(L,8,N,K,C,y).double()
            with torch.no_grad():v.A.normal_(0,.1);v.P_s.normal_(0,.1)
            h=torch.randn(2,C,8,N,dtype=torch.float64,requires_grad=True);z=torch.randn(2,C,L,dtype=torch.float64,requires_grad=True);x=torch.randn(2,L,C,dtype=torch.float64,requires_grad=True)
            out=v(h,z,x);a=v.route(z,x);kernel=torch.einsum('bcr,dr,rk->bcdk',a,v.A,v.T)
            ref=F.conv1d(h.reshape(1,48,N),kernel.reshape(48,1,v.Kv),padding=v.Kv//2,groups=48).reshape_as(h)
            torch.testing.assert_close(out,ref,atol=1e-10,rtol=1e-9)
            weight=torch.randn_like(out);pars=(h,z,*v.parameters())
            aa=torch.autograd.grad((out*weight).sum(),pars,retain_graph=True);bb=torch.autograd.grad((ref*weight).sum(),pars,retain_graph=True)
            for a,b in zip(aa,bb):torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-9)
            # X statistics are intentionally float32 even in this double fixture.
            ga=torch.autograd.grad((out*weight).sum(),x,retain_graph=True)[0];gb=torch.autograd.grad((ref*weight).sum(),x)[0]
            torch.testing.assert_close(ga,gb,atol=2e-6,rtol=2e-5)
            self.assertGreater(float(ga[:,:,y].abs().sum()),0)
            self.assertEqual(float(ga[:,:,[j for j in range(C) if j!=y]].abs().sum()),0)

    def test_actual_inputs_observer_no_batch_context(self):
        p=f.profile('PJM',H=24);m=adapter.build(p,use_u=True,use_v=True).eval();opened(m,state=True)
        X=torch.randn(2,96,3);original=X.clone();state=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone();seen={};counts={}
        def observe(name):
            def hook(mod,inp,out):
                seen[name]=(inp,out);counts[name]=counts.get(name,0)+1
                if name=='revin' and inp[1]=='norm':seen['Z']=out.permute(0,2,1).clone()
            return hook
        block=m.model.stages[0].blocks[0]
        hooks=[mod.register_forward_hook(observe(name)) for name,mod in [('U',m.u),('V',m.v),('drop2',block.ffn1drop2),('stem',m.model.downsample_layers[0]),('revin',m.model.revin_layer)]]
        with torch.inference_mode():y=m(X)
        for hook in hooks:hook.remove()
        h=seen['drop2'][1].reshape(2,3,64,24)
        self.assertEqual(seen['U'][0][0].data_ptr(),h.data_ptr());self.assertEqual(seen['V'][0][0].data_ptr(),h.data_ptr())
        self.assertEqual(seen['V'][0][2].data_ptr(),X.data_ptr());self.assertFalse(torch.equal(h,seen['stem'][1].reshape_as(h)));self.assertFalse(torch.equal(h,seen['U'][1]));self.assertEqual(counts['drop2'],1);self.assertEqual(counts['revin'],2)
        f.equal(self,seen['V'][0][1],seen['Z'])
        with torch.inference_mode():
            z,d=m(X,return_diagnostics=True);f.equal(self,y,z);f.equal(self,state,m.state_dict());f.equal(self,rng,torch.get_rng_state());f.equal(self,X,original)
            a=m.v.route(seen['V'][0][1],X);s=m.v.target_state(X);m.use_u=False
            off=m(X);f.equal(self,s,m.v.target_state(X));f.equal(self,a,m.v.route(seen['V'][0][1],X));m.use_u=True
            m(torch.randn(1,96,3));f.equal(self,y,m(X))
        self.assertEqual(d['V']['target_idx'],2);self.assertIn('target_state',d['V']);self.assertNotIn('native_BN',d)
        f.output('observer',dict(output_input_persistent_state_RNG='exact',temporary_RevIN='per-forward transient cache, not persistent',diagnostics=d))

    def test_shape_parameter_loss_schema(self):
        rows=[]
        for ds,variant,H in [('UrbanEV',v,3) for v in ('F1','F2','F3','F4')]+[('PJM','MS',24)]:
            p=f.profile(ds,variant,H);m=adapter.build(p,use_u=True,use_v=True).eval();X=torch.randn(2,p['T'],p['C']);y=adapter.prediction(m,X)
            self.assertEqual(y.shape,(2,p['pred_len'],1));self.assertEqual(m.v.target_idx,p['target_idx']);self.assertEqual(m.u.target,p['target_idx']);self.assertEqual(list(m.u.aux),p['aux_idx']);self.assertGreater(len(m.u.aux),0)
            U=sum(p.numel() for p in m.u.parameters());Vn=sum(p.numel() for p in m.v.parameters())
            self.assertEqual(U,1536+25*p['C']-1);self.assertEqual(Vn,16*p['T']+64+4*(64+m.v.Kv)+32)
            self.assertRaises(ValueError,adapter.loss,y,torch.zeros(2,p['pred_len'],p['C']))
            rows.append(dict(dataset=ds,variant=variant,C=p['C'],N=m.spec.value['N'],Kv=m.v.Kv,B=sum(p.numel() for n,p in m.named_parameters() if not n.startswith(('u.','v.'))),U=U,V=Vn))
        f.output('parameter-counts',rows)

    def test_cross_version_synthetic_state_rejection_before_write(self):
        p=f.profile();m=adapter.build(p,use_u=True,use_v=True);before=copy.deepcopy(m.state_dict())
        ck=adapter.synthetic_checkpoint(m)
        for version in ('v1','v2'):
            bad=copy.deepcopy(ck);bad['format']='moderntcn-uv-synthetic-'+version;bad['spec']['candidate_id']='moderntcn-uv-dev-'+version
            self.assertRaises(ValueError,adapter.load_synthetic_checkpoint,m,bad);f.equal(self,before,m.state_dict())
            identity=dict(candidate_id='moderntcn-uv-dev-'+version,structural_spec=bad['spec'])
            with patch('timemixer_dev.training_state.validate',side_effect=AssertionError('must reject before state writes')):
                self.assertRaises(ValueError,training_state.validate,{},m,None,None,identity)
        path=f.EVIDENCE/'synthetic-fixtures'/('checkpoint-'+__import__('os').environ['MTCN_TEST_RUN']+'.pt')
        torch.save(ck,path);loaded=torch.load(path,map_location='cpu',weights_only=False);adapter.load_synthetic_checkpoint(m,loaded);f.equal(self,before,m.state_dict())
        for key,value in [('target_idx',2),('aux_idx',[2,1])]:
            bad=copy.deepcopy(ck);bad['spec'][key]=value;self.assertRaises(ValueError,adapter.load_synthetic_checkpoint,m,bad);f.equal(self,before,m.state_dict())

class Metadata(unittest.TestCase):
    def test_scope_reuse_protocol_and_rejection(self):
        from moderntcn_dev_v3.training_config import validate
        from moderntcn_dev_v3.scope import verified_u_records,expected_waves
        from moderntcn_dev_v2.training_config import validate as previous_validate
        c=validate(f.STORED);old=json.loads((f.ROOT/'configs/moderntcn_uv_fold6_epf_parallel_v2.json').read_text());previous_validate(old)
        self.assertEqual(len(c['tasks']),18);self.assertEqual(len(verified_u_records(c)),9);self.assertEqual(len(c['future_input_scope']),12)
        self.assertEqual([len(w) for w in expected_waves()],[4,4,1,4,4,1]);self.assertEqual(sum(t['arithmetic']['max_optimizer_steps'] for t in c['tasks']),1245080)
        self.assertEqual(sum(t['arithmetic']['max_epochs'] for t in c['tasks']),360)
        for a,b in zip(c['tasks'],old['tasks']):
            for key in ['structural_source_profile','data_config','data_stat','profile_sha','arithmetic','baseline_training']:self.assertEqual(a[key],b[key])
        for mutate in [lambda c:c['tasks'].reverse(),lambda c:c['tasks'].append(c['tasks'][0]),lambda c:c['tasks'][0].update(fold=1),lambda c:c.update(B_ref='paper_C1_r8_v1')]:
            bad=copy.deepcopy(c);mutate(bad);self.assertRaises(ValueError,validate,bad)
        self.assertRaises(ValueError,validate,old)

    def test_summary_missing_ties_and_reference_attribution(self):
        from moderntcn_dev_v3.summary import aggregate
        from moderntcn_dev_v3.scope import verified_u_records
        c=f.STORED;refs=verified_u_records(c);empty=aggregate(c,[],reuse_records=refs)
        self.assertEqual(empty['completed'],0);self.assertEqual(empty['U_equivalent_reuse'],9);self.assertEqual(empty['goals']['goal2']['numerical_status'],'Not evaluated')
        records=[]
        for t in c['tasks']:
            b=c['existing_B'][t['baseline_pair_cell_id']];records.append((t,dict(task_id=t['id'],mse=b['mse']*.8,mae=b['mae']*(1 if t['dataset'] in ('PJM','NP','BE') else .8))))
        result=aggregate(c,records,reuse_records=refs);self.assertEqual(result['goals']['goal2']['winning_cells'],7);self.assertEqual(result['goals']['goal3']['numerical_status'],'Not evaluated')
        self.assertEqual(aggregate(c,records[:-1],reuse_records=refs)['goals']['goal2']['numerical_status'],'Not evaluated');self.assertRaises(ValueError,aggregate,c,records+[records[0]],reuse_records=refs)

    def test_template_nonexecuting_and_foreign_authorization(self):
        from moderntcn_dev_v3.scope import validate_launch_scope
        a=json.loads((f.EVIDENCE/'non-execution-preparation/launch-binding.template.json').read_text());c=f.STORED;ids=[t['id'] for t in c['tasks']]
        validate_launch_scope(c,ids,a);self.assertIsNone(a['commit']);self.assertFalse(a['review_approved'])
        with patch('moderntcn_dev_v3.launcher.git',side_effect=AssertionError('reject before Git/network')):
            self.assertRaises(PermissionError,launcher.validate_authorization,c,'unused',ids,a)
            for version in ('v1','v2'):
                bad=copy.deepcopy(a);bad.update(review_approved=True,candidate_id='moderntcn-uv-dev-'+version)
                self.assertRaises(PermissionError,launcher.validate_authorization,c,'unused',ids,bad)
        for mutate in [lambda a:a.update(max_optimizer_updates=1867620),lambda a:a['task_profile_bindings'].clear(),lambda a:a['U_reuse'].clear()]:
            bad=copy.deepcopy(a);mutate(bad);self.assertRaises(ValueError,validate_launch_scope,c,ids,bad)

if __name__=='__main__':unittest.main()
