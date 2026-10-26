"""Bounded synthetic differential tests; no real data/checkpoint or training CLI."""
import ast,copy,json,math,os,unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from tests import moderntcn_v8_fixtures as f
from moderntcn_dev_v8 import adapter,training_state,launcher
from moderntcn_dev_v8.modules import LaggedAuxiliaryTransport
from moderntcn_dev_v3 import adapter as previous
from moderntcn_dev_v8.guards import *
def gradients(m):return {n:p.grad for n,p in m.named_parameters()}
def common(m):return {n:p for n,p in m.state_dict().items() if not n.startswith(('u.','v.'))}
def close(a,b):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)

def reference(u,z):
    B,C,D,N=z.shape;J=u.J;K=u.K
    # Independent per-variable/per-position formula, no production gather/masks.
    norm=torch.stack([torch.stack([F.layer_norm(z[b,c].T,(D,),eps=1e-5) for c in range(C)]) for b in range(B)])
    q=torch.stack([F.linear(norm[b,u.target],u.W_Q) for b in range(B)])
    k=torch.stack([torch.stack([F.linear(norm[b,c],u.W_K[j]) for j,c in enumerate(u.aux)]) for b in range(B)])
    val=torch.stack([torch.stack([F.linear(z[b,c].T,u.W_V[j]) for j,c in enumerate(u.aux)]) for b in range(B)])
    score=torch.stack([torch.stack([torch.stack([sum((q[b,n]*k[b,j,n-l]).sum() for n in range(l,N))/((N-l)*math.sqrt(8)) for l in range(K)]) for j in range(J)]) for b in range(B)])
    weights=[];messages=[]
    for b in range(B):
        wb=[];mb=[]
        for n in range(N):
            valid=[(j,l) for j in range(J) for l in range(K) if l<=n]
            w=torch.softmax(torch.stack([score[b,j,l] for j,l in valid]),dim=0)
            mb.append(sum(w[i]*val[b,j,n-l] for i,(j,l) in enumerate(valid)))
            lookup={p:i for i,p in enumerate(valid)}
            wb.append(torch.stack([w[lookup[j,l]] if (j,l) in lookup else score.new_zeros(()) for j in range(J) for l in range(K)]).reshape(J,K))
        weights.append(torch.stack(wb));messages.append(torch.stack(mb))
    alpha=torch.stack(weights);message=torch.stack(messages);delta=F.linear(message,u.W_O).permute(0,2,1)
    out=z.clone();out[:,u.target]=z[:,u.target]+delta
    return out,q,k,val,score,alpha,message

class Differential(unittest.TestCase):
    def setUp(self):torch.manual_seed(401)

    def test_loop_forward_gradient_Urban_EPF(self):
        rows=[]
        for C,N,target in [(11,12,0),(3,24,2)]:
            u=LaggedAuxiliaryTransport(C,64,N,target,[i for i in range(C) if i!=target]);u.W_O.data.copy_(torch.linspace(-.1,.1,512).reshape(64,8));z=torch.randn(2,C,64,N,requires_grad=True)
            actual=u(z);q,k,v=u.project(z);msg,score,alpha=u.transport(q,k,v);expected,qr,kr,vr,sr,ar,mr=reference(u,z)
            errors={}
            for name,a,b in zip(['output','q','k','value','score','alpha','message'],[actual,q,k,v,score,alpha,msg],[expected,qr,kr,vr,sr,ar,mr]):
                close(a,b);errors[name]=float((a-b).abs().max())
            weight=torch.linspace(-.5,.5,actual.numel()).reshape_as(actual)/actual.numel()
            ga=torch.autograd.grad((actual*weight).sum(),[z,*u.parameters()],retain_graph=True);gb=torch.autograd.grad((expected*weight).sum(),[z,*u.parameters()])
            for a,b in zip(ga,gb):close(a,b)
            errors['gradient']=max(float((a-b).abs().max()) for a,b in zip(ga,gb));self.assertTrue(torch.equal(alpha.masked_select(~u.valid[None,:,None,:].expand_as(alpha)),torch.zeros_like(alpha.masked_select(~u.valid[None,:,None,:].expand_as(alpha)))))
            close(alpha.sum((2,3)),torch.ones(2,N));rows.append(dict(C=C,N=N,max_abs=errors,atol=2e-6,rtol=2e-5))
        f.output('loop-reference',rows)

    def test_fixed_lag_direction_and_no_wrap(self):
        u=LaggedAuxiliaryTransport(3,64,12,0,[1,2]);q=torch.zeros(1,12,8);k=torch.zeros(1,2,12,8);value=torch.zeros_like(k)
        q[0,4,0]=8;k[0,0,2,0]=8;value[0,0,2,0]=3;value[0,1,-1,0]=100
        message,score,alpha=u.transport(q,k,value)
        self.assertEqual(int(score[0,0].argmax()),2)
        close(score[0,0,2],torch.tensor(64/(10*math.sqrt(8))))
        self.assertEqual(float(message[0,0,0]),0);self.assertGreater(float(message[0,4,0]),0)
        changed=value.clone();changed[0,0,2,0]=7;msg2,s2,a2=u.transport(q,k,changed);f.equal(self,score,s2);f.equal(self,alpha,a2);self.assertGreater(float(msg2[0,4,0]),float(message[0,4,0]))
        # Poison only invalid safe gather fills; no legitimate source is changed.
        safe_k=k[:,:,u.source_index,:].clone();safe_v=value[:,:,u.source_index,:].clone();invalid=~u.valid[None,None,:,:,None]
        safe_k.masked_fill_(invalid,999);safe_v.masked_fill_(invalid,-777)
        s=torch.einsum('bnr,bjnkr->bjk',q,safe_k.masked_fill(invalid,0))/(u.overlap*math.sqrt(8))[None,None,:]
        m=torch.einsum('bnjk,bjnkr->bnr',alpha,safe_v);f.equal(self,s,score);f.equal(self,m,message)
        f.output('lag-fixture',dict(predefined_query_position=4,key_position=2,expected_lag=2,score=score.tolist(),joint_mask=True,value_effect=True,tail_does_not_wrap=True,invalid_fills_inert=True))

    def test_schema_order_target_batch_and_finite(self):
        rows=[]
        for C,N,t in [(11,12,0),(3,24,2)]:
            u=LaggedAuxiliaryTransport(C,64,N,t,[i for i in range(C) if i!=t]);u.W_O.data.fill_(.02);z=torch.randn(2,C,64,N);a=u(z);q,k,v=u.project(z)
            b=z.clone();b[:,t]=torch.flip(b[:,t],[1]);q2,k2,v2=u.project(b);self.assertFalse(torch.equal(q,q2));f.equal(self,k,k2);f.equal(self,v,v2)
            f.equal(self,a[:,u.aux],z[:,u.aux]);f.equal(self,u(z.flip(0)).flip(0),a)
            bad=z.clone();bad[0,0,0,0]=float('nan');self.assertRaises(ValueError,u,bad);self.assertRaises(ValueError,u,z[:,:,:,:-1]);self.assertRaises(ValueError,LaggedAuxiliaryTransport,C,64,N,C,[])
            self.assertRaises(ValueError,LaggedAuxiliaryTransport,C,64,N,t,list(reversed(u.aux)))
            rows.append(dict(target=t,aux=list(u.aux),target_query_not_value=True,batch_permutation_same_shape_exact=True,aux_slots_exact=True))
        empty=LaggedAuxiliaryTransport(1,64,12,0,[]);z=torch.randn(2,1,64,12);empty.W_O.data.fill_(1);f.equal(self,empty(z),z)
        f.output('schema',dict(cases=rows,no_aux_zero=True))

    def test_nonzero_sequential_wiring_and_observer(self):
        rows=[]
        for ds,H in [('UrbanEV',6),('PJM',24)]:
            p=f.profile(ds,H=H);m=adapter.build(p,use_u=True,use_v=True);f.open_branches(m);m.eval();X=torch.randn(2,p['T'],p['C']);original=X.clone();state=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone();out=m(X);seen={};calls={};hooks=[];block=m.model.stages[0].blocks[0]
            def hook(name):
                def observe(mod,args,result):seen[name]=(args,result);calls[name]=calls.get(name,0)+1
                return observe
            def rev(mod,args,result):
                name='revin_'+args[1];seen[name]=(args,result.clone());calls[name]=calls.get(name,0)+1
            hooks.append(m.model.revin_layer.register_forward_hook(rev))
            for name,mod in [('u',m.u),('v',m.v),('stem',m.model.downsample_layers[0]),('h',block.ffn1drop2),('ffn1drop1',block.ffn1drop1),('ffn2pw1',block.ffn2pw1),('ffn2drop1',block.ffn2drop1),('ffn2act',block.ffn2act),('ffn2pw2',block.ffn2pw2),('ffn2drop2',block.ffn2drop2),('BN',block.norm),('head',m.model.head)]:hooks.append(mod.register_forward_hook(hook(name)))
            actual,diag=m(X,return_diagnostics=True)
            for h in hooks:h.remove()
            f.equal(self,out,actual);f.equal(self,state,m.state_dict());f.equal(self,original,X);f.equal(self,rng,torch.get_rng_state());self.assertTrue(all(n==1 for n in calls.values()))
            h=seen['h'][1].reshape(2,p['C'],64,-1);v=seen['v'][1];z=h+v;s=seen['u'][1]
            f.equal(self,seen['v'][0][0],h);f.equal(self,seen['v'][0][1],seen['revin_norm'][1].permute(0,2,1));f.equal(self,seen['v'][0][2],X);f.equal(self,seen['u'][0][0],z);self.assertGreater(float((s-z).abs().sum()),0);f.equal(self,s[:,p['aux_idx']],z[:,p['aux_idx']])
            f.equal(self,seen['ffn2pw1'][0][0],s.permute(0,2,1,3).reshape(2,64*p['C'],-1));f.equal(self,seen['revin_denorm'][0][0],seen['head'][1].permute(0,2,1));self.assertIn('U_lagged_auxiliary',diag);self.assertNotIn('U_tied_component',diag)
            m.use_v=False;self.assertEqual(m(X).shape,(2,p['pred_len'],p['C']));m.use_u=False
            with patch.object(m.u,'forward',side_effect=AssertionError('disabled U')):m(X)
            rows.append(dict(dataset=ds,calls=calls,nonzero=True,V_reads_h=True,U_reads_postV=True,aux_slots_unchanged=True,single_RevIN=True,observer='exact'))
        f.output('wiring',rows)

    def test_exit_then_internal_loss_gradient_opening(self):
        rows=[]
        for ds,H in [('UrbanEV',3),('PJM',24)]:
            p=f.profile(ds,H=H);m=adapter.build(p,use_u=True,use_v=True);opt=torch.optim.Adam(m.parameters(),lr=.01,weight_decay=1e-7);X=torch.randn(2,p['T'],p['C']);target=torch.randn(2,p['pred_len'],1);history=[]
            for step in range(3):
                opt.zero_grad();adapter.loss(adapter.prediction(m,X),target).backward();g={n:float(p.grad.abs().sum()) for n,p in m.named_parameters() if n.startswith(('u.','v.'))};history.append(g)
                if step==0:
                    self.assertGreater(g['u.W_O'],0)
                    for name in ['u.W_Q','u.W_K','u.W_V']:self.assertEqual(g[name],0)
                    self.assertGreater(g['v.A'],0)
                self.assertGreater(float(m.model.stages[0].blocks[0].ffn1pw1.weight.grad.abs().sum()),0);self.assertGreater(float(m.model.stages[0].blocks[0].ffn2pw1.weight.grad.abs().sum()),0);opt.step()
            for name in ['u.W_Q','u.W_K','u.W_V','v.P_s','v.W1.weight']:self.assertGreater(history[-1][name],0)
            rows.append(dict(dataset=ds,loss_gradients_before_Adam=history))
        f.output('gradient-opening',rows)

    def test_profiles_parameters_target_loss(self):
        rows=[]
        for ds,var,H,U in [('UrbanEV','F1',3,6144),('UrbanEV','F2',6,8192),('UrbanEV','F3',9,9216),('UrbanEV','F4',12,11264),('PJM','F4',24,3072)]:
            p=f.profile(ds,var,H);m=adapter.build(p,use_u=True,use_v=True);out=adapter.prediction(m,torch.randn(2,p['T'],p['C']));self.assertEqual(out.shape,(2,p['pred_len'],1));self.assertEqual(m.u.target,p['target_idx']);self.assertEqual(list(m.u.aux),p['aux_idx']);self.assertGreater(m.u.J,0)
            u=sum(p.numel() for p in m.u.parameters());v=sum(p.numel() for p in m.v.parameters());self.assertEqual(u,U);self.assertEqual(v,572 if ds=='UrbanEV' else 2076);self.assertRaises(ValueError,adapter.loss,out,torch.zeros(2,p['pred_len'],p['C']))
            self.assertEqual(set(dict(m.u.named_parameters())),{'W_Q','W_K','W_V','W_O'})
            rows.append(dict(dataset=ds,variant=var,C=p['C'],L=p['T'],P=p['pred_len'],H=H,N=m.spec.value['N'],aux=p['aux_idx'],B=sum(p.numel() for n,p in m.named_parameters() if not n.startswith(('u.','v.'))),U=u,V=v,UV=u+v,buffer_bytes=sum(b.numel()*b.element_size() for b in m.u.buffers())))
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

    def test_same_v8_guard_exact_and_fault_boundaries(self):
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
            f.output('guard-integration',dict(full_v8_updates_per_path=2,all_outputs_gradients_states_RNG='exact',fault_boundaries='same order, no extra update',cases=rows,previous_GPU_evidence_reused=True))

    def test_version_rejection_before_write_and_synthetic_IO(self):
            m=adapter.build(f.profile(),use_u=True,use_v=True);state=copy.deepcopy(m.state_dict());ck=adapter.synthetic_checkpoint(m)
            for version in ['v1','v2','v3','v4','v5','v6','v7']:
                bad=copy.deepcopy(ck);bad['format']='moderntcn-uv-synthetic-'+version;bad['spec']['candidate_id']='moderntcn-uv-dev-'+version
                self.assertRaises(ValueError,adapter.load_synthetic_checkpoint,m,bad);f.equal(self,state,m.state_dict())
                with patch('timemixer_dev.training_state.restore',side_effect=AssertionError('must reject before read/write')):self.assertRaises(ValueError,training_state.restore,'forbidden-real-path',m,None,None,None,dict(candidate_id=bad['spec']['candidate_id'],structural_spec=bad['spec']))
            path=f.EVIDENCE/'tests'/('synthetic-state-'+os.environ['MTCN_TEST_RUN']+'.pt');torch.save(ck,path);loaded=torch.load(path,map_location='cpu',weights_only=False);adapter.load_synthetic_checkpoint(m,loaded);f.equal(self,state,m.state_dict())
            bad=copy.deepcopy(ck);bad['spec']['target_idx']=2;self.assertRaises(ValueError,adapter.load_synthetic_checkpoint,m,bad);f.equal(self,state,m.state_dict())

class Metadata(unittest.TestCase):
    def test_scope_summary_duplicates_budget_and_old_config(self):
            from moderntcn_dev_v8.training_config import validate
            from moderntcn_dev_v8.scope import verified_v_records,expected_waves
            from moderntcn_dev_v8.summary import aggregate
            from moderntcn_dev_v7.training_config import validate as old_validate
            c=validate(f.STORED);old=json.loads((f.ROOT/'configs/moderntcn_uv_fold6_epf_tied_component_v7.json').read_text());old_validate(old);refs=verified_v_records(c)
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
            from moderntcn_dev_v8.scope import validate_launch_scope
            a=json.loads((f.EVIDENCE/'non-execution-preparation/launch-binding.template.json').read_text());c=f.STORED;ids=[t['id'] for t in c['tasks']];validate_launch_scope(c,ids,a)
            self.assertIsNone(a['commit']);self.assertFalse(a['review_approved']);self.assertFalse(c['execution_authorized'])
            with patch('moderntcn_dev_v8.launcher.git',side_effect=AssertionError('reject before Git/network')):
                self.assertRaises(PermissionError,launcher.validate_authorization,c,'unused',ids,a)
                for version in ['v1','v2','v3','v4','v5','v6','v7']:
                    bad=copy.deepcopy(a);bad.update(review_approved=True,candidate_id='moderntcn-uv-dev-'+version);self.assertRaises(PermissionError,launcher.validate_authorization,c,'unused',ids,bad)
            old=(f.EVIDENCE.parent/'M6-ModernTCN-UV-v4-postrun-review-diagnosis-r1/performance/guards.py').read_text();new=(f.ROOT/'moderntcn_dev_v8/guards.py').read_text()
            def functions(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
            self.assertEqual(functions(old),functions(new))
            s=(f.ROOT/'moderntcn_dev_v8/worker.py').read_text();seq=['loss_guard(loss)','loss.backward()','gradients_collected(named)','opt.step();steps+=1','parameters_collected(named)','scheduler.after_successful_update()'];pos=[s.index(k) for k in seq];self.assertEqual(pos,sorted(pos));self.assertNotIn('torch.cuda.synchronize',s)
            for mutate in [lambda a:a.update(max_optimizer_updates=1),lambda a:a['task_profile_bindings'].clear(),lambda a:a['V_reuse'].clear()]:
                bad=copy.deepcopy(a);mutate(bad);self.assertRaises(ValueError,validate_launch_scope,c,ids,bad)
            f.output('guard-AST',dict(all_functions_and_error_class='exact AST',worker_boundaries=seq,extra_CUDA_sync=False,legacy_AUTH_rejected=True))
