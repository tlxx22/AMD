"""Targeted first acceptance of Goal3 execution scope, not a model redesign."""
import ast,copy,json,subprocess,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from moderntcn_input_ablation_v10 import scope,training_config as tc,summary,launcher,training_state as ts,worker
from moderntcn_dev_v10 import adapter
from utils.ch3_data import load
from utils.ch3_onecycle import Schedule
from utils.feature_schema import get_feature_schema
from tests.moderntcn_input_ablation_v10_fixtures import E,case,fixture
RESULTS=[]

class InputScope(unittest.TestCase):
    def test_01_scope_schema_budget(self):
        c=tc.derive();self.assertEqual(len(c['tasks']),12);self.assertEqual([len(w) for w in c['waves']],[4,4,4]);self.assertEqual(sum(t['arithmetic']['max_optimizer_steps'] for t in c['tasks']),1782120)
        for t in c['tasks']:
            p=t['structural_source_profile'];self.assertEqual(p['features'],list(get_feature_schema(t['input_variant']).feature_names));self.assertEqual(t['arithmetic']['train_batches'],{3:7435,6:7429,9:7422,12:7416}[t['h']]);self.assertEqual(p['training'],t['baseline_training'])
        for key,value in [('arm','U'),('h',24),('fold',1),('input_variant','F4'),('task_scope','development_fold6_epf_horizon_coupled_spline_v10')]:
            bad=copy.deepcopy(c);bad['tasks'][0][key]=value
            with self.assertRaises(ValueError):tc.validate(bad)
        bad=copy.deepcopy(c);bad['tasks'].append(bad['tasks'][0])
        with self.assertRaises(ValueError):tc.validate(bad)

    def test_02_same_loader_windows_scaler(self):
        folder=case('data');evidence=[]
        for h in scope.HORIZONS:
            data=[]
            for v in ('F1','F2','F3'):
                c,t,path=fixture(folder,v,h,(30,60,90));sets,m=load(t['data_config'],t,resolved_profile=t['structural_source_profile'],path_override=path);data.append((t,sets,m))
            # F4 data-only reference, with the same synthetic history; never a model/test.
            t=copy.deepcopy(data[0][0]);p=t['structural_source_profile'];names=list(get_feature_schema('F4').feature_names);t['input_variant']='F4';p.update(input_variant='F4',C=11,features=names,aux_idx=list(range(1,11)))
            sets,m=load(t['data_config'],t,resolved_profile=p,path_override=path);data.append((t,sets,m))
            for t,sets,m in data:
                self.assertFalse(m['test_observations_accessed']);self.assertEqual(m['scaler'],data[-1][2]['scaler']);self.assertEqual(m['window_counts'],data[-1][2]['window_counts'])
                for split in ('train','validation'):
                    ds=sets[split];ref=data[-1][1][split];self.assertEqual(len(ds),len(ref));self.assertEqual(ds.label_horizon,h)
                    for index in (0,1,len(ds)-1):
                        x,y=ds[index];xr,yr=ref[index];self.assertTrue(torch.equal(y,yr));self.assertEqual(tuple(x.shape),(12,len(t['structural_source_profile']['features'])))
                        for i,name in enumerate(t['structural_source_profile']['features']):self.assertTrue(torch.equal(x[:,i],xr[:,names.index(name)]))
                        w,node=divmod(index,2);pos=ds.start+w+12+h-1;self.assertTrue(torch.equal(y,torch.from_numpy(ds.y[pos,node:node+1].reshape(1,1))))
                evidence.append(dict(variant=t['input_variant'],h=h,windows=m['window_counts'],scaler_sha=tc.digest(m['scaler']),features=t['structural_source_profile']['features']))
        with (E/'tests'/'data-alignment.json').open('x') as f:json.dump(evidence,f,indent=2)

    def test_03_three_actual_model_short_chains(self):
        for variant in scope.VARIANTS:
            folder=case(variant);c,t,path=fixture(folder,variant);root=folder/'results'
            if variant=='F1':
                paused=worker.run_worker(c,t['id'],root,'synthetic-input',device='cpu',path_override=path,pause_after_epoch=1)
                self.assertEqual(paused['status'],'paused');self.assertFalse((Path(paused['staging'])/'test-claim.json').exists())
                r=worker.run_worker(c,t['id'],root,'synthetic-input',device='cpu',path_override=path,resume=True)
            else:r=worker.run_worker(c,t['id'],root,'synthetic-input',device='cpu',path_override=path)
            m=r['manifest'];self.assertEqual(m['identity']['task_scope'],scope.SCOPE);self.assertEqual(m['steps'],4);self.assertEqual(m['test_calls'],1);self.assertEqual(m['routes'],dict(train=4,validation=4,test=2));self.assertEqual(m['identity']['structural_spec']['C'],{'F1':6,'F2':8,'F3':9}[variant]);self.assertEqual(r['metrics']['elements'],4)
            self.assertEqual(worker.audit(r['path']),r);self.assertEqual(summary.summarize(c,root,'synthetic-input')['fresh_completed'],1)
            RESULTS.append((c,t,r,root,path))
        with (E/'tests'/'short-chain-receipts.json').open('x') as f:json.dump([r for c,t,r,root,path in RESULTS],f,indent=2)

    def test_04_state_scope_before_live_write(self):
        c,t,r,root,path=RESULTS[0];state=ts.read(Path(r['path'])/'last.pt');p=t['structural_source_profile'];m=adapter.build(p,use_u=True,use_v=True,task='MS',fold=6);tr=p['training'];opt=torch.optim.Adam(m.parameters(),lr=tr['lr'],betas=tuple(tr['betas']),eps=tr['eps'],weight_decay=tr['weight_decay']);sch=Schedule(opt,tr['scheduler']);identity=state['identity']
        before=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone()
        self.assertIs(ts.validate(state,m,opt,sch,identity),state)
        for mutation in ('scope','F','H','knots'):
            bad=copy.deepcopy(state)
            if mutation=='scope':bad['identity']['task_scope']='old-epf-scope'
            elif mutation=='F':bad['identity']['task']['input_variant']='F2'
            elif mutation=='H':bad['identity']['task']['h']=6
            else:bad['model']['u.knots'][0]+=1
            with self.assertRaises(ValueError):ts.validate(bad,m,opt,sch,identity)
            self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        with self.assertRaises(ValueError):ts._require_model_identity(m,dict(identity,task_scope='old'))
        # Existing registered success cannot be published twice, without constructing another model.
        with self.assertRaises(FileExistsError):worker.publish(r['path'],r['path'],r['manifest'],r['metrics'])

    def test_05_f4_bound_reference_and_rejection(self):
        c=tc.derive();refs=scope.verified_f4_records(c);self.assertEqual(len(refs),4);self.assertEqual({t['h'] for t,m in refs},set(scope.HORIZONS))
        for replacement in (c['F4_reuse'][:-1],c['F4_reuse']+[c['F4_reuse'][0]]):
            b=copy.deepcopy(c);b['F4_reuse']=replacement
            with self.assertRaises(ValueError):scope.verified_f4_records(b)
        b=copy.deepcopy(c);b['F4_reuse'][0]['producer_commit']='0'*40
        with self.assertRaises(ValueError):scope.verified_f4_records(b)
        with self.assertRaises(ValueError):summary.aggregate(c,[],f4_records=[])
        s=summary.aggregate(c,[]);self.assertEqual(s['input_evaluation_cells_available'],4);self.assertEqual(s['goal3']['numerical_status'],'Not evaluated');self.assertEqual(s['module_goals']['goal1']['numerical_status'],'Met');self.assertEqual(s['module_goals']['goal2']['winning_cells'],7)

    def test_06_goal3_complete_missing_ties_no_f2_f3_order(self):
        def cells(values):return {v:{h:dict(mse=x,mae=x) for h in scope.HORIZONS} for v,x in zip(('F1','F2','F3','F4'),values)}
        for values in ((4,2,3,1),(4,3,2,1)):
            self.assertEqual(summary.goal3(cells(values))['numerical_status'],'Met')
        for values in ((4,1,3,1),(4,2,4,1),(4,2,3,3)):
            self.assertEqual(summary.goal3(cells(values))['numerical_status'],'Not met')
        z=cells((4,2,3,1));z['F1'][3]=None;self.assertEqual(summary.goal3(z)['numerical_status'],'Not evaluated')
        z=cells((4,2,3,1));z['F1'][3]['mae']=0;self.assertEqual(summary.goal3(z)['numerical_status'],'Not met')
        c=tc.derive();t=c['tasks'][0];m=dict(task_id=t['id'],mse=1.,mae=1.)
        with self.assertRaises(ValueError):summary.aggregate(c,[(t,m),(t,m)])
        bad=copy.deepcopy(t);bad['arm']='U'
        with self.assertRaises(ValueError):summary.aggregate(c,[(bad,m)])
        # Per-H reversal allowed: only equal-four-H means govern.
        z=cells((4,2,3,1));z['F4'][3]=dict(mse=3.,mae=3.)
        self.assertEqual(summary.goal3(z)['numerical_status'],'Met')

    def test_07_nonexecution_before_network_model_and_scope(self):
        c=tc.derive();ids=[t['id'] for t in c['tasks']]
        with patch.object(launcher,'git',side_effect=AssertionError('network forbidden')),patch.object(launcher,'gpu_startup',side_effect=AssertionError('GPU forbidden')):
            with self.assertRaises(PermissionError):launcher.validate_authorization(c,'unused',ids,dict(commit=None,review_approved=False))
            with self.assertRaises(ValueError):launcher.validate_authorization(c,'unused',ids,dict(commit='x',review_approved=True,task_scope='development_fold6_epf_horizon_coupled_spline_v10'))
        for changed in (ids[:-1],ids[::-1],ids+[ids[0]]):
            with self.assertRaises(ValueError):scope.validate_launch_scope(c,changed,dict(task_scope=scope.SCOPE))
        with self.assertRaises(PermissionError):launcher.stop('/tmp','unrelated')
        with self.assertRaises(ValueError):launcher.status('/tmp','../outside')

    def test_08_original_model_guard_and_namespace_static(self):
        import moderntcn_dev_v10.worker as old
        self.assertIs(worker.adapter,adapter);self.assertIs(worker.gradients_collected,old.gradients_collected);self.assertIs(worker.parameters_collected,old.parameters_collected);self.assertIs(worker.loss_guard,old.loss_guard)
        for name in ('evaluate','environment','request_stop','append','hashlib_of_tensor'):
            import inspect
            self.assertEqual(ast.dump(ast.parse(inspect.getsource(getattr(worker,name)))),ast.dump(ast.parse(inspect.getsource(getattr(old,name)))))
        text=Path(worker.__file__).read_text();self.assertLess(text.index('loss_guard(loss)'),text.index('loss.backward()'));self.assertLess(text.index('gradients_collected(named)'),text.index('opt.step()'));self.assertLess(text.index('parameters_collected(named)'),text.index('scheduler.after_successful_update()'))
        self.assertNotEqual(worker.RUN_NAMESPACE,old.RUN_NAMESPACE)
        with self.assertRaises(ValueError):adapter.loss(torch.zeros(2,1,1),torch.zeros(2,1))
        self.assertEqual(subprocess.run(['bash','-n',str(tc.ROOT/'scripts/ch3/start_moderntcn_input_ablation_v10.sh')]).returncode,0)
        for path in (tc.ROOT/'moderntcn_input_ablation_v10').glob('*.py'):ast.parse(path.read_text())
        self.assertEqual(tc.derive()['source_binding'],worker.source_binding())

if __name__=='__main__':unittest.main()
