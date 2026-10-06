"""Three-ring preparation fixtures. Real model/Adam/GPU computation is forbidden."""
import ast,copy,json,math,os,subprocess,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack,nullcontext
from tests.ch3_historical_type1_v2_231 import s,q
from utils import ch3_type1_upstream as u
from utils.ch3_type1_scaled import Schedule,configuration,coefficient,lr_used,validate_probe_trace
from utils.ch3_contract import ROOT,profile,digest,BestState,step_arithmetic
from utils.ch3_native_recovery_records import ref,bound,exclusive,sha

class FakeOptimizer:
    def __init__(self):self.param_groups=[dict(lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=1e-7)]

class V2Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests import ch3_historical_type1_v2_147 as frozen
        cls.binding=patch.dict(globals(),s=frozen.s,q=frozen.q,bound=frozen.bound);cls.binding.start();cls.cs=q.configs()
    @classmethod
    def tearDownClass(cls):cls.binding.stop()
    def test_counts(self):self.assertEqual([len(c['tasks'])for c in self.cs.values()],[28,35,84]);self.assertEqual(sum(len(c['tasks'])for c in self.cs.values()),147)
    def test_models(self):self.assertEqual(s.MODELS,('AMD','DLinear','PatchTST','iTransformer','TimeMixer','ModernTCN','TimeXer'))
    def test_urban_exact(self):self.assertEqual({(t['model'],t['fold'],t['h'])for t in self.cs['URBAN_SUBSET']['tasks']},{(m,f,h)for m in s.MODELS for f in (1,2)for h in (3,12)})
    def test_epf_exact(self):self.assertEqual({(t['model'],t['dataset'],t['h'])for t in self.cs['EPF_ALL']['tasks']},{(m,d,24)for m in s.MODELS for d in ('PJM','NP','BE','FR','DE')})
    def test_m_exact(self):self.assertEqual({(t['model'],t['dataset'],t['h'])for t in self.cs['M_ALL']['tasks']},{(m,d,h)for m in s.MODELS for d in ('ETTh1','Weather','Exchange')for h in (96,192,336,720)})
    def test_exclusions(self):self.assertFalse(any(t['model']in ('J','N','S')or t['dataset']=='ECL'for c in self.cs.values()for t in c['tasks']))
    def test_batches_and_epochs(self):
        for stage,c in self.cs.items():
            expected=(128,128,20,5)if stage=='URBAN_SUBSET'else(32,32,10,3)if stage=='EPF_ALL'else(128,128,10,None)
            for t in c['tasks']:
                p=profile(c,t);self.assertEqual(tuple(p['training'][k]for k in ('batch','eval_batch','epochs','patience')),expected)
    def test_T_shape(self):
        for stage,c in self.cs.items():
            for t in c['tasks']:
                p=profile(c,t);self.assertEqual(p['T'],12 if stage=='URBAN_SUBSET' else 168 if stage=='EPF_ALL' else 96)
                if stage=='URBAN_SUBSET':self.assertEqual((p['C'],p['pred_len']),(11,1))
    def test_diff_whitelist(self):
        for stage,c in self.cs.items():
            for old,t in zip(s.selected(stage),c['tasks']):
                p=copy.deepcopy(profile(c,t));prior=copy.deepcopy(s.parent(stage)['resolved_profiles'][old['id']])
                for x in (p,prior):
                    x.pop('T')
                    for k in ('batch','eval_batch','lr','epochs','patience','scheduler'):x['training'].pop(k)
                self.assertEqual(p,prior)
    def test_fixed_optimizer_and_data_fields(self):
        for stage,c in self.cs.items():
            for old,t in zip(s.selected(stage),c['tasks']):
                p=profile(c,t);prior=s.parent(stage)['resolved_profiles'][old['id']]
                for k in ('betas','eps','weight_decay','accumulation','threads','workers','dtype','train_shuffle','train_drop_last','eval_drop_last','seed'):self.assertEqual(p['training'][k],prior['training'][k])
                self.assertEqual(p['training']['accumulation'],1);self.assertFalse(p['training']['eval_drop_last'])
    def test_numeric_inheritance(self):
        for stage,c in self.cs.items():
            for t in c['tasks']:self.assertEqual(s.numeric_policy(c,t),s.parent(stage)['baseline_unified']['numeric_policies'][t['model']+'-'+t['dataset']])
    def test_moderntcn_rules(self):
        c=self.cs['EPF_ALL'];self.assertTrue(all(s.numeric_policy(c,t)['state_atol']==5e-4 for t in c['tasks']if t['model']=='ModernTCN'))
        c=self.cs['URBAN_SUBSET'];self.assertTrue(all(s.numeric_policy(c,t)is None for t in c['tasks']if t['model']=='ModernTCN'))
    def test_m_channels(self):
        for t in self.cs['M_ALL']['tasks']:
            p=profile(self.cs['M_ALL'],t);self.assertEqual((t['task'],t['metric_scope'],p['task'],p['metric_scope']),('M','all_channels','M','all_channels'));self.assertEqual(p['supervised_channels'],list(range(p['C'])));self.assertEqual(p['output_order'],p['features'])
    def test_metadata_binding(self):
        from utils.ch3_time_marks import metadata
        for c in self.cs.values():
            data=bound(c['baseline_unified']['data_ref']);self.assertFalse(data['test_observations_accessed'])
            for t in c['tasks']:
                m=data['metadata'][t['dataset']][t['id']];a=step_arithmetic(c,t)
                self.assertEqual(m['window_counts'],dict(train=a['train_windows'],validation=a['validation_windows']));self.assertEqual(digest(m),data['data_bindings'][t['dataset']][t['id']]);self.assertFalse(m['test_observations_accessed'])
                for k,v in metadata(profile(c,t)).items():self.assertEqual(m[k],v)
    def test_source_state_projection(self):
        from utils.ch3_m6 import source_states
        for stage,c in self.cs.items():
            data=bound(c['baseline_unified']['data_ref']);prior=bound(s.parent(stage)['baseline_unified']['data_ref'])
            actual=source_states(c)
            self.assertEqual(data['source_states'],{p:prior['source_states'][p]for p in actual})
            self.assertEqual(data['source_states'],actual)
    def test_all_scheduler_exact(self):
        for c in self.cs.values():
            for t in c['tasks']:
                p=profile(c,t);self.assertEqual(p['training']['scheduler'],configuration(p['training']['epochs'],step_arithmetic(c,t)['train_batches']))
    def test_formal_budget(self):
        expected=[dict(runs=28,run_epochs=560,adam=1028440,backward=1028440,forward=1143128),dict(runs=35,run_epochs=350,adam=399000,backward=399000,forward=467845),dict(runs=84,run_epochs=840,adam=108080,backward=108080,forward=128877)]
        self.assertEqual([s.formal_budget(c)['total']for c in self.cs.values()],expected)
        self.assertEqual({k:sum(r[k]for r in expected)for k in expected[0]},dict(runs=147,run_epochs=1750,adam=1535520,backward=1535520,forward=1739850))
    def test_budget_independent_arithmetic(self):
        for c in self.cs.values():
            for t in c['tasks']:
                p=profile(c,t);d=c['datasets'][t['dataset']];train=p['training'];a=s.formal_budget(c)['tasks'][t['id']]
                if t['dataset']=='UrbanEV':start,end,test=c['urban_folds'][t['fold']-1];n=(start-12-t['h']+1)*275;v=(end-start-12-t['h']+1)*275;z=(test-end-12-t['h']+1)*275
                else:start,end,test=d['endpoints'];n=start-p['T']-p['pred_len']+1;v=end-start-p['pred_len']+1;z=test-end-p['pred_len']+1
                self.assertEqual(a['adam'],n//train['batch']*train['epochs']);self.assertEqual(a['forward'],train['epochs']*(n//train['batch']+math.ceil(v/train['eval_batch']))+math.ceil(z/train['eval_batch']))
    def test_probe_plans(self):
        groups=[s.probe_groups(c)for c in self.cs.values()];self.assertEqual([len(g)for g in groups],[7,8,21]);self.assertEqual([sum(len(g['representatives'])for g in rows)for rows in groups],[28,35,84])
    def test_probe_caps(self):
        expected=[(336,464,504,696),(420,560,618,824),(1008,1344,1512,2016)]
        for c,(a,f,ac,fc)in zip(self.cs.values(),expected):b=s.probe_budget(c);self.assertEqual(b['nominal'],dict(adam=a,backward=a,forward=f));self.assertEqual(b['caps'],dict(adam=ac,backward=ac,forward=fc))
    def test_timexer_structure_groups(self):
        groups=[g for g in s.probe_groups(self.cs['EPF_ALL'])if g['model']=='TimeXer'];self.assertEqual([g['planned_q']for g in groups],[4,2]);self.assertEqual([len(g['representatives'])for g in groups],[3,2]);self.assertNotEqual(groups[0]['id'],groups[1]['id'])
    def test_m_actual_four_h_waves(self):
        for m,waves in s.plan(self.cs['M_ALL'])['formal_waves'].items():self.assertEqual([len(w)for w in waves],[4,4,4])
    def test_old_namespace_refused(self):
        from utils.ch3_type1_execution import read_config
        old=json.loads((ROOT/'configs/ch3_type1_urban_subset.json').read_text())
        with self.assertRaises(ValueError):s.validate(old)
        with self.assertRaises(PermissionError):read_config(dict(unified_stage='URBAN_SUBSET',type1_scope=old['type1_followup'],protocol_file=str(ROOT/'configs/ch3_type1_urban_subset.json')))
    def test_nonexecuting_template(self):
        a=q.start_template();self.assertTrue(all(a[k]is False for k in ('reviewed','execution_permitted','structure_frozen','m6_authorized','budget_authorized')));self.assertIsNone(a['closure_commit']);self.assertEqual(a['result_review'],'pending')
    def test_old_approval_refused(self):
        a=bound(ref(s.PACKAGE.with_name('baseline-type1-followup-v1')/'start-review.json'))
        with patch.object(q,'closure',return_value=a['closure_commit']):
            with self.assertRaises(PermissionError):q.validate_start(a)
    def test_source_recipe(self):self.assertEqual(bound(s.AUTHOR_RECIPE)['commit'],'76011909357972bd55a27adba2e1be994d81b327');self.assertTrue(all(sha(p)==h for p,h in bound(s.AUTHOR_RECIPE)['files'].items()))
    def test_source_batch_proof(self):
        evidence=bound(ref(s.PACKAGE/'batch-source-proof.json'));self.assertEqual({(r['dataset'],r['batch'],r['eval_batch'])for r in evidence['source']},{('ETTh1',128,128),('Weather',128,128),('Exchange',512,512)})
    def test_fresh_result_namespace(self):self.assertFalse(s.RESULT.exists());self.assertFalse((s.PACKAGE/'start-review.json').exists());self.assertNotEqual(s.RESULT,u.OLD_RESULT)

class ScaledScheduleTests(unittest.TestCase):
    def test_E10_author_type1_equivalence(self):
        from utils.ch3_type1 import lr_used as old
        self.assertEqual([lr_used(e,10)for e in range(1,11)],[old(e)for e in range(1,11)])
    def test_E20_exact_curve(self):self.assertEqual([lr_used(e,20)for e in range(1,21)],[1e-4*.5**(max(e-2,0)*(8/18))for e in range(1,21)])
    def test_first_two_and_end(self):
        for E in (10,20):self.assertEqual([lr_used(e,E)for e in (1,2,E)],[1e-4,1e-4,3.90625e-7])
    def test_monotone(self):self.assertTrue(all(lr_used(e,20)>lr_used(e+1,20)for e in range(2,20)))
    def test_fixed_coefficient(self):self.assertEqual((coefficient(10),coefficient(20)),(1.,8/18))
    def test_no_per_batch_decay(self):
        o=FakeOptimizer();schedule=Schedule(o,configuration(20,7))
        for _ in range(7):schedule.after_successful_update();self.assertEqual(o.param_groups[0]['lr'],1e-4)
    def test_fixed_betas_all_epochs(self):
        o=FakeOptimizer();schedule=Schedule(o,configuration(20,2))
        for e in range(1,21):
            for _ in range(2):schedule.after_successful_update()
            schedule.after_epoch(e,e<20);self.assertEqual(o.param_groups[0]['betas'],(.9,.999))
    def test_resume_every_boundary(self):
        for E in (10,20):
            for stop in range(1,E):
                a=FakeOptimizer();full=Schedule(a,configuration(E,3))
                for e in range(1,stop+1):
                    for _ in range(3):full.after_successful_update()
                    full.after_epoch(e,True)
                state=copy.deepcopy(full.state_dict());b=FakeOptimizer();part=Schedule(b,configuration(E,3));b.param_groups=copy.deepcopy(a.param_groups);part.load_state_dict(state,stop*3)
                for e in range(stop+1,E+1):
                    self.assertEqual(full.state_dict(),part.state_dict())
                    for _ in range(3):full.after_successful_update();part.after_successful_update()
                    full.after_epoch(e,e<E);part.after_epoch(e,e<E)
                self.assertEqual(full.state_dict(),part.state_dict())
    def test_early_stop_does_not_rescale(self):
        a=Schedule(FakeOptimizer(),configuration(20,1))
        for e in range(1,8):a.after_successful_update();a.after_epoch(e,e<7)
        self.assertEqual(a.state_dict()['current_lr'],lr_used(7,20));self.assertEqual(a.config['epochs'],20)
    def test_repeat_boundary_rejected(self):
        a=Schedule(FakeOptimizer(),configuration(20,1));a.after_successful_update();a.after_epoch(1,True)
        with self.assertRaises(ValueError):a.after_epoch(1,True)
    def test_partial_epoch_restore_rejected(self):
        a=Schedule(FakeOptimizer(),configuration(20,7));a.after_successful_update()
        with self.assertRaises(ValueError):Schedule(FakeOptimizer(),a.config).load_state_dict(a.state_dict(),1)
    def test_old_type1_checkpoint_rejected(self):
        from utils.ch3_type1 import Schedule as Old,configuration as oldcfg
        with self.assertRaises(ValueError):Schedule(FakeOptimizer(),configuration(20,3)).load_state_dict(Old(FakeOptimizer(),oldcfg(20,3)).state_dict(),0)
    def test_onecycle_checkpoint_rejected(self):
        with self.assertRaises(ValueError):Schedule(FakeOptimizer(),configuration(20,3)).load_state_dict(dict(schema='ch3-onecycle-state-v1'),0)
    def test_wrong_E_rejected(self):
        v=Schedule(FakeOptimizer(),configuration(20,3)).state_dict()
        with self.assertRaises(ValueError):Schedule(FakeOptimizer(),configuration(10,3)).load_state_dict(v,0)
    def test_wrong_lr_beta_rejected(self):
        for k,v in (('lr',.1),('betas',(.8,.999))):
            o=FakeOptimizer();x=Schedule(o,configuration(20,1));o.param_groups[0][k]=v
            with self.assertRaises(ValueError):x.validate()
    def test_build_routing(self):
        from utils.ch3_onecycle import build
        p=dict(training=dict(scheduler=configuration(20,3)));self.assertIsInstance(build(FakeOptimizer(),p),Schedule)
    def test_exact_probe_scheduler(self):
        x=Schedule(FakeOptimizer(),configuration(20,100));rows=[]
        for _ in range(6):x.after_successful_update();rows.append(x.state_dict())
        validate_probe_trace(x.config,rows,[.9,.999]);rows[4]['updates']=6
        with self.assertRaises(ValueError):validate_probe_trace(x.config,rows,[.9,.999])
    def test_distinct_checkpoint_schema(self):self.assertEqual(Schedule.checkpoint_schema,'ch3-state-v4-type1-horizon-scaled')
    def test_no_val_test_scheduler_steps(self):
        import ch3_runner
        f=next(n for n in ast.parse(Path(ch3_runner.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='evaluate');self.assertFalse(any(isinstance(n,ast.Attribute)and n.attr in ('after_epoch','after_successful_update')for n in ast.walk(f)))

class ThreeRingLifecycleTests(unittest.TestCase):
    def test_actual_three_ring_dispatch(self):
        with tempfile.TemporaryDirectory(dir=s.PACKAGE)as tmp,patch.object(s,'RESULT',Path(tmp)/'synthetic-result'),patch.object(q,'CONTROL',Path(tmp)/'controller'),patch.object(q,'stop_check'),patch.object(q,'dynamic',return_value={}):
            q.CONTROL.mkdir();approval=exclusive(Path(tmp)/'approval.json',dict(synthetic=True,execution_permitted=False));order=[];predecessors=[]
            def permit(c,start,probe,summary_ref=None,boundary_ref=None):
                predecessors.append((c['baseline_unified']['stage'],set(boundary_ref or {})));return exclusive(s.context(c)['control']/('probe-permit.json'if probe else'formal-permit.json'),dict(synthetic=True,execution_permitted=False))
            def wait(stage,value,probe,model=None,runtime_ref=None):
                order.append((stage,probe,model))
                if not probe:exclusive(s.context(q.configs()[stage])['control']/('group-'+model)/'complete.json',dict(synthetic=True,technical_complete=True,result_review='pending',model=model,task_ids=[t['id']for t in q.configs()[stage]['tasks']if t['model']==model],mse=1e100))
            with patch.object(q,'wait_upstream',return_value=dict(synthetic=True)),patch.object(q,'validate_start'),patch.object(q,'create_permit',side_effect=permit),patch.object(q,'wait_owned',side_effect=wait),patch.object(q,'audit_probe',side_effect=lambda c:exclusive(s.context(c)['control']/'admission-summary.json',dict(synthetic=True))),patch.object(q,'seal_runtime',side_effect=lambda c,a:exclusive(s.context(c)['control']/'runtime-admission.json',dict(synthetic=True))):q.run(approval)
            final=bound(ref(q.CONTROL/'complete.json'));self.assertEqual(final['total_runs'],231);self.assertEqual([m for _,probe,m in order if not probe],list(s.MODELS)*3);self.assertEqual(final['result_review'],'pending')
            for stage,keys in predecessors:self.assertEqual(keys,set(s.STAGES[:s.STAGES.index(stage)]))
            for stage,key in (('URBAN_SUBSET','URBAN_boundary'),('EPF_ALL','EPF_boundary'),('M_ALL','M_boundary')):self.assertEqual(len(q.validate_boundary_light(final[key],stage)['task_ids']),{'URBAN_SUBSET':28,'EPF_ALL':35,'M_ALL':168}[stage])
    def test_failure_stops_all_later_rings(self):
        seen=[];actions={state:lambda r,k=state:seen.append(k)for state in q.STATES if state!='COMPLETE'}
        def failure(r):raise ValueError('numeric synthetic')
        actions['EPF_AUTO_AUDIT']=failure
        with self.assertRaises(ValueError):q.drive(actions,lambda x:None,lambda:None)
        self.assertNotIn('M_RESOURCE_NUMERIC_PROBE',seen)
    def test_wait_no_GPU_no_lock(self):
        f=next(n for n in ast.parse(Path(q.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='wait_upstream');self.assertNotIn('GPULock',ast.unparse(f));self.assertNotIn('wait_owned',ast.unparse(f))
    def test_old_running_wait_allowed(self):
        with patch.object(u,'health'),patch.object(u,'same',return_value=True),patch.object(Path,'exists',return_value=False):self.assertEqual(u.status()['state'],'WAIT_V3_COMPLETE_AND_RELEASED')
    def test_owner_missing_no_complete_rejected(self):
        with patch.object(u,'health'),patch.object(u,'same',return_value=False),patch.object(Path,'exists',return_value=False):
            with self.assertRaises(RuntimeError):u.status()
    def test_old_failure_rejected(self):
        with patch.object(u,'records'),patch.object(Path,'exists',return_value=True):
            with self.assertRaises(RuntimeError):u.health()
    def test_old_anchors_preserved(self):self.assertEqual(u.BASE,'df6a16403e10d51097db8c88829909c533d15652');self.assertEqual(u.START_REF['sha256'],'4c39ec5e025ca8ea492129d459ba395c70e68d1d2a6a698b2b142518bf8e632d');self.assertEqual(u.CONTROLLER_REF['sha256'],'1b8e6098450326577a3f42091ac09237a2c84caac12d2aff7a9b6403645dda79')
    def test_old_stop_cannot_stop_v3(self):
        with tempfile.TemporaryDirectory(dir=s.PACKAGE)as tmp,patch.object(q,'CONTROL',Path(tmp)),patch.object(q,'same',return_value=False),patch.object(q,'signal_owned')as send:
            exclusive(Path(tmp)/'controller.json',dict(scope=s.ID,owner=dict(pid=999999,start_ticks='1')));r=q.safe_stop();self.assertFalse(r['old_chain_signal_sent']);send.assert_not_called()
    def test_foreign_controller_stop_rejected(self):
        with tempfile.TemporaryDirectory(dir=s.PACKAGE)as tmp,patch.object(q,'CONTROL',Path(tmp)):
            exclusive(Path(tmp)/'controller.json',dict(scope=u.OLD_SCOPE,owner=u.OWNER))
            with self.assertRaises(PermissionError):q.safe_stop()
            self.assertFalse((Path(tmp)/'STOP').exists())
    def test_output_conflict_rejected(self):
        with tempfile.TemporaryDirectory(dir=s.PACKAGE)as tmp,patch.object(s,'RESULT',Path(tmp)),patch.object(q,'closure',return_value='mock'),patch.object(q,'validate_start'),patch.object(q,'dynamic',return_value={}),patch.object(q,'upstream_status',return_value={'state':'WAIT'}):self.assertTrue(any('retained' in x for x in q.readiness({})))
    def test_current_legacy_audit_exclusion(self):
        text=(ROOT/'utils/ch3_native_execution.py').read_text();self.assertIn("not a.get('recovery_scope') and not a.get('unified_scope') and not a.get('type1_scope')",text)
    def test_boundary_wrong_count_rejected(self):
        with tempfile.TemporaryDirectory(dir=s.PACKAGE)as tmp,patch.object(s,'RESULT',Path(tmp)):
            c=q.configs()['EPF_ALL'];path=s.context(c)['control']/'technical-boundary.json';r=exclusive(path,dict(purpose='baseline_type1_EPF_ALL_boundary_v1',scope=s.ID,technical_complete=True,result_review='pending',protocol_sha=digest(c),task_ids=['wrong']*35))
            with self.assertRaises(ValueError):q.validate_boundary_light(r,'EPF_ALL')
    def test_summary_M_route_and_no_selection(self):
        from tests.ch3_historical_type1_v2_231 import summary
        result_index=summary.result_index
        rows=result_index(q.configs());self.assertEqual(len(rows['rows']),231);self.assertTrue(rows['no_old_new_selection']);self.assertTrue(all('/M/formal-'in r['v3']['path']for r in rows['rows']if r['ring']=='M_ALL'))
    def test_formal_metadata_excludes_raw(self):
        from tests.ch3_historical_type1_v2_231 import execution
        metadata_files=execution.metadata_files
        for c in q.configs().values():
            refs=metadata_files(c,dict(start_authorization_ref=ref(s.PACKAGE/'start-approval.template.json')))
            self.assertFalse(any('/serial/'in p or '/q4/'in p or p.endswith('.bin')for p in refs))
    def test_final_test_once_and_not_earlystop(self):
        text=ast.parse((ROOT/'ch3_runner.py').read_text());f=next(x for x in text.body if isinstance(x,ast.FunctionDef)and x.name=='formal_worker');calls=[n for n in ast.walk(f)if isinstance(n,ast.Call)and any(k.arg=='test_capability'for k in n.keywords)];self.assertEqual(len(calls),1)
        self.assertGreater(calls[0].lineno,max(n.lineno for n in ast.walk(f)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='update'and isinstance(n.func.value,ast.Name)and n.func.value.id=='best'))
    def test_unknown_test_recovery_rejected(self):
        from ch3_runner import audit_resume
        with tempfile.TemporaryDirectory()as tmp:
            with self.assertRaises(PermissionError):audit_resume(tmp,dict(resume_audits={'x':dict(mode='resume',test_access_status='unknown')}),'x',protocol={'type1_followup':s.ID,'tasks':[{'id':'x'}]})
    def test_eval_weighted_tail_math_unchanged(self):
        source=(ROOT/'ch3_runner.py').read_text();f=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef)and x.name=='evaluate');text=ast.unparse(f);self.assertIn('sse / count',text);self.assertIn('sae / count',text);self.assertIn('numel()',text)
    def test_fresh_no_resume(self):
        from tests.ch3_historical_type1_v2_231 import execution
        make_config=execution.make_config
        c=q.configs()['M_ALL'];t=c['tasks'][0]
        with self.assertRaises(PermissionError):make_config(c,'ch3_formal',Path('/tmp/not-created-type1-resume'),task=t['id'],approval={},resume=True)

class ThreeRingAuditTests(unittest.TestCase):
    observations=[]
    def _fixture(self,stage,scope_kind='type1',failure=None,serial_failure=False,one_group=False,full_audit=False):
            from contextlib import contextmanager,ExitStack
            from types import SimpleNamespace
            import ch3_runner,m5_formal_entry as tool
            from tests.ch3_historical_type1_v2_231 import native as e

            @contextmanager
            def fixture():
                c=q.configs()[stage];groups=s.probe_groups(c)
                if one_group:groups=groups[:1]
                with tempfile.TemporaryDirectory(prefix='probe-audit-ownership-',dir=s.PACKAGE)as tmp,ExitStack()as stack:
                    root=Path(tmp);ctx=dict(s.context(c),probe_root=root/'probe',control=root/'stage',result_root=root/'results',fixture=root/'fixture')
                    for key in ('probe_root','control','fixture'):ctx[key].mkdir()
                    controller=root/'controller';controller.mkdir();exclusive(controller/'controller.json',dict(owner=q.owner(),scope=s.ID))
                    a=dict(commit='synthetic-no-closure',protocol_sha=digest(c),code={},environment={},hardware=dict(cpu_affinity=[1,2]),synthetic=True,execution_permitted=False)
                    if scope_kind=='type1':a['type1_scope']=s.ID
                    elif scope_kind=='recovery':a['recovery_scope']='synthetic-recovery-scope'
                    elif scope_kind=='unified':a['unified_scope']='synthetic-unified-scope'
                    elif scope_kind!='legacy':raise AssertionError('explicit branch fixture')
                    exclusive(ctx['probe_root']/'approval.json',a);events=[]

                    def make_config(cc,purpose,out,*,task,approval,**kwargs):
                        self.assertIs(cc,c);self.assertEqual(purpose,'ch3_probe');out=Path(out);out.mkdir(parents=True)
                        t=next(t for t in c['tasks']if t['id']==task);p=profile(c,t);schedule=Schedule(FakeOptimizer(),p['training']['scheduler']);scheduler=[];points=[]
                        for step in range(1,7):
                            schedule.after_successful_update();scheduler.append(schedule.state_dict())
                            sf=out/('full-step'+str(step)+'.json');df=out/('full-step'+str(step)+'.bin');sf.write_text('{"synthetic":true}');df.write_bytes(b'synthetic-no-model-payload')
                            points.append(dict(step=step,schema_file=str(sf),data_file=str(df)))
                        tr=dict(id=task,profile_sha=digest(p),initial='same-init',initial_rng='same-rng',batch_ids=list(range(6)),validation_tail=1,steps=6,final_rng='same-rng',finite=True,
                            trajectory=[dict(step=i,loss=1.,state='same-state',optimizer='same-adam')for i in range(1,7)],validation=dict(mse=1.,mae=1.,sse=1.,sae=1.,elements=1),final='same-state',scheduler_trace=scheduler,M_full_state_trace=points)
                        rule=s.numeric_policy(c,t)
                        if rule:tr.update(numeric_policy_sha=digest(rule),full_numeric_trace=copy.deepcopy(points))
                        pid=2000000+c['tasks'].index(t);tr.update(time_mark=p.get('time_mark'),threads=4,affinity=a['hardware']['cpu_affinity'],memory=[dict(allocated=1,rss_before_hash=1,rss_after_hash=1)for _ in range(6)])
                        tr['memory_review']=ch3_runner.memory_growth_review(tr['memory'])
                        if failure and task==groups[0]['representatives'][0] and ((out.parent.name=='serial')==serial_failure):
                            if failure=='finite':tr['finite']=False
                            elif failure=='identity':
                                if serial_failure:tr['profile_sha']='wrong-profile'
                                else:tr['initial_rng']='wrong-rng'
                            elif failure=='numeric':tr['trajectory'][0]['loss']=2.
                        exclusive(out/'trajectory.json',tr);counts=s.worker_counts(c,t);exclusive(out/'budget.json',dict(counts=counts,by_pid={str(pid):counts},synthetic=True))
                        exclusive(out/'runtime.json',dict(pid=pid,task=task,error=None,synthetic=True));(out/'audit.jsonl').write_text('{"event":"synthetic-permitted"}\n')
                        cfg=dict(purpose=purpose,task=task,output=str(out),budget_file=str(out/'budget.json'),synthetic=True,approval=a,protocol_sha=digest(c),successor_scope=ctx['probe_scope'],successor_phase=out.parent.name,prefix_files={},limits=dict(**counts,seconds=1800))
                        exclusive(out/'config.json',cfg);events.append(dict(event='make_config',task=task,phase=out.parent.name));return cfg

                    def run_configs(configs,out,monitor):
                        out=Path(out);out.mkdir(parents=True);is_probe=configs[0]['purpose']=='ch3_probe'
                        row=dict(failure=None,returncodes=[0]*len(configs),resource_admission=True,elapsed=2. if out.parent.name=='serial'else 1.,synthetic=True)
                        pids=[str(2000000+c['tasks'].index(next(t for t in c['tasks']if t['id']==v['task'])))for v in configs]
                        row.update(exit_transitions_resolved=True,process_attribution='Measured',process_peaks={pid:1 for pid in pids},cpu_peaks={pid:1 for pid in pids})
                        if is_probe and ((failure=='resource' and out.parent.name=='serial') or (failure=='resource-q4' and out.parent.name=='q4')):row.update(failure='fixture resource failure',failure_kind='resource',resource_admission=False,returncodes=[1])
                        exclusive(out/'process.json',row);(out/'memory.jsonl').write_text(json.dumps(dict(synthetic=True,owned_pid_metadata={pid:dict(start_ticks='1')for pid in pids}))+'\n')
                        events.append(dict(event='dispatch',phase=out.parent.name,group=out.parent.parent.name,ids=[v['task']for v in configs]));return row

                    stack.enter_context(patch.object(s,'context',return_value=ctx));stack.enter_context(patch.object(s,'probe_groups',return_value=groups));stack.enter_context(patch.object(q,'CONTROL',controller))
                    if scope_kind=='recovery':
                        from utils import ch3_native_recovery as recovery
                        stack.enter_context(patch.object(recovery,'CONTROL',controller))
                    stack.enter_context(patch.dict(os.environ,{q.SECRET:'synthetic-audit-ownership-only'}))
                    stack.enter_context(patch.object(tool,'make_config',side_effect=make_config));stack.enter_context(patch.object(tool,'run_configs',side_effect=run_configs))
                    numeric=stack.enter_context(patch.object(ch3_runner,'_compare_full_numeric_files',return_value=dict(passed=True,state_max_abs=0.,compared_elements=1,exact=True,failures=[])))
                    stack.enter_context(patch('utils.ch3_urban_capture.compare_traces',return_value=dict(synthetic=True)))
                    endpoint=stack.enter_context(patch('utils.ch3_urban_confirmation.compare_measured',return_value=dict(passed=True,synthetic=True)))
                    immediate=stack.enter_context(patch.object(e,'compare',wraps=e.compare))
                    full=stack.enter_context(patch.object(e,'validate_probe_completion',wraps=e.validate_probe_completion))if full_audit else stack.enter_context(patch.object(e,'validate_probe_completion',side_effect=lambda cc,report:report))
                    yield SimpleNamespace(c=c,ctx=ctx,a=a,e=e,root=root,groups=groups,events=events,full=full,immediate=immediate,numeric=numeric,endpoint=endpoint,stack=stack)
            return fixture()
    def _success(self,stage):
            with self._fixture(stage)as f:
                report=f.e.run_probe(f.c,f.a)
                self.assertEqual(f.full.call_count,0);self.assertTrue((f.ctx['probe_root']/'complete.json').is_file())
                reps=sum(len(g['representatives'])for g in f.groups);self.assertEqual(f.immediate.call_count,2*reps)
                self.assertEqual(set(report['decisions']),{g['id']for g in f.groups});self.assertTrue(all(d['status']=='Passed'for d in report['decisions'].values()))
                self.assertEqual(report['budget']['actual'],s.probe_budget(f.c)['nominal']);self.assertFalse(report['budget']['refund'])
                before=f.full.call_count;summary_ref=q.audit_probe(f.c);summary=bound(summary_ref)
                self.assertEqual(f.full.call_count-before,1);self.assertTrue(summary['technical_admission']);self.assertEqual(summary['complete_ref'],ref(f.ctx['probe_root']/'complete.json'))
                self.assertEqual(set(summary['decisions']),{g['id']for g in f.groups})
                self.observations.append(dict(stage=stage,case='full-success',groups=len(f.groups),representatives=reps,run_probe_tail_full_audits=before,AUTO_AUDIT_full_audits=f.full.call_count-before,immediate_compare_calls=f.immediate.call_count,endpoint_compute_calls=f.endpoint.call_count,dispatches=[v for v in f.events if v['event']=='dispatch'],run_probe_mocked=False,audit_probe_mocked=False))
                return f.full.call_count
    def test_urban_one_full_audit(self):self.assertEqual(self._success('URBAN_SUBSET'),1)
    def test_epf_one_full_audit(self):self.assertEqual(self._success('EPF_ALL'),1)
    def test_m_one_full_audit(self):self.assertEqual(self._success('M_ALL'),1)
    def test_M_saved_evidence_actual_full_auditor(self):
        with self._fixture('M_ALL',one_group=True,full_audit=True)as f:
            f.e.run_probe(f.c,f.a);self.assertEqual(f.full.call_count,0)
            value=q.audit_probe(f.c);self.assertEqual(f.full.call_count,1);self.assertTrue(bound(value)['technical_admission'])
            self.observations.append(dict(stage='M_ALL',case='actual-saved-evidence-validator',validate_probe_completion_mocked=False,numeric_tensor_comparator_stubbed=True,run_probe_tail_full_audits=0,AUTO_AUDIT_full_audits=1))
    def test_three_ring_total_three_not_six(self):self.assertEqual(sum(self._success(stage)for stage in s.STAGES),3)
    def test_resource_only_q4_to_q2_keeps_batch(self):
        with self._fixture('URBAN_SUBSET',failure='resource-q4',one_group=True)as f:
            before=digest(f.c);report=f.e.run_probe(f.c,f.a)
            self.assertEqual(next(iter(report['decisions'].values()))['concurrency'],2);self.assertEqual(digest(f.c),before)
            self.assertEqual([e['phase']for e in f.events if e['event']=='dispatch'],['serial']*4+['q4']+['q2']*2)
            self.assertFalse(report['budget']['refund'])
    def test_q1_resource_fixed_batch_blocker(self):
        with self._fixture('M_ALL',failure='resource',one_group=True)as f:
            before=digest(f.c)
            with self.assertRaises(RuntimeError):f.e.run_probe(f.c,f.a)
            self.assertEqual(digest(f.c),before);self.assertEqual([x['phase']for x in f.events if x['event']=='dispatch'],['serial'])
    def test_legacy_recovery_unified_tail_behavior_unchanged(self):
            for branch,expected in (('legacy',1),('recovery',0),('unified',0)):
                with self.subTest(branch=branch),self._fixture('URBAN_SUBSET',scope_kind=branch,one_group=True)as f:
                    f.e.run_probe(f.c,f.a);self.assertEqual(f.full.call_count,expected);self.assertEqual(f.immediate.call_count,8)
                    self.observations.append(dict(case=branch+'-tail-branch',run_probe_tail_full_audits=expected,immediate_compare_calls=f.immediate.call_count,branch_fixture_only=True))
    def test_formal_light_paths_and_runtime_scan_rule_preserved(self):
            from contextlib import nullcontext
            import ch3_runner
            from tests.ch3_historical_type1_v2_231 import execution
            for stage in s.STAGES:
                with self.subTest(stage=stage),self._fixture(stage)as f:
                    f.e.run_probe(f.c,f.a);summary_ref=q.audit_probe(f.c);summary=bound(summary_ref);self.assertEqual(f.full.call_count,1)
                    upstream=exclusive(f.root/'controller/upstream-fixture.json',dict(synthetic=True,execution_permitted=False))
                    a=dict(f.a,purpose='baseline_type1_formal_permit_v1',type1_scope=s.ID,successor_scope=f.ctx['formal_scope'],execution_permitted=True,manual_review=False,reviewed=False,review_mode='preauthorized_machine_gate',start_authorization_ref=ref(f.ctx['probe_root']/'approval.json'),authorized_task_ids=[t['id']for t in f.c['tasks']],profile_shas={t['id']:digest(profile(f.c,t))for t in f.c['tasks']},data_binding_ref=f.c['baseline_unified']['data_ref'],caps=s.formal_budget(f.c)['total'],budget_refund=False,additional_search=0,from_scratch=True,upstream_boundary_ref=upstream,summary_ref=summary_ref,manifest_ref=summary['manifest_ref'])
                    a['predecessor_boundaries']={k:upstream for k in s.STAGES[:s.STAGES.index(stage)]}
                    f.stack.enter_context(patch.object(q,'validate_start',return_value=f.a));f.stack.enter_context(patch.object(q,'dynamic',return_value={k:f.a[k]for k in ('commit','protocol_sha','code','environment','hardware')}))
                    f.stack.enter_context(patch.object(q,'validate_upstream_boundary_light'));f.stack.enter_context(patch.object(q,'validate_boundary_light'))
                    f.stack.enter_context(patch('utils.ch3_m_execution.gpu_environment_reasons',return_value=[]));f.stack.enter_context(patch.dict(os.environ,{'TMPDIR':str(f.ctx['fixture'])}))
                    permit_ref=exclusive(f.ctx['control']/'formal-permit.json',a);self.assertEqual(q.validate_permit(f.c,a),a)
                    scans=f.stack.enter_context(patch.object(q,'scan_manifest',wraps=q.scan_manifest));runtime=q.seal_runtime(f.c,permit_ref);self.assertEqual(scans.call_count,1)
                    numeric_before=f.numeric.call_count;immediate_before=f.immediate.call_count
                    wave=s.formal_waves(f.c,summary,'AMD')[0];cfgs=[]
                    for run in wave:
                        out=f.ctx['result_root']/'formal-AMD'/run;cfg=execution.make_config(f.c,'ch3_formal',out,task=run,approval=a,runtime_ref=runtime);execution.validate_worker(f.c,cfg);cfgs.append(cfg)
                        self.assertIsNone(cfg['approval']);self.assertNotIn(str(f.ctx['probe_root']/'complete.json'),cfg['metadata_files']);self.assertFalse(any('/serial/'in p or '/q4/'in p or p.endswith('.bin')for p in cfg['metadata_files']))
                    execution.validate_wave(f.c,cfgs,f.ctx['control']/'fixture-wave')
                    f.stack.enter_context(patch.object(ch3_runner,'GPULock',return_value=nullcontext()))
                    f.stack.enter_context(patch.object(execution,'technical_group',return_value=dict(technical_complete=True,synthetic=True)))
                    execution.run_group(f.c,a,'DLinear',runtime)
                    self.assertEqual(f.full.call_count,1);self.assertEqual(scans.call_count,1);self.assertEqual(f.numeric.call_count,numeric_before);self.assertEqual(f.immediate.call_count,immediate_before)
                    self.observations.append(dict(stage=stage,case='actual-formal-light-paths',formal_permit_full_audits=0,formal_config_full_audits=0,formal_worker_full_audits=0,formal_group_wave_full_audits=0,manifest_scans_per_lifecycle=1,per_task_manifest_scans=0,formal_numeric_payload_comparisons=0,formal_raw_probe_payload_reads=0,structural_entrypoints_executed=['validate_permit','seal_runtime','make_config','validate_worker','validate_wave','run_group']))

    def test_immediate_failed_group_stops_later_groups(self):
            for stage in s.STAGES:
                for kind in ('numeric','finite','identity'):
                    with self.subTest(stage=stage,kind=kind),self._fixture(stage,failure=kind)as f:
                        with self.assertRaises((ValueError,RuntimeError)):f.e.run_probe(f.c,f.a)
                        dispatch=[v for v in f.events if v['event']=='dispatch'];self.assertEqual({v['group']for v in dispatch},{f.groups[0]['id']});self.assertEqual({v['phase']for v in dispatch},{'serial','q4'})
                        self.assertFalse((f.ctx['probe_root']/'complete.json').exists());failure=bound(ref(f.ctx['probe_root']/'failure.json'));self.assertFalse(failure['automatic_retry']);self.assertFalse(failure['budget']['refund']);self.assertEqual(f.full.call_count,0)
                        self.observations.append(dict(stage=stage,case=kind+'-failure',later_groups_dispatched=0,resource_fallbacks=0,full_audits=0,immediate_compare_calls=f.immediate.call_count,failure_retained=True))

    def test_serial_self_gate_stops_before_parallel(self):
            for kind in ('finite','identity','resource'):
                with self.subTest(kind=kind),self._fixture('URBAN_SUBSET',failure=kind,serial_failure=True)as f:
                    with self.assertRaises((ValueError,RuntimeError)):f.e.run_probe(f.c,f.a)
                    dispatch=[v for v in f.events if v['event']=='dispatch'];self.assertEqual(len(dispatch),1);self.assertEqual(dispatch[0]['phase'],'serial');self.assertEqual(f.full.call_count,0)
                    self.observations.append(dict(stage='URBAN_SUBSET',case='serial-'+kind+'-failure',dispatches=1,parallel_dispatched=0,later_groups_dispatched=0,full_audits=0))

class CPUPlumbingTests(unittest.TestCase):
    def test_actual_M_technical_closeout(self):
        from tests.ch3_historical_type1_v2_231 import execution,native_tasks
        technical_group=execution.technical_group;result_path=native_tasks.result_path
        c=q.configs()['M_ALL'];data=bound(c['baseline_unified']['data_ref'])
        with tempfile.TemporaryDirectory(dir=s.PACKAGE)as tmp,patch.object(s,'RESULT',Path(tmp)):
            approval=dict(commit='synthetic-M-closeout',data_bindings=data['data_bindings'])
            for t in (t for t in c['tasks']if t['model']=='AMD'):
                out=result_path(c,t);out.mkdir(parents=True);p=profile(c,t);a=step_arithmetic(c,t);history=[]
                for e in range(1,11):
                    n=a['validation_windows']*p['pred_len']*p['C'];history.append(dict(epoch=e,steps=e*a['train_batches'],validation=dict(mse=1.,mae=1.,sse=float(n),sae=float(n),elements=n),best_epoch=1))
                (out/'history.jsonl').write_text(''.join(json.dumps(row)+'\n'for row in history));(out/'best.pt').write_bytes(b'synthetic-best');(out/'last.pt').write_bytes(b'synthetic-last')
                exclusive(out/'manifest.json',dict(task=t,profile=p,identity=dict(commit=approval['commit'],profile_sha=digest(p),protocol_sha=digest(c),data_sha=data['data_bindings'][t['dataset']][t['id']])))
                elements=a['test_windows_arithmetic_only']*p['pred_len']*p['C']
                exclusive(out/'result.json',dict(id=t['id'],commit=approval['commit'],protocol_sha=digest(c),profile_sha=digest(p),scientific_protocol=s.PROTOCOL,scheduler_updates=history[-1]['steps'],scheduler_sha=digest(p['training']['scheduler']),metric_scope='all_channels',elements=elements,mse=1.,mae=1.,best_epoch=1,final_test=dict(calls=1,selected='best.pt',sha256=sha(out/'best.pt'),epoch=1)))
                counts=dict(adam=history[-1]['steps'],backward=history[-1]['steps'],forward=s.formal_budget(c)['tasks'][t['id']]['forward']);exclusive(out/'budget.json',dict(counts=counts,by_pid={'2000000':counts}));exclusive(out/'runtime.json',dict(pid=2000000,task=t['id'],error=None))
            receipt=technical_group(c,'AMD',approval);self.assertTrue(receipt['technical_complete']);self.assertEqual(len(receipt['task_ids']),24)
    def test_weighted_M_tail_actual_evaluator(self):
        import torch,ch3_runner
        from utils.ch3_m_tasks import evaluate_all
        p=copy.deepcopy(profile(q.configs()['M_ALL'],q.configs()['M_ALL']['tasks'][0]));p.update(C=2,features=['a','b'],target_idx=1)
        class EvaluationFixture:
            def eval(self):pass
        rows=[(torch.zeros(2,1,2),torch.zeros(2,1,2)),(torch.full((1,1,2),3.),torch.zeros(1,1,2))]
        with patch('models.ch3_adapter.target_prediction',side_effect=lambda model,x,p,mark:(x,None)):
            r=evaluate_all(EvaluationFixture(),rows,p,'cpu')
        self.assertEqual((r['sse'],r['elements'],r['mse'],r['mae']),(18.,6,3.,1.));self.assertEqual([r['mse']for r in r['channels']],[3.,3.])
    def test_new_scheduler_best_last_CPU_checkpoint(self):
        import torch,ch3_runner
        class StateFixture:
            def state_dict(self):return dict(synthetic=torch.zeros(1))
            def load_state_dict(self,state,strict=True):self.state=state
        class OptimizerFixture(FakeOptimizer):
            def state_dict(self):return copy.deepcopy(self.param_groups)
            def load_state_dict(self,state):self.param_groups=copy.deepcopy(state)
        o=OptimizerFixture();x=Schedule(o,configuration(20,3));g=torch.Generator().manual_seed(2024)
        # Represent two completed epochs with the exact same boundary events as the runner.
        for e in (1,2):
            for _ in range(3):x.after_successful_update()
            x.after_epoch(e,True)
        with tempfile.TemporaryDirectory(dir=s.PACKAGE)as tmp:
            for name in ('best.pt','last.pt'):
                path=Path(tmp)/name;ch3_runner.save_state(path,StateFixture(),o,dict(protocol=s.PROTOCOL),BestState(5),2,6,g,x)
                raw=torch.load(path,map_location='cpu');self.assertEqual(raw['schema'],Schedule.checkpoint_schema);self.assertEqual(raw['scheduler'],x.state_dict())
                other=OptimizerFixture();z=Schedule(other,x.config);best,epoch,steps=ch3_runner.restore_state(path,StateFixture(),other,dict(protocol=s.PROTOCOL),torch.Generator(),z)
                self.assertEqual((epoch,steps),(2,6));self.assertEqual(z.state_dict(),x.state_dict());self.assertEqual(other.param_groups[0]['lr'],lr_used(3,20))
    def test_M_prefix_dataset_all_channels_without_test(self):
        from utils.ch3_data import load
        c=q.configs()['M_ALL']
        for d in ('ETTh1','Weather','Exchange'):
            t=next(t for t in c['tasks']if t['dataset']==d and t['model']=='iTransformer'and t['h']==96);p=profile(c,t);datasets,meta=load(c,t)
            self.assertEqual(set(datasets),{'train','validation'});self.assertEqual(digest(meta),bound(c['baseline_unified']['data_ref'])['data_bindings'][d][t['id']]);x,y,*marks=datasets['train'][0]
            self.assertEqual(tuple(x.shape),(96,p['C']));self.assertEqual(tuple(y.shape),(96,p['C']));self.assertFalse(meta['test_observations_accessed'])
    def test_synthetic_tmux_owned_waiter_stop(self):
        import shlex
        from m6_remaining_entry import identity,same
        session='fixture-type1-v2-'+str(os.getpid())+'-'+str(time.time_ns())
        with tempfile.TemporaryDirectory(dir=s.PACKAGE)as tmp:
            root=Path(tmp);pidfile=root/'pid';log=root/'log'
            code="import os,time;from pathlib import Path;Path("+repr(str(pidfile))+").write_text(str(os.getpid()));time.sleep(30)"
            args=[q.PYTHON,'-B','-c',code,str(ROOT/'m6_type1_followup_entry.py'),'start'];command=shlex.join(args)+' >'+shlex.quote(str(log))+' 2>&1'
            subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),command],check=True)
            for _ in range(100):
                if pidfile.exists():break
                time.sleep(.02)
            self.assertTrue(pidfile.exists());owner={k:v for k,v in identity(int(pidfile.read_text())).items()if k in ('pid','start_ticks')}
            self.assertTrue(same(owner));exclusive(root/'controller.json',dict(scope=s.ID,owner=owner))
            with patch.object(q,'CONTROL',root):result=q.safe_stop()
            self.assertTrue(result['supervisor_signal_sent']);self.assertFalse(result['old_chain_signal_sent'])
            for _ in range(100):
                if not same(owner):break
                time.sleep(.02)
            self.assertFalse(same(owner))
            for _ in range(100):
                if subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0:break
                time.sleep(.02)
            self.assertNotEqual(subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode,0)
