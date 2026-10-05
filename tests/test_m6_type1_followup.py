"""No-model contracts and lifecycle fixtures for isolated type1 handoff."""
import ast,copy,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from utils.ch3_contract import ROOT,digest,BestState
from tests.ch3_historical_type1_v1 import s,q,u,profile
from utils.ch3_type1 import Schedule,configuration,lr_used,validate_probe_trace
from utils.ch3_native_recovery_records import exclusive,ref,bound,manifest_projection,scan_manifest,sha

class FakeOptimizer:
    def __init__(self):self.param_groups=[dict(lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=1e-7)]

class Type1Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cs=q.configs();cls.old=s.parent()
    def test_matrix(self):
        self.assertEqual([len(self.cs[k]['tasks'])for k in s.STAGES],[28,35]);self.assertEqual(sum(len(c['tasks'])for c in self.cs.values()),63)
    def test_models_order(self):self.assertEqual(s.MODELS,('AMD','DLinear','PatchTST','iTransformer','TimeMixer','ModernTCN','TimeXer'))
    def test_urban_subset(self):
        self.assertEqual({(t['model'],t['fold'],t['h'])for t in self.cs['URBAN_SUBSET']['tasks']},{(m,f,h)for m in s.MODELS for f in (1,6)for h in (3,12)})
    def test_epf_matrix(self):self.assertEqual({(t['model'],t['dataset'],t['h'])for t in self.cs['EPF_ALL']['tasks']},{(m,d,24)for m in s.MODELS for d in ('PJM','NP','BE','FR','DE')})
    def test_excluded(self):
        for c in self.cs.values():self.assertFalse(any(t['model']in ('J','N','S')or t['dataset']in ('ECL','ETTh1','Weather','Exchange')or t['task']!='MS'for t in c['tasks']))
    def test_source_profiles(self):
        for stage,c in self.cs.items():
            for old,t in zip(s.selected(stage),c['tasks']):
                p=profile(c,t);prior=copy.deepcopy(self.old['resolved_profiles'][old['id']]);prior['T']=12 if t['dataset']=='UrbanEV'else 168;prior['training']['lr']=1e-4;prior['training']['scheduler']=p['training']['scheduler'];self.assertEqual(p,prior)
    def test_fixed_20_5(self):
        for c in self.cs.values():
            for t in c['tasks']:self.assertEqual((profile(c,t)['training']['epochs'],profile(c,t)['training']['patience']),(20,5))
    def test_urban_shape(self):
        for t in self.cs['URBAN_SUBSET']['tasks']:
            p=profile(self.cs['URBAN_SUBSET'],t);self.assertEqual((p['T'],p['pred_len'],p['C'],p['training']['batch'],p['training']['eval_batch']),(12,1,11,128,128))
    def test_epf_batches_inherited(self):
        for old,t in zip(s.selected('EPF_ALL'),self.cs['EPF_ALL']['tasks']):
            p=profile(self.cs['EPF_ALL'],t);prior=self.old['resolved_profiles'][old['id']];self.assertEqual(p['T'],168);self.assertEqual(p['pred_len'],24);self.assertEqual(p['training']['batch'],prior['training']['batch']);self.assertEqual(p['structure'],prior['structure'])
    def test_policies_inherited(self):
        for c in self.cs.values():
            for t in c['tasks']:self.assertEqual(s.numeric_policy(c,t),self.old['baseline_unified']['numeric_policies'][t['model']+'-'+t['dataset']])
    def test_moderntcn(self):
        for c in self.cs.values():
            for t in c['tasks']:
                if t['model']=='ModernTCN':self.assertIsNone(s.numeric_policy(c,t))if t['dataset']=='UrbanEV'else self.assertEqual(s.numeric_policy(c,t)['state_atol'],5e-4)
    def test_budget_urban(self):self.assertEqual(s.formal_budget(self.cs['URBAN_SUBSET'])['total'],dict(runs=28,run_epochs=560,adam=2414440,backward=2414440,forward=2711114))
    def test_budget_epf(self):self.assertEqual(s.formal_budget(self.cs['EPF_ALL'])['total'],dict(runs=35,run_epochs=700,adam=672860,backward=672860,forward=778932))
    def test_budget_total(self):
        rows=[s.formal_budget(c)['total']for c in self.cs.values()];self.assertEqual({k:sum(r[k]for r in rows)for k in rows[0]},dict(runs=63,run_epochs=1260,adam=3087300,backward=3087300,forward=3490046))
    def test_probe_budget_urban(self):
        b=s.probe_budget(self.cs['URBAN_SUBSET']);self.assertEqual(b['nominal'],dict(adam=336,backward=336,forward=464));self.assertEqual(b['caps'],dict(adam=504,backward=504,forward=696))
    def test_probe_budget_epf(self):
        b=s.probe_budget(self.cs['EPF_ALL']);self.assertEqual(b['nominal'],dict(adam=420,backward=420,forward=560));self.assertEqual(b['caps'],dict(adam=618,backward=618,forward=824))
    def test_probe_15_63(self):
        groups=[g for c in self.cs.values()for g in s.probe_groups(c)];self.assertEqual(len(groups),15);self.assertEqual(sum(len(g['representatives'])for g in groups),63)
    def test_urban_real_crossfold_wave(self):
        for g in s.probe_groups(self.cs['URBAN_SUBSET']):self.assertEqual(len(g['representatives']),4);self.assertEqual(g['planned_q'],4);self.assertTrue(any('-f6-'in r for r in g['representatives']))
    def test_epf_real_waves(self):
        c=self.cs['EPF_ALL'];plan=s.plan(c)
        for model in s.MODELS[:-1]:self.assertEqual([len(w)for w in plan['formal_waves'][model]],[4,1])
        self.assertEqual([len(w)for w in plan['formal_waves']['TimeXer']],[3,2])
    def test_no_profile_mutation(self):
        c=copy.deepcopy(self.cs['URBAN_SUBSET']);c['resolved_profiles'][c['tasks'][0]['id']]['training']['batch']=64
        with self.assertRaises(ValueError):s.validate(c)
    def test_exact_source_recipe(self):
        r=bound(s.AUTHOR_RECIPE);self.assertEqual(r['commit'],'76011909357972bd55a27adba2e1be994d81b327');self.assertTrue(all(sha(f)==h for f,h in r['files'].items()))
    def test_no_old_result_alias(self):
        for c in self.cs.values():
            for t in c['tasks']:self.assertNotIn(t['id'],{x['id']for x in self.old['tasks']});self.assertNotEqual(s.context(c)['result_root'],u.OLD_RESULT)
    def test_train_only_metadata(self):
        for c in self.cs.values():
            data=bound(c['baseline_unified']['data_ref']);self.assertFalse(data['test_observations_accessed'])
            for rows in data['metadata'].values():
                for m in rows.values():self.assertFalse(m['test_observations_accessed']);self.assertEqual(m['endpoints_read'][1],m['parsed_records']if 'parsed_records'in m else m['endpoints_read'][1])
    def test_metadata_T_dependent_mark_shape(self):
        from utils.ch3_time_marks import metadata
        for c in self.cs.values():
            data=bound(c['baseline_unified']['data_ref'])
            for t in c['tasks']:
                m=data['metadata'][t['dataset']][t['id']]
                for k,v in metadata(profile(c,t)).items():self.assertEqual(m[k],v)
    def test_patience_strict_tie(self):
        best=BestState(5);self.assertTrue(best.update(1.,1))
        for e in range(2,7):self.assertFalse(best.update(1.,e));self.assertEqual(best.epoch,1)
        self.assertTrue(best.stopped)
    def test_template_nonexecutable(self):
        a=q.start_template();self.assertTrue(all(a[k]is False for k in ('reviewed','execution_permitted','structure_frozen','m6_authorized','budget_authorized')));self.assertIsNone(a['closure_commit'])
    def test_old_start_cannot_authorize(self):
        with patch.object(q,'closure',return_value='candidate'):
            with self.assertRaises(PermissionError):q.validate_start(bound(u.START_REF))
    def test_branch_runtime_root(self):self.assertEqual(ROOT.name,'AMD-type1-followup-v1');self.assertEqual(q.ROOT,ROOT)

class Type1ScheduleTests(unittest.TestCase):
    def test_all_20_epoch_lrs(self):self.assertEqual([lr_used(e)for e in range(1,21)],[1e-4,1e-4]+[1e-4*.5**n for n in range(1,19)])
    def test_no_batch_decay(self):
        o=FakeOptimizer();s=Schedule(o,configuration(20,6))
        for _ in range(6):s.after_successful_update();self.assertEqual(o.param_groups[0]['lr'],1e-4)
    def test_epoch2_not_halved(self):
        o=FakeOptimizer();s=Schedule(o,configuration(20,1));s.after_successful_update();s.after_epoch(1,True);self.assertEqual(o.param_groups[0]['lr'],1e-4);s.after_successful_update();s.after_epoch(2,True);self.assertEqual(o.param_groups[0]['lr'],5e-5)
    def test_fixed_betas_20_epochs(self):
        o=FakeOptimizer();s=Schedule(o,configuration(20,1))
        for e in range(1,21):s.after_successful_update();s.after_epoch(e,e<20);self.assertEqual(o.param_groups[0]['betas'],(.9,.999))
    def test_resume_each_boundary(self):
        cfg=configuration(20,3)
        for stop in range(1,20):
            o=FakeOptimizer();schedule=Schedule(o,cfg)
            for e in range(1,stop+1):
                for _ in range(3):schedule.after_successful_update()
                schedule.after_epoch(e,True)
            state=schedule.state_dict();restored_opt=FakeOptimizer();restored=Schedule(restored_opt,cfg);restored_opt.param_groups=copy.deepcopy(o.param_groups);restored.load_state_dict(state,3*stop)
            for e in range(stop+1,21):
                self.assertEqual(schedule.state_dict(),restored.state_dict())
                for _ in range(3):schedule.after_successful_update();restored.after_successful_update()
                schedule.after_epoch(e,e<20);restored.after_epoch(e,e<20)
            self.assertEqual(schedule.state_dict(),restored.state_dict())
    def test_duplicate_decay_refused(self):
        x=Schedule(FakeOptimizer(),configuration(20,1));x.after_successful_update();x.after_epoch(1,True)
        with self.assertRaises(ValueError):x.after_epoch(1,True)
    def test_partial_epoch_not_committed(self):
        x=Schedule(FakeOptimizer(),configuration(20,2));x.after_successful_update()
        with self.assertRaises(ValueError):x.after_epoch(1,True)
    def test_earlystop_no_decay(self):
        x=Schedule(FakeOptimizer(),configuration(20,1))
        for e in range(1,7):x.after_successful_update();x.after_epoch(e,e<6)
        self.assertFalse(x.state_dict()['continuing']);self.assertEqual(x.state_dict()['current_lr'],lr_used(6))
    def test_invalid_betas_rejected(self):
        o=FakeOptimizer();x=Schedule(o,configuration(20,1));o.param_groups[0]['betas']=(.8,.999)
        with self.assertRaises(ValueError):x.validate()
    def test_wrong_scheduler_rejected(self):
        x=Schedule(FakeOptimizer(),configuration(20,1));value=x.state_dict();value['schema']='ch3-onecycle-state-v1'
        with self.assertRaises(ValueError):x.load_state_dict(value,0)
    def test_wrong_checkpoint_profile(self):
        x=Schedule(FakeOptimizer(),configuration(20,1));v=x.state_dict();v['config']['epochs']=10
        with self.assertRaises(ValueError):x.load_state_dict(v,0)
    def test_lr_tamper_rejected(self):
        o=FakeOptimizer();x=Schedule(o,configuration(20,1));o.param_groups[0]['lr']=.01
        with self.assertRaises(ValueError):x.validate()
    def test_short_trace_exact(self):
        cfg=configuration(20,100);x=Schedule(FakeOptimizer(),cfg);traces=[]
        for _ in range(6):x.after_successful_update();traces.append(x.state_dict())
        validate_probe_trace(cfg,traces,[.9,.999]);self.assertTrue(all(t['completed_epoch']==0 for t in traces))
        traces[4]['groups'][0]['lr']=.0002
        with self.assertRaises(ValueError):validate_probe_trace(cfg,traces,[.9,.999])
    def test_full_update_cap(self):
        x=Schedule(FakeOptimizer(),configuration(1,1));x.after_successful_update()
        with self.assertRaises(ValueError):x.after_successful_update()
    def test_validation_test_no_steps(self):
        import ch3_runner
        a=ast.parse(Path(ch3_runner.__file__).read_text());evaluate=next(n for n in a.body if isinstance(n,ast.FunctionDef)and n.name=='evaluate');self.assertFalse(any(isinstance(n,ast.Attribute)and n.attr in ('after_epoch','after_successful_update')for n in ast.walk(evaluate)))
    def test_checkpoint_distinct_schema(self):
        import ch3_runner
        text=Path(ch3_runner.__file__).read_text();self.assertIn("getattr(scheduler,'checkpoint_schema','ch3-state-v2-onecycle')",text);self.assertEqual(Schedule.checkpoint_schema,'ch3-state-v3-type1')
    def test_final_test_once(self):
        import ch3_runner
        tree=ast.parse(Path(ch3_runner.__file__).read_text());f=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='formal_worker');calls=[n for n in ast.walk(f)if isinstance(n,ast.Call)and any(k.arg=='test_capability'for k in n.keywords)];self.assertEqual(len(calls),1)
    def test_test_unknown_resume_blocker(self):
        from ch3_runner import audit_resume
        with tempfile.TemporaryDirectory()as tmp:
            with self.assertRaises(PermissionError):audit_resume(tmp,dict(resume_audits={'x':dict(mode='resume',test_access_status='unknown')}),'x',protocol={'type1_followup':s.ID,'tasks':[{'id':'x'}]})

class HandoffTests(unittest.TestCase):
    def test_anchor_exact(self):self.assertEqual(u.anchors()['owner'],dict(pid=36799,start_ticks='171530843'));self.assertEqual(u.CONTROLLER_REF['sha256'],'1b8e6098450326577a3f42091ac09237a2c84caac12d2aff7a9b6403645dda79')
    def test_running_old_waits_without_gpu(self):
        with tempfile.TemporaryDirectory()as tmp,patch.object(u,'OLD_RESULT',Path(tmp)),patch.object(u,'health',return_value=True),patch.object(u,'same',return_value=True):self.assertFalse(u.status(full=True)['READY_FOR_GPU_EXECUTION'])
    def test_dead_without_complete_refused(self):
        with tempfile.TemporaryDirectory()as tmp,patch.object(u,'OLD_RESULT',Path(tmp)),patch.object(u,'health',return_value=True),patch.object(u,'same',return_value=False):
            with self.assertRaises(RuntimeError):u.status(full=True)
    def test_live_owner_blocks_enable(self):
        with tempfile.TemporaryDirectory()as tmp,patch.object(u,'OLD_RESULT',Path(tmp)),patch.object(u,'health',return_value=True),patch.object(u,'same',return_value=True):
            p=Path(tmp)/'queue/controller/complete.json';p.parent.mkdir(parents=True);p.write_text('{}');self.assertFalse(u.status(full=True)['READY_FOR_GPU_EXECUTION'])
    def test_only_ms_complete_refused(self):
        with patch.object(u,'records',return_value=({},{})):
            with self.assertRaises(ValueError):u.validate_completion(dict(scope=u.OLD_SCOPE,technical_complete=True,result_review='pending',MS_boundary={},M_boundary={},total_runs=203))
    def test_forged_scope_refused(self):
        with patch.object(u,'records',return_value=({},{})):
            with self.assertRaises(ValueError):u.validate_completion(dict(scope='fake',technical_complete=True,result_review='pending',MS_boundary={},M_boundary={},total_runs=287))
    def test_bad_sha_refused(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'x.json';p.write_text('{}')
            with self.assertRaises(ValueError):u.expected_ref(dict(path=str(p),sha256='0'*64),p)
    def test_wrong_path_refused(self):
        with self.assertRaises(ValueError):u.expected_ref(dict(path='/tmp/a',sha256='0'*64),Path('/tmp/b'))
    def test_old_stop_refused(self):
        with tempfile.TemporaryDirectory()as tmp,patch.object(u,'OLD_RESULT',Path(tmp)),patch.object(u,'records',return_value=({},{})):
            p=Path(tmp)/'queue/controller/STOP';p.parent.mkdir(parents=True);p.touch()
            with self.assertRaises(RuntimeError):u.health()
    def test_old_failure_refused(self):
        with tempfile.TemporaryDirectory()as tmp,patch.object(u,'OLD_RESULT',Path(tmp)),patch.object(u,'records',return_value=({},{})):
            p=Path(tmp)/'probe/M/failure.json';p.parent.mkdir(parents=True);p.touch()
            with self.assertRaises(RuntimeError):u.health()
    def test_registered_owned_worker_alive_blocks(self):
        with tempfile.TemporaryDirectory()as tmp,patch.object(u,'OLD_RESULT',Path(tmp)),patch.object(u,'health',return_value=True),patch.object(u,'validate_completion',return_value={}),patch.object(u,'owned_refs',return_value=[dict(pid=99,start_ticks='1')]),patch.object(u,'same',side_effect=lambda x:x['pid']==99):
            p=Path(tmp)/'queue/controller/complete.json';p.parent.mkdir(parents=True);p.write_text('{}');self.assertFalse(u.status(full=True)['READY_FOR_GPU_EXECUTION'])
    def test_complete_and_release_enables(self):
        with tempfile.TemporaryDirectory()as tmp,patch.object(u,'OLD_RESULT',Path(tmp)),patch.object(u,'health',return_value=True),patch.object(u,'validate_completion',return_value={}),patch.object(u,'owned_refs',return_value=[]),patch.object(u,'same',return_value=False):
            p=Path(tmp)/'queue/controller/complete.json';p.parent.mkdir(parents=True);p.write_text('{}');self.assertTrue(u.status(full=True)['READY_FOR_GPU_EXECUTION'])
    def test_arm_readiness_is_separate(self):
        with patch.object(q,'readiness',return_value=[]),patch.object(q,'upstream_status',return_value=dict(state='WAIT',READY_FOR_GPU_EXECUTION=False)):
            v=q.readiness_report();self.assertTrue(v['READY_TO_ARM_HANDOFF']);self.assertFalse(v['READY_FOR_GPU_EXECUTION'])
    def test_wait_loop_no_lock_gpu_or_dispatch(self):
        tree=ast.parse(Path(q.__file__).read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='wait_upstream');text=ast.unparse(f);self.assertNotIn('GPULock',text);self.assertNotIn('resource_check',text);self.assertNotIn('wait_owned',text)
    def test_synthetic_chain_effect_independent(self):
        visited=[];actions={state:(lambda r,state=state:visited.append(state)or dict(technical_complete=True,mse=1e100))for state in q.STATES if state!='COMPLETE'};result=q.drive(actions,lambda x:None,lambda:None);self.assertEqual(visited,list(q.STATES[:-1]));self.assertIn('EPF_FORMAL_ALL_BASELINES',result)
    def test_technical_failure_stops_chain(self):
        visited=[];actions={x:(lambda r,x=x:visited.append(x))for x in q.STATES if x!='COMPLETE'}
        def fail(r):raise ValueError('numeric failure')
        actions['URBAN_AUTO_AUDIT']=fail
        with self.assertRaises(ValueError):q.drive(actions,lambda x:None,lambda:None)
        self.assertNotIn('EPF_RESOURCE_NUMERIC_PROBE',visited)
    def test_safe_stop_new_waiter_only(self):
        with tempfile.TemporaryDirectory()as tmp,patch.object(q,'CONTROL',Path(tmp)),patch.object(q,'same',return_value=False),patch.object(q,'signal_owned')as send:
            exclusive(Path(tmp)/'controller.json',dict(scope=s.ID,owner=dict(pid=999999,start_ticks='1')));r=q.safe_stop();self.assertFalse(r['old_chain_signal_sent']);self.assertTrue((Path(tmp)/'STOP').exists());send.assert_not_called()
    def test_fresh_output_conflict(self):
        with tempfile.TemporaryDirectory()as tmp,patch.object(s,'RESULT',Path(tmp)),patch.object(q,'closure',return_value='mock'),patch.object(q,'validate_start'),patch.object(q,'dynamic',return_value={}),patch.object(q,'upstream_status',return_value={'state':'WAIT'}):self.assertIn('retained unified execution/output/launcher; fresh repeat forbidden',q.readiness({}))
    def test_actual_run_sequence_synthetic_both_rings(self):
        with tempfile.TemporaryDirectory()as tmp,patch.object(s,'RESULT',Path(tmp)/'synthetic-result'),patch.object(q,'CONTROL',Path(tmp)/'controller'),patch.object(q,'stop_check'),patch.object(q,'dynamic',return_value={}):
            q.CONTROL.mkdir();approval=exclusive(Path(tmp)/'approval.json',dict(synthetic=True,execution_permitted=False));order=[]
            def permit(c,start,probe,summary_ref=None,boundary_ref=None):return exclusive(s.context(c)['control']/('probe-permit.json'if probe else'formal-permit.json'),dict(synthetic=True,execution_permitted=False))
            def wait(stage,value,probe,model=None,runtime_ref=None):
                order.append((stage,probe,model))
                if not probe:exclusive(s.context(q.configs()[stage])['control']/('group-'+model)/'complete.json',dict(synthetic=True,technical_complete=True,result_review='pending',model=model,task_ids=[t['id']for t in q.configs()[stage]['tasks']if t['model']==model],mse=1e100))
            with patch.object(q,'wait_upstream',return_value=dict(synthetic=True)),patch.object(q,'validate_start'),patch.object(q,'create_permit',side_effect=permit),patch.object(q,'wait_owned',side_effect=wait),patch.object(q,'audit_probe',side_effect=lambda c:exclusive(s.context(c)['control']/'admission-summary.json',dict(synthetic=True))),patch.object(q,'seal_runtime',side_effect=lambda c,a:exclusive(s.context(c)['control']/'runtime-admission.json',dict(synthetic=True))):q.run(approval)
            final=bound(ref(q.CONTROL/'complete.json'));self.assertEqual(final['total_runs'],63);self.assertTrue(final['technical_complete']);self.assertEqual(final['result_review'],'pending');self.assertEqual([m for stage,probe,m in order if not probe],list(s.MODELS)*2)
    def test_missing_and_duplicate_old_stage_indices_refused(self):
        start,_=u.records()
        with tempfile.TemporaryDirectory()as tmp,patch.object(u,'OLD_RESULT',Path(tmp)),patch.object(u,'records',return_value=(start,{})):
            top=dict(scope=u.OLD_SCOPE,technical_complete=True,result_review='pending',MS_boundary={},M_boundary={},total_runs=287)
            c=bound(start['config_refs']['MS']);ids=[t['id']for t in c['tasks']]
            for broken in (ids[:-1],ids[:-1]+[ids[0]]):
                p=Path(tmp)/'queue/MS/technical-boundary.json'
                if p.exists():p.unlink() # Only this explicitly created synthetic fixture, never a historical artifact.
                top['MS_boundary']=exclusive(p,dict(purpose='baseline_unified_MS_boundary_v1',scope=u.OLD_SCOPE,technical_complete=True,result_review='pending',task_ids=broken,protocol_sha=digest(c),commit=u.BASE))
                with self.assertRaises(ValueError):u.validate_completion(top)
    def test_no_user_permission_in_middle(self):self.assertFalse(any('REVIEW' in state for state in q.STATES));self.assertNotIn('TMARK_PROBE_RUNNING',q.STATES)

class AuditArchitectureTests(unittest.TestCase):
    def test_formal_paths_no_full_probe_audit(self):
        from tests.ch3_historical_type1_v1 import execution as e
        tree=ast.parse(Path(e.__file__).read_text())
        for f in (n for n in tree.body if isinstance(n,ast.FunctionDef)):
            self.assertNotIn('validate_probe_completion',ast.unparse(f));self.assertNotIn('_compare_full_numeric_files',ast.unparse(f))
    def test_one_scan_at_runtime_only(self):
        tree=ast.parse(Path(q.__file__).read_text());calls=[(f.name,n)for f in tree.body if isinstance(f,ast.FunctionDef)for n in ast.walk(f)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='scan_manifest'];self.assertEqual([f for f,n in calls],['seal_runtime'])
    def test_arbitrary_upstream_ready_json_cannot_authorize(self):
        with tempfile.TemporaryDirectory()as tmp,patch.object(q,'CONTROL',Path(tmp)):
            value=exclusive(Path(tmp)/'upstream-technical-boundary.json',dict(READY_FOR_GPU_EXECUTION=True,anchors=q.upstream_anchors(),handoff_scope=s.ID,successor_owner=u.OWNER))
            with self.assertRaises(PermissionError):q.validate_upstream_boundary_light(value)
    def test_each_group_computational_shapes_compatible(self):
        for c in q.configs().values():
            for g in s.probe_groups(c):
                shapes=set()
                for run in g['representatives']:
                    p=profile(c,next(t for t in c['tasks']if t['id']==run));train=copy.deepcopy(p['training']);train.pop('scheduler');shapes.add(digest(dict(T=p['T'],pred_len=p['pred_len'],C=p['C'],structure=p['structure'],training=train)))
                self.assertEqual(len(shapes),1)
    def test_runtime_crossprocess_mac(self):
        text=Path(q.__file__).read_text();self.assertIn('hmac.compare_digest',text);self.assertIn("v['owner']!=json.loads((CONTROL/'controller.json').read_text())['owner']",text)
    def test_raw_probe_not_worker_metadata(self):
        from tests.ch3_historical_type1_v1 import execution as e
        c=q.configs()['URBAN_SUBSET'];refs=e.metadata_files(c,dict(start_authorization_ref=ref(s.PACKAGE/'start-approval.template.json')))
        self.assertFalse(any('numeric-full' in p or '/serial/' in p or '/q4/' in p or p.endswith('probe/complete.json')for p in refs))
    def test_summary_light_no_raw_replay(self):
        tree=ast.parse(Path(q.__file__).read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='validate_summary_light');self.assertNotIn('validate_probe_completion',ast.unparse(f));self.assertNotIn('scan_manifest',ast.unparse(f))
    def test_boundary_light_no_rebuild(self):
        tree=ast.parse(Path(q.__file__).read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='validate_boundary_light');self.assertNotIn('technical_group',ast.unparse(f))
    def test_generic_fullstate_dedup_branch(self):
        from tests.ch3_historical_type1_v1 import native
        compare=native.compare
        import inspect
        text=inspect.getsource(compare);self.assertIn('if not generic_full and not _compare_full_numeric_files',text);self.assertIn('compare_measured',text)
    def test_endpoint_gate_preserved(self):
        from tests.ch3_historical_type1_v1 import native
        confirmation_endpoint=native.confirmation_endpoint
        c=q.configs()['URBAN_SUBSET'];t=next(t for t in c['tasks']if t['model']=='TimeMixer');self.assertTrue(confirmation_endpoint(c,dict(task=t['id'],purpose='ch3_probe',successor_scope=s.context(c)['probe_scope'])))
    def test_manifest_not_selfsigned(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);p=root/'payload.bin';p.write_bytes(b'original');report=dict(artifacts={str(p):ref(p)})
            complete=root/'complete.json';exclusive(complete,report);cr=ref(complete);p.write_bytes(b'changed')
            with self.assertRaises(ValueError):manifest_projection(report,cr,root)
    def test_manifest_tamper_scan_rejects(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);p=root/'payload.bin';p.write_bytes(b'original');report=dict(artifacts={str(p):ref(p)});cr=exclusive(root/'complete.json',report);manifest=manifest_projection(report,cr,root);p.write_bytes(b'changed')
            with self.assertRaises(ValueError):scan_manifest(manifest,cr,root)
    def test_numeric_failure_no_resource_fallback(self):
        from utils.ch3_m_execution import resource_fallback
        self.assertFalse(resource_fallback(dict(failure_kind='numeric',failure='numeric',returncodes=[1])))
    def test_resource_schedule_widths(self):
        from utils.ch3_native_tasks import attempt_widths
        self.assertEqual(list(attempt_widths(dict(planned_q=4))),[4,2]);self.assertEqual(list(attempt_widths(dict(planned_q=2))),[2])
    def test_bound_ref_tamper_rejected(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'a.json';value=exclusive(p,dict(synthetic=True));p.write_text('{}')
            with self.assertRaises(ValueError):bound(value)
    def test_compact_config_no_inline_formal(self):
        import inspect
        from tests.ch3_historical_type1_v1 import execution as e
        text=inspect.getsource(e.make_config);self.assertIn("s['approval']=None",text);self.assertIn('formal_permit_ref',text);self.assertIn('runtime_admission_ref',text)
    def test_generic_payload_compared_once_bounded_and_exact(self):
        from tests.ch3_historical_type1_v1 import native
        compare=native.compare
        import ch3_runner
        for model in ('iTransformer','DLinear'):
            c=q.configs()['URBAN_SUBSET'];t=next(t for t in c['tasks']if t['model']==model);p=profile(c,t);o=FakeOptimizer();schedule=Schedule(o,p['training']['scheduler']);traces=[]
            for _ in range(6):schedule.after_successful_update();traces.append(schedule.state_dict())
            points=[dict(step=i,schema_file=str(s.context(c)['probe_root']/(str(i)+'.json')),data_file=str(s.context(c)['probe_root']/(str(i)+'.bin')))for i in range(1,7)]
            x=dict(id=t['id'],profile_sha=digest(p),initial='i',initial_rng='r',batch_ids=list(range(6)),validation_tail=1,steps=6,final_rng='r',finite=True,trajectory=[dict(step=i,loss=1.,state='s',optimizer='o')for i in range(1,7)],validation=dict(mse=1.,mae=1.,sse=1.,sae=1.,elements=1),final='s',scheduler_trace=traces,M_full_state_trace=points)
            rule=s.numeric_policy(c,t)
            if rule:x.update(numeric_policy_sha=digest(rule),full_numeric_trace=points)
            with patch.object(ch3_runner,'_compare_full_numeric_files',return_value=dict(passed=True,state_max_abs=0.,compared_elements=10,exact=True,failures=[]))as gate:
                self.assertTrue(compare(c,t,x,copy.deepcopy(x))['passed']);self.assertEqual(gate.call_count,6)
    def test_compact_real_config_fixture(self):
        from tests.ch3_historical_type1_v1 import execution as e
        c=q.configs()['URBAN_SUBSET'];t=c['tasks'][0]
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);ctx=s.context(c);ctx=dict(ctx,result_root=root/'results',control=root/'control',fixture=root/'fixture');ctx['control'].mkdir();ctx['fixture'].mkdir()
            fake=exclusive(ctx['control']/'formal-permit.json',dict(synthetic=True,execution_permitted=False));runtime=exclusive(ctx['control']/'runtime-admission.json',dict(synthetic=True,execution_permitted=False));a=dict(start_authorization_ref=ref(s.PACKAGE/'start-approval.template.json'))
            out=ctx['result_root']/('formal-'+t['model'])/t['id']
            with patch.object(s,'context',return_value=ctx),patch.object(e,'authorization_reasons',return_value=[]),patch.object(q,'validate_runtime'),patch.object(q,'stop_check'):
                cfg=e.make_config(c,'ch3_formal',out,task=t['id'],approval=a,runtime_ref=runtime)
            self.assertIsNone(cfg['approval']);self.assertEqual(cfg['formal_permit_ref'],fake);self.assertEqual(cfg['runtime_admission_ref'],runtime);self.assertNotIn('probe_report',cfg);self.assertLess((out/'config.json').stat().st_size,100000)
            self.assertEqual(cfg['limits']['adam'],s.formal_budget(c)['tasks'][t['id']]['adam'])
    def test_old_implementation_globals_preserved(self):
        from utils import ch3_baseline_unified_tasks as old
        self.assertEqual(old.PROTOCOL,'baseline-unified96-onecycle001-v3');self.assertEqual(old.ID,'m6-baseline-unified96-oc01-v3')
    def test_summary_supplements_and_no_selection(self):
        from tests.ch3_historical_type1_v1 import summary as x
        text=Path(x.__file__).read_text();self.assertIn('supplements/epf4-timemixer-v1',text);self.assertIn('no_old_new_selection=True',text)


class ProbeAuditOwnershipTests(unittest.TestCase):
    """Exercise real scheduling/audit entrypoints; replace only compute/launch leaves."""
    observations=[]

    def _fixture(self,stage,scope_kind='type1',failure=None,serial_failure=False,one_group=False):
        from contextlib import contextmanager,ExitStack
        from types import SimpleNamespace
        import ch3_runner,m5_formal_entry as tool
        from tests.ch3_historical_type1_v1 import native as e

        @contextmanager
        def fixture():
            c=q.configs()[stage];groups=s.probe_groups(c)
            if one_group:groups=groups[:1]
            with tempfile.TemporaryDirectory(prefix='probe-audit-ownership-',dir=s.PACKAGE)as tmp,ExitStack()as stack:
                root=Path(tmp);ctx=dict(s.context(c),probe_root=root/'probe',control=root/'stage',result_root=root/'results',fixture=root/'fixture')
                for key in ('probe_root','control','fixture'):ctx[key].mkdir()
                controller=root/'controller';controller.mkdir();exclusive(controller/'controller.json',dict(owner=q.owner(),scope=s.ID))
                a=dict(commit='synthetic-no-closure',protocol_sha=digest(c),code={},environment={},hardware={},synthetic=True,execution_permitted=False)
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
                    if failure and task==groups[0]['representatives'][0] and ((out.parent.name=='serial')==serial_failure):
                        if failure=='finite':tr['finite']=False
                        elif failure=='identity':
                            if serial_failure:tr['profile_sha']='wrong-profile'
                            else:tr['initial_rng']='wrong-rng'
                        elif failure=='numeric':tr['trajectory'][0]['loss']=2.
                    exclusive(out/'trajectory.json',tr);counts=s.worker_counts(c,t);exclusive(out/'budget.json',dict(counts=counts,synthetic=True))
                    cfg=dict(purpose=purpose,task=task,output=str(out),budget_file=str(out/'budget.json'),synthetic=True)
                    exclusive(out/'config.json',cfg);events.append(dict(event='make_config',task=task,phase=out.parent.name));return cfg

                def run_configs(configs,out,monitor):
                    out=Path(out);out.mkdir(parents=True);is_probe=configs[0]['purpose']=='ch3_probe'
                    row=dict(failure=None,returncodes=[0]*len(configs),resource_admission=True,elapsed=2. if out.parent.name=='serial'else 1.,synthetic=True)
                    if is_probe and failure=='resource' and out.parent.name=='serial':row.update(failure='fixture resource failure',failure_kind='resource',resource_admission=False,returncodes=[1])
                    exclusive(out/'process.json',row);(out/'memory.jsonl').write_text('{"synthetic":true}\n')
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
                full=stack.enter_context(patch.object(e,'validate_probe_completion',side_effect=lambda cc,report:report))
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

    def test_urban_probe_tail_zero_auto_audit_one(self):self.assertEqual(self._success('URBAN_SUBSET'),1)
    def test_epf_probe_tail_zero_auto_audit_one(self):self.assertEqual(self._success('EPF_ALL'),1)
    def test_two_ring_success_total_two_full_audits(self):
        self.assertEqual(sum(self._success(stage)for stage in s.STAGES),2)

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

    def test_legacy_recovery_unified_tail_behavior_unchanged(self):
        for branch,expected in (('legacy',1),('recovery',0),('unified',0)):
            with self.subTest(branch=branch),self._fixture('URBAN_SUBSET',scope_kind=branch,one_group=True)as f:
                f.e.run_probe(f.c,f.a);self.assertEqual(f.full.call_count,expected);self.assertEqual(f.immediate.call_count,8)
                self.observations.append(dict(case=branch+'-tail-branch',run_probe_tail_full_audits=expected,immediate_compare_calls=f.immediate.call_count,branch_fixture_only=True))

    def test_formal_light_paths_and_runtime_scan_rule_preserved(self):
        from contextlib import nullcontext
        import ch3_runner
        from tests.ch3_historical_type1_v1 import execution
        for stage in s.STAGES:
            with self.subTest(stage=stage),self._fixture(stage)as f:
                f.e.run_probe(f.c,f.a);summary_ref=q.audit_probe(f.c);summary=bound(summary_ref);self.assertEqual(f.full.call_count,1)
                upstream=exclusive(f.root/'controller/upstream-fixture.json',dict(synthetic=True,execution_permitted=False))
                a=dict(f.a,purpose='baseline_type1_formal_permit_v1',type1_scope=s.ID,successor_scope=f.ctx['formal_scope'],execution_permitted=True,manual_review=False,reviewed=False,review_mode='preauthorized_machine_gate',start_authorization_ref=ref(f.ctx['probe_root']/'approval.json'),authorized_task_ids=[t['id']for t in f.c['tasks']],profile_shas={t['id']:digest(profile(f.c,t))for t in f.c['tasks']},data_binding_ref=f.c['baseline_unified']['data_ref'],caps=s.formal_budget(f.c)['total'],budget_refund=False,additional_search=0,from_scratch=True,upstream_boundary_ref=upstream,summary_ref=summary_ref,manifest_ref=summary['manifest_ref'])
                if stage=='EPF_ALL':a['ms_boundary_ref']=upstream
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

if __name__=='__main__':unittest.main()
