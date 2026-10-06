"""112 amendments -> 371 sources -> 231 followups; synthetic leaves, no training."""
import ast,copy,json,os,subprocess,tempfile,unittest
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
from utils import ch3_type1_tasks as s,ch3_type1_chain as q,ch3_round2_amendment as a,ch3_type1_upstream as u
from utils.ch3_contract import ROOT,digest,profile,step_arithmetic
from utils.ch3_native_recovery_records import bound,ref,exclusive,sha

def changes(x,y,path=''):
    if isinstance(x,dict) and isinstance(y,dict):
        return sum((changes(x.get(k),y.get(k),path+'.'+k if path else k)for k in sorted(set(x)|set(y))),[])
    return [] if x==y else [path]

class MatrixAndProfiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cs=q.configs()
    def test_exact_112_28_35_168(self):self.assertEqual([len(c['tasks'])for c in self.cs.values()],[112,28,35,168])
    def test_exact_112_dataset_H_models(self):self.assertEqual({(t['model'],t['dataset'],t['h'])for t in self.cs['M_AMEND']['tasks']},{(m,d,h)for m in s.MODELS for d in a.DATASETS for h in (96,192,336,720)})
    def test_per_model_fixed_amendment_order(self):self.assertEqual([t['dataset']for t in self.cs['M_AMEND']['tasks']if t['model']=='TimeMixer'],[d for d in a.DATASETS for _ in range(4)])
    def test_effective_371_distinct(self):self.assertEqual((len(a.expected_cells()),len(set(a.expected_cells()))),(371,371))
    def test_no_excluded_models_or_scopes(self):self.assertFalse(any(t['model']in ('J','N','S')or t['dataset']=='ECL'or t['seed']!=2024 for c in self.cs.values()for t in c['tasks']))
    def test_Weather_OneCycle_only_two_fields(self):
        for t in self.cs['M_AMEND']['tasks']:
            if t['dataset']!='Weather':continue
            old=a.parent()['resolved_profiles'][a.source_task(t)['id']];new=profile(self.cs['M_AMEND'],t)
            self.assertEqual(changes(old,new),['training.epochs','training.scheduler.epochs']);self.assertIsNone(new['training']['patience']);self.assertEqual(new['training']['epochs'],20)
    def test_Weather_own_original_batch_retained(self):
        for t in self.cs['M_AMEND']['tasks']:
            if t['dataset']=='Weather':
                before=a.parent()['resolved_profiles'][a.source_task(t)['id']];p=profile(self.cs['M_AMEND'],t)
                self.assertEqual([p['training'][k]for k in ('batch','eval_batch')],[before['training'][k]for k in ('batch','eval_batch')])
    def test_amendment_ETT_own_ETTh1_structure(self):
        for t in self.cs['M_AMEND']['tasks']:
            if t['dataset']=='Weather':continue
            before=copy.deepcopy(a.parent()['resolved_profiles'][a.source_task(t)['id']]);new=profile(self.cs['M_AMEND'],t)
            for p in (before,new):
                p.pop('dataset');p.pop('time_mark',None);p['structure'].pop('freq',None)
                for k in ('batch','eval_batch','epochs','patience'):p['training'].pop(k)
                p['training']['scheduler'].pop('steps_per_epoch')
            self.assertEqual(before,new)
    def test_ETT_training_and_scheduler(self):
        from utils.ch3_onecycle import configuration
        for t in self.cs['M_AMEND']['tasks']:
            p=profile(self.cs['M_AMEND'],t);tr=p['training'];self.assertEqual(tr['scheduler'],configuration(tr['epochs'],step_arithmetic(self.cs['M_AMEND'],t)['train_batches']))
            if t['dataset']!='Weather':self.assertEqual([tr[k]for k in ('batch','eval_batch','epochs','patience','accumulation')],[128,128,10,None,1])
    def test_third_Weather_exact_three_differences(self):
        old=json.loads((ROOT/'configs/ch3_type1_m_all_v2.json').read_text());changed=0
        for prior,t in zip(old['tasks'],self.cs['M_ALL']['tasks']):
            paths=changes(old['resolved_profiles'][prior['id']],profile(self.cs['M_ALL'],t))
            self.assertEqual(paths,['training.epochs','training.scheduler.coefficient','training.scheduler.epochs']if t['dataset']=='Weather'else[]);changed+=bool(paths)
        self.assertEqual(changed,28)
    def test_third_other_203_science_unchanged(self):
        count=0
        for stage,c in list(self.cs.items())[1:]:
            old=json.loads((ROOT/'configs'/('ch3_type1_'+stage.lower()+'_v2.json')).read_text())
            for prior,t in zip(old['tasks'],c['tasks']):
                if t['dataset']=='Weather':continue
                self.assertEqual(profile(c,t),old['resolved_profiles'][prior['id']]);count+=1
        self.assertEqual(count,203)
    def test_third_numeric_registry_zero_difference(self):
        for stage,c in list(self.cs.items())[1:]:self.assertEqual(c['baseline_unified']['numeric_policies'],json.loads((ROOT/'configs'/('ch3_type1_'+stage.lower()+'_v2.json')).read_text())['baseline_unified']['numeric_policies'])
    def test_amendment_numeric_M_template_only(self):
        for t in self.cs['M_AMEND']['tasks']:
            rule=s.numeric_policy(self.cs['M_AMEND'],t);old=copy.deepcopy(a.parent()['baseline_unified']['numeric_policies'][t['model']+'-'+('Weather'if t['dataset']=='Weather'else'ETTh1')])
            if rule is not None:
                rule.pop('dataset',None);rule.pop('id',None);old.pop('dataset',None);old.pop('id',None)
            self.assertEqual(rule,old)
    def test_metadata_preserves_exact_parent_scaler(self):
        for c in self.cs.values():
            data=bound(c['baseline_unified']['data_ref']);self.assertFalse(data['test_observations_accessed'])
            for row in data['mapping']:
                t=next(t for t in c['tasks']if t['id']==row['task_id']);old=bound(row['source_ref'])['metadata'][t['dataset']][row['source_task_id']]
                self.assertEqual(data['metadata'][t['dataset']][t['id']],old);self.assertEqual(digest(old),row['metadata_sha'])
    def test_hour_minute_independent_scalers_and_M_output(self):
        c=self.cs['M_AMEND'];data=bound(c['baseline_unified']['data_ref'])
        for t in c['tasks']:
            p=profile(c,t);self.assertEqual((p['task'],p['metric_scope']),('M','all_channels'));self.assertEqual(p['supervised_channels'],list(range(p['C'])));self.assertEqual(p['output_order'],p['features'])
            if t['dataset']=='Weather':continue
            m=data['metadata'][t['dataset']][t['id']];self.assertEqual(m['scaler_fit_records'],34560 if t['dataset'].startswith('ETTm')else 8640)
            if p.get('time_mark'):self.assertEqual(p['time_mark']['K'],5 if t['dataset'].startswith('ETTm')else 4)
            self.assertNotEqual(c['datasets'][t['dataset']]['source_admission']['local_sha256_inherited'],a.parent()['datasets']['ETTh1']['source_admission']['local_sha256_inherited'])
    def test_precise_formal_and_probe_budgets(self):
        b=bound(ref(a.PACKAGE/'budget.json'));self.assertEqual(b['amendment'],dict(runs=112,run_epochs=1400,adam=608000,backward=608000,forward=734673));self.assertEqual(b['third_round'],dict(runs=231,run_epochs=2870,adam=1781920,backward=1781920,forward=2057615));self.assertEqual(b['new_execution_total'],dict(runs=343,run_epochs=4270,adam=2389920,backward=2389920,forward=2792288));self.assertEqual(b['probe_total_caps'],dict(adam=6162,backward=6162,forward=8240))
    def test_85_groups_343_representatives(self):
        groups=[s.probe_groups(c)for c in self.cs.values()];self.assertEqual([len(x)for x in groups],[28,7,8,42]);self.assertEqual(sum(len(g['representatives'])for gs in groups for g in gs),343)
    def test_old_start_cannot_authorize_new_scope(self):
        with patch.object(q,'closure',return_value=s.BASE):
            with self.assertRaises(PermissionError):q.validate_start(bound(ref(s.PACKAGE.parent/'baseline-type1-followup-v2/start-review.json')))
    def test_current_template_cannot_execute(self):
        x=q.start_template();self.assertTrue(all(x[k]is False for k in ('reviewed','execution_permitted','m6_authorized','budget_authorized','structure_frozen')));self.assertIsNone(x['closure_commit'])
    def test_new_outputs_and_permissions_absent(self):self.assertFalse(a.RESULT.exists());self.assertFalse(s.RESULT.exists());self.assertFalse((s.PACKAGE/'start-review.json').exists())
    def test_old287_boundary_unchanged(self):self.assertEqual(u.anchors()['expected_runs'],dict(MS=203,M=84,total=287))

class RevisionIndex(unittest.TestCase):
    def rows(self):
        start,_=u.records();old=[*bound(start['config_refs']['MS'])['tasks'],*a.parent()['tasks']];old.extend(t for t in a.tasks()if t['dataset']!='Weather')
        return dict(effective_counts=dict(MS=203,M=168,total=371),planned_formal_executions=399,cells=[dict(cell_id=a.key(t),task=t['task'],model=t['model'],dataset=t['dataset'],fold=t['fold'],H=t['h'],seed=t['seed'],origin='amend1'if t['dataset']in a.DATASETS else'original_v3',scientific_protocol=a.PROTOCOL if t['dataset']in a.DATASETS else u.OLD_PROTOCOL,supersedes={'path':'historical Weather10'}if t['dataset']=='Weather'else None,mse=1e100)for t in old])
    def test_all_Weather_selected_even_worse(self):v=a.validate_index(self.rows());self.assertEqual(len([r for r in v['cells']if r['dataset']=='Weather'and r['origin']=='amend1']),28)
    def test_no_partial_Weather_fallback(self):
        v=self.rows();next(r for r in v['cells']if r['dataset']=='Weather')['origin']='original_v3'
        with self.assertRaises(ValueError):a.validate_index(v)
    def test_missing_cell_rejected(self):
        v=self.rows();v['cells'].pop()
        with self.assertRaises(ValueError):a.validate_index(v)
    def test_duplicate_cell_rejected(self):
        v=self.rows();v['cells'][0]=v['cells'][1]
        with self.assertRaises(ValueError):a.validate_index(v)
    def test_287_cannot_claim_371(self):
        v=self.rows();v['cells']=v['cells'][:287]
        with self.assertRaises(ValueError):a.validate_index(v)
    def test_wrong_execution_protocol_rejected(self):
        v=self.rows();next(r for r in v['cells']if r['origin']=='amend1')['scientific_protocol']=u.OLD_PROTOCOL
        with self.assertRaises(ValueError):a.validate_index(v)
    def test_wrong_dataset_source_label_rejected(self):
        v=self.rows();next(r for r in v['cells']if r['dataset']=='Weather')['dataset']='ETTh1'
        with self.assertRaises(ValueError):a.validate_index(v)
    def test_pending_never_relabels_Weather10(self):self.assertEqual(a.summary()['revision'],'Pending');self.assertIn('no Weather10',a.summary()['Weather20'])
    def test_third_summary_default_revision_source_is_pending(self):
        from utils.ch3_type1_summary import result_index
        result=result_index(q.configs());self.assertEqual(len(result['rows']),231)
        self.assertTrue(all(r['second_round_revised']==dict(status='revision_pending')for r in result['rows']))
    def test_real_build_index_sealed_refs_399_to_371(self):
        start,_=u.records();cs={k:bound(start['config_refs'][k])for k in ('MS','M')};new=q.configs()['M_AMEND']
        with tempfile.TemporaryDirectory()as tmp,patch.object(u,'OLD_RESULT',Path(tmp)/'old'),patch.object(a,'RESULT',Path(tmp)/'amend'):
            def boundary(c,stage,root,commit,science):
                rows={}
                for t in c['tasks']:
                    out=root/stage/('formal-'+t['model'])/t['id'];p=profile(c,t)
                    m=exclusive(out/'manifest.json',dict(task=t,profile=p,identity=dict(commit=commit)))
                    r=exclusive(out/'result.json',dict(id=t['id'],commit=commit,profile_sha=digest(p),protocol_sha=digest(c),scientific_protocol=science,final_test=dict(calls=1),mse=1e100 if science==a.PROTOCOL else 1.,mae=1.))
                    runtime=exclusive(out/'runtime.json',dict(synthetic=True));rows[t['id']]={'manifest.json':m,'result.json':r,'runtime.json':runtime}
                receipt=exclusive(root/(stage+'-receipt.json'),dict(artifacts=rows));return exclusive(root/(stage+'-boundary.json'),dict(task_ids=list(rows),receipts={'synthetic':receipt}))
            old={k:boundary(c,k,u.OLD_RESULT,u.BASE,u.OLD_PROTOCOL)for k,c in cs.items()};amend=boundary(new,a.STAGE,a.RESULT,'actual-future-fixture-commit',a.PROTOCOL)
            value=a.build_index(old,amend,new);self.assertEqual(len(value['cells']),371);self.assertEqual(sum(r['origin']=='original_v3'for r in value['cells']),259)
            weather=[r for r in value['cells']if r['dataset']=='Weather'];self.assertTrue(all(r['mse']==1e100 and r['supersedes']for r in weather));self.assertTrue(all(r['execution_commit']=='actual-future-fixture-commit'for r in weather))

class DispatchAndAudit(unittest.TestCase):
    observations=[]
    def test_actual_dispatch_wait_amend_seal_then_three_rings(self):
        cs=q.configs();original=s.context
        with tempfile.TemporaryDirectory()as tmp,ExitStack()as stack:
            root=Path(tmp);stack.enter_context(patch.object(q,'CONTROL',root/'controller'));q.CONTROL.mkdir();order=[]
            stack.enter_context(patch.object(s,'context',side_effect=lambda c:dict(original(c),control=root/c['baseline_unified']['stage'],probe_root=root/'probe'/c['baseline_unified']['stage'])))
            approval=exclusive(root/'approval.json',dict(synthetic=True,execution_permitted=False))
            stack.enter_context(patch.object(q,'validate_start'));stack.enter_context(patch.object(q,'stop_check'));stack.enter_context(patch.object(q,'dynamic',return_value={}))
            stack.enter_context(patch.object(q,'wait_upstream',side_effect=lambda:order.append('released-287')or{}))
            def permit(c,start,probe,summary_ref=None,boundary_ref=None,round2_ref=None):
                stage=c['baseline_unified']['stage'];self.assertEqual(set(boundary_ref),set(s.STAGES[:s.STAGES.index(stage)]));self.assertEqual(round2_ref is None,stage=='M_AMEND')
                return exclusive(s.context(c)['control']/('probe-permit.json'if probe else'formal-permit.json'),dict(synthetic=True,execution_permitted=False))
            def wait(stage,value,probe,model=None,runtime_ref=None):
                order.append((stage,probe,model))
                if not probe:exclusive(s.context(cs[stage])['control']/('group-'+model)/'complete.json',dict(model=model,technical_complete=True,result_review='pending',task_ids=[t['id']for t in cs[stage]['tasks']if t['model']==model],mse=1e100))
            stack.enter_context(patch.object(q,'create_permit',side_effect=permit));stack.enter_context(patch.object(q,'wait_owned',side_effect=wait))
            audit=stack.enter_context(patch.object(q,'audit_probe',side_effect=lambda c:exclusive(s.context(c)['control']/'admission-summary.json',dict(synthetic=True))))
            stack.enter_context(patch.object(q,'seal_runtime',side_effect=lambda c,pr:exclusive(s.context(c)['control']/'runtime-admission.json',dict(synthetic=True))))
            stack.enter_context(patch.object(q,'seal_round2_boundary',side_effect=lambda value:order.append('sealed-371')or exclusive(root/'round2.json',dict(synthetic=True))))
            q.run(approval);self.assertEqual(audit.call_count,4);self.assertLess(order.index('sealed-371'),order.index(('URBAN_SUBSET',True,None)));self.assertEqual([m for x in order if isinstance(x,tuple)for stage,probe,m in [x]if not probe],list(s.MODELS)*4)
            complete=bound(ref(q.CONTROL/'complete.json'));self.assertEqual((complete['total_runs'],complete['third_round_runs'],complete['round2_effective_runs']),(343,231,371))
    def test_amend_failure_stops_third(self):
        seen=[];actions={state:lambda r,k=state:seen.append(k)for state in q.STATES if state!='COMPLETE'}
        actions['AMEND_AUTO_AUDIT']=lambda r:(_ for _ in ()).throw(ValueError('numeric fixture'))
        with self.assertRaises(ValueError):q.drive(actions,lambda _:None,lambda:None)
        self.assertNotIn('URBAN_RESOURCE_NUMERIC_PROBE',seen)
    def test_old287_cannot_skip112_to_enable_third(self):
        c=q.configs()['URBAN_SUBSET'];fake=dict(path=str(q.CONTROL/'upstream-technical-boundary.json'),sha256='0'*64)
        with self.assertRaises((ValueError,OSError)):q.validate_round2_boundary_light(fake)
    def test_four_stages_real_probe_tail_zero_AUTO_AUDIT_one(self):
        import tests.test_m6_type1_followup_v2 as fixtures
        import tests.ch3_historical_type1_v2_231 as historical
        from utils import ch3_native_execution as native,ch3_onecycle as onecycle
        from utils.ch3_type1_scaled import Schedule as Scaled
        from tests.test_ch3_onecycle import optimizer
        def schedule(opt,cfg):
            if cfg['name']=='OneCycleLR':
                value=onecycle.Schedule(optimizer(),cfg);advance=value.after_successful_update
                def event():value.optimizer.step();advance()
                value.after_successful_update=event;return value
            return Scaled(opt,cfg)
        helper=fixtures.ThreeRingAuditTests()
        with patch.dict(fixtures.__dict__,s=s,q=q,Schedule=schedule),patch.object(historical,'native',native):
            for stage in s.STAGES:
                with helper._fixture(stage,full_audit=True)as f:
                    f.e.run_probe(f.c,f.a);self.assertEqual(f.full.call_count,0);immediate=f.immediate.call_count
                    self.assertEqual(immediate,2*len(f.c['tasks']))
                    q.audit_probe(f.c);self.assertEqual(f.full.call_count,1);self.assertEqual(f.immediate.call_count,5*len(f.c['tasks']))
                    self.observations.append(dict(stage=stage,tail=0,AUTO_AUDIT=1,immediate_gates=immediate,final_audit_compare_calls=f.immediate.call_count-immediate,run_probe_mocked=False,audit_probe_mocked=False,full_validator_mocked=False))
    def test_formal_compact_paths_do_not_replay_probe(self):
        from utils.ch3_type1_execution import metadata_files
        for c in q.configs().values():
            values=metadata_files(c,dict(start_authorization_ref=ref(s.PACKAGE/'start-approval.template.json')))
            self.assertFalse(any('/serial/'in x or '/q4/'in x or x.endswith('.bin')for x in values))
        import inspect
        self.assertNotIn('validate_probe_completion',inspect.getsource(q.validate_permit))
    def test_manifest_scan_runtime_once(self):
        import inspect
        self.assertEqual(inspect.getsource(q.seal_runtime).count('scan_manifest('),1);self.assertNotIn('scan_manifest(',inspect.getsource(q.validate_runtime))
    def test_formal_paths_real_light_validation_and_one_runtime_scan(self):
        import tests.test_m6_type1_followup_v2 as fixtures
        import tests.ch3_historical_type1_v2_231 as historical
        from utils import ch3_native_execution as native,ch3_onecycle as onecycle,ch3_type1_execution as execution
        from utils.ch3_type1_scaled import Schedule as Scaled
        from tests.test_ch3_onecycle import optimizer
        import ch3_runner
        from contextlib import nullcontext
        def schedule(opt,cfg):
            if cfg['name']=='OneCycleLR':
                value=onecycle.Schedule(optimizer(),cfg);advance=value.after_successful_update
                def event():value.optimizer.step();advance()
                value.after_successful_update=event;return value
            return Scaled(opt,cfg)
        helper=fixtures.ThreeRingAuditTests()
        with patch.dict(fixtures.__dict__,s=s,q=q,Schedule=schedule),patch.object(historical,'native',native):
            for stage in s.STAGES:
                with helper._fixture(stage,one_group=False)as f:
                    f.e.run_probe(f.c,f.a);summary_ref=q.audit_probe(f.c);summary=bound(summary_ref)
                    upstream=exclusive(f.root/'upstream.json',dict(synthetic=True,execution_permitted=False))
                    permit=dict(f.a,purpose='baseline_type1_formal_permit_v1',type1_scope=s.ID,successor_scope=f.ctx['formal_scope'],execution_permitted=True,manual_review=False,reviewed=False,review_mode='preauthorized_machine_gate',start_authorization_ref=ref(f.ctx['probe_root']/'approval.json'),authorized_task_ids=[t['id']for t in f.c['tasks']],profile_shas={t['id']:digest(profile(f.c,t))for t in f.c['tasks']},data_binding_ref=f.c['baseline_unified']['data_ref'],caps=s.formal_budget(f.c)['total'],budget_refund=False,additional_search=0,from_scratch=True,upstream_boundary_ref=upstream,summary_ref=summary_ref,manifest_ref=summary['manifest_ref'],round2_boundary_ref=None if stage=='M_AMEND'else upstream,predecessor_boundaries={k:upstream for k in s.STAGES[:s.STAGES.index(stage)]})
                    f.stack.enter_context(patch.object(q,'validate_start',return_value=f.a));f.stack.enter_context(patch.object(q,'dynamic',return_value={k:f.a[k]for k in ('commit','protocol_sha','code','environment','hardware')}))
                    for name in ('validate_upstream_boundary_light','validate_boundary_light','validate_round2_boundary_light'):f.stack.enter_context(patch.object(q,name))
                    f.stack.enter_context(patch('utils.ch3_m_execution.gpu_environment_reasons',return_value=[]));f.stack.enter_context(patch.dict(os.environ,TMPDIR=str(f.ctx['fixture'])))
                    permit_ref=exclusive(f.ctx['control']/'formal-permit.json',permit);q.validate_permit(f.c,permit)
                    scans=f.stack.enter_context(patch.object(q,'scan_manifest',wraps=q.scan_manifest));runtime=q.seal_runtime(f.c,permit_ref)
                    numeric=f.numeric.call_count;immediate=f.immediate.call_count;configs=[]
                    for run in s.formal_waves(f.c,summary,'AMD')[0]:
                        cfg=execution.make_config(f.c,'ch3_formal',f.ctx['result_root']/'formal-AMD'/run,task=run,approval=permit,runtime_ref=runtime);execution.validate_worker(f.c,cfg);configs.append(cfg)
                        self.assertIsNone(cfg['approval']);self.assertFalse(any('/serial/'in p or '/q4/'in p or p.endswith('.bin')for p in cfg['metadata_files']))
                    execution.validate_wave(f.c,configs,f.ctx['control']/'synthetic-wave')
                    f.stack.enter_context(patch.object(ch3_runner,'GPULock',return_value=nullcontext()));f.stack.enter_context(patch.object(execution,'technical_group',return_value=dict(synthetic=True,technical_complete=True)))
                    execution.run_group(f.c,permit,'DLinear',runtime)
                    self.assertEqual((f.full.call_count,scans.call_count,f.numeric.call_count,f.immediate.call_count),(1,1,numeric,immediate))
                    self.observations.append(dict(stage=stage,formal_full_audits=0,formal_raw_probe_reads=0,formal_numeric_comparisons=0,formal_manifest_scans=1,per_task_scans=0,entrypoints=['validate_permit','seal_runtime','make_config','validate_worker','validate_wave','run_group']))
    def test_OneCycle20_rejects_old10_checkpoint_and_exact_trace(self):
        from tests.test_ch3_onecycle import optimizer
        from utils import ch3_onecycle as o
        import torch
        opt=optimizer();other=optimizer();current=o.Schedule(opt,o.configuration(20,3));official=torch.optim.lr_scheduler.OneCycleLR(other,steps_per_epoch=3,pct_start=.2,epochs=20,max_lr=.01)
        for _ in range(60):
            self.assertEqual((opt.param_groups[0]['lr'],opt.param_groups[0]['betas']),(other.param_groups[0]['lr'],other.param_groups[0]['betas']));opt.step();current.after_successful_update();other.step();official.step()
        self.assertEqual(current.updates,60)
        old=o.Schedule(optimizer(),o.configuration(10,3));new=o.Schedule(optimizer(),o.configuration(20,3))
        with self.assertRaises(ValueError):new.load_state_dict(old.state_dict(),0)
    def test_new_actual_tmux_waiter_owned_stop(self):
        import tests.test_m6_type1_followup_v2 as historical
        with patch.dict(historical.__dict__,s=s,q=q):historical.CPUPlumbingTests().test_synthetic_tmux_owned_waiter_stop()
    def test_upstream_running_and_dead_without_complete(self):
        with patch.object(u,'health'),patch.object(Path,'exists',return_value=False),patch.object(u,'same',return_value=True):self.assertFalse(u.status(full=True)['READY_FOR_GPU_EXECUTION'])
        with patch.object(u,'health'),patch.object(Path,'exists',return_value=False),patch.object(u,'same',return_value=False):
            with self.assertRaises(RuntimeError):u.status(full=True)
    def test_foreign_owner_safe_stop_refused(self):
        with tempfile.TemporaryDirectory()as tmp,patch.object(q,'CONTROL',Path(tmp)),patch.object(q,'signal_owned')as send:
            exclusive(Path(tmp)/'controller.json',dict(scope=u.OLD_SCOPE,owner=u.OWNER))
            with self.assertRaises(PermissionError):q.safe_stop()
            send.assert_not_called();self.assertFalse((Path(tmp)/'STOP').exists())
    def test_unknown_test_resume_refused(self):
        from ch3_runner import audit_resume
        with tempfile.TemporaryDirectory()as tmp:
            with self.assertRaises(PermissionError):audit_resume(tmp,dict(resume_audits={'x':dict(mode='resume',test_access_status='unknown')}),'x',protocol={'type1_followup':s.ID,'tasks':[{'id':'x'}]})
    def test_training_math_unchanged_function_AST(self):
        old=ast.parse(subprocess.check_output(['git','show',s.BASE+':ch3_runner.py'],cwd=ROOT,text=True));new=ast.parse((ROOT/'ch3_runner.py').read_text())
        for name in ('formal_worker','evaluate','save_state','restore_state','audit_resume'):
            get=lambda tree:ast.dump(next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name==name),include_attributes=False)
            self.assertEqual(get(old),get(new))
    def test_scheduler_and_data_math_files_unchanged(self):
        for file in ('utils/ch3_type1_scaled.py','utils/ch3_onecycle.py','utils/ch3_data.py','utils/ch3_time_marks.py','models/ch3_adapter.py'):
            raw=subprocess.check_output(['git','show',s.BASE+':'+file],cwd=ROOT);self.assertEqual(raw,(ROOT/file).read_bytes())

if __name__=='__main__':unittest.main()
