"""No-load recovery contract tests; all fixtures remain in the new package."""
import ast
import copy
import json
import os
import struct
import subprocess
import sys
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack
from utils import ch3_native_recovery as r
from utils import ch3_native_recovery_records as rec
from utils import ch3_native_recovery_execution as w
from utils import ch3_native_execution as e
from utils import ch3_native_tasks as scope
from utils.ch3_contract import digest,profile,BestState


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cs=r.original_configs()
    def setUp(self):
        base=Path(os.environ.get('CH3_RECOVERY_FIXTURE_ROOT',str(r.PACKAGE/'acceptance/fixtures')))
        self.root=base/(self._testMethodName+'-'+uuid.uuid4().hex);self.root.mkdir(parents=True)
    def store(self,name,value):return rec.exclusive(self.root/name,value)
    def manifest(self):
        root=self.root/'probe';root.mkdir();(root/'payload.bin').write_bytes(b'original expected bytes')
        expected=rec.ref(root/'payload.bin');report=dict(artifacts={expected['path']:expected})
        source=self.store('original-complete.json',report)
        return root,source,rec.manifest_projection(report,source,root)
    def fixture_inventory(self):
        value=copy.deepcopy(r.prepared('inventory'))
        for row in value['rows']:
            row['original_root']=str(self.root/('original-'+row['task_id']))
        return value
    def actions(self,fail=None,drain=None):
        calls=[];states=[];control=self.root/('control-'+uuid.uuid4().hex);control.mkdir()
        def action(state):
            def run(done):
                calls.append(state)
                if state==fail:raise ValueError('synthetic technical failure')
                if state==drain:r.write_marker(control/'DRAIN_STOP')
                return dict(technical_complete=True,result_review='pending',effect='poor')
            return run
        return control,calls,states,{s:action(s) for s in r.STATES[:-1]}

    def test_inventory_exact_87_and_test_status(self):
        inv=r.prepared('inventory');rec.validate_inventory(self.cs['tmark'],inv)
        self.assertEqual(inv['counts'],dict(A=8,B=4,C=0,D=75,E=0))
        self.assertEqual(sum(inv['counts'].values()),87)
        for row in inv['rows']:
            if row['classification']=='B':self.assertEqual(row['checkpoint_committed_epoch'],7);self.assertEqual(row['test_access_status'],'not_observed_under_guard')

    def test_actual_cost_exceeds_checkpoint_no_refund_or_double_add(self):
        b=rec.budget_projection(self.cs['tmark'],r.prepared('inventory'))
        self.assertEqual(b['required_repeated_work'],dict(adam=8407,backward=8408,forward=8410))
        for k in rec.KINDS:self.assertEqual(b['total_execution_worst_case'][k],b['historical_actual_consumed'][k]+b['recovery_future_worst_case'][k])
        self.assertEqual(b['total_execution_worst_case']['adam'],3695937);self.assertFalse(b['budget_refund'])
        self.assertEqual(b['checkpoint_committed_work'],dict(adam=250874,backward=250874,forward=278486))
        self.assertEqual(b['charged_work_after_checkpoint'],b['required_repeated_work'])
        self.assertEqual(b['authorized_remaining']['adam'],3687530-259281)

    def test_inconsistent_unknown_test_or_wrong_identity_rejected(self):
        inv=self.fixture_inventory();rec.validate_inventory(self.cs['tmark'],inv)
        b=next(i for i,x in enumerate(inv['rows']) if x['classification']=='B')
        for key,value in [('classification','E'),('test_access_status','unknown'),('profile_sha','wrong'),('original_commit','wrong')]:
            wrong=copy.deepcopy(inv);wrong['rows'][b][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):rec.validate_inventory(self.cs['tmark'],wrong)

    def test_C_zero_work_and_D_fresh_A_inherited_map(self):
        inv=self.fixture_inventory();row=next(x for x in inv['rows'] if x['classification']=='D');row.update(classification='C',test_access_status='no_model_work_no_test')
        rec.validate_inventory(self.cs['tmark'],inv);mapping=rec.effective_map(self.cs['tmark'],inv,self.root/'new')
        self.assertEqual(mapping['rows'][row['task_id']]['origin'],'recovery_fresh')
        for x in inv['rows']:
            if x['classification']=='A':self.assertEqual(mapping['rows'][x['task_id']]['effective_root'],x['original_root'])
        row['actual_charged_counts']['forward']=1
        with self.assertRaises(ValueError):rec.validate_inventory(self.cs['tmark'],inv)

    def test_old_wave_filter_never_repack(self):
        waves=[['A','B'],['C','D']];inv=dict(rows=[dict(task_id='A',classification='A'),dict(task_id='D',classification='A')])
        self.assertEqual(rec.pending_waves(waves,inv),[(0,['B']),(1,['C'])])

    def test_manifest_expected_sha_is_original_trust_root(self):
        root,source,manifest=self.manifest();self.assertEqual(manifest['source_probe_complete_ref'],source)
        (root/'payload.bin').write_bytes(b'tampered but valid-looking')
        with self.assertRaises(ValueError):rec.manifest_projection(rec.bound(source),source,root)
        with self.assertRaises(ValueError):rec.scan_manifest(manifest,source,root)

    def test_manifest_wrong_source_symlink_and_duplicate_ref_reject(self):
        root,source,manifest=self.manifest()
        with self.assertRaises(ValueError):rec.scan_manifest(manifest,dict(source,sha256='wrong'),root)
        bad=copy.deepcopy(manifest);bad['rows']*=2;bad['artifact_count']=2
        with self.assertRaises(ValueError):rec.scan_manifest(bad,source,root)
        (root/'link.bin').symlink_to(root/'payload.bin')
        report=dict(artifacts={str(root/'link.bin'):rec.ref(root/'link.bin')})
        with self.assertRaises(ValueError):rec.manifest_projection(report,source,root)

    def test_process_approval_refs_covered_without_overwriting_primary(self):
        primary=r.prepared('manifest');extra=r.prepared('process_manifest')
        self.assertEqual(primary['source_probe_complete_ref'],r.TRUST['complete'])
        self.assertEqual(extra['source_probe_complete_ref'],r.TRUST['complete'])
        self.assertEqual(extra['artifact_count'],33)
        combined=rec.merge_manifests([primary,extra]);self.assertEqual(combined['artifact_count'],893)
        with self.assertRaises(ValueError):rec.merge_manifests([primary,primary])

    def test_exclusive_manifest_and_lifecycle_no_overwrite(self):
        value={'actual':1};x=self.store('sealed.json',value);self.assertEqual(self.store('sealed.json',value),x)
        with self.assertRaises(FileExistsError):self.store('sealed.json',dict(actual=2))

    def test_runtime_one_scan_per_lifecycle_and_cross_process_ref(self):
        root,source,manifest=self.manifest();mref=self.store('manifest.json',manifest)
        permit=dict(protocol_sha=digest(self.cs['tmark']),controller_execution_commit='fixture',worker_execution_commit='fixture',
            science_baseline_commit=r.BASE,science_computation_fingerprint='fixed',environment={},hardware={},source_states={},code={})
        pref=self.store('permit.json',permit);summary=self.store('summary.json',dict(original_complete_ref=source))
        actual_scan=rec.scan_manifest
        with patch.object(r,'PACKAGE',self.root),patch.object(r,'validate_permit_light',side_effect=lambda c,a:a),patch.object(r,'validate_summary_light',side_effect=lambda c,s:s),patch.object(r,'stop_check'),patch.dict(os.environ,{r.RUNTIME_SECRET_ENV:'fixture-secret'}),patch.object(rec,'scan_manifest',side_effect=lambda m,s,p:actual_scan(m,s,root)) as scan:
            runtime=r.seal_runtime(self.cs['tmark'],pref,summary,mref,'life1',dict(pid=1,start_ticks='1'))
            with self.assertRaises(FileExistsError):r.seal_runtime(self.cs['tmark'],pref,summary,mref,'life1',{})
            self.assertEqual(scan.call_count,1)
            code='import json,sys; v=json.load(open(sys.argv[1])); assert v["scan_count"]==1 and v["formal_permit_ref"]["sha256"]==sys.argv[2]'
            subprocess.run([sys.executable,'-B','-c',code,runtime['path'],pref['sha256']],check=True)
            with patch.object(r,'dynamic_binding'),patch.object(r,'stop_check'):
                r.validate_runtime_light(self.cs['tmark'],runtime,pref,require_owner=False)
                bad=rec.bound(runtime);bad['integrity_scan_passed']=False
                badref=self.store('bad-runtime.json',bad)
                with self.assertRaises(ValueError):r.validate_runtime_light(self.cs['tmark'],badref,pref,False)

    def test_runtime_current_chain_authentication_not_arbitrary_JSON(self):
        c=self.cs['tmark'];v=dict(purpose='native_recovery_runtime_admission_v1',scope=r.ID,formal_permit_ref={},integrity_scan_passed=True,stage='tmark',protocol_sha=digest(c),lifecycle='x',launch_identity='x')
        path=self.root/'lifecycles/x/tmark/runtime-admission.json';ref=rec.exclusive(path,v)
        with patch.object(r,'PACKAGE',self.root),patch.dict(os.environ,{r.RUNTIME_SECRET_ENV:'actual-chain'}):
            with self.assertRaises(PermissionError):r.validate_runtime_light(c,ref,{},False)

    def test_formal_record_validation_has_no_probe_replay(self):
        functions=[r.validate_summary_light,r.validate_permit_light,r.validate_runtime_light,w.validate_worker,w.validate_wave,w.make_formal_config,w.run_group]
        for function in functions:
            tree=ast.parse(__import__('inspect').getsource(function));names=[n.func.id if isinstance(n.func,ast.Name) else n.func.attr if isinstance(n.func,ast.Attribute) else '' for n in ast.walk(tree) if isinstance(n,ast.Call)]
            self.assertNotIn('validate_probe_completion',names,function.__name__);self.assertNotIn('_compare_full_numeric_files',names,function.__name__);self.assertNotIn('scan_manifest',names,function.__name__)

    def test_compact_worker_metadata_never_follows_raw_reports(self):
        a=dict(technical_admission_ref={'path':str(self.root/'summary.json'),'sha256':'x'},probe_manifest_ref={'path':str(self.root/'manifest.json'),'sha256':'x'})
        meta=r.metadata_files(self.cs['tmark'],a)
        self.assertEqual(set(meta),{str(self.root/'summary.json'),str(self.root/'manifest.json')})
        self.assertTrue(all(not p.endswith('.bin') for p in meta))
        summary=r.prepared('summary');self.assertLess(len(json.dumps(summary)),30000)
        self.assertNotIn('serial',summary['decisions'][next(iter(summary['decisions']))])

    def test_complete_run_config_rejected_before_output_creation(self):
        inv=self.fixture_inventory();row=next(x for x in inv['rows'] if x['classification']=='A');ref=self.store('inventory.json',inv)
        with r.activate('tmark') as c,patch.object(r,'TMARK_RESULT',self.root/'new'),patch.object(r,'validate_permit_light'),patch.object(r,'validate_runtime_light'):
            with self.assertRaises(PermissionError):w.make_formal_config(c,row['task_id'],dict(inventory_ref=ref),{}, {})
        self.assertFalse((self.root/'new').exists())

    def test_B_read_old_write_new_preserves_optimizer_rng_best(self):
        old=self.root/'old';new=self.root/'new';old.mkdir();new.mkdir();(old/'best.pt').write_bytes(b'exact inherited best')
        identity=dict(commit=r.BASE);(old/'manifest.json').write_text(json.dumps(dict(identity=identity)))
        row=dict(original_root=str(old),checkpoint_committed_epoch=7,checkpoint_committed_steps=21,
            files={'best.pt':dict(**rec.ref(old/'best.pt'),size=25),'manifest.json':rec.ref(old/'manifest.json')})
        best=BestState();best.epoch=4
        c=self.cs['tmark'];t=c['tasks'][8]
        model,opt,generator=object(),object(),object()
        with patch.object(w,'resume_row',return_value=row),patch('ch3_runner.restore_state',return_value=(best,7,21)) as restore:
            result=w.resume_before_training(c,t,new,{},model,opt,generator)
            restore.assert_called_once_with(old/'last.pt',model,opt,identity,generator)
            self.assertEqual(result,(best,7,21));self.assertEqual((old/'best.pt').read_bytes(),(new/'best.pt').read_bytes())
            self.assertEqual(w.best_identity(c,t,new,{},dict(commit='new')),identity)
            (new/'best.pt').write_bytes(b'new better checkpoint')
            self.assertEqual(w.best_identity(c,t,new,{},dict(commit='new')),dict(commit='new'))
        tree=ast.parse((r.ROOT/'ch3_runner.py').read_text());f=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='restore_state')
        text=ast.unparse(f)
        for word in ('load_state_dict','setstate','set_rng_state','set_rng_state_all','generator.set_state','resume identity mismatch'):self.assertIn(word,text)

    def test_history_old_prefix_and_new_suffix_no_duplicate_epoch(self):
        prefix=self.root/'old-history.jsonl';prefix.write_text(json.dumps(dict(epoch=7,steps=21))+'\n')
        out=self.root/'new';out.mkdir();rec.exclusive(out/'resume-source.json',dict(row=dict(files={'history.jsonl':rec.ref(prefix)})))
        (out/'history.jsonl').write_text(json.dumps(dict(epoch=8,steps=24))+'\n')
        history,n=w.effective_history(out);self.assertEqual([x['epoch'] for x in history],[7,8]);self.assertEqual(n,1)
        self.assertEqual(prefix.read_text(),json.dumps(dict(epoch=7,steps=21))+'\n')

    def test_wrong_resume_task_profile_data_and_sha_refused(self):
        inv=self.fixture_inventory();b=next(x for x in inv['rows'] if x['classification']=='B');old=Path(b['original_root']);old.mkdir()
        t=next(t for t in self.cs['tmark']['tasks'] if t['id']==b['task_id'])
        for n in ('manifest.json','last.pt','best.pt','history.jsonl'):
            p=old/n;p.write_text('{}');b['files'][n]=dict(**rec.ref(p),size=p.stat().st_size)
        invref=self.store('bad-inventory.json',inv)
        with self.assertRaises((ValueError,KeyError)):w.resume_row(self.cs['tmark'],dict(inventory_ref=invref),t['id'])

    def test_generic_full_float_compared_once_per_payload(self):
        c=self.cs['m'];t=next(t for t in c['tasks'] if t['model']=='TimeMixer')
        points=[dict(schema_file=str(self.root/'schema'),data_file=str(self.root/('data'+str(i)))) for i in range(6)]
        trace=dict(M_full_state_trace=points,full_numeric_trace=points)
        def generic(c,t,x,y):
            from ch3_runner import _compare_full_numeric_files
            for a,b in zip(x['full_numeric_trace'],y['full_numeric_trace']):_compare_full_numeric_files({},a,b)
            return dict(passed=True)
        with patch('ch3_runner.compare_probe_trajectories',side_effect=generic),patch('ch3_runner._compare_full_numeric_files',return_value=dict(passed=True)) as compare,patch.object(scope,'context',return_value=dict(probe_root=self.root)):
            self.assertTrue(e.compare(c,t,trace,trace)['passed']);self.assertEqual(compare.call_count,6)

    def test_policy_None_full_state_exact_and_unknown_policy_refused(self):
        c=self.cs['m'];t=c['tasks'][0];points=[dict(schema_file=str(self.root/'s'),data_file=str(self.root/'d'))]*6;trace=dict(M_full_state_trace=points)
        with patch('ch3_runner.compare_probe_trajectories',return_value=dict(passed=True)),patch('ch3_runner._compare_full_numeric_files',return_value=dict(passed=True)) as compare,patch.object(scope,'context',return_value=dict(probe_root=self.root)),patch('utils.ch3_contract.numeric_probe_policy',return_value=None):
            e.compare(c,t,trace,trace);self.assertEqual(compare.call_count,6);self.assertTrue(all(x.args[0]['state_atol']==0 for x in compare.call_args_list))
        with patch('ch3_runner.compare_probe_trajectories',return_value=dict(passed=True)),patch('utils.ch3_contract.numeric_probe_policy',return_value=dict(kind='unsupported')):
            with self.assertRaises(ValueError):e.compare(c,t,trace,trace)

    def test_TimeMixer_Urban_endpoint_gate_preserved(self):
        c=self.cs['tmark'];t=next(t for t in c['tasks'] if t['model']=='TimeMixer' and t['dataset']=='UrbanEV')
        points=[dict(schema_file=str(self.root/'s'),data_file=str(self.root/'d'))]*6;trace=dict(M_full_state_trace=points,full_numeric_trace=points)
        with patch('ch3_runner.compare_probe_trajectories',return_value=dict(passed=True)),patch.object(scope,'context',return_value=dict(probe_root=self.root)),patch('utils.ch3_urban_capture.compare_traces',return_value={}) as endpoint,patch('utils.ch3_urban_confirmation.compare_measured',return_value=dict(passed=False)):
            self.assertFalse(e.compare(c,t,trace,trace)['passed']);endpoint.assert_called_once()

    def test_full_state_schema_all_fields_and_nonfloat_exact(self):
        import ch3_runner
        entries=[];offset=0
        for name in ('model/parameter/a','model/buffer/b','gradient/a','optimizer/a/exp_avg','optimizer/a/exp_avg_sq'):
            entries.append(dict(path=name,mode='bounded',dtype='torch.float32',shape=[1],offset=offset,nbytes=4,numel=1));offset+=4
        entries.append(dict(path='optimizer/a/step',mode='exact',dtype='torch.int64',shape=[1],offset=offset,nbytes=8,numel=1))
        schema=self.root/'schema.json';schema.write_text(json.dumps(dict(entries=entries)))
        def point(name,step):
            data=self.root/name;data.write_bytes(struct.pack('<5fq',*(1.,)*5,step))
            return dict(schema_file=str(schema),schema_sha=rec.sha(schema),data_file=str(data),data_sha=rec.sha(data),bytes=data.stat().st_size)
        x=point('x.bin',6);y=point('y.bin',7)
        self.assertFalse(ch3_runner._compare_full_numeric_files(dict(state_atol=1e-4),x,y)['passed'])
        z=point('z.bin',6);self.assertTrue(ch3_runner._compare_full_numeric_files(dict(state_atol=0),x,z)['passed'])

    def test_sealed_boundary_light_no_rebuild_and_tamper_refused(self):
        c=self.cs['tmark'];v=dict(purpose='native_recovery_replacement_boundary_v1',scope=r.ID,technical_complete=True,result_review='pending',task_ids=[t['id'] for t in c['tasks']],protocol_sha=digest(c),science_baseline_commit=r.BASE,old_retirement_boundary_ref=r.OLD_RETIREMENT)
        ref=self.store('sealed-boundary.json',v);r.validate_boundary_light(ref)
        with patch.object(w,'technical_task',side_effect=AssertionError('must not rebuild')),patch.object(r,'PACKAGE',self.root):
            rec.exclusive(self.root/'native-time-mark-replacement-boundary-recovery1.json',v)
            w.closeout(c,{}, {},'retry')
        Path(ref['path']).write_text('{}')
        with self.assertRaises(ValueError):r.validate_boundary_light(ref)

    def test_unsealed_closeout_can_retry_in_new_lifecycle(self):
        c=self.cs['tmark'];mapping=dict(rows={t['id']:dict(effective_root=str(self.root/t['id']),origin='recovery_fresh') for t in c['tasks']})
        mref=self.store('map.json',mapping);invref=self.store('inv.json',dict(rows=[]))
        with patch.object(r,'PACKAGE',self.root),patch.object(r,'stop_check'),patch.object(w,'technical_task',side_effect=ValueError('interrupted unsealed closeout')):
            with self.assertRaises(ValueError):w.closeout(c,dict(inventory_ref=invref),mref,'life1')
        self.assertFalse((self.root/'native-time-mark-replacement-boundary-recovery1.json').exists())

    def test_budget_insufficient_prevents_actual_permit(self):
        a=r.approval_template();self.assertFalse(a['execution_permitted']);self.assertIsNone(a['controller_execution_commit'])
        with patch.object(r,'controller_binding',return_value=dict(controller_execution_commit='closed')):
            with self.assertRaises(ValueError):r.validate_start_authorization(a)
        self.assertLess(rec.budget_projection(self.cs['tmark'],r.prepared('inventory'))['deficit_or_headroom']['adam'],0)

    def test_science_worker_controller_identities_separate(self):
        import ch3_runner
        c=self.cs['tmark'];t=c['tasks'][0];a=dict(commit='new-worker',science_baseline_commit=r.BASE,worker_execution_commit='new-worker',controller_execution_commit='new-controller',science_computation_fingerprint='unchanged',recovery_scope=r.ID)
        with patch.object(ch3_runner,'code_binding',return_value={}):identity=ch3_runner.formal_identity(c,t,{},a)
        self.assertEqual(identity['commit'],'new-worker');self.assertEqual(identity['science_baseline_commit'],r.BASE);self.assertEqual(identity['controller_execution_commit'],'new-controller')

    def test_full_recovery_chain_no_tmark_probe_or_manual_wait(self):
        control,calls,states,actions=self.actions()
        with patch.object(r,'CONTROL',control):r.drive(actions,lambda s,v:states.append(s),lambda:r.stop_check(control,drain=True))
        self.assertEqual(states,list(r.STATES));self.assertEqual(calls,list(r.STATES[:-1]));self.assertNotIn('TMARK_PROBE_RUNNING',calls)

    def test_failure_or_STOP_never_dispatches_next_stage(self):
        for failure in ('REMAINING_TMARK_FORMAL','M_PROBE','M_AUTO_AUDIT'):
            control,calls,states,actions=self.actions(fail=failure)
            with self.assertRaises(ValueError):r.drive(actions,lambda s,v:states.append(s),lambda:r.stop_check(control,drain=True))
            self.assertEqual(calls[-1],failure);self.assertNotIn('COMPLETE',states)

    def test_drain_before_wave_dispatch_does_not_spawn(self):
        control=self.root/'control';control.mkdir();r.write_marker(control/'DRAIN_STOP')
        with patch.object(r,'CONTROL',control),patch.object(r,'PACKAGE',self.root):
            with self.assertRaises(r.DrainStop):
                with r.dispatch_guard():self.fail('wave dispatched')

    def test_drain_active_wave_finishes_and_blocks_next(self):
        control=self.root/'control';control.mkdir();events=[]
        with patch.object(r,'CONTROL',control),patch.object(r,'PACKAGE',self.root):
            with r.dispatch_guard():events.append('wave-started')
            r.write_marker(control/'DRAIN_STOP');events.append('wave-completed')
            with self.assertRaises(r.DrainStop):
                with r.dispatch_guard():events.append('next-wave')
        self.assertEqual(events,['wave-started','wave-completed'])

    def test_drain_group_handoff_and_before_COMPLETE(self):
        for state in ('REMAINING_TMARK_FORMAL','M_FORMAL'):
            control,calls,states,actions=self.actions(drain=state)
            with self.assertRaises(r.DrainStop):r.drive(actions,lambda s,v:states.append(s),lambda:r.stop_check(control,drain=True))
            self.assertNotIn('COMPLETE',states)

    def test_drain_resume_eligible_is_not_unconditional_resume(self):
        control=self.root/'control';control.mkdir();rec.exclusive(control/'controller.json',dict(pid=0,start_ticks='0'))
        with patch.object(r,'CONTROL',control),patch.object(r.os,'kill') as signal:
            value=r.safe_stop(drain=True);signal.assert_not_called()
            self.assertTrue(value['resume_eligible']);self.assertFalse(value['unconditional_resume'])

    def test_owned_synchronous_child_real_exit_and_failure_retained(self):
        control=self.root/'control';control.mkdir()
        with patch.object(r,'CONTROL',control),patch.object(r,'PACKAGE',self.root):
            receipt=r.wait_owned([sys.executable,'-B','-c','import time;time.sleep(.15)'],'synthetic-child')
            self.assertEqual(receipt['exit_code'],0);self.assertFalse(r.same(receipt['child']))
            with self.assertRaises(RuntimeError):r.wait_owned([sys.executable,'-B','-c','import time;time.sleep(.1);raise SystemExit(7)'],'synthetic-failure')
            self.assertTrue((control/'synthetic-failure.log').exists())

    def test_tmux_synthetic_lifecycle_no_real_entry_or_worker(self):
        session='ch3-recovery-fixture-'+uuid.uuid4().hex[:12]
        script=self.root/'lifecycle.py'
        script.write_text('import json,os,time\nfrom pathlib import Path\np=Path(__file__).parent\n(p/"ready.json").write_text(json.dumps({"pid":os.getpid(),"start_ticks":Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()[19]}))\ntime.sleep(.4)\n(p/"done.json").write_text(json.dumps({"exit":0,"models":0,"GPU":0}))\n')
        # Multiple argv bypass the server's slow interactive profile; the
        # lifecycle assertion and fixed two-second bound stay unchanged.
        self.assertEqual(subprocess.run(['tmux','new-session','-d','-s',session,sys.executable,'-B',str(script)],check=False).returncode,0)
        for _ in range(100):
            if (self.root/'done.json').exists():break
            time.sleep(.02)
        self.assertTrue((self.root/'done.json').exists())
        for _ in range(100):
            if subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:break
            time.sleep(.02)
        self.assertNotEqual(subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode,0)
        owner=json.loads((self.root/'ready.json').read_text());self.assertFalse(r.same(owner))

    def test_original_science_bytes_and_training_AST_unchanged(self):
        protected=['models/ch3_adapter.py','utils/ch3_data.py','utils/ch3_time_marks.py','configs/ch3_formal_profiles.json','configs/ch3_native_time_mark_profiles.json']
        for path in protected:self.assertEqual((r.ROOT/path).read_bytes(),subprocess.check_output(['git','-C',str(r.ROOT),'show',r.BASE+':'+path]))
        old=ast.parse(subprocess.check_output(['git','-C',str(r.ROOT),'show',r.BASE+':ch3_runner.py']))
        new=ast.parse((r.ROOT/'ch3_runner.py').read_text())
        a={x.name:x for x in old.body if isinstance(x,ast.FunctionDef)};b={x.name:x for x in new.body if isinstance(x,ast.FunctionDef)}
        for name in ('init_training','update','evaluate','save_state','restore_state','probe_worker'):
            self.assertEqual(ast.dump(a[name],include_attributes=False),ast.dump(b[name],include_attributes=False),name)
        loop=lambda f:next(n for n in f.body if isinstance(n,ast.For))
        self.assertEqual(ast.dump(loop(a['formal_worker']),include_attributes=False),ast.dump(loop(b['formal_worker']),include_attributes=False))
        self.assertEqual(digest(self.cs['m']),'831163583475c86604286db561cc309b10b31a6d4df10f3b57c0a52f6c74b3fa')
        self.assertEqual(len(scope.probe_groups(self.cs['m'])),21)

    def test_old_permits_and_retained_STOP_do_not_authorize_recovery(self):
        old=rec.bound(r.TRUST['formal_permit'])
        with self.assertRaises(ValueError):r.validate_permit_light(self.cs['tmark'],old)
        self.assertTrue((scope.PACKAGE/'chain-execution-v1/STOP').exists())
        self.assertNotEqual(r.CONTROL,scope.PACKAGE/'chain-execution-v1')

    def test_retirement_fixed_original_reference_not_current_self_signature(self):
        b=r.validate_retirement_light()
        self.assertFalse(b['technical_complete']);self.assertTrue(b['retirement_accepted'])
        self.assertEqual(r.OLD_RETIREMENT,rec.bound(r.TRUST['formal_permit'])['old_boundary'])
        bad=self.store('changed-retirement.json',dict(b,technical_complete=True))
        with patch.object(r,'OLD_RETIREMENT',dict(bad,sha256='0'*64)):
            with self.assertRaises(ValueError):r.validate_retirement_light()
        with patch.object(r,'OLD_RETIREMENT',bad):
            with self.assertRaises(ValueError):r.validate_retirement_light()

    def test_actual_accepted_data_binding_schema_both_stages(self):
        from utils.ch3_native_chain import PREPARATION
        for stage,c in self.cs.items():
            accepted=rec.bound(PREPARATION['data_'+stage])
            self.assertNotIn('protocol_sha',accepted)
            value=r.data_binding(c)
            self.assertEqual(value['data_bindings'],accepted['data_bindings'])
            with patch('utils.ch3_m6.source_states',return_value={'changed':'state'}):
                with self.assertRaises(ValueError):r.data_binding(c)

    def compact_fixture(self):
        from ch3_runner import code_binding
        from utils.ch3_native_chain import PREPARATION
        c=self.cs['tmark'];data=rec.bound(PREPARATION['data_tmark']);prior=rec.bound(PREPARATION['environment_source'])
        auth=self.store('synthetic-start-authorization.json',dict(reviewed=True,execution_permitted=True,scope=r.ID,
            controller_execution_commit='synthetic-reviewed-closure',science_proof_ref=r.PROOF,
            authorized_execution_caps=rec.budget_projection(c,r.prepared('inventory'))['total_execution_worst_case']))
        a=dict(purpose='ch3_formal',recovery_scope=r.ID,successor_scope=r.ID+'-tmark-formal',reviewed=False,manual_review=False,
            review_mode=rec.MODE,execution_permitted=True,task='MS',metric_scope='target_only',science_baseline_commit=r.BASE,
            seed_list=[2024],additional_search=0,protocol_sha=digest(c),authorized_task_ids=[t['id'] for t in c['tasks']],
            profile_shas={t['id']:digest(profile(c,t)) for t in c['tasks']},run_budget=dict(runs=87,run_epochs=1020),
            max_optimizer_steps=3687530,numeric_policy_shas=r.policy_shas(c),structure_frozen=True,m6_authorized=True,
            technical_admission=True,result_review='pending',commit='synthetic-reviewed-closure',
            controller_execution_commit='synthetic-reviewed-closure',worker_execution_commit='synthetic-reviewed-closure',
            science_computation_fingerprint='fixture-math-unchanged',code=code_binding(),environment=prior['environment'],hardware=prior['hardware'],
            data_bindings=data['data_bindings'],data_binding_ref=PREPARATION['data_tmark'],source_states=data['source_states'],
            technical_admission_ref=r.PREPARED['summary'],probe_manifest_ref=r.PREPARED['manifest'],
            probe_metadata_manifest_ref=r.PREPARED['process_manifest'],
            compact_decisions=r.prepared('summary')['decisions'],probe_budget=r.prepared('summary')['budget'],
            start_authorization_ref=auth,inventory_ref=r.PREPARED['inventory'],effective_map_ref=r.PREPARED['effective_map'],
            old_retirement_boundary_ref=r.OLD_RETIREMENT,
            authorized_execution_caps=rec.bound(auth)['authorized_execution_caps'],synthetic_fixture=True)
        pref=self.store('synthetic-formal-permit.json',a)
        return c,a,pref

    def test_positive_compact_permit_and_exact_scope_mutations(self):
        c,a,pref=self.compact_fixture()
        with r.activate('tmark'),patch.object(r,'dynamic_binding'),patch.object(e,'validate_probe_completion',side_effect=AssertionError('no replay')):
            r.validate_permit_light(c,a)
            for key in ('protocol_sha','science_baseline_commit','profile_shas','data_bindings','compact_decisions','numeric_policy_shas','run_budget','worker_execution_commit'):
                wrong=dict(a);wrong[key]='wrong'
                with self.subTest(key=key),self.assertRaises((ValueError,KeyError,TypeError)):r.validate_permit_light(c,wrong)

    def test_positive_compact_config_worker_wave_and_zero_raw_reads(self):
        c,a,pref=self.compact_fixture();case=next(x for x in r.prepared('inventory')['rows'] if x['classification']=='D')
        secret='synthetic-runtime-secret';counts=[]
        actual_runtime=r.validate_runtime_light
        with patch.object(r,'PACKAGE',self.root),patch.object(r,'CONTROL',self.root/'control'),patch.object(r,'TMARK_RESULT',self.root/'new-results'),patch.dict(os.environ,{r.RUNTIME_SECRET_ENV:secret}),r.activate('tmark'),patch.object(r,'dynamic_binding'),patch.object(e,'validate_probe_completion',side_effect=AssertionError('formal full audit forbidden')),patch('ch3_runner._compare_full_numeric_files',side_effect=AssertionError('formal comparator forbidden')),patch.object(rec,'scan_manifest',side_effect=lambda *args:(counts.append('startup-scan') or dict(files=1,bytes=1,integrity_scan_passed=True))):
            runtime=r.seal_runtime(c,pref,r.PREPARED['summary'],r.PREPARED['manifest'],'fixture-life',r.identity(os.getpid()))
            with patch.object(r,'validate_runtime_light',side_effect=lambda c,rr,pr:actual_runtime(c,rr,pr,False)):
                cfg=w.make_formal_config(c,case['task_id'],a,pref,runtime)
                Path(cfg['fixture_root']).mkdir(exist_ok=True)
                with patch.dict(os.environ,{'TMPDIR':cfg['fixture_root']}):w.validate_worker(c,cfg)
                self.assertIsNone(cfg['approval']);self.assertNotIn('full_state',json.dumps(cfg));self.assertNotIn('trajectory',json.dumps(cfg))
                self.assertEqual(counts,['startup-scan'])
                # Consume the exact original pending wave without reconstructing evidence.
                model=next(t['model'] for t in c['tasks'] if t['id']==case['task_id'])
                waves=rec.pending_waves(scope.formal_waves(c,dict(decisions=a['compact_decisions']),model),r.prepared('inventory'))
                member=next(ids for _,ids in waves if case['task_id'] in ids)
                configs=[dict(cfg,task=run,output=str(scope.result_path(c,next(t for t in c['tasks'] if t['id']==run)))) for run in member]
                w.validate_wave(c,configs,scope.context(c)['control']/model/'wave-synthetic')
                self.assertEqual(counts,['startup-scan'])
            rec.exclusive(self.root/'validation-counts.json',dict(full_numeric=0,full_probe_audit=0,raw_per_task=0,startup_manifest_scans=len(counts),config_bytes=Path(cfg['output'],'config.json').stat().st_size))

    def test_compact_summary_and_permit_SHA_tamper_rejected(self):
        c,a,pref=self.compact_fixture();Path(pref['path']).write_text('{}')
        with self.assertRaises(ValueError):r.worker_permit(dict(formal_permit_ref=pref))
        summary=copy.deepcopy(r.prepared('summary'));summary['original_complete_ref']['sha256']='other'
        with self.assertRaises(ValueError):r.validate_summary_light(c,summary)

    def test_worker_raw_allowlist_and_wrong_task_refused(self):
        c,a,pref=self.compact_fixture()
        with self.assertRaises(PermissionError):w.read_config(dict(recovery_scope=r.ID,protocol_file='/other',successor_scope=r.ID+'-tmark-formal'))
        with self.assertRaises(PermissionError):w.read_config(dict(recovery_scope=r.ID,protocol_file=str(scope.PROFILE_FILE),protocol_sha=digest(c),successor_scope=r.ID+'-tmark-probe',purpose='ch3_probe'))

    def test_fixed_launcher_token_current_log_stale_wrong_and_duplicate(self):
        from utils.ch3_m_launch import file_identity
        log=self.root/'log';log.touch();token='a'*64;approval=self.store('approval.json',{})
        launch=dict(scope=r.ID,approval_ref=approval,log=str(log),log_identity=file_identity(log),token_sha=__import__('hashlib').sha256(token.encode()).hexdigest())
        rec.exclusive(str(log)+'.launch.json',launch)
        with patch.object(r,'LOG',log),patch('utils.ch3_m_launch.tmux_view',return_value={}):
            r.verify_launch(approval,token)
            for bad in (None,'b'*64):
                with self.assertRaises((ValueError,PermissionError)):r.verify_launch(approval,bad)
            rec.exclusive(str(log)+'.claimed.json',{})
            with self.assertRaises(FileExistsError):r.verify_launch(approval,token)

    def test_safe_stop_foreign_child_and_supervisor_refused(self):
        control=self.root/'control';control.mkdir();owner=dict(pid=123,start_ticks='fixture')
        rec.exclusive(control/'controller.json',owner)
        rec.exclusive(control/'current.json',dict(child=dict(pid=124,start_ticks='fixture',owner=owner,command=['foreign'])))
        with patch.object(r,'CONTROL',control),patch.object(r,'same',return_value=True),patch.object(r.os,'kill') as signals:
            with self.assertRaises(ValueError):r.safe_stop()
            signals.assert_not_called();self.assertTrue((control/'STOP').exists())


from tests.test_m6_native_chain import SavedProbeTests as _SavedProbeTests


class ProbeRegressionTests(_SavedProbeTests):
    """Same existing immediate gates, isolated permanent fixture evidence."""
    def setUp(self):
        base=Path(os.environ.get('CH3_RECOVERY_FIXTURE_ROOT',str(r.PACKAGE/'acceptance/fixtures')))
        self.root=base/(self._testMethodName+'-'+uuid.uuid4().hex);self.root.mkdir(parents=True)
        self.c=copy.deepcopy(type(self).c);self.c['groups']=self.c['groups'][:1]
        self.ctx=scope.context(self.c);self.ctx.update(probe_root=self.root/'probe',control=self.root/'formal');self.ctx['probe_root'].mkdir()
        self.a=dict(commit='synthetic',protocol_sha=digest(self.c),code={},environment={},hardware={'cpu_affinity':[]})
        rec.exclusive(self.ctx['probe_root']/'approval.json',self.a)
