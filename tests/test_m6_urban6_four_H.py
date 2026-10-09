"""Four-H science, withdrawal isolation and real CPU-only admission audit."""
import contextlib,copy,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from utils import ch3_third_round_after_selection as b,ch3_type1_tasks as s,ch3_type1_chain as q
from utils.ch3_contract import profile,digest,step_arithmetic,numeric_probe_policy,task_by_id
from utils.ch3_native_recovery_records import bound,ref,exclusive,manifest_projection,scan_manifest
b.activate()


class UrbanFourH(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cs=q.configs();cls.c=cls.cs['URBAN_SUBSET']

    def test_exact168_model_fold_horizon_and_unique_profile_identity(self):
        c=self.c;self.assertEqual(len(c['tasks']),168)
        self.assertEqual(len({t['id']for t in c['tasks']}),168);self.assertEqual(len({t['profile']for t in c['tasks']}),168)
        for model in s.MODELS:
            for fold in range(1,7):self.assertEqual([t['h']for t in c['tasks']if(t['model'],t['fold'])==(model,fold)],[3,6,9,12])
        self.assertEqual({h:sum(t['h']==h for t in c['tasks'])for h in(3,6,9,12)},{3:42,6:42,9:42,12:42})
        for old,t in zip(s.selected('URBAN_SUBSET'),c['tasks']):self.assertEqual([old[k]for k in('model','fold','h')],[t[k]for k in('model','fold','h')])

    def test_original84_profiles_fully_equal_and_new84_uniquely_derived(self):
        parent=bound(b.contract()['urban_parent_ref']);frozen=s.parent();counts={True:0,False:0}
        for t in self.c['tasks']:
            p=profile(self.c,t);prior=next(x for x in frozen['tasks']if x['dataset']=='UrbanEV'and(x['model'],x['fold'],x['h'])==(t['model'],t['fold'],t['h']))
            self.assertEqual(p,s.inherited_profile(prior));self.assertEqual((p['T'],p['pred_len'],p['C']),(12,1,11))
            self.assertEqual((p['training']['epochs'],p['training']['patience'],p['training']['batch']),(20,5,128))
            self.assertEqual(p['training']['scheduler']['name'],'type1_horizon_scaled_v1')
            if t['h']in(3,12):
                old=next(x for x in parent['tasks']if(x['model'],x['fold'],x['h'])==(t['model'],t['fold'],t['h']))
                self.assertEqual(p,profile(parent,old))
            counts[t['h']in(3,12)]+=1
        self.assertEqual(counts,{True:84,False:84})

    def test_all168_exact_fold_data_scaler_node_order_windows_and_steps(self):
        c=self.c;data=bound(c['baseline_unified']['data_ref']);source=bound(s.parent()['baseline_unified']['data_ref'])
        for t in c['tasks']:
            prior=c['baseline_unified']['parent_refs'][t['id']]['task_id'];m=data['metadata']['UrbanEV'][t['id']]
            self.assertEqual(m,source['metadata']['UrbanEV'][prior]);self.assertEqual(data['data_bindings']['UrbanEV'][t['id']],digest(m))
            arithmetic=step_arithmetic(c,t)
            self.assertEqual(m['window_counts'],dict(train=arithmetic['train_windows'],validation=arithmetic['validation_windows']))
            self.assertEqual(profile(c,t)['training']['scheduler']['steps_per_epoch'],arithmetic['train_batches'])
        self.assertFalse(data['test_observations_accessed']);self.assertEqual(len(data['mapping']),168)

    def test_numeric_policies_unchanged_except_explicit_H_applicability(self):
        old=bound(b.contract()['urban_parent_ref'])['baseline_unified']['numeric_policies'];now=self.c['baseline_unified']['numeric_policies']
        self.assertEqual(set(now),set(old))
        for k,v in old.items():self.assertEqual(now[k],dict(v,horizons=[3,6,9,12]));self.assertEqual(now[k]['state_atol'],1e-4)

    def test_wrong_H_duplicate_run_policy_profile_and_source_SHA_rejected(self):
        for kind in('horizon','duplicate','layers','pred_len','policy','data_sha'):
            bad=copy.deepcopy(self.c);t=bad['tasks'][0]
            if kind=='horizon':t['h']=9
            elif kind=='duplicate':bad['tasks'][1]=copy.deepcopy(t)
            elif kind=='layers':bad['resolved_profiles'][t['id']]['structure']['e_layers']=99
            elif kind=='pred_len':bad['resolved_profiles'][t['id']]['pred_len']=6
            elif kind=='policy':next(iter(bad['baseline_unified']['numeric_policies'].values()))['state_atol']=1
            else:bad['baseline_unified']['data_ref']['sha256']='0'*64
            with self.assertRaises((ValueError,PermissionError),msg=kind):b.validate_revision(bad)

    def test_EPF_M_unchanged_and371_5670_exact(self):
        cs=self.cs;self.assertEqual([len(cs[x]['tasks'])for x in b.STAGES],[168,35,168])
        total=sum(profile(c,t)['training']['epochs']for c in cs.values()for t in c['tasks']);self.assertEqual(total,5670)
        self.assertEqual(ref(s.file('M_ALL'))['sha256'],'03c88ba46fd871329b27ab7b6bd7e4f167fee750783221a8698783ac1a3e5989')
        for stage in('EPF_ALL','M_ALL'):self.assertEqual(cs[stage],bound(b.contract()['config_refs'][stage]))
        for t in cs['M_ALL']['tasks']:
            if t['model']=='PatchTST':self.assertEqual(profile(cs['M_ALL'],t)['structure']['d_model'],128 if t['dataset']=='Weather'else 4)
        self.assertEqual(b.selected_main()['retained_cells'],351)
        self.assertEqual(b.completion_counts()['third_round_runs'],371)

    def test_fresh_all168_representatives_and_no_old_admission(self):
        groups=s.probe_groups(self.c);self.assertEqual(len(groups),7)
        self.assertEqual(sum(len(g['representatives'])for g in groups),168)
        self.assertTrue(all(len(g['representatives'])==24 and g['planned_q']==4 for g in groups))
        for g in groups:self.assertEqual(set(g['coverage']),set(g['representatives']))
        self.assertEqual(s.probe_budget(self.c)['caps'],dict(adam=3024,backward=3024,forward=4176))
        self.assertRaises(PermissionError,b.adopt_urban_probe,{})
        self.assertNotIn('URBAN_SUBSET_PROBE',b.continuation_actions());self.assertEqual(b.SEED_STAGES,())

    def test_old_B_stop_cost_test_access_and_readonly_protection(self):
        stop=b.verify_stopped_B();self.assertTrue(stop['owned_processes_exited']);self.assertTrue(stop['session_absent'])
        self.assertEqual((stop['completed_formal'],stop['interrupted_formal'],stop['running_formal']),(0,4,0))
        self.assertEqual(stop['last_persisted_actual'],dict(adam=33997,backward=34000,forward=37244))
        self.assertEqual(stop['validation_completed_records'],19);self.assertEqual(stop['formal_test_calls'],0)
        self.assertFalse(stop['old_formal_eligibility']);self.assertFalse(stop['old_probe_eligibility_for_new_queue']);self.assertFalse(stop['may_resume'])
        self.assertEqual(b.anchors()['old_B_completed_credit'],0)
        self.assertTrue(all(not row['budget_finalized']for row in stop['rows']))

    def test_old_permission_and_synthetic_completed_receipt_cannot_credit_new_B(self):
        a=bound(b.verify_stopped_B()['authorization_ref'])
        with patch.object(q,'closure',return_value=a['closure_commit']):self.assertRaises(PermissionError,q.validate_start,a)
        fake=b.completion_record({q.STAGE_STATES[k][3]:{'synthetic':True}for k in b.STAGES})
        self.assertRaises((ValueError,KeyError),b.validate_complete,fake)
        fake['withdrawn_B_completed_credit']=4;self.assertRaises(ValueError,b.validate_complete,fake)

    def test_exact_current_manifest_accepts_and_old_process_SHA_symlink_reject(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-four-H-manifest-',dir=b.PACKAGE/'fixtures')as raw:
            root=Path(raw);p=root/'process.json';exclusive(p,{'synthetic':True});complete=exclusive(root/'complete.json',{'synthetic':True})
            report=dict(scope=b.ID+'-URBAN_SUBSET-probe',artifacts={str(p):ref(p)})
            m=manifest_projection(report,complete,root);self.assertTrue(scan_manifest(m,complete,root)['integrity_scan_passed'])
            bad=copy.deepcopy(report);bad['artifacts'][str(p)]['sha256']='0'*64;self.assertRaises(ValueError,manifest_projection,bad,complete,root)
            old=b.verify_stopped_B()['refs']['controller.json'];bad=dict(report,artifacts={old['path']:old});self.assertRaises(ValueError,manifest_projection,bad,complete,root)
            link=root/'alias.json';link.symlink_to(p);bad=dict(report,artifacts={str(link):ref(link)});self.assertRaises(ValueError,manifest_projection,bad,complete,root)

    def test_real_summary_reports_full_four_H_and_no_old_B_current_results(self):
        from utils.ch3_type1_summary import result_index
        v=result_index(self.cs);self.assertEqual(len(v['rows']),371);self.assertEqual(v['coverage']['UrbanEV'],'168/168: folds1–6 and H3,6,9,12')
        old_root=b.verify_stopped_B()['old_result_root']
        self.assertTrue(all(row['new']['status']=='not_available'and old_root not in row['new']['path']for row in v['rows']))
        self.assertEqual(v['result_review'],'pending')

    def test_real_stage_seals_and_complete371_reject_missing_or_old_tasks(self):
        from tests.test_m6_B_startup_recovery import fixture,synthetic_gpu
        from utils import ch3_event_resources as event
        with fixture()as(root,auth,start,stack):
            stack.enter_context(patch.dict(os.environ,{q.SECRET:'CPU-synthetic-complete371-owner-only'}))
            with patch.object(subprocess,'check_output',side_effect=synthetic_gpu),event.prestart(auth):pass
            exclusive(q.CONTROL/'controller.json',dict(owner=q.owner(),scope=b.ID,authorization=start));event.seal_startup(start)
            receipts={}
            for stage,c in self.cs.items():
                ctx=s.context(c);groups={}
                for model in ctx['models']:
                    groups[model]=exclusive(ctx['control']/('group-'+model)/'complete.json',dict(synthetic=True,technical_complete=True,model=model,task_ids=[t['id']for t in c['tasks']if t['model']==model],**b.resource_binding()))
                receipts[q.STAGE_STATES[stage][3]]=q.seal_boundary(c,groups)
            v=b.completion_record(receipts);self.assertEqual(b.validate_complete(v),v)
            self.assertEqual(v['executed_third_formal_runs'],371);self.assertEqual(v['result_review'],'pending')
            bad=copy.deepcopy(v);bad['stage_boundaries'].pop('EPF_ALL');self.assertRaises(ValueError,b.validate_complete,bad)
            stage='URBAN_SUBSET';path=Path(receipts[q.STAGE_STATES[stage][3]]['path']);payload=json.loads(path.read_text());payload['task_ids'][0]=b.verify_stopped_B()['rows'][0]['task_id'];path.write_text(json.dumps(payload)+'\n')
            bad=copy.deepcopy(v);bad['stage_boundaries'][stage]=ref(path);self.assertRaises(ValueError,b.validate_complete,bad)

    def test_real_CPU_owned_stop_does_not_signal_unrelated_fixture(self):
        root=Path(tempfile.mkdtemp(prefix='synthetic-four-H-stop-',dir=b.PACKAGE/'fixtures'))
        script='''import sys,os,json,subprocess;from pathlib import Path
from utils import ch3_third_round_after_selection as b,ch3_type1_chain as q
from utils.ch3_native_recovery_records import exclusive
b.activate();q.CONTROL=Path(sys.argv[1]);q.CONTROL.mkdir();child=None;other=None
try:
 child=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(20)',str(b.ENTRY),'group-child'])
 other=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(20)'])
 exclusive(q.CONTROL/'controller.json',dict(scope=b.ID,owner=q.owner()))
 exclusive(q.CONTROL/'current.json',dict(owner=q.owner(),child=q.owner(child.pid),command=child.args))
 result=q.safe_stop(signal_supervisor=False);child.wait(timeout=5)
 assert result['STOP']and result['child_signal_sent']and child.returncode!=0 and other.poll()is None
 print(json.dumps(dict(safe_stop=result,owned_exit=child.returncode,unrelated_alive=True)))
finally:
 for p in(child,other):
  if p is not None and p.poll()is None:p.terminate();p.wait(timeout=5)
'''
        r=subprocess.run([sys.executable,'-B','-c',script,str(root/'controller'),str(b.ENTRY),'start'],cwd=b.ROOT,capture_output=True,text=True,timeout=20)
        (root/'stdout.txt').write_text(r.stdout);(root/'stderr.txt').write_text(r.stderr);self.assertEqual(r.returncode,0,r.stderr)
        self.assertTrue(json.loads(r.stdout)['unrelated_alive'])

    def test_real_full168_native_probe_AUTO_AUDIT_and_compact_formal_permit(self):
        from tests.test_m6_B_startup_recovery import fixture,synthetic_gpu
        from tests.test_m6_probe_schema_recovery import trajectory
        from utils import ch3_native_execution as native,ch3_event_resources as event
        from tools.restricted_regression import m5_formal_entry as tool,resource_budget
        import ch3_runner as runner
        real_query=subprocess.check_output
        with fixture()as(root,auth,start,stack):
            c=self.c;old_context=s.context;ctx=dict(old_context(c),probe_root=root/'probe',control=root/'queue',fixture=root/'fixture',result_root=root/'formal')
            ctx['fixture'].mkdir();ctx['control'].mkdir()
            stack.enter_context(patch.object(s,'context',side_effect=lambda cc:ctx if cc==c else old_context(cc)))
            stack.enter_context(patch.dict(os.environ,{q.SECRET:'CPU-four-H-only-not-an-execution-grant'}))
            stack.enter_context(patch.dict(sys.modules,{'m5_formal_entry':tool,'resource_budget':resource_budget}))
            # Keep the real lock behavior, with only its fixture path isolated.
            lock=runner.GPULock;stack.enter_context(patch.object(runner,'GPULock',side_effect=lambda cc:lock({'execution':{'evidence':str(root)}})))
            with patch.object(subprocess,'check_output',side_effect=synthetic_gpu),event.prestart(auth):pass
            exclusive(q.CONTROL/'controller.json',dict(owner=q.owner(),scope=b.ID,authorization=start));hw=event.seal_startup(start);b.import_ms()
            a=bound(q.create_permit(c,start,True));exclusive(ctx['probe_root']/'approval.json',a)
            spawned=[];calls=[]
            def spawn(cfg):
                t=task_by_id(c,cfg['task']);out=Path(cfg['output']);h=(out/'worker.log').open('x')
                delay=.15 if cfg['successor_phase']=='serial'else .01
                child=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep('+str(delay)+')'],stdout=h,stderr=h);spawned.append(child)
                tr=trajectory(c,t,out,numeric_probe_policy(c,t));tr.update(b.resource_binding(),allocated=None,reserved=None)
                for row in tr['memory']:row.update(allocated=None,reserved=None,resource_mode=event.MODE)
                tr['affinity']=a['hardware']['cpu_affinity'];tr['memory_review']=runner.memory_growth_review(tr['memory'],event.MODE)
                if t['model']=='TimeMixer':
                    from utils.ch3_urban_confirmation import policy
                    diagnostic=[]
                    for step in range(7):
                        point=copy.deepcopy(tr['M_full_state_trace'][max(step,1)-1]);point['step']=step
                        meta=exclusive(out/('diagnostic-meta-'+str(step)+'.json'),dict(step=step,rng='rng',absent_gradients=[],empty_optimizer_state=[]))
                        point.update(meta_file=meta['path'],meta_sha=meta['sha256']);diagnostic.append(point)
                    tr['urban_diagnostic_trace']=diagnostic
                    snapshot=dict(model=tr['final'],optimizer=tr['trajectory'][-1]['optimizer'],rng=tr['final_rng'])
                    endpoint=dict(metrics=tr['validation'],batch_ids=['CPU-cached-validation-A','CPU-cached-validation-B'],before=snapshot,after=snapshot,mode_restored=True,target=dict(index=profile(c,t)['target_idx'],pred_len=1,C=11,metric_space='train-standardized target-only'))
                    tr['urban_confirmation']=dict(policy_sha=digest(policy(c,t)),evaluations={'2':copy.deepcopy(endpoint),'6':copy.deepcopy(endpoint)})
                exclusive(out/'trajectory.json',tr);counts=s.worker_counts(c,t)
                (out/'budget.json').write_text(json.dumps(dict(counts=counts,by_pid={str(child.pid):counts}))+'\n')
                exclusive(out/'runtime.json',dict(pid=child.pid,task=t['id'],error=None));(out/'audit.jsonl').write_text('{"event":"CPU-synthetic-only"}\n')
                calls.append(t['id']);return child,h
            def no_GPU(args,**kwargs):
                if args[0]=='nvidia-smi':raise AssertionError('runtime GPU query forbidden')
                return real_query(args,**kwargs)
            stack.enter_context(patch.object(tool,'spawn',side_effect=spawn))
            stack.enter_context(patch.object(tool,'read_profiles',return_value=dict(execution=dict(evidence=str(ctx['probe_root'])))))
            stack.enter_context(patch.object(subprocess,'check_output',side_effect=no_GPU))
            for name in('gpu_sample','owned_pid_metadata','ExitObservation'):stack.enter_context(patch.object(tool,name,side_effect=AssertionError('runtime telemetry forbidden')))
            report=native.run_probe(c,a);self.assertEqual(len(calls),336);self.assertTrue(all(p.poll()is not None for p in spawned))
            self.assertEqual(set(calls),{t['id']for t in c['tasks']});self.assertTrue(all(calls.count(t['id'])==2 for t in c['tasks']))
            with patch.object(native,'validate_probe_completion',wraps=native.validate_probe_completion)as audit:
                summary_ref=q.audit_probe(c);self.assertEqual(audit.call_count,1)
                summary=q.validate_summary_light(c,summary_ref);self.assertEqual(summary['startup_hardware_ref'],hw)
                formal=q.create_permit(c,start,False,summary_ref);runtime=q.seal_runtime(c,formal);q.validate_runtime(c,runtime,formal)
                self.assertEqual(audit.call_count,1)
            bad=copy.deepcopy(report);bad['evidence'].pop(next(iter(bad['evidence'])));self.assertRaises(ValueError,native.validate_probe_completion,c,bad)
            self.assertFalse(list(ctx['probe_root'].rglob('memory.jsonl')));self.assertFalse(b.RESULT.exists())


if __name__=='__main__':unittest.main()
