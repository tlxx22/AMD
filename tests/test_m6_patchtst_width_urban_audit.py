"""Synthetic provenance/audit/queue tests and exact width-only scientific deltas."""
import contextlib,copy,json,math,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from utils import ch3_patchtst_depth_urban6_recovery as old,ch3_patchtst_width_reproduction as w
from utils import ch3_type1_chain as q,ch3_type1_tasks as s,ch3_native_execution as native
from utils.ch3_contract import ROOT,digest,profile,numeric_probe_policy,step_arithmetic
from utils.ch3_native_recovery_records import bound,ref,exclusive,manifest_projection,scan_manifest,process_refs
w.activate()


class WidthContract(unittest.TestCase):
    def test_exact40_unique_width_only_profiles_and_parents(self):
        seen=[]
        for stage,c in q.configs().items():
            width=int(stage[-1]);self.assertEqual(len(c['tasks']),20)
            self.assertEqual({(t['dataset'],t['h'])for t in c['tasks']},{(d,h)for d in w.DATASETS for h in w.HORIZONS})
            for t in c['tasks']:
                p=c['baseline_unified']['parent_refs'][t['id']];parent=bound(p['config_ref']);prior=next(x for x in parent['tasks']if x['id']==p['task_id']);before=profile(parent,prior);expected=copy.deepcopy(before);expected['structure']['d_model']=width
                self.assertEqual(profile(c,t),expected);self.assertEqual(numeric_probe_policy(c,t),numeric_probe_policy(parent,prior));self.assertEqual(before['structure']['n_heads'],4);self.assertEqual(t['variant'],f'encoder-2-d_model-{width}');seen.append(t['id'])
            w.validate_revision(c)
        self.assertEqual(len(set(seen)),40)

    def test_Weather8_parent_profiles_exact128_16_2_256_20_10(self):
        proof=bound(ref(w.PACKAGE/'Weather-preservation.json'))
        self.assertEqual(len(proof['rows']),8)
        for row in proof['rows']:
            c=bound(row['config_ref']);t=next(t for t in c['tasks']if t['id']==row['task']);p=profile(c,t)
            self.assertEqual(p,row['profile']);self.assertEqual(digest(p),row['profile_sha']);self.assertEqual([p['structure'][k]for k in('d_model','n_heads','e_layers','d_ff')],[128,16,2,256]);self.assertEqual((p['training']['epochs'],p['training']['patience']),(20,10))
        self.assertTrue(all(t['dataset']!='Weather'for c in q.configs().values()for t in c['tasks']))

    def test_science_identity_batch_seed_LR_epoch_no_cross_resume(self):
        for c in q.configs().values():
            self.assertEqual(s.expected_tasks(c['baseline_unified']['stage']),c['tasks'])
            for t in c['tasks']:
                p=profile(c,t);a=p['training'];self.assertEqual(t['seed'],2024);self.assertEqual((p['T'],a['batch'],a['eval_batch'],a['epochs'],a['patience']),(96,128,128,10,None));self.assertEqual(a['scheduler']['name'],'OneCycleLR');self.assertEqual(a['scheduler']['max_lr'],.01);self.assertTrue(a['from_scratch'])

    def test_all_other_science_and_old133_numeric_policies_unchanged(self):
        proof=bound(ref(w.PACKAGE/'producer-delta-proof.final.json'))
        for name,h in proof['protected_exact'].items():self.assertEqual(ref(ROOT/name)['sha256'],h)
        self.assertEqual(sum(len(bound(v)['baseline_unified']['numeric_policies'])for v in old.contract()['old_config_refs'].values()),133)

    def test_max400_epochs_formal_and_probe_caps_independently_derived(self):
        budgets=[s.formal_budget(c)['total']for c in q.configs().values()]
        self.assertEqual({k:sum(b[k]for b in budgets)for k in budgets[0]},dict(runs=40,run_epochs=400,adam=55840,backward=55840,forward=75292))
        for c in q.configs().values():self.assertEqual(s.probe_budget(c)['caps'],dict(adam=360,backward=360,forward=480))

    def test_tamper_width_heads_Weather_task_scheduler_numeric_all_rejected(self):
        c=q.configs()['PATCH_DM4']
        for field,value in(('d_model',8),('n_heads',1),('e_layers',3),('d_ff',256)):
            bad=copy.deepcopy(c);bad['resolved_profiles'][bad['tasks'][0]['id']]['structure'][field]=value
            self.assertRaises(ValueError,w.validate_revision,bad)
        for kind in('weather','numeric','scheduler'):
            bad=copy.deepcopy(c)
            if kind=='weather':bad['tasks'][0]['dataset']='Weather'
            elif kind=='numeric':bad['baseline_unified']['numeric_policies']['PatchTST-ETTh1']['state_atol']=1
            else:bad['resolved_profiles'][bad['tasks'][0]['id']]['training']['scheduler']['epochs']=9
            self.assertRaises(ValueError,w.validate_revision,bad)

    def test_old_or_other_width_attempt_permit_rejected(self):
        c=q.configs()['PATCH_DM4']
        for a in(dict(execution_attempt=old.ATTEMPT,probe_recovery_ref=old.REUSE_REF),dict(execution_attempt=w.ATTEMPT,probe_recovery_ref=w.REUSE_REF,base287_boundary_ref=old.SOURCE_REF)):
            self.assertRaises(PermissionError,w.validate_permit_link,c,a,True)

    def test_new_width_own_serial_and_q_gate_not_inherited(self):
        for c in q.configs().values():
            self.assertEqual(len(s.probe_groups(c)),5)
            for g in s.probe_groups(c):self.assertEqual(g['planned_q'],4);self.assertEqual(len(g['representatives']),4)
        self.assertEqual(w.SEED_STAGES,());self.assertEqual(w.retained_refs(dict(probe_recovery_ref=w.REUSE_REF)),{})

    def test_paper_actual_Table2_and_Exchange_no_reference(self):
        paper=bound(w.contract()['paper_ref'])
        self.assertEqual(paper['table'],2);self.assertEqual(paper['horizons'],[96,192,336,720]);self.assertIsNone(paper['Exchange']);self.assertFalse(paper['width_confirmed_by_table']);self.assertFalse(paper['author_width_clue']['unique_setting_confirmed'])
        self.assertEqual(paper['PatchTST']['ETTh1'],dict(mse=.516,mae=.484))

    def test_CPU_real40_construction_forward_evidence_no_updates(self):
        rows=[]
        for width in(4,8):
            for d in w.DATASETS:
                capture=bound(ref(w.PACKAGE/f'cpu-smoke-dm{width}-{d}.json'));self.assertEqual(capture['exit_code'],0,capture['stderr']);v=json.loads(capture['stdout']);self.assertEqual((v['construction'],v['forward']), (4,4));self.assertEqual((v['backward'],v['adam'],v['GPU'],v['test']), (0,0,0,0));rows.extend(v['rows'])
        self.assertEqual(len(rows),40);self.assertEqual(len({x['task_id']for x in rows}),40)
        for x in rows:self.assertEqual(x['encoder_blocks'],2);self.assertEqual(x['heads'],4);self.assertTrue(x['finite'])

    def test_r6_sources_completed48_and_original_failure_still_failure(self):
        source=w.verify_source();self.assertEqual(source['depth_completed_new_formal'],48);self.assertEqual(source['old_completed_new_formal'],196);self.assertEqual(source['third_formal'],0)
        self.assertEqual(bound(source['refs']['queue/controller/failure.json'])['error'],"ValueError('probe artifact path/symlink outside exact scope')")

    def test_full_fixed_queue_stops_after_two_width_boundaries(self):
        events=[];actions={state:lambda receipts,state=state:events.append(state)or {'synthetic':state}for state in q.STATES if state!='COMPLETE'}
        receipts=q.drive(actions,lambda state:None,lambda:None)
        self.assertEqual(events,[x for x in q.STATES if x!='COMPLETE']);self.assertFalse(any('URBAN' in x or 'M_ALL' in x or 'ADOPT' in x for x in events))
        complete=w.completion_record(receipts);self.assertTrue(complete['awaiting_user_width_selection']);self.assertFalse(complete['automatic_adoption']);self.assertTrue(complete['third_round_paused']);self.assertEqual(complete['total_runs'],40)

    def test_failure_and_STOP_block_second_width(self):
        for stop in(False,True):
            events=[]
            def action(receipts):events.append('first');raise InterruptedError('STOP')if stop else ValueError('numeric fault')
            actions={state:action for state in q.STATES if state!='COMPLETE'}
            self.assertRaises((ValueError,InterruptedError),q.drive,actions,lambda state:None,lambda:None);self.assertEqual(events,['first'])

    def test_foreign_resource_and_wrong_completion_rejected(self):
        self.assertRaises(PermissionError,w.monitor_binding,[dict(purpose='ch3_probe',approval={},resource_mode='anything')])
        self.assertRaises(ValueError,w.validate_complete,dict(scope=w.ID,technical_complete=True,total_runs=40,third_round_paused=False))


@contextlib.contextmanager
def urban_fixture():
    """Small independent records; no old binary/test/checkpoint access."""
    with tempfile.TemporaryDirectory(prefix='m6-urban-audit-synthetic-')as tmp,contextlib.ExitStack()as stack:
        root=Path(tmp);oldroot=root/'old';newroot=root/'new';newroot.mkdir()
        run='AMD-UrbanEV-F4-type1-followup-v3-recovery1-f1-h3-s2024';scope='m6-baseline-type1-followup-v3-recovery1-URBAN_SUBSET-probe'
        oldconfig=bound(old.contract()['old_config_refs']['URBAN_SUBSET']);task=next(t for t in oldconfig['tasks']if t['id']==run);p=profile(oldconfig,task);cr=exclusive(root/'old-config.json',oldconfig)
        producer=dict(commit=old.PRODUCER,protocol_sha=digest(oldconfig),successor_scope=scope,authorized_task_ids=[t['id']for t in oldconfig['tasks']])
        exclusive(oldroot/'queue/URBAN_SUBSET/probe-permit.json',producer)
        serial=oldroot/'probe/URBAN_SUBSET/AMD-cross-fold-f1-f2-H3-H12/serial';process=exclusive(serial/'wave-0/process.json',dict(returncodes=[0],failure=None,resource_admission=True))
        out=serial/run;artifacts={}
        for name,value in dict(config=dict(task=run,unified_stage='URBAN_SUBSET',purpose='ch3_probe',protocol_sha=digest(oldconfig),approval=producer),runtime=dict(task=run,error=None),trajectory=dict(id=run,profile_sha=digest(p),steps=6,finite=True),budget=dict(counts=dict(adam=6,backward=6,forward=8))).items():
            f=exclusive(out/(name+'.json'),value);artifacts[f['path']]=f
        row=dict(process=process,task_ids=[run],producer=producer,artifacts=artifacts,self_comparison=dict(passed=True))
        saved=dict(old_config_ref=cr,accepted={run:row})
        report=dict(probe_recovery_ref=old.REUSE_REF,scope=scope,artifacts=artifacts,execution_complete=True,reviewed=False,evidence={'AMD-cross-fold-f1-f6-H3-H12/serial/0':dict(process=process,task_ids=[run])},approval=exclusive(newroot/'approval.json',{}))
        stack.enter_context(patch.object(old,'OLD_RESULT',oldroot));stack.enter_context(patch.object(old,'evidence',return_value=saved));stack.enter_context(patch.object(q,'PROBE_RECOVERY',old))
        yield root,newroot,report,row,saved,stack


class UrbanReadonly(unittest.TestCase):
    def test_accepted_exact_process_manifest_and_scan(self):
        with urban_fixture()as(root,newroot,report,row,saved,stack):
            readonly=old.retained_refs(report);self.assertEqual(readonly[row['process']['path']],row['process'])
            complete=exclusive(newroot/'complete.json',report);m=manifest_projection(report,complete,newroot,process_refs(report));self.assertEqual(m['artifact_count'],6);self.assertTrue(scan_manifest(m,complete,newroot)['integrity_scan_passed'])

    def test_wrong_process_path_even_same_bytes_rejected(self):
        with urban_fixture()as(root,newroot,report,row,saved,stack):
            row['process']=exclusive(root/'unknown/process.json',dict(returncodes=[0],failure=None,resource_admission=True));self.assertRaises(ValueError,old.retained_refs,report)

    def test_wrong_SHA_and_process_tamper_rejected(self):
        with urban_fixture()as(root,newroot,report,row,saved,stack):
            row['process']['sha256']='0'*64;self.assertRaises(ValueError,old.retained_refs,report)
        with urban_fixture()as(root,newroot,report,row,saved,stack):
            Path(row['process']['path']).write_text('{}\n');self.assertRaises(ValueError,old.retained_refs,report)

    def test_wrong_task_and_stage_rejected(self):
        with urban_fixture()as(root,newroot,report,row,saved,stack):
            row['task_ids']=['wrong'];self.assertRaises(ValueError,old.retained_refs,report)
        with urban_fixture()as(root,newroot,report,row,saved,stack):
            report['scope']='foreign-URBAN_SUBSET-probe';self.assertRaises(PermissionError,old.retained_refs,report)

    def test_unknown_producer_and_failed_resource_rejected(self):
        with urban_fixture()as(root,newroot,report,row,saved,stack):
            row['producer']['commit']='f'*40;self.assertRaises(ValueError,old.retained_refs,report)
        with urban_fixture()as(root,newroot,report,row,saved,stack):
            failed=dict(returncodes=[0],failure=None,resource_admission=False);path=Path(row['process']['path']);path.write_text(json.dumps(failed));row['process']=ref(path)
            self.assertRaises(ValueError,old.retained_refs,report)

    def test_symlink_and_outside_extra_reference_rejected(self):
        with urban_fixture()as(root,newroot,report,row,saved,stack):
            path=Path(row['process']['path']);raw=path.read_bytes();path.unlink();target=root/'target';target.write_bytes(raw);path.symlink_to(target);self.assertRaises(ValueError,old.retained_refs,report)
        with urban_fixture()as(root,newroot,report,row,saved,stack):
            outside=exclusive(root/'foreign.json',{});complete=exclusive(newroot/'complete.json',report)
            self.assertRaises(ValueError,manifest_projection,report,complete,newroot,[outside])

    def test_wrong_evidence_attachment_cannot_reuse_registered_process(self):
        with urban_fixture()as(root,newroot,report,row,saved,stack):
            report['evidence']['AMD-cross-fold-f1-f6-H3-H12/serial/0']['task_ids']=['wrong'];self.assertRaises(ValueError,old.retained_refs,report)

    def test_AUTO_AUDIT_real_projection_and_compact_MAC_path(self):
        # This fixture has already-validated numeric completion. The separate
        # full width probe fixture executes the actual numerical validator.
        with urban_fixture()as(root,newroot,report,row,saved,stack):
            c=dict(tasks=[],baseline_unified=dict(numeric_policies={}));report.update(commit='f'*40,protocol_sha=digest(c),code={},environment={},hardware={},budget={})
            report['decisions']={};complete=exclusive(newroot/'complete.json',report)
            stack.enter_context(patch.object(q.s,'context',return_value=dict(probe_root=newroot,control=root/'control',probe_scope=report['scope'])))
            stack.enter_context(patch.object(native,'validate_probe_completion',side_effect=lambda cc,rr:None))
            stack.enter_context(patch.object(q,'stop_check'));stack.enter_context(patch.dict(os.environ,{q.SECRET:'CPU-synthetic-only'}))
            summary=q.audit_probe(c);value=bound(summary);self.assertTrue(value['technical_admission']);m=bound(value['manifest_ref']);self.assertTrue(any(x['path']==row['process']['path']for x in m['rows']));self.assertIn('mac',value)

class WidthPipeline(unittest.TestCase):
    def test_real_probe_numeric_AUTO_AUDIT_permit_and_compact_formal_failure(self):
        from tests.test_m6_event_driven_resources import fixture,forbidden_query
        from tests.test_m6_probe_schema_recovery import trajectory
        from tools.restricted_regression import m5_formal_entry as tool
        from utils import ch3_type1_execution as execution
        from utils.ch3_contract import task_by_id
        with fixture('PATCH_DM4')as(root,c,ctx,partial,hw,stack,queries):
            w.import_ms()
            pr=q.create_permit(c,partial['start_authorization_ref'],True);a=bound(pr);exclusive(ctx['probe_root']/'approval.json',a)
            calls=[]
            def spawn(cfg):
                t=task_by_id(c,cfg['task']);out=Path(cfg['output']);h=(out/'worker.log').open('x')
                p=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(.12)'],stdout=h,stderr=h)
                tr=trajectory(c,t,out,numeric_probe_policy(c,t));tr.update(w.resource_binding(),allocated=None,reserved=None)
                for row in tr['memory']:row.update(allocated=None,reserved=None,resource_mode='exclusive_gpu_event_driven_v1')
                from ch3_runner import memory_growth_review
                tr['affinity']=a['hardware']['cpu_affinity'];tr['memory_review']=memory_growth_review(tr['memory'],'exclusive_gpu_event_driven_v1');exclusive(out/'trajectory.json',tr)
                counts=s.worker_counts(c,t);(out/'budget.json').write_text(json.dumps(dict(counts=counts,by_pid={str(p.pid):counts}))+'\n')
                exclusive(out/'runtime.json',dict(pid=p.pid,task=t['id'],error=None));(out/'audit.jsonl').write_text('{"event":"CPU-synthetic-only"}\n')
                calls.append(t['id']);return p,h
            stack.enter_context(patch.object(tool,'spawn',side_effect=spawn))
            stack.enter_context(patch.object(tool,'read_profiles',return_value=dict(execution=dict(evidence=str(ctx['probe_root'])))))
            stack.enter_context(patch.object(subprocess,'check_output',side_effect=forbidden_query))
            stack.enter_context(patch.object(tool,'gpu_sample',side_effect=AssertionError('runtime sampling forbidden')))
            stack.enter_context(patch.object(tool,'owned_pid_metadata',side_effect=AssertionError('attribution forbidden')))
            report=native.run_probe(c,a);self.assertEqual(len(calls),40);self.assertEqual(len(report['decisions']),5)
            self.assertEqual(report['budget']['actual'],dict(adam=240,backward=240,forward=320));self.assertNotIn('numeric_reference_ref',report)
            audit=stack.enter_context(patch.object(native,'validate_probe_completion',wraps=native.validate_probe_completion))
            summary_ref=q.audit_probe(c);summary=q.validate_summary_light(c,summary_ref);self.assertEqual(audit.call_count,1)
            formal=q.create_permit(c,a['start_authorization_ref'],False,summary_ref);runtime=q.seal_runtime(c,formal);q.validate_runtime(c,runtime,formal)
            formal_calls=[]
            def fail(cfg):
                execution.validate_worker(c,cfg);formal_calls.append(cfg['task']);out=Path(cfg['output']);h=(out/'worker.log').open('x')
                return subprocess.Popen([sys.executable,'-B','-c',"print('RuntimeError: CUDA out of memory',flush=True);raise SystemExit(1)"],stdout=h,stderr=h),h
            with patch.object(tool,'spawn',side_effect=fail):self.assertRaisesRegex(RuntimeError,'unified formal technical/resource failure',execution.run_group,c,bound(formal),'PatchTST',runtime)
            self.assertEqual(len(formal_calls),4);self.assertEqual(audit.call_count,1);self.assertFalse((ctx['control']/'group-PatchTST/complete.json').exists());self.assertEqual(len(queries),1)
            bad=copy.deepcopy(report);bad['decisions'].pop(next(iter(bad['decisions'])));self.assertRaises(ValueError,native.validate_probe_completion,c,bad)

    def test_installed_width_worker_read_config_no_all_stage_rescan(self):
        from utils import ch3_type1_execution as execution
        c=q.configs()['PATCH_DM8'];cfg=dict(unified_stage='PATCH_DM8',type1_scope=w.ID,protocol_file=str(s.file('PATCH_DM8')),protocol_sha=digest(c),probe_schema_recovery_ref=w.REUSE_REF,approval=dict(execution_attempt=w.ATTEMPT))
        class Guard:
            _STATE=True
            @staticmethod
            def require_installed():return cfg
        with patch.dict(sys.modules,{'restricted_io_guard':Guard}),patch.object(q,'configs',side_effect=AssertionError('installed worker cannot scan stages')):
            self.assertEqual(execution.read_config(cfg),c)
        bad=dict(cfg,protocol_sha='wrong');self.assertRaises(PermissionError,execution.read_config,bad)

    def test_full_Urban_native_validator_and_AUTO_AUDIT_without_stubs(self):
        source=bound_source=Path(w.PACKAGE/'urban-full-audit-synthetic-source.final.py')
        self.assertEqual(ref(source)['sha256'],'baf4a0deb0d583979e8fe4ea38726abaf38e7e3925753eeef7f949736441e34a')
        x=subprocess.run([sys.executable,'-B','-c',source.read_text()],cwd=ROOT,capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},timeout=120)
        self.assertEqual(x.returncode,0,x.stderr[-4000:]);v=json.loads(x.stdout)
        self.assertTrue(v['full_Urban_numerical_validator']);self.assertTrue(v['AUTO_AUDIT']);self.assertEqual((v['GPU'],v['model'],v['adam'],v['backward'],v['test']),(0,0,0,0,0))

    def test_real_width_guard_install_binding_and_original_unauthorized_denial(self):
        source=Path(w.PACKAGE/'width-guard-fixture-source-v2.py');self.assertEqual(ref(source)['sha256'],'5398780d4df0907495fa00786c605cc77dba7f8d58b26da731e919ceaa91bf6d')
        with tempfile.TemporaryDirectory(prefix='m6-width-permanent-guard-')as raw:
            x=subprocess.run([sys.executable,'-B','-c',source.read_text(),'good',str(Path(raw)/'fixture')],cwd=ROOT,capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''},timeout=120)
            self.assertEqual(x.returncode,0,x.stderr[-4000:]);v=json.loads(x.stdout)
            self.assertTrue(v['real_guard_installed']);self.assertFalse(v['old_path_in_metadata']);self.assertEqual(v['calls']['all_stage_after_install'],0)

    def test_comparison_real_saved_refs_domain_means_conflict_no_choice(self):
        with tempfile.TemporaryDirectory(prefix='m6-width-comparison-synthetic-')as raw,patch.object(s,'RESULT',Path(raw)),patch.object(q,'CONTROL',Path(raw)/'controller'):
            cs=q.configs();reference=bound(w.contract()['paper_ref'])['PatchTST'];boundaries={}
            for stage,c in cs.items():
                artifacts={};width=int(stage[-1]);ctx=s.context(c)
                for t in c['tasks']:
                    p=profile(c,t);base=reference.get(t['dataset'],dict(mse=1.,mae=1.));result=dict(id=t['id'],commit='f'*40,protocol_sha=digest(c),final_test=dict(calls=1),mse=base['mse']+(.1 if width==4 else .2),mae=base['mae']+(.3 if width==4 else .1));out=ctx['result_root']/'formal-PatchTST'/t['id']
                    artifacts[t['id']]={'result.json':exclusive(out/'result.json',result),'manifest.json':exclusive(out/'manifest.json',dict(task=t,profile=p,identity=dict(commit='f'*40,profile_sha=digest(p)))),'budget.json':exclusive(out/'budget.json',dict(counts=dict(adam=1,backward=1,forward=1)))}
                receipt=exclusive(ctx['control']/'group-PatchTST/complete.json',dict(artifacts=artifacts));boundaries[stage]=exclusive(ctx['control']/'technical-boundary.json',dict(purpose='baseline_type1_'+stage+'_boundary_v1',scope=w.ID,technical_complete=True,result_review='pending',commit='f'*40,protocol_sha=digest(c),task_ids=[t['id']for t in c['tasks']],receipts={'PatchTST':receipt},**w.resource_binding()))
            complete=w.completion_record({q.STAGE_STATES[k][3]:v for k,v in boundaries.items()});exclusive(q.CONTROL/'complete.json',complete)
            report=w.width_results();self.assertIsNone(report['selected_width']);self.assertTrue(report['third_round_paused']);self.assertTrue(report['closer']['ETTh1']['metric_conflict']);self.assertEqual(report['closer']['ETTh1']['by_metric'],dict(mse=4,mae=8));self.assertIsNone(report['candidates']['PATCH_DM4']['Exchange']['paper'])
            self.assertEqual(sum(len(x['rows'])for domains in report['candidates'].values()for x in domains.values()),40)

    def test_final_source_proof_rejects_protected_math_change(self):
        original=w.sha
        with patch.object(w,'sha',side_effect=lambda p:'0'*64 if str(p).endswith('ch3_runner.py')else original(p)):
            self.assertRaisesRegex(ValueError,'frozen computation',w.verify_production_inheritance)


if __name__=='__main__':unittest.main()
