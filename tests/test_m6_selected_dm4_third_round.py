"""CPU source/config/adoption/lifecycle checks; no model, GPU, or test access."""
import contextlib,copy,json,os,sys,tempfile,unittest,subprocess
from pathlib import Path
from unittest.mock import patch
from utils import ch3_third_round_after_selection as b,ch3_type1_chain as q,ch3_type1_tasks as s
from utils.ch3_contract import digest,profile
from utils.ch3_native_recovery_records import bound,ref,exclusive,manifest_projection,scan_manifest
b.activate()


class SelectedDM4(unittest.TestCase):
    def test_source40_complete_counts_checksums_test_once(self):
        a=bound(b.contract()['width_source_audit_ref'])
        self.assertTrue(a['technical_complete']);self.assertTrue(a['python_system_checksum_match']);self.assertEqual(a['artifact_files'],280)
        self.assertEqual(a['actual_counts'],dict(adam=55840,backward=55840,forward=75292));self.assertEqual(a['actual_run_epochs'],400)
        self.assertEqual((a['new_test_calls'],a['checkpoint_deserializations'],a['formal_test_calls']),(0,0,40))
        self.assertEqual([len(a['stages'][x]['rows'])for x in('PATCH_DM4','PATCH_DM8')],[20,20])

    def test_371_exact20_replaced_351_identical(self):
        v=b.selected_main();old=bound(v['original_fixed_enc2_main_ref'])
        changes=[(x,y)for x,y in zip(old['cells'],v['cells'])if x!=y]
        self.assertEqual(len(changes),20);self.assertEqual(len({x['cell_id']for x in v['cells']}),371)
        for x,y in changes:self.assertEqual((x['model'],x['task']),( 'PatchTST','M'));self.assertNotEqual(x['dataset'],'Weather');self.assertEqual(y['d_model'],4);self.assertEqual(y['supersedes'],x['result_ref'])

    def test_reject_dm8_dm16_mixed_unknown_refs_and_Weather_modification(self):
        good=b.selected_main();i=next(i for i,x in enumerate(good['cells'])if x.get('d_model')==4)
        for width in(8,16):
            v=copy.deepcopy(good);v['cells'][i]['d_model']=width;self.assertRaises(ValueError,b.selected_main,v)
        for field,value in(('result_ref',b.SOURCE_REF),('profile_sha','0'*64),('run_id','wrong'),('cell_id','duplicate')):
            v=copy.deepcopy(good);v['cells'][i][field]=value;self.assertRaises(ValueError,b.selected_main,v)
        v=copy.deepcopy(good);next(x for x in v['cells']if x['dataset']=='Weather')['mse']=0;self.assertRaises(ValueError,b.selected_main,v)

    def test_M_ALL20_width_and148_full_task_profile_unchanged(self):
        c=q.configs()['M_ALL'];old=bound(b.contract()['parent_M_ALL_ref']);changes=0
        for before,t in zip(old['tasks'],c['tasks']):
            p=profile(old,before);actual=profile(c,t)
            if before['model']=='PatchTST'and before['dataset']!='Weather':
                p['structure']['d_model']=4;changes+=1;self.assertEqual(actual,p);self.assertNotEqual(t['id'],before['id']);self.assertEqual(actual['training']['scheduler']['name'],'type1_horizon_scaled_v1')
            else:self.assertEqual(t,before);self.assertEqual(actual,p)
        self.assertEqual(changes,20);self.assertEqual(len(c['tasks']),168)

    def test_Weather128_whole_profile_metadata_identity_retained(self):
        c=q.configs()['M_ALL'];old=bound(b.contract()['parent_M_ALL_ref']);data=bound(c['baseline_unified']['data_ref']);prior=bound(old['baseline_unified']['data_ref'])
        ts=[t for t in old['tasks']if t['model']=='PatchTST'and t['dataset']=='Weather'];self.assertEqual(len(ts),4)
        for t in ts:
            p=profile(c,t);self.assertEqual(p,profile(old,t));self.assertEqual([p['structure'][k]for k in('d_model','n_heads','e_layers','d_ff')],[128,16,2,256]);self.assertEqual((p['training']['epochs'],p['training']['patience']),(20,10));self.assertEqual(data['metadata']['Weather'][t['id']],prior['metadata']['Weather'][t['id']]);self.assertEqual(data['data_bindings']['Weather'][t['id']],prior['data_bindings']['Weather'][t['id']])

    def test_config_group_task_references_follow_only20_new_identities(self):
        c=q.configs()['M_ALL'];old=bound(b.contract()['parent_M_ALL_ref']);ids={t['id']for t in c['tasks']}
        for g,prior in zip(c['groups'],old['groups']):
            self.assertTrue(set(g['task_ids'])<=ids);self.assertTrue(set(g['representatives'])<=ids)
            self.assertTrue(all(set(wave)<=ids for wave in g['waves']))
            if g['model']!='PatchTST'or g['dataset']=='Weather':self.assertEqual(g,prior)

    def test_new_config_rejects_heads_scheduler_numeric_or_extra_task(self):
        good=q.configs()['M_ALL'];tid=next(t['id']for t in good['tasks']if t.get('variant')=='encoder-2-d_model-4-user-selected')
        for kind in('width','heads','scheduler','numeric','extra'):
            v=copy.deepcopy(good)
            if kind=='width':v['resolved_profiles'][tid]['structure']['d_model']=8
            elif kind=='heads':v['resolved_profiles'][tid]['structure']['n_heads']=1
            elif kind=='scheduler':v['resolved_profiles'][tid]['training']['scheduler']['name']='OneCycleLR'
            elif kind=='numeric':next(iter(v['baseline_unified']['numeric_policies'].values()))['state_atol']=1
            else:v['tasks'].append(v['tasks'][0])
            self.assertRaises(ValueError,b.validate_revision,v)

    def test_summary_source_slots_remain_exact168_without_duplicate_ETT_expansion(self):
        c=q.configs()['M_ALL'];prior=s.selected('M_ALL');self.assertEqual(len(prior),168)
        for x,y in zip(prior,c['tasks']):self.assertEqual([x[k]for k in('model','dataset','fold','h','seed')],[y[k]for k in('model','dataset','fold','h','seed')])

    def test_second_onecycle_third_type1_separated(self):
        a=bound(b.contract()['width_source_audit_ref'])
        self.assertTrue(all(x['profile']['training']['scheduler']['name']=='OneCycleLR'for x in a['stages']['PATCH_DM4']['rows']))
        c=q.configs()['M_ALL'];self.assertTrue(all(profile(c,t)['training']['scheduler']['name']=='type1_horizon_scaled_v1'for t in c['tasks']))

    def test_new_shape_has_own_serial_q4_q2_q1_policy(self):
        c=q.configs()['M_ALL'];g=[g for g in s.probe_groups(c)if g['model']=='PatchTST'];self.assertEqual(len(g),6)
        self.assertEqual(sum(len(x['representatives'])for x in g),24)
        self.assertTrue(all(x['planned_q']==4 for x in g));from utils.ch3_native_tasks import attempt_widths;self.assertEqual(attempt_widths(g[0]),(4,2)) # q1 consumes measured serial
        self.assertEqual(b.SEED_STAGES,());self.assertEqual(b.COMPLETED_STAGES,())

    def test_B_rejects_A_old_attempt_and_old_MAC(self):
        c=q.configs()['M_ALL']
        for bad in(dict(execution_attempt='PatchTST-enc2-width4-8-reproduction-r1'),dict(execution_attempt=b.ATTEMPT,probe_recovery_ref=b.REUSE_REF,base287_boundary_ref=b.SOURCE_REF)):
            self.assertRaises(PermissionError,b.validate_permit_link,c,bad,True)
        self.assertRaises((ValueError,PermissionError),q.validate_start,None)

    def test_exact_Urban_registry_and_wrong_stage_scope_rejected(self):
        r=dict(scope=b.ID+'-URBAN_SUBSET-probe',probe_recovery_ref=b.REUSE_REF)
        refs=b.retained_refs(r);self.assertGreater(len(refs),10)
        self.assertIn('fc56e22a37ae93fe3f2ce351d55ed8824739291d20d1d2ff8fd73770a73a23d5',{x['sha256']for x in refs.values()})
        self.assertRaises(PermissionError,b.retained_refs,dict(r,scope='foreign-URBAN_SUBSET-probe'))
        self.assertRaises(PermissionError,b.retained_refs,dict(r,probe_recovery_ref=b.SOURCE_REF))
        self.assertEqual(b.retained_refs(dict(r,scope=b.ID+'-EPF_ALL-probe')), {})

    def test_real_manifest_and_scan_exact_refs_SHA_and_symlink(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-m6-selected-manifest-')as raw:
            p=Path(raw);root=p/'probe';root.mkdir();outside=p/'registered-process.json';outside.write_text('{}\n');value=ref(outside)
            report=dict(scope=b.ID+'-URBAN_SUBSET-probe',probe_recovery_ref=b.REUSE_REF,artifacts={str(outside):value})
            complete=exclusive(root/'complete.json',{'synthetic':True});self.assertRaises(ValueError,manifest_projection,report,complete,root)
            with patch.object(b,'retained_refs',return_value={str(outside):value}):
                m=manifest_projection(report,complete,root);self.assertTrue(scan_manifest(m,complete,root)['integrity_scan_passed'])
                outside.write_text('{"tampered":true}\n');self.assertRaises(ValueError,scan_manifest,m,complete,root)
                report['artifacts'][str(outside)]=ref(outside);bad=dict(report,artifacts={str(outside):dict(value,sha256='0'*64)});self.assertRaises(ValueError,manifest_projection,bad,complete,root)
                link=p/'link.json';link.symlink_to(outside);report['artifacts']={str(link):ref(link)};self.assertRaises(ValueError,manifest_projection,report,complete,root)

    def test_registry_cannot_expand_to_unknown_producer_or_stage(self):
        original=b.bound;r=dict(scope=b.ID+'-URBAN_SUBSET-probe',probe_recovery_ref=b.REUSE_REF);spec=b.contract()
        for key,value in(('producer_commit','0'*40),('stage','M_ALL'),('purpose','forged')):
            bad=original(spec['urban_readonly_ref']);bad[key]=value
            def read(x):return bad if x==spec['urban_readonly_ref']else original(x)
            with patch.object(b,'bound',side_effect=read):self.assertRaises(ValueError,b.retained_refs,r)

    def test_drive287_order_no_A_depth_or_old_prefix_workers(self):
        events=[];actions={x:lambda r,x=x:events.append(x)or {'synthetic':True}for x in q.STATES if x!='COMPLETE'}
        receipts=q.drive(actions,lambda _:None,lambda:None);v=b.completion_record(receipts)
        self.assertEqual(v['executed_third_formal_runs'],287);self.assertEqual(set(v['stage_boundaries']),set(b.STAGES))
        self.assertFalse(any('PATCH_ENC' in x or 'PATCH_DM' in x or 'M_BASE' in x or 'M_AMEND' in x for x in events))
        self.assertEqual([x for x in events if x.endswith('_FORMAL')],['URBAN_SUBSET_FORMAL','EPF_ALL_FORMAL','M_ALL_FORMAL'])

    def test_failure_STOP_numeric_budget_block_following_stages(self):
        for exc in(ValueError('numeric finite/budget'),InterruptedError('STOP'),RuntimeError('worker failure')):
            events=[]
            def fail(r):events.append('first');raise exc
            actions={x:fail for x in q.STATES if x!='COMPLETE'}
            self.assertRaises(type(exc),q.drive,actions,lambda _:None,lambda:None);self.assertEqual(events,['first'])

    def test_actual_shared_run_287_and_Urban_no_probe_redispatch(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-m6-B-driver-')as raw,contextlib.ExitStack()as stack:
            root=Path(raw);stack.enter_context(patch.object(s,'RESULT',root/'results'));stack.enter_context(patch.object(q,'CONTROL',root/'controller'));q.CONTROL.mkdir()
            start=exclusive(root/'synthetic-start.json',{'synthetic':True});events=[];runs=[]
            stack.enter_context(patch.object(q,'validate_start',side_effect=lambda v:v));stack.enter_context(patch.object(q,'stop_check'))
            stack.enter_context(patch.object(b,'import_ms',side_effect=lambda:exclusive(root/'synthetic-adoption.json',{'synthetic':True})))
            stack.enter_context(patch.object(b,'adopt_urban_probe',side_effect=lambda r:events.append(('URBAN_SUBSET','adopt_saved_probe'))or exclusive(root/'synthetic-urban-adoption.json',{'synthetic':True})))
            def permit(c,start_ref,probe,*args,**kwargs):
                stage=c['baseline_unified']['stage'];events.append((stage,'probe'if probe else'formal'))
                return exclusive(root/(stage+('-probe'if probe else'-formal')+'.json'),{'synthetic':True,'stage':stage})
            def wait(stage,value,probe,model=None,runtime=None):
                if not probe:
                    runs.extend(t['id']for t in q.configs()[stage]['tasks']if t['model']==model)
                    exclusive(s.context(q.configs()[stage])['control']/('group-'+model)/'complete.json',{'synthetic':True})
            stack.enter_context(patch.object(q,'create_permit',side_effect=permit));stack.enter_context(patch.object(q,'wait_owned',side_effect=wait))
            stack.enter_context(patch.object(q,'audit_probe',side_effect=lambda c:exclusive(root/(c['baseline_unified']['stage']+'-audit.json'),{'synthetic':True})))
            stack.enter_context(patch.object(q,'seal_runtime',side_effect=lambda c,value:value));stack.enter_context(patch.object(q,'seal_boundary',side_effect=lambda c,r:exclusive(root/(c['baseline_unified']['stage']+'-boundary.json'),{'synthetic':True})))
            q.run(start);complete=bound(ref(q.CONTROL/'complete.json'))
            self.assertEqual(len(runs),287);self.assertEqual(len(set(runs)),287)
            self.assertEqual(events,[('URBAN_SUBSET','adopt_saved_probe'),('URBAN_SUBSET','formal'),('EPF_ALL','probe'),('EPF_ALL','formal'),('M_ALL','probe'),('M_ALL','formal')])
            self.assertEqual(complete['executed_third_formal_runs'],287);self.assertEqual(complete['adopted_depth_formal_runs'],48)

    def test_new_handoff_MAC_bound_to_current_lifecycle(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-m6-B-handoff-')as raw,patch.object(q,'CONTROL',Path(raw)),patch.object(q,'closure',return_value='f'*40),patch.dict(os.environ,{q.SECRET:'synthetic-B-only-secret'}),patch.object(b,'verify_source',return_value=b.contract()):
            exclusive(q.CONTROL/'controller.json',dict(owner=q.owner(),synthetic=True))
            v=b.import_ms();self.assertEqual(q.validate_upstream_boundary_light(v)['anchors'],b.anchors())
            with patch.dict(os.environ,{q.SECRET:'different-owner-secret'}):self.assertRaises(PermissionError,q.validate_upstream_boundary_light,v)

    def test_real_B_worker_guard_and_resource_binding_before_model(self):
        raw=tempfile.mkdtemp(prefix='synthetic-B-guard-',dir=b.PACKAGE/'fixtures')
        # Retain the final installed-guard fixture and access log for byte review.
        with contextlib.nullcontext(raw)as raw:
            root=Path(raw);r=subprocess.run([sys.executable,'-B',str(b.SCIENCE_PACKAGE/'cpu-worker-binding-fixture.py'),str(root/'run')],cwd=b.ROOT,capture_output=True,text=True,timeout=120,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''})
            (root/'stdout.txt').write_text(r.stdout);(root/'stderr.txt').write_text(r.stderr);(root/'exit.json').write_text(json.dumps(dict(synthetic=True,exit_code=r.returncode))+'\n')
            self.assertEqual(r.returncode,0,r.stdout+r.stderr);v=json.loads(r.stdout)
            self.assertTrue(v['real_guard_installed']);self.assertTrue(v['synthetic_lifecycle']);self.assertEqual(v['calls']['all_stage_after_install'],0);self.assertEqual(v['calls']['binding'],1);self.assertEqual(v['calls']['synthetic_card_samples'],0);self.assertFalse(v['old_path_in_metadata'])
            self.assertEqual([v[k]for k in('real_model','GPU','forward','backward','Adam','validation_test','checkpoint_load')],[0]*7)

    def test_real_saved_Urban_AUTO_AUDIT_integration_evidence(self):
        v=bound(ref(b.SCIENCE_PACKAGE/'urban-B-auto-audit-integration.json'))
        self.assertTrue(v['synthetic_lifecycle']);self.assertEqual(v['files'],2843);self.assertEqual(v['new_gpu_queries'],0)
        self.assertIn('native.validate_probe_completion',v['actual_functions']);self.assertIn('chain.audit_probe',v['actual_functions'])
        self.assertEqual(v['real_saved_numeric_resource_source'],bound(b.contract()['urban_audit_ref'])['source_complete'])

    def test_complete_pending_is_not_result_review_Passed(self):
        self.assertRaises(ValueError,b.validate_complete,dict(scope=b.ID,technical_complete=True,result_review='Passed',executed_third_formal_runs=287))
        self.assertFalse(b.RESULT.exists());self.assertFalse(b.LOG.exists());self.assertFalse((b.PACKAGE/'start-review.json').exists())


if __name__=='__main__':unittest.main()
