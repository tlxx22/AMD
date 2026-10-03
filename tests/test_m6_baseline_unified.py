"""Exact source/profile and synthetic orchestration regressions, no model imports."""
import ast,copy,json,os,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from utils import ch3_baseline_unified_tasks as s
from utils import ch3_baseline_unified_chain as q
from utils import ch3_baseline_unified_execution as e
from utils.ch3_contract import digest,profile,step_arithmetic,validate_manifest,generate_tasks
from utils.ch3_native_recovery_records import bound,exclusive,ref,sha


class UnifiedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cs=q.configs()
    def test_exact_matrix(self):
        self.assertEqual([len(self.cs[k]['tasks']) for k in ('MS','M')],[203,84])
        ids=[t['id'] for c in self.cs.values() for t in c['tasks']]
        self.assertEqual(len(set(ids)),287)
        for c in self.cs.values():self.assertEqual(generate_tasks(c),c['tasks']);self.assertIs(validate_manifest(c),c)
    def test_domains_and_exclusions(self):
        self.assertEqual(set(self.cs['MS']['datasets']),set(s.MS_DOMAINS));self.assertEqual(set(self.cs['M']['datasets']),set(s.M_DOMAINS))
        for c in self.cs.values():self.assertEqual({t['model'] for t in c['tasks']},set(s.MODELS));self.assertNotIn('ECL',c['datasets'])
    def test_only_authorized_direct_parent_fields_changed(self):
        for stage,c in self.cs.items():
            parent=s.direct_parent(stage)
            for previous,t in zip(parent['tasks'],c['tasks']):
                old=copy.deepcopy(parent['resolved_profiles'][previous['id']]);new=profile(c,t)
                if t['dataset']=='UrbanEV':
                    self.assertEqual((new['training']['epochs'],new['training']['patience'],new['training']['scheduler']['epochs']),(20,5,20))
                    new['training']['epochs']=old['training']['epochs'];new['training']['patience']=old['training']['patience'];new['training']['scheduler']['epochs']=old['training']['scheduler']['epochs']
                self.assertEqual(old,new)
    def test_T_urban_exception_and_onecycle_everywhere(self):
        for c in self.cs.values():
            for t in c['tasks']:
                p=profile(c,t);self.assertEqual(p['T'],12 if t['dataset']=='UrbanEV' else 96)
                self.assertEqual(p['training']['lr'],.01);cfg=p['training']['scheduler'];self.assertEqual(cfg['steps_per_epoch'],step_arithmetic(c,t)['train_batches']);self.assertEqual(cfg['epochs'],p['training']['epochs'])
    def test_numeric_policies_exact_authorized_registry(self):
        for stage,c in self.cs.items():
            for row,t in zip(s.parent_rows(stage),c['tasks']):self.assertEqual(s.numeric_policy(c,t),s.expected_numeric_policy(row))
    def test_M_21_registry(self):self.assertEqual(len(s.probe_groups(self.cs['M'])),21)
    def test_MS_15_minimal_groups_and_63_reps(self):
        groups=s.probe_groups(self.cs['MS']);self.assertEqual(len(groups),15);self.assertEqual(sum(len(g['representatives']) for g in groups),63)
        coverage=[x for g in groups for ids in g['coverage'].values() for x in ids];self.assertEqual(len(coverage),203);self.assertEqual(set(coverage),{t['id'] for t in self.cs['MS']['tasks']})
    def test_compatible_market_group_identity(self):
        c=self.cs['MS']
        for g in s.probe_groups(c):
            if 'EPF' in g['id']:self.assertEqual(len({digest(s.compute_identity(c,next(t for t in c['tasks'] if t['id']==r),True)) for r in g['representatives']}),1)
    def test_new_budgets_proposal(self):
        self.assertEqual(s.formal_budget(self.cs['MS'])['total'],dict(runs=203,run_epochs=4060,adam=15274280,backward=15274280,forward=17173829))
        self.assertEqual(s.formal_budget(self.cs['M'])['total'],dict(runs=84,run_epochs=840,adam=300150,backward=300150,forward=359040))
    def test_probe_budgets(self):
        self.assertEqual(s.probe_budget(self.cs['MS'])['caps'],dict(adam=1122,forward=1520,backward=1122));self.assertEqual(s.probe_budget(self.cs['M'])['caps'],dict(adam=1512,forward=2016,backward=1512))
    def test_fresh_namespaces_no_old_result_reuse(self):
        for c in self.cs.values():
            from utils.ch3_native_tasks import result_path
            for t in c['tasks']:
                self.assertIn(s.PROTOCOL,str(result_path(c,t)));self.assertNotIn('native-time-mark-v4',str(result_path(c,t)));self.assertNotIn('parent_run_id',t)
    def test_J_NS_files_protected(self):
        protected=json.loads((s.PACKAGE/'protected-before.json').read_text())
        # Snapshot structure is checked dynamically against its actual fields below.
        def visit(value):
            if isinstance(value,dict):
                for k,v in value.items():
                    if isinstance(v,str) and len(v)==64 and Path(k).is_absolute():self.assertEqual(sha(k),v)
                    else:visit(v)
            elif isinstance(value,list):
                for v in value:visit(v)
        visit(protected)
    def test_parent_config_frozen(self):
        from utils.ch3_baseline_unified_tasks import catalog
        for row in catalog()['parents'].values():self.assertEqual(sha(row['ref']['path']),row['ref']['sha256'])
    def test_frozen_recipe_source(self):
        r=bound(s.AUTHOR_RECIPE)
        self.assertTrue(r['source_refs'])
        for value in r['source_refs']:self.assertEqual(sha(value['path']),value['sha256'])

    def test_legal_synthetic_start_approval(self):
        a=q.start_template();a.update(reviewed=True,execution_permitted=True,structure_frozen=True,m6_authorized=True,budget_authorized=True,closure_commit='fixture-closure',authorization_basis='fixture explicit approval')
        with patch.object(q,'closure',return_value='fixture-closure'):self.assertEqual(q.validate_start(a),a)
        a['formal_caps']['MS']['adam']+=1
        with patch.object(q,'closure',return_value='fixture-closure'):
            with self.assertRaises(PermissionError):q.validate_start(a)

    def test_missing_token_direct_start_refused(self):
        with self.assertRaises(PermissionError):q.verify_launch(s.RETIREMENT,None)

    def test_stale_log_rejected_matching_current_log_allowed(self):
        with tempfile.TemporaryDirectory(dir=s.PACKAGE,prefix='fixture-launch-') as root:
            log=Path(root)/'launcher.log';log.touch()
            with patch.object(q,'LOG',log),patch.object(q,'closure',return_value='fixture'),patch.object(q,'validate_start'),patch.object(q,'dynamic',return_value={}):
                self.assertTrue(any('launcher' in x for x in q.readiness(None)))
                self.assertFalse(any('launcher' in x for x in q.readiness(None,launch=True)))

    def test_synthetic_sync_child_real_exit(self):
        import sys
        with tempfile.TemporaryDirectory(dir=s.PACKAGE,prefix='fixture-owned-') as root:
            control=Path(root);real=subprocess.Popen;called=[]
            def fixture_process(args,**kwargs):
                called.append(args);return real([sys.executable,'-c','import time; time.sleep(.15)'],**kwargs)
            with patch.object(q,'CONTROL',control),patch.object(q.subprocess,'Popen',side_effect=fixture_process):
                result=q.wait_owned('MS',s.RETIREMENT,True)
                self.assertEqual(result['exit_code'],0);self.assertFalse(q.same(result['child']));self.assertEqual(len(called),1)

    def test_group_child_failure_never_advances(self):
        import sys
        with tempfile.TemporaryDirectory(dir=s.PACKAGE,prefix='fixture-owned-') as root:
            real=subprocess.Popen
            with patch.object(q,'CONTROL',Path(root)),patch.object(q.subprocess,'Popen',side_effect=lambda args,**kw:real([sys.executable,'-c','raise SystemExit(7)'],**kw)):
                with self.assertRaises(RuntimeError):q.wait_owned('MS',s.RETIREMENT,True)

    def test_actual_synthetic_chain_permit_runtime_and_model_order(self):
        with tempfile.TemporaryDirectory(dir=s.PACKAGE,prefix='fixture-chain-') as root:
            root=Path(root);calls=[];cs=self.cs
            def context(c):
                stage=c['baseline_unified']['stage'];return dict(stage=stage,control=root/stage,probe_root=root/('probe-'+stage))
            def permit(c,start_ref,probe,summary_ref=None,boundary_ref=None):
                stage=c['baseline_unified']['stage'];calls.append(('permit',stage,probe));return exclusive(root/(stage+('-probe' if probe else '-formal')+'.json'),{'fixture':True})
            def wait(stage,value,probe,model=None,runtime_ref=None):
                calls.append(('child',stage,probe,model))
                if not probe:exclusive(root/stage/('group-'+model)/'complete.json',{'fixture':True})
            def seal(c,receipts):return exclusive(root/(c['baseline_unified']['stage']+'-boundary.json'),{'fixture':True})
            def audit(c):calls.append(('audit',c['baseline_unified']['stage']));return s.RETIREMENT
            with patch.object(q,'CONTROL',root),patch.object(s,'context',side_effect=context),patch.object(q,'validate_start'),patch.object(q,'bound',return_value={}),patch.object(q,'create_permit',side_effect=permit),patch.object(q,'audit_probe',side_effect=audit),patch.object(q,'seal_runtime',return_value=s.RETIREMENT),patch.object(q,'wait_owned',side_effect=wait),patch.object(q,'seal_boundary',side_effect=seal):q.run(s.RETIREMENT)
            self.assertTrue((root/'complete.json').exists())
            for stage in ('MS','M'):self.assertEqual([v[3] for v in calls if v[:3]==('child',stage,False)],list(s.MODELS))
            self.assertEqual([v for v in calls if v[0]=='audit'],[('audit','MS'),('audit','M')])

    def test_resource_fallback_never_numeric(self):
        from utils.ch3_m_execution import resource_fallback
        self.assertFalse(resource_fallback(dict(failure='numeric gate failure',returncodes=[0],resource=True)))

    def test_scheduler_comparison_exact(self):
        from utils import ch3_native_execution as native
        c=self.cs['M'];t=c['tasks'][0];cfg=profile(c,t)['training']['scheduler']
        trace=[dict(config=cfg,config_sha=digest(cfg),updates=i,scheduler=dict(last_epoch=i,_step_count=i+1,total_steps=cfg['epochs']*cfg['steps_per_epoch'])) for i in range(1,7)]
        x={'scheduler_trace':trace};y=copy.deepcopy(x);y['scheduler_trace'][3]['scheduler']['last_epoch']=99
        with patch('ch3_runner.compare_probe_trajectories',return_value={'passed':True}):
            with self.assertRaises(ValueError):native.compare(c,t,x,y)

    def test_summary_excludes_J_and_old_protocol(self):
        c=copy.deepcopy(self.cs['MS']);c['tasks']=c['tasks'][:1];t=c['tasks'][0]
        with tempfile.TemporaryDirectory(dir=s.PACKAGE,prefix='fixture-summary-') as root:
            ctx={'result_root':Path(root)};path=Path(root)/('formal-'+t['model'])/t['id']/'result.json';path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(scientific_protocol='legacy',id=t['id'])))
            with patch.object(s,'context',return_value=ctx):
                with self.assertRaises(ValueError):s.result_index({'MS':c})
        index=s.result_index(self.cs);self.assertEqual(index['missing'],287);self.assertFalse(any(r['model']in ('J','N','S') for r in index['rows']))

    def test_signal_handler_does_not_signal_itself(self):
        import inspect
        self.assertIn('safe_stop(signal_supervisor=False)',inspect.getsource(q.start))

    def test_full_float_payload_not_compared_twice(self):
        from utils import ch3_native_execution as native
        c=self.cs['M'];t=next(t for t in c['tasks'] if t['model']=='TimeMixer');cfg=profile(c,t)['training']['scheduler']
        root=s.context(c)['probe_root'];points=[dict(schema_file=str(root/'schema.json'),data_file=str(root/('step'+str(i)+'.bin'))) for i in range(6)]
        x=dict(M_full_state_trace=points,full_numeric_trace=points,scheduler_trace=[dict(config=cfg,config_sha=digest(cfg),updates=i,scheduler=dict(last_epoch=i,_step_count=i+1,total_steps=cfg['epochs']*cfg['steps_per_epoch'])) for i in range(1,7)])
        with patch('ch3_runner.compare_probe_trajectories',return_value={'passed':True}) as generic,patch('ch3_runner._compare_full_numeric_files') as duplicate:
            self.assertTrue(native.compare(c,t,x,x)['passed']);generic.assert_called_once();duplicate.assert_not_called()

    def test_None_policy_full_state_exact_still_compared(self):
        from utils import ch3_native_execution as native
        c=self.cs['M'];t=c['tasks'][0];cfg=profile(c,t)['training']['scheduler'];root=s.context(c)['probe_root']
        x=dict(M_full_state_trace=[dict(schema_file=str(root/'schema.json'),data_file=str(root/('step'+str(i)+'.bin'))) for i in range(6)],scheduler_trace=[dict(config=cfg,config_sha=digest(cfg),updates=i,scheduler=dict(last_epoch=i,_step_count=i+1,total_steps=cfg['epochs']*cfg['steps_per_epoch'])) for i in range(1,7)])
        with patch('ch3_runner.compare_probe_trajectories',return_value={'passed':True}),patch('ch3_runner._compare_full_numeric_files',return_value={'passed':True}) as full:
            self.assertTrue(native.compare(c,t,x,x)['passed']);self.assertEqual(full.call_count,6)
            self.assertTrue(all(call.args[0]=={'state_atol':0} for call in full.call_args_list))

    def test_runtime_scan_once_children_light_and_tamper(self):
        c=self.cs['MS'];a=dict(summary_ref=s.RETIREMENT,manifest_ref=s.RETIREMENT,commit='fixture',code={})
        with tempfile.TemporaryDirectory(dir=s.PACKAGE,prefix='fixture-runtime-') as root:
            root=Path(root);exclusive(root/'controller.json',{'owner':q.owner()})
            with patch.object(q,'CONTROL',root),patch.object(s,'context',return_value={'control':root,'stage':'MS','probe_root':root}),patch.object(q,'validate_permit',return_value=a),patch.object(q,'validate_summary_light',return_value={'complete_ref':s.RETIREMENT}),patch.object(q,'scan_manifest') as scan,patch.dict(os.environ,{q.SECRET:'fixture-secret'}):
                r=q.seal_runtime(c,s.RETIREMENT)
                for _ in range(5):q.validate_runtime(c,r,s.RETIREMENT)
                scan.assert_called_once()
                v=json.loads(Path(r['path']).read_text());v['integrity_scan_passed']=False;Path(r['path']).write_text(json.dumps(v))
                with self.assertRaises(PermissionError):q.validate_runtime(c,ref(r['path']),s.RETIREMENT)

    def test_actual_manifest_tamper_rejected(self):
        from utils.ch3_native_recovery_records import manifest_projection,scan_manifest
        with tempfile.TemporaryDirectory(dir=s.PACKAGE,prefix='fixture-manifest-') as root:
            root=Path(root);p=root/'numeric.bin';p.write_bytes(b'fixture');r=ref(p);complete={'artifacts':{str(p):r}}
            manifest=manifest_projection(complete,s.RETIREMENT,root);scan_manifest(manifest,s.RETIREMENT,root)
            p.write_bytes(b'tampered')
            with self.assertRaises(ValueError):scan_manifest(manifest,s.RETIREMENT,root)

    def test_owned_summary_positive_and_tamper_rejected(self):
        import hmac,hashlib
        c=self.cs['MS'];groups=s.probe_groups(c)
        with tempfile.TemporaryDirectory(dir=s.PACKAGE,prefix='fixture-summary-') as root:
            root=Path(root);ctx=dict(control=root,probe_root=root/'probe',stage='MS',probe_scope=s.ID+'-MS-probe',caps=s.probe_budget(c)['caps'])
            exclusive(root/'controller.json',{'owner':q.owner()})
            value=dict(purpose='baseline_unified_technical_admission_v1',scope=ctx['probe_scope'],owner=q.owner(),technical_admission=True,manual_review=False,reviewed=False,result_review='pending',
                complete_ref={'path':str(ctx['probe_root']/'complete.json'),'sha256':'0'*64},manifest_ref={'path':str(root/'probe-artifact-manifest.json'),'sha256':'0'*64},protocol_sha=digest(c),policy_sha=digest(c['baseline_unified']['numeric_policies']),profile_shas={t['id']:digest(profile(c,t)) for t in c['tasks']},
                decisions={g['id']:{'status':'Passed','concurrency':g['planned_q'],'coverage':g['coverage']} for g in groups},budget={'caps':ctx['caps'],'refund':False,'actual':dict.fromkeys(ctx['caps'],0),'reserved':dict.fromkeys(ctx['caps'],0)})
            value['mac']=hmac.new(b'fixture-secret',digest(value).encode(),hashlib.sha256).hexdigest();r=exclusive(root/'admission-summary.json',value)
            with patch.object(q,'CONTROL',root),patch.object(s,'context',return_value=ctx),patch.dict(os.environ,{q.SECRET:'fixture-secret'}):
                self.assertTrue(q.validate_summary_light(c,r)['technical_admission'])
                value['decisions'][groups[0]['id']]['concurrency']=1;Path(r['path']).write_text(json.dumps(value))
                with self.assertRaises(PermissionError):q.validate_summary_light(c,ref(r['path']))

    def test_machine_template_execution_false_refused(self):
        c=self.cs['MS'];ctx=s.context(c)
        a=dict(purpose='baseline_unified_probe_permit_v1',unified_scope=s.ID,successor_scope=ctx['probe_scope'],execution_permitted=False,manual_review=False,reviewed=False,review_mode='preauthorized_machine_gate',start_authorization_ref=s.RETIREMENT)
        with patch.object(q,'validate_start',return_value={}),patch.object(q,'dynamic',return_value={}):
            with self.assertRaises(PermissionError):q.validate_permit(c,a,True)
    def test_order_and_wave_no_backfill(self):
        for c in self.cs.values():
            plan=s.plan(c);self.assertEqual(plan['models'],list(s.MODELS))
            self.assertEqual([t['model'] for t in c['tasks']][0],'AMD')
            for model,waves in plan['formal_waves'].items():
                flattened=[r for wave in waves for r in wave];self.assertEqual(set(flattened),{t['id'] for t in c['tasks'] if t['model']==model})
    def test_unmeasured_q_rejected(self):
        c=self.cs['MS'];decisions={g['id']:dict(status='Passed',concurrency=4) for g in s.probe_groups(c)}
        with self.assertRaises(ValueError):s.formal_waves(c,dict(decisions=decisions),'TimeXer')
    def test_template_rejected(self):
        with patch.object(q,'closure',return_value='fixture-closure'):
            with self.assertRaises(PermissionError):q.validate_start(q.start_template())
    def test_wrong_task_or_parameter_rejected(self):
        c=copy.deepcopy(self.cs['MS']);c['resolved_profiles'][c['tasks'][0]['id']]['structure']['k']=99
        with self.assertRaises(ValueError):s.validate(c)
    def test_historical_retirement_no_refund(self):
        r=bound(s.RETIREMENT);self.assertFalse(r['budget_refund']);self.assertFalse(r['scientific_failure'])
    def test_synthetic_MS_then_M_complete(self):
        calls=[];actions={state:(lambda receipts,k=state:calls.append(k) or dict(technical_complete=True,effect='poor')) for state in q.STATES if state!='COMPLETE'}
        values=q.drive(actions,lambda state:None,lambda:None)
        self.assertEqual(calls,list(q.STATES[:-1]));self.assertEqual(len(values),8)
    def test_technical_failure_stops_later_stages(self):
        calls=[]
        def fail(r):calls.append('failed');raise RuntimeError('numeric failure')
        actions={state:(lambda r,k=state:calls.append(k)) for state in q.STATES[:-1]};actions['MS_AUTO_AUDIT']=fail
        with self.assertRaises(RuntimeError):q.drive(actions,lambda state:None,lambda:None)
        self.assertNotIn('MS_FORMAL_ALL_BASELINES',calls);self.assertNotIn('M_RESOURCE_NUMERIC_PROBE',calls)
    def test_no_effect_cancel(self):
        calls=[];actions={state:(lambda r,k=state:calls.append(k) or {'mse':99999999}) for state in q.STATES[:-1]}
        q.drive(actions,lambda state:None,lambda:None);self.assertIn('M_FORMAL_ALL_BASELINES',calls)
    def test_STOP_between_stages(self):
        count=[0];calls=[]
        def check():
            count[0]+=1
            if count[0]==4:raise InterruptedError('STOP')
        actions={state:(lambda r,k=state:calls.append(k)) for state in q.STATES[:-1]}
        with self.assertRaises(InterruptedError):q.drive(actions,lambda state:None,check)
        self.assertNotIn('MS_AUTO_AUDIT',calls)
    def test_bound_record_tamper_rejected(self):
        with tempfile.TemporaryDirectory(dir=s.PACKAGE,prefix='fixture-') as root:
            p=Path(root)/'record.json';r=exclusive(p,dict(x=1));p.write_text('{}')
            with self.assertRaises(ValueError):bound(r)
    def test_exclusive_different_repeat_refused(self):
        with tempfile.TemporaryDirectory(dir=s.PACKAGE,prefix='fixture-') as root:
            p=Path(root)/'record.json';exclusive(p,dict(x=1))
            with self.assertRaises(FileExistsError):exclusive(p,dict(x=2))
    def test_no_full_probe_audit_in_formal_light_paths(self):
        for f in (q.validate_permit,q.validate_summary_light,q.validate_runtime,e.make_config,e.validate_worker,e.validate_wave,e.run_group):
            import inspect
            self.assertNotIn('validate_probe_completion(',inspect.getsource(f));self.assertNotIn('_compare_full_numeric_files(',inspect.getsource(f))
    def test_worker_raw_sidecar_excluded(self):
        c=self.cs['MS'];a=dict(start_authorization_ref=s.RETIREMENT)
        files=e.metadata_files(c,a)
        self.assertFalse(any('.bin' in p or 'trajectory.json'in p for p in files))
    def test_update_evaluate_model_data_math_AST_unchanged(self):
        old=ast.parse(subprocess.check_output(['git','show',s.BASE+':ch3_runner.py'],cwd=s.ROOT,text=True));new=ast.parse((s.ROOT/'ch3_runner.py').read_text())
        def functions(tree):return {x.name:ast.dump(x,include_attributes=False) for x in tree.body if isinstance(x,ast.FunctionDef)}
        a,b=functions(old),functions(new)
        for name in ('update','evaluate','init_training'):self.assertEqual(a[name],b[name])
        self.assertEqual(sha(s.ROOT/'models/ch3_adapter.py'),__import__('hashlib').sha256(subprocess.check_output(['git','show',s.BASE+':models/ch3_adapter.py'],cwd=s.ROOT)).hexdigest())
    def test_scheduler_only_successful_update_and_test_once_AST(self):
        tree=ast.parse((s.ROOT/'ch3_runner.py').read_text());fn=next(f for f in tree.body if isinstance(f,ast.FunctionDef) and f.name=='formal_worker')
        calls=[ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n,ast.Call)]
        self.assertEqual(calls.count('scheduler.after_successful_update'),1)
        self.assertEqual(sum('ch3_formal_test' in ast.unparse(n) for n in ast.walk(fn) if isinstance(n,ast.Dict)),1)
    def test_new_checkpoint_schema_old_compatible(self):
        import inspect,ch3_runner
        self.assertIn('ch3-state-v2-onecycle',inspect.getsource(ch3_runner.save_state));self.assertIn('ch3-state-v1',inspect.getsource(ch3_runner.restore_state))


if __name__=='__main__':unittest.main()
