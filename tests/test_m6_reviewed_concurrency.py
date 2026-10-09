"""No-model CPU fixtures for the real reviewed-plan permit/runtime/formal gates."""
import contextlib,copy,hashlib,json,math,os,runpy,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from utils import ch3_third_round_after_selection as b,ch3_type1_chain as q,ch3_type1_tasks as s
from utils import ch3_type1_execution as x,ch3_reviewed_concurrency as concurrency,ch3_event_resources as event
from utils.ch3_contract import digest,profile,step_arithmetic
from utils.ch3_native_recovery_records import bound,ref,exclusive
from tools.restricted_regression import m5_formal_entry as tool,resource_budget
b.activate()
REAL_QUERY=subprocess.check_output


def no_gpu(args,*a,**kw):
    if 'nvidia-smi'in str(args)or'nvml'in str(args).lower():raise AssertionError('runtime GPU query forbidden')
    return REAL_QUERY(args,*a,**kw)


def startup_sample(args,**kw):
    if args[0]!='nvidia-smi':return REAL_QUERY(args,**kw)
    assert kw['timeout']==10
    gpu=event.frozen_hardware()['gpu'];total=float(gpu.split(',')[2]);return gpu+f', 1, {total-1}, 0\n'


@contextlib.contextmanager
def fixture(width=None,root=None,adopt_source=True):
    root=Path(root)if root else Path(tempfile.mkdtemp(prefix='CPU-reviewed-',dir=b.PACKAGE/'fixtures'))
    root.mkdir(exist_ok=True);(root/'SYNTHETIC_ONLY.json').write_text('{"synthetic":true,"GPU":0,"training":0,"real_permission":false}\n')
    cs=q.configs();prior_context=s.context
    with contextlib.ExitStack()as stack:
        stack.enter_context(patch.object(q,'configs',return_value=cs))
        stack.enter_context(patch.object(s,'RESULT',root/'synthetic-result'))
        stack.enter_context(patch.object(q,'CONTROL',root/'controller'))
        stack.enter_context(patch.object(q,'LOG',root/'synthetic-launcher.log'))
        stack.enter_context(patch.object(q,'closure',return_value='f'*40))
        import ch3_runner as runner
        git=runner.git
        stack.enter_context(patch.object(runner,'git',side_effect=lambda *args:'f'*40 if args==('rev-parse','HEAD')else git(*args)))
        # Independent tests below execute the actual source validators. Repeated
        # fixture lifecycles need only substitute the already checked history I/O.
        stack.enter_context(patch.object(b,'verify_source',return_value=b.contract()))
        stack.enter_context(patch('utils.ch3_m_execution.gpu_environment_reasons',return_value=[]))
        from tools.restricted_regression import run_restricted
        stack.enter_context(patch.dict(sys.modules,{'m5_formal_entry':tool,'resource_budget':resource_budget,'run_restricted':run_restricted}))
        if width is not None:
            spec=copy.deepcopy(b.contract());plan=bound(spec['concurrency_plan_ref'])
            for row in plan['stages'].values():
                for group in row['groups']:group['q']=min(width,group['q'])
            spec['concurrency_plan_ref']=exclusive(root/'synthetic-reviewed-plan.json',plan)
            stack.enter_context(patch.object(b,'contract',return_value=spec))
        def context(c):
            ctx=prior_context(c);f=root/'fixtures'/c['baseline_unified']['stage'];f.mkdir(parents=True,exist_ok=True)
            return dict(ctx,fixture=f)
        stack.enter_context(patch.object(s,'context',side_effect=context))
        stack.enter_context(patch.dict(os.environ,{q.SECRET:'CPU-synthetic-invalid-for-real-HEAD'}))
        auth=q.start_template();auth.update(reviewed=True,execution_permitted=True,structure_frozen=True,m6_authorized=True,budget_authorized=True,closure_commit='f'*40,authorization_basis='CPU fixture only; cannot authorize real current HEAD')
        start=exclusive(root/'synthetic-start.json',auth)
        event.invalidate_startup();event._PRESTART=None
        with patch.object(subprocess,'check_output',side_effect=startup_sample),event.prestart(auth):pass
        exclusive(q.CONTROL/'controller.json',dict(owner=q.owner(),scope=b.ID,authorization=start,synthetic=True))
        event.seal_startup(start)
        if adopt_source:b.import_ms()
        stack.enter_context(patch.object(subprocess,'check_output',side_effect=no_gpu))
        yield root,cs,start,stack
        event.invalidate_startup();event._PRESTART=None


def formal_permit(c,start):
    # Later-stage unit fixtures need prior-stage envelopes too. Seal those
    # explicitly synthetic receipts through the real boundary validators;
    # never remove or mock the production predecessor requirement.
    predecessors={}
    for stage in s.STAGES[:s.STAGES.index(c['baseline_unified']['stage'])]:
        prior=q.configs()[stage];ctx=s.context(prior);receipts={}
        for model in s.MODELS:
            receipts[model]=exclusive(ctx['control']/('group-'+model)/'complete.json',
                dict(model=model,task_ids=[t['id']for t in prior['tasks']if t['model']==model],
                    technical_complete=True,result_review='pending',synthetic=True,
                    fixture_role='predecessor envelope only; no model execution',
                    **b.resource_binding(),**concurrency.binding(prior)))
        predecessors[stage]=q.seal_boundary(prior,receipts)
        q.validate_boundary_light(predecessors[stage],stage)
    s.context(c)['control'].mkdir(parents=True,exist_ok=False)
    permit=q.create_permit(c,start,False,boundary_ref=predecessors);runtime=q.seal_runtime(c,permit)
    return bound(permit),permit,runtime


def write_artifacts(c,t,a,pid,initialized_budget=False):
    from utils.ch3_native_tasks import result_path
    out=result_path(c,t);out.mkdir(parents=True,exist_ok=True);p=profile(c,t);ar=step_arithmetic(c,t)
    for name in('best.pt','last.pt'):(out/name).write_text('SYNTHETIC CHECKPOINT PLACEHOLDER; no tensors\n')
    rows=[];epochs=p['training']['epochs'];elements=ar['validation_windows']*p['pred_len']*(p['C']if t['task']=='M'else 1)
    for i in range(1,epochs+1):
        sse=elements/i;sae=elements/(i+1)
        rows.append(dict(epoch=i,steps=i*ar['train_batches'],validation=dict(elements=elements,sse=sse,sae=sae,mse=sse/elements,mae=sae/elements),best_epoch=i))
    (out/'history.jsonl').write_text(''.join(json.dumps(v)+'\n'for v in rows))
    ident=dict(commit=a['commit'],profile_sha=digest(p),protocol_sha=digest(c),data_sha=bound(a['data_binding_ref'])['data_bindings'][t['dataset']][t['id']],**b.resource_binding(),**concurrency.binding(c))
    exclusive(out/'manifest.json',dict(task=t,profile=p,identity=ident,synthetic=True))
    steps=epochs*ar['train_batches'];forward=epochs*(ar['train_batches']+math.ceil(ar['validation_windows']/p['training']['eval_batch']))+math.ceil(ar['test_windows_arithmetic_only']/p['training']['eval_batch'])
    counts=dict(adam=steps,backward=steps,forward=forward)
    result=dict(id=t['id'],commit=a['commit'],protocol_sha=digest(c),profile_sha=digest(p),scientific_protocol=c['baseline_unified']['id'],scheduler_updates=steps,scheduler_sha=digest(p['training']['scheduler']),mse=1.,mae=1.,best_epoch=epochs,seed=2024,final_test=dict(calls=1,selected='best.pt',sha256=ref(out/'best.pt')['sha256'],epoch=epochs),synthetic=True)
    if t['task']=='M':result.update(metric_scope='all_channels',elements=ar['test_windows_arithmetic_only']*p['pred_len']*p['C'])
    exclusive(out/'result.json',result);exclusive(out/'runtime.json',dict(task=t['id'],pid=pid,error=None,synthetic=True))
    budget=dict(counts=counts,by_pid={str(pid):counts},synthetic=True)
    if initialized_budget:
        # make_config genuinely initialized this new CPU fixture's counter.
        # Retain that zero-work snapshot before simulating its final artifact;
        # no historical or real experiment file is writable through this helper.
        assert (s.RESULT.parent/'SYNTHETIC_ONLY.json').is_file()
        assert out.resolve().is_relative_to(s.RESULT.resolve())
        initial=bound(ref(out/'budget.json'));assert initial['counts']==dict(adam=0,backward=0,forward=0)
        exclusive(out/'synthetic-initial-budget.json',initial)
        (out/'budget.json').write_text(json.dumps(budget)+'\n')
    else:exclusive(out/'budget.json',budget)


def synthetic_process(c,a,ids,identities):
    # These are explicitly synthetic task labels for genuinely exited CPU fixture
    # processes. They cannot be source-adopted: different root, fake HEAD and MAC.
    return dict(returncodes=[0]*len(ids),elapsed=.01,failure=None,failure_kind=None,resource_admission=True,
        owned_workers_exited=True,original_failures=[],cleanup_terminations=[],runtime_gpu_queries=0,
        telemetry='not_collected_startup_only',**b.resource_binding(),startup_hardware_ref=a['startup_hardware_ref'],
        worker_lifecycles=[dict(pid=v['pid'],start_ticks=v['start_ticks'],task_id=t,returncode=0)for t,v in zip(ids,identities)],synthetic=True)


def cpu_identities(n=4):
    ps=[subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(.2)'])for _ in range(n)]
    rows=[q.owner(p.pid)for p in ps]
    assert all(p.wait(timeout=10)==0 for p in ps)
    return rows


def guard_fixture(root):
    sys.path.insert(0,str(b.ROOT/'tools/restricted_regression'))
    import restricted_io_guard as guard
    with fixture(root=root)as(root,cs,start,stack):
        c=cs['EPF_ALL'];a,permit,runtime=formal_permit(c,start);t=next(t for t in c['tasks']if t['model']=='TimeXer')
        from utils.ch3_native_tasks import result_path
        stack.enter_context(patch.dict(os.environ,{'TMPDIR':str(s.context(c)['fixture'])}))
        cfg=x.make_config(c,'ch3_formal',result_path(c,t),task=t['id'],approval=a,runtime_ref=runtime)
        path=Path(cfg['output'])/'config.json';os.environ.update(AMD_RR_CONFIG=str(path),AMD_RR_CONFIG_SHA256=ref(path)['sha256'])
        guard.install(str(path),ref(path)['sha256']);state=guard.require_installed()
        stack.enter_context(patch.object(q,'configs',side_effect=AssertionError('no full-stage revalidation under guard')))
        actual=x.read_config(state)
        import ch3_runner as runner
        ident=runner.formal_identity(actual,t,bound(a['data_binding_ref'])['metadata'][t['dataset']][t['id']],a)
        assert ident['concurrency_plan_ref']==a['concurrency_plan_ref']
        old=b.ROOT.parent/'amd-execution-evidence/m6/m6-formal-launch-dhozikhu/baseline-unified96-onecycle001-v3/ms-data-bindings.json'
        try:guard.check_access(str(old))
        except guard.ForbiddenAccess:pass
        else:raise AssertionError('unauthorized historical evidence must remain forbidden')
        print(json.dumps(dict(real_guard=True,formal_identity=True,old_metadata_denied=True,all_stage_after_install=0,GPU_queries=0,training=0)))


class ReviewedConcurrency(unittest.TestCase):
    def test_startup_epoch_budget_equals_science_and_actual_three_configs(self):
        startup=b.startup_contract();science=b.contract();cs=q.configs()
        self.assertEqual(startup['max_run_epochs'],science['max_run_epochs'])
        self.assertEqual(startup['max_run_epochs'],5670)
        self.assertEqual({k:len(c['tasks'])for k,c in cs.items()},dict(URBAN_SUBSET=168,EPF_ALL=35,M_ALL=168))
        epochs={k:sum(profile(c,t)['training']['epochs']for t in c['tasks'])for k,c in cs.items()}
        self.assertEqual(epochs,dict(URBAN_SUBSET=3360,EPF_ALL=350,M_ALL=1960))
        self.assertEqual(sum(epochs.values()),startup['max_run_epochs'])
        old_ref=ref(b.PACKAGE/'startup-recovery.contract.json')
        self.assertEqual(old_ref['sha256'],'d9a412facd4f6d6c4e5e42cad806715d12e43dd6de73b425af0f6cef47473ef1')
        old=bound(old_ref);self.assertEqual(old['max_run_epochs'],3990)
        prior=bound(ref(b.PACKAGE/'startup-recovery.contract.v2.json'))
        self.assertEqual(prior,dict(old,max_run_epochs=5670))
        self.assertEqual(startup,dict(prior,science_contract_ref=b.CONTRACT_REF))
        self.assertEqual(science['concurrency_plan_ref']['sha256'],'9a39937dc45a55eea2191e08c56e0cdccee113a1129294530e8cc5633a82ebad')
        self.assertFalse((b.PACKAGE/'start-review.json').exists());self.assertFalse(b.RESULT.exists())

    def test_startup_epoch_budget_3990_rejected_by_real_contract_validator(self):
        root=Path(tempfile.mkdtemp(prefix='synthetic-budget-',dir=b.PACKAGE/'timexer-epf-four-plus-one-v1/fixtures'))
        (root/'SYNTHETIC_ONLY.json').write_text('{"synthetic":true,"GPU":0,"training":0,"permission":false}\n')
        wrong=copy.deepcopy(b.startup_contract());wrong['max_run_epochs']=3990
        wrong_ref=exclusive(root/'startup-budget-3990.json',wrong)
        with patch.object(b,'STARTUP_RECOVERY_REF',wrong_ref):
            self.assertRaisesRegex(PermissionError,'exact B technical startup contract',b.startup_contract)
        self.assertEqual(b.startup_contract()['max_run_epochs'],5670)
        self.assertFalse((b.PACKAGE/'start-review.json').exists());self.assertFalse(b.RESULT.exists())

    def test_actual_source_checks_and_original66(self):
        self.assertEqual(b.verify_production_inheritance()['protected_function_count'],66)
        self.assertTrue(b.verify_source());self.assertEqual(b.verify_stopped_four_H()['new_queue_completed_from_old_attempt'],0)

    def test_scientific_profiles_numeric_tables_Weather_and371_budget_unchanged(self):
        cs=q.configs();self.assertEqual(sum(len(c['tasks'])for c in cs.values()),371)
        self.assertEqual(sum(s.formal_budget(c)['total']['run_epochs']for c in cs.values()),5670)
        for stage,c in cs.items():
            old=json.loads(subprocess.check_output(['git','show','HEAD:'+str(s.file(stage).relative_to(b.ROOT))],text=True));self.assertEqual(c,old)
            self.assertEqual(s.probe_budget(c)['max_workers'],0)
        parent=bound(b.contract()['parent_M_ALL_ref']);c=cs['M_ALL']
        for t in c['tasks']:
            if t['model']=='PatchTST':
                if t['dataset']=='Weather':self.assertEqual(profile(c,t),profile(parent,t));self.assertEqual(profile(c,t)['structure']['d_model'],128)
                else:self.assertEqual(profile(c,t)['structure']['d_model'],4)
        self.assertEqual(b.selected_main()['retained_cells'],351)

    def test_fixed_plan56_all_q4_and_no_new_Passed_claims(self):
        rows=[g for c in q.configs().values()for g in concurrency.stage_plan(c)['groups']]
        self.assertEqual(len(rows),56);self.assertTrue(all(g['q']==4 for g in rows))
        self.assertTrue(all('status'not in g and g['limitations']for g in rows))
        self.assertFalse(any('PROBE'in state or'AUTO_AUDIT'in state for state in q.STATES))
        template=q.start_template()
        for stage,c in q.configs().items():self.assertEqual(bound(template['plan_refs'][stage]),s.plan(c))
        root=Path(tempfile.mkdtemp(prefix='synthetic-old-stage-plan-',dir=b.PACKAGE/'timexer-epf-four-plus-one-v1/fixtures'))
        (root/'SYNTHETIC_ONLY.json').write_text('{"synthetic":true,"GPU":0,"permission":false}\n')
        for stage in b.STAGES:(root/(stage.lower()+'-plan.json')).write_bytes(Path(template['plan_refs'][stage]['path']).read_bytes())
        (root/'epf_all-plan.json').write_bytes((b.PACKAGE/'epf_all-plan.json').read_bytes())
        with patch.object(b,'PLAN_PACKAGE',root):self.assertRaisesRegex(ValueError,'current stage plan projection',b.verify_source)

    def test_generic_future_EPF_default4_plus1_and_explicit_historical_read(self):
        api=runpy.run_path(s.__file__);c=q.configs()['EPF_ALL'];future=copy.deepcopy(c)
        future['baseline_unified']['id']='CPU-future-new-EPF-protocol';future['tasks'].reverse()
        groups=api['probe_groups'](future);self.assertEqual(len(groups),7)
        tasks={t['id']:t for t in future['tasks']}
        for g in groups:
            self.assertEqual(g['planned_q'],4);self.assertEqual(g['grouping_policy'],s.EPF_GROUPING_POLICY)
            self.assertEqual([tasks[t]['dataset']for t in g['representatives']],['PJM','NP','BE','FR','DE'])
        old_api=runpy.run_path(str(b.PACKAGE/'timexer-epf-four-plus-one-v1/before-candidate/utils/ch3_type1_tasks.py'))
        legacy=api['historical_epf_probe_groups_v1'](c)
        self.assertEqual(legacy,old_api['probe_groups'](c))
        tx=[g for g in legacy if g['model']=='TimeXer']
        self.assertEqual([g['planned_q']for g in tx],[4,2]);self.assertEqual([len(g['representatives'])for g in tx],[3,2])
        self.assertRaises(PermissionError,api['historical_epf_probe_groups_v1'],future)
        self.assertEqual(len(api['probe_groups'](future)),7)
        # Different future shapes are visible risks; the default never silently
        # returns to a historical3+2 split.
        changed=copy.deepcopy(future);t=next(t for t in changed['tasks']if t['model']=='TimeXer'and t['dataset']=='NP')
        changed['resolved_profiles'][t['id']]['training']['batch']=64
        tx=next(g for g in api['probe_groups'](changed)if g['model']=='TimeXer')
        self.assertTrue(tx['resource_review_required']);self.assertTrue(tx['resource_risks']);self.assertEqual(len(tx['representatives']),5);self.assertEqual(tx['planned_q'],4)
        uniform=copy.deepcopy(future)
        for p in uniform['resolved_profiles'].values():p['training'].update(batch=64,eval_batch=64)
        self.assertTrue(all(g['resource_review_required']and g['planned_q']==4 for g in api['probe_groups'](uniform)))
        uniform=copy.deepcopy(future)
        for t in uniform['tasks']:
            if t['model']=='TimeXer':uniform['resolved_profiles'][t['id']]['structure']['d_model']+=4
        tx=next(g for g in api['probe_groups'](uniform)if g['model']=='TimeXer')
        self.assertTrue(tx['resource_review_required']);self.assertEqual(len(tx['representatives']),5)

    def test_EPF14_waves_TimeXer_order_and_other_stages_unchanged(self):
        cs=q.configs();old=bound(ref(b.PACKAGE/'reviewed-concurrency-plan.json'));new=bound(b.contract()['concurrency_plan_ref'])
        self.assertEqual(new['epf_grouping_policy'],s.EPF_GROUPING_POLICY)
        for stage,n in [('URBAN_SUBSET',7),('EPF_ALL',7),('M_ALL',42)]:self.assertEqual(len(new['stages'][stage]['groups']),n)
        for stage in ['URBAN_SUBSET','M_ALL']:self.assertEqual(new['stages'][stage],old['stages'][stage])
        c=cs['EPF_ALL'];row=concurrency.stage_plan(c);tasks={t['id']:t for t in c['tasks']};seen=[];waves=[]
        for model in s.MODELS:
            w=s.formal_waves(c,row,model);self.assertEqual([len(ids)for ids in w],[4,1])
            self.assertEqual([[tasks[tid]['dataset']for tid in ids]for ids in w],[['PJM','NP','BE','FR'],['DE']])
            self.assertTrue(all(tasks[tid]['model']==model for ids in w for tid in ids));waves.extend(w);seen.extend(tid for ids in w for tid in ids)
        self.assertEqual(len(waves),14);self.assertEqual(len(seen),35);self.assertEqual(len(set(seen)),35)
        for g in row['groups']:
            if g['model']=='TimeXer':
                self.assertEqual(g['evidence_class'],'similar_shape_inference');self.assertFalse(g['shape_matches_history'])
            else:
                prior=next(x for x in old['stages']['EPF_ALL']['groups']if x['id']==g['id'])
                self.assertEqual({k:v for k,v in g.items()if k not in('grouping_policy','resource_review_required','resource_risks')},prior)
        history=bound(ref(b.PACKAGE/'timexer-epf-four-plus-one-v1/historical-TimeXer-batch-identity.json'))
        self.assertEqual({(r['batch'],r['recorded_q'])for r in history['rows']},{(16,4),(4,2)})
        for r in history['rows']:bound(r['worker_config_ref']);bound(r['protocol_file_ref'])

    def test_EPF_grouping_version_order_q_task_and_SHA_tamper_rejected(self):
        c=q.configs()['EPF_ALL'];good=concurrency.stage_plan(c)
        for kind in ['order','q','task','version']:
            bad=copy.deepcopy(good);g=bad['groups'][-1]
            if kind=='order':g['task_ids'][0],g['task_ids'][1]=g['task_ids'][1],g['task_ids'][0]
            elif kind=='q':g['q']=2
            elif kind=='task':g['task_ids'][0]='foreign-task'
            else:bad['grouping_policy']='historical3+2'
            self.assertRaises((PermissionError,ValueError),concurrency.formal_waves,c,bad,'TimeXer')
        wrong=dict(b.contract()['concurrency_plan_ref'],sha256='0'*64)
        self.assertRaises(ValueError,bound,wrong)

    def test_TimeXer_EPFFormal5_permit_runtime_worker_manifest_complete_bound(self):
        with fixture()as(root,cs,start,stack):
            c=cs['EPF_ALL'];a,permit,runtime=formal_permit(c,start);plan=x.formal_plan(c,a)
            waves=s.formal_waves(c,plan,'TimeXer');self.assertEqual([len(ids)for ids in waves],[4,1])
            group=s.context(c)['control']/'group-TimeXer';group.mkdir()
            stack.enter_context(patch.dict(os.environ,{'TMPDIR':str(s.context(c)['fixture'])}))
            from utils.ch3_native_tasks import result_path
            for i,ids in enumerate(waves):
                owned=cpu_identities(len(ids))
                for tid,identity in zip(ids,owned):
                    t=next(t for t in c['tasks']if t['id']==tid)
                    cfg=x.make_config(c,'ch3_formal',result_path(c,t),task=tid,approval=a,runtime_ref=runtime)
                    x.validate_worker(c,cfg);tool.validate_config(cfg);self.assertEqual(cfg['concurrency_plan_ref'],b.contract()['concurrency_plan_ref'])
                    write_artifacts(c,t,a,identity['pid'],initialized_budget=True)
                    manifest=bound(ref(result_path(c,t)/'manifest.json'));self.assertEqual(manifest['identity']['concurrency_plan_ref'],a['concurrency_plan_ref'])
                exclusive(group/('wave-'+str(i))/'process.json',synthetic_process(c,a,ids,owned))
            receipt=x.technical_group(c,'TimeXer',dict(a,data_bindings=bound(a['data_binding_ref'])['data_bindings']))
            self.assertTrue(receipt['technical_complete']);self.assertEqual(receipt['concurrency_plan_ref'],a['concurrency_plan_ref'])
            exclusive(group/'complete.json',receipt)
            self.assertEqual(len(waves[0])+len(waves[1]),5);self.assertNotIn('summary_ref',a);self.assertFalse(receipt['probe_admission_claim'])

    def test_plan_rejects_q_shape_profile_group_Git_and_fake_Passed(self):
        c=q.configs()['URBAN_SUBSET'];good=concurrency.stage_plan(c)
        for kind in('q','shape','profile','duplicate','fake'):
            v=copy.deepcopy(good)
            if kind=='q':v['groups'][0]['q']=8
            elif kind=='shape':next(iter(v['groups'][0]['shapes'].values()))['C']=1
            elif kind=='profile':v['profile_shas'][c['tasks'][0]['id']]='0'*64
            elif kind=='duplicate':v['groups'][0]['task_ids'].append(c['tasks'][0]['id'])
            else:v['groups'][0]['status']='Passed'
            self.assertRaises(ValueError,concurrency.validate_stage,c,v)
        with fixture()as(root,cs,start,stack):
            a,permit,runtime=formal_permit(cs['URBAN_SUBSET'],start)
            for field,value in(('commit','0'*40),('concurrency_plan_ref',b.SOURCE_REF),('execution_attempt','THIRD-PatchTST-enc2-dm4-Urban6-four-H-r1')):
                bad=dict(a);bad[field]=value;self.assertRaises((ValueError,PermissionError),q.validate_permit,cs['URBAN_SUBSET'],bad)
            bad=dict(a,summary_ref=b.SOURCE_REF);self.assertRaises(PermissionError,concurrency.validate_permit,cs['URBAN_SUBSET'],bad)
            self.assertRaises(PermissionError,q.create_permit,cs['URBAN_SUBSET'],start,True)

    def test_real_permit_runtime_and_worker_gate_without_probe_summary(self):
        with fixture()as(root,cs,start,stack):
            c=cs['URBAN_SUBSET'];a,permit,runtime=formal_permit(c,start)
            self.assertNotIn('summary_ref',a);self.assertNotIn('manifest_ref',a);self.assertNotIn('technical_admission',a)
            v=q.validate_runtime(c,runtime,permit);self.assertNotIn('integrity_scan_passed',v)
            self.assertFalse(v['probe_admission_claim']);t=c['tasks'][0]
            from utils.ch3_native_tasks import result_path
            stack.enter_context(patch.dict(os.environ,{'TMPDIR':str(s.context(c)['fixture'])}))
            cfg=x.make_config(c,'ch3_formal',result_path(c,t),task=t['id'],approval=a,runtime_ref=runtime)
            x.validate_worker(c,cfg);tool.validate_config(cfg)
            bad=copy.deepcopy(cfg);bad['concurrency_plan_ref']=b.SOURCE_REF;self.assertRaises(PermissionError,x.validate_worker,c,bad)
            wrong=bound(runtime);wrong['owner']['start_ticks']='0';rr=exclusive(root/'bad-runtime.json',wrong);self.assertRaises(PermissionError,q.validate_runtime,c,rr,permit)

    def test_actual_installed_guard_and_formal_binding_in_CPU_child(self):
        root=Path(tempfile.mkdtemp(prefix='CPU-guard-',dir=b.PACKAGE/'fixtures'))
        result=subprocess.run([sys.executable,'-B',__file__,'--guard-fixture',str(root)],cwd=b.ROOT,capture_output=True,text=True,timeout=180,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(b.ROOT)))
        (root/'stdout.txt').write_text(result.stdout);(root/'stderr.txt').write_text(result.stderr)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr);self.assertTrue(json.loads(result.stdout)['real_guard'])

    def test_q1_q2_q4_real_run_configs_and_resource_audit_zero_GPU_queries(self):
        from utils.ch3_native_execution import wave_resource_identities
        for width in(1,2,4):
            with fixture(width)as(root,cs,start,stack):
                c=cs['URBAN_SUBSET'];a,permit,runtime=formal_permit(c,start);ids=s.formal_waves(c,concurrency.stage_plan(c),'AMD')[0]
                self.assertEqual(len(ids),width);from utils.ch3_native_tasks import result_path
                configs=[x.make_config(c,'ch3_formal',result_path(c,next(t for t in c['tasks']if t['id']==tid)),task=tid,approval=a,runtime_ref=runtime)for tid in ids]
                spawned=[]
                def spawn(cfg):
                    h=(Path(cfg['output'])/'worker.log').open('x');p=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(.2)'],stdout=h,stderr=h);spawned.append(p);return p,h
                with patch.object(tool,'spawn',side_effect=spawn),patch.object(tool,'gpu_sample',side_effect=AssertionError('no GPU sampler')),patch.object(tool,'owned_pid_metadata',side_effect=AssertionError('no namespace')),patch.object(tool,'ExitObservation',side_effect=AssertionError('no attribution')):
                    out=s.context(c)['control']/'group-AMD/wave-0';value=tool.run_configs(configs,out,monitor=True)
                self.assertTrue(value['resource_admission']);self.assertEqual(len(wave_resource_identities(value,a,out/'memory.jsonl',ids)),width)
                self.assertEqual(value['runtime_gpu_queries'],0);self.assertFalse((out/'memory.jsonl').exists());self.assertTrue(all(p.poll()==0 for p in spawned))
                self.assertRaises(FileExistsError,tool.run_configs,configs,out,True)

    def test_complete371_real_scheduler_permits_audits_boundaries_no_probe(self):
        with fixture(adopt_source=False)as(root,cs,start,stack):
            # The real q.run continuation creates its source-adoption boundary;
            # only model computation is replaced by synthetic task artifacts.
            self.assertFalse((q.CONTROL/'upstream-technical-boundary.json').exists())
            identities=cpu_identities();seen=[];stage_seen={stage:[]for stage in b.STAGES}
            def produce(stage,permit,probe,model=None,runtime=None):
                self.assertFalse(probe);c=cs[stage];a=q.validate_permit(c,bound(permit));q.validate_runtime(c,runtime,permit)
                group=s.context(c)['control']/('group-'+model);group.mkdir()
                for i,ids in enumerate(s.formal_waves(c,x.formal_plan(c,a),model)):
                    seen.extend(ids);stage_seen[stage].extend(ids)
                    for tid,identity in zip(ids,identities):write_artifacts(c,next(t for t in c['tasks']if t['id']==tid),a,identity['pid'])
                    exclusive(group/('wave-'+str(i))/'process.json',synthetic_process(c,a,ids,identities))
                receipt=x.technical_group(c,model,dict(a,data_bindings=bound(a['data_binding_ref'])['data_bindings']))
                exclusive(group/'complete.json',receipt)
            stack.enter_context(patch.object(q,'wait_owned',side_effect=produce))
            stack.enter_context(patch.object(q,'audit_probe',side_effect=AssertionError('no AUTO_AUDIT')))
            q.run(start);done=bound(ref(q.CONTROL/'complete.json'));b.validate_complete(done)
            self.assertEqual(len(seen),371);self.assertEqual(len(set(seen)),371);self.assertEqual(done['result_review'],'pending')
            self.assertEqual({stage:len(ids)for stage,ids in stage_seen.items()},b.CURRENT_STAGE_RUNS)
            self.assertEqual(set(done['stage_boundaries']),set(b.STAGES))
            self.assertEqual(done['epf_grouping_policy'],b.contract()['epf_grouping_policy'])
            self.assertEqual(done['concurrency_plan_ref'],b.contract()['concurrency_plan_ref'])
            self.assertEqual((done['total_third_formal_runs'],done['executed_third_formal_runs']),(371,371))
            self.assertEqual((done['withdrawn_B_completed_credit'],done['stopped_four_H_completed_credit']),(0,0))
            self.assertTrue((q.CONTROL/'upstream-technical-boundary.json').is_file())
            for stage,r in done['stage_boundaries'].items():
                self.assertEqual(len(q.validate_boundary_light(r,stage)['task_ids']),b.CURRENT_STAGE_RUNS[stage])
            # Mutations stay in isolated synthetic documents. The real final
            # validator must reject policy/plan/credit/count/review substitutions.
            cases={
                'wrong-epf-policy':dict(done,epf_grouping_policy='historical3-plus2'),
                'missing-epf-policy':{k:v for k,v in done.items()if k!='epf_grouping_policy'},
                'wrong-plan-sha':dict(done,concurrency_plan_ref=dict(done['concurrency_plan_ref'],sha256='0'*64)),
                'old57-plan':dict(done,concurrency_plan_ref=ref(b.PACKAGE/'reviewed-concurrency-plan.json')),
                'withdrawn-twoH-credit':dict(done,withdrawn_B_completed_credit=1),
                'stopped-fourH-credit':dict(done,stopped_four_H_completed_credit=1),
                'wrong-total':dict(done,total_third_formal_runs=370),
                'wrong-executed':dict(done,executed_third_formal_runs=370),
                'premature-result-review':dict(done,result_review='Passed'),
            }
            for label,bad in cases.items():
                exclusive(root/('rejected-complete-'+label+'.json'),bad)
                with self.subTest(complete_tamper=label):
                    self.assertRaises((ValueError,PermissionError),b.validate_complete,bad)
            exclusive(root/'CPU-complete371-acceptance.json',dict(synthetic=True,
                stages={stage:len(ids)for stage,ids in stage_seen.items()},unique_tasks=len(set(seen)),
                complete_ref=ref(q.CONTROL/'complete.json'),validate_complete='Passed',
                complete_tamper_rejections=sorted(cases),result_review=done['result_review'],
                GPU_queries=0,real_training=False,real_test=False))
            # Corrupt only this synthetic producer's artifacts to exercise the
            # unchanged finite/test-once/budget gates after plan-based dispatch.
            from utils.ch3_native_tasks import result_path
            c=cs['URBAN_SUBSET'];t=c['tasks'][0];path=result_path(c,t)/'result.json';original=path.read_text();a=bound(ref(s.context(c)['control']/'formal-permit.json'));a['data_bindings']=bound(a['data_binding_ref'])['data_bindings']
            for kind in('nonfinite','test-twice','wrong-best','wrong-scheduler'):
                bad=json.loads(original)
                if kind=='nonfinite':bad['mse']=float('nan')
                elif kind=='test-twice':bad['final_test']['calls']=2
                elif kind=='wrong-best':bad['best_epoch']=1
                else:bad['scheduler_sha']='0'*64
                (root/('rejected-'+kind+'.json')).write_text(json.dumps(bad)+'\n');path.write_text(json.dumps(bad)+'\n')
                self.assertRaises(ValueError,x.technical_group,c,'AMD',a);path.write_text(original)
            self.assertFalse((s.RESULT/'probe').exists());self.assertFalse(list(s.RESULT.glob('**/admission-summary.json')))
            bad=dict(done,stopped_four_H_completed_credit=1);self.assertRaises(PermissionError,b.validate_complete,bad)
            from utils.ch3_type1_summary import result_index
            # Only the new fixture paths are populated. Historical indexes are
            # real read-only sources and contain no checkpoint/test operations.
            value=result_index(cs);self.assertEqual(len(value['rows']),371);self.assertFalse(value['probe_admission_claim'])

    def test_missing_artifact_nonfinite_and_test_once_are_rejected(self):
        with fixture()as(root,cs,start,stack):
            c=cs['URBAN_SUBSET'];a,permit,runtime=formal_permit(c,start)
            self.assertRaises(FileNotFoundError,x.technical_group,c,'AMD',a)
        for exc in(ValueError('numeric finite'),RuntimeError('CUDA out of memory'),InterruptedError('STOP')):
            with fixture()as(root,cs,start,stack):
                calls=[]
                def fail(*args,**kw):calls.append(args);raise exc
                with patch.object(b,'import_ms',return_value=ref(q.CONTROL/'upstream-technical-boundary.json')),patch.object(q,'wait_owned',side_effect=fail):self.assertRaises(type(exc),q.run,start)
                self.assertEqual(len(calls),1)

    def test_old_auths_rejected_and_no_executable_new_lifecycle(self):
        old=bound(ref(b.PREVIOUS_PACKAGE/'start-review.json'))
        with patch.object(q,'closure',return_value=old['closure_commit']):self.assertRaises(PermissionError,q.validate_start,old)
        self.assertFalse(b.RESULT.exists());self.assertFalse(b.LOG.exists());self.assertFalse((b.PACKAGE/'start-review.json').exists())
        self.assertFalse(any('PROBE'in x or'AUTO_AUDIT'in x for x in q.STATES))

    def test_formal_OOM_nonzero_STOP_and_timeout_stop_without_fallback_or_foreign_kill(self):
        for case in('OOM','business','STOP','timeout'):
            with fixture()as(root,cs,start,stack):
                c=cs['EPF_ALL'];a,permit,runtime=formal_permit(c,start);spawned=[]
                other=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(20)'])
                def spawn(cfg):
                    x.validate_worker(c,cfg)
                    h=(Path(cfg['output'])/'worker.log').open('x')
                    script='import time;time.sleep(2)'
                    if not spawned and case in('OOM','business'):
                        script="raise RuntimeError('CUDA out of memory')"if case=='OOM'else"raise ValueError('independent numeric business failure')"
                    if case=='timeout':cfg['limits']['seconds']=.05
                    p=subprocess.Popen([sys.executable,'-B','-c',script],stdout=h,stderr=h);spawned.append(p)
                    if case=='STOP':(q.CONTROL/'STOP').write_text('CPU fixture STOP\n')
                    return p,h
                try:
                    with patch.object(tool,'spawn',side_effect=spawn),patch.object(tool,'gpu_sample',side_effect=AssertionError('no GPU telemetry')):
                        self.assertRaises((RuntimeError,InterruptedError),x.run_group,c,a,'TimeXer',runtime)
                    self.assertLessEqual(len(spawned),4);self.assertTrue(all(p.poll()is not None for p in spawned));self.assertIsNone(other.poll())
                    self.assertFalse((s.context(c)['control']/'group-TimeXer/complete.json').exists())
                finally:
                    other.terminate();other.wait(timeout=10)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--guard-fixture':guard_fixture(sys.argv[2])
    else:unittest.main()
