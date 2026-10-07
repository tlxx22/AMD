"""Scoped policy, immutable retained sources and actual no-model resume/audit path."""
import ast,copy,json,os,struct,subprocess,sys,tempfile,unittest
from contextlib import ExitStack,contextmanager
from pathlib import Path
from unittest.mock import patch
import ch3_runner as runner
from utils import ch3_moderntcn_etth1_recovery as r,ch3_probe_schema_recovery as previous
from utils import ch3_native_execution as native,ch3_type1_tasks as s,ch3_type1_chain as q
from utils.ch3_contract import ROOT,digest,profile,task_by_id,numeric_probe_policy
from utils.ch3_native_recovery_records import ref,bound,exclusive,scan_manifest
from tests.test_m6_probe_schema_recovery import trajectory

@contextmanager
def attempt():
    from utils import ch3_ms_seal_recovery as ms,ch3_round2_amendment as amendment
    with ExitStack()as stack:
        for module,names in ((s,('RESULT','context','package','file')),(ms,('RESULT','M_FILE')),(amendment,('RESULT',)),(q,('CONTROL','LOG','SESSION','ENTRY','WRAPPER','PROBE_RECOVERY','QUEUE_LOCK')),(r,('ACTIVE',))):
            for name in names:stack.enter_context(patch.object(module,name,getattr(module,name)))
        r.ACTIVE=False;r.activate();yield

class Policy(unittest.TestCase):
    def test_science_and_other_registries_remain_exact(self):
        old=bound(dict(path=str(r.COLLECTION_FILE),sha256=r.CONFIG_SHA));new=bound(r.CONFIG_REF)
        self.assertEqual(old['resolved_profiles'],new['resolved_profiles']);self.assertEqual(old['tasks'],new['tasks'])
        self.assertEqual(old['datasets'],new['datasets']);self.assertEqual(old['sources'],new['sources'])
        r.validate_revision(new)
        keys=set(old['baseline_unified']['numeric_policies'])
        self.assertEqual(set(k for k in keys if old['baseline_unified']['numeric_policies'][k]!=new['baseline_unified']['numeric_policies'][k]),keys)
        with attempt():
            cs=q.configs();self.assertEqual([len(c['tasks'])for c in cs.values()],[84,112,28,35,168])
            for stage in cs:self.assertEqual(cs[stage]['baseline_unified']['numeric_revision_ref'],r.ALL_M_POLICY_REF)
            for stage in cs:
                prior=bound(bound(r.ALL_M_POLICY_REF)['stages'][stage]['old_config_ref']);value=cs[stage]
                self.assertEqual(prior['resolved_profiles'],value['resolved_profiles']);self.assertEqual(prior['tasks'],value['tasks'])
                self.assertEqual(prior['datasets'],value['datasets']);self.assertEqual(prior['sources'],value['sources'])
                self.assertEqual(s.formal_budget(prior),s.formal_budget(value));self.assertEqual(s.probe_budget(prior),s.probe_budget(value))
                from utils.ch3_contract import validate_baseline_numeric_registry
                self.assertEqual(validate_baseline_numeric_registry(value),value['baseline_unified']['numeric_policies'])
            self.assertEqual(s.formal_budget(old),s.formal_budget(new));self.assertEqual(s.probe_budget(old),s.probe_budget(new))
    def test_wrong_dataset_scope_or_threshold_cannot_inherit(self):
        new=bound(r.CONFIG_REF)
        for key in ('ModernTCN-Weather','ModernTCN-Exchange','TimeMixer-ETTh1'):
            c=copy.deepcopy(new)
            if c['baseline_unified']['numeric_policies'][key]is None:c['baseline_unified']['numeric_policies'][key]={'state_atol':2e-4}
            else:c['baseline_unified']['numeric_policies'][key]['state_atol']=3e-4
            with self.assertRaises(ValueError):r.validate_revision(c)
        for field,value in [('state_atol',3e-4),('metric_atol',2e-6),('loss_rtol',1e-6)]:
            c=copy.deepcopy(new);c['baseline_unified']['numeric_policies']['ModernTCN-ETTh1'][field]=value
            with self.assertRaises(ValueError):r.validate_revision(c)
    def test_all_M_except_Exchange_exact_matrix_and_thresholds(self):
        # Retain the historical test method; the latest authorization includes Exchange.
        fields=dict(kind='full_float_state',state_atol=2e-4,loss_atol=1e-6,metric_atol=1e-6,loss_rtol=0,rtol=0,equal_nan=False)
        rows=[];rules=[]
        with attempt():
            for stage,c in q.configs().items():
                if stage not in ('M_BASE','M_AMEND','M_ALL'):continue
                for t in c['tasks']:
                    if t['model']!='ModernTCN':continue
                    rule=numeric_probe_policy(c,t)
                    self.assertEqual({k:rule[k]for k in fields},fields);self.assertEqual((rule['task'],rule['dataset'],rule['model']),('M',t['dataset'],'ModernTCN'))
                    self.assertEqual(rule['horizons'],[96,192,336,720]);rows.append((stage,t['id']))
                rules.extend((stage,key)for key in c['baseline_unified']['numeric_policies']if key.startswith('ModernTCN-'))
        self.assertEqual(len(rows),52);self.assertEqual(len(set(rows)),52);self.assertEqual(len(rules),13)
    def test_Exchange_and_MS_cannot_receive_the_M_extension(self):
        # Wrong Exchange identity and MS adoption stay rejected after the authorized expansion.
        with attempt():
            cs=q.configs()
            for stage in ('M_BASE','M_ALL'):
                c=copy.deepcopy(cs[stage]);self.assertEqual(c['baseline_unified']['numeric_policies']['ModernTCN-Exchange']['dataset'],'Exchange')
                c['baseline_unified']['numeric_policies']['ModernTCN-Exchange']=copy.deepcopy(c['baseline_unified']['numeric_policies']['ModernTCN-ETTh1'])
                with self.assertRaises(ValueError):s.validate(c)
                c=copy.deepcopy(cs[stage]);c['baseline_unified']['numeric_policies']['ModernTCN-Exchange']=None
                with self.assertRaises(ValueError):s.validate(c)
            for stage in ('URBAN_SUBSET','EPF_ALL'):
                c=copy.deepcopy(cs[stage]);key=next(iter(c['baseline_unified']['numeric_policies']))
                c['baseline_unified']['numeric_policies'][key]=copy.deepcopy(cs['M_BASE']['baseline_unified']['numeric_policies']['ModernTCN-ETTh1'])
                with self.assertRaises(ValueError):s.validate(c)
    def test_frozen_legacy_configs_keep_their_original_policies(self):
        for stage in r.CONFIG_REFS:
            c=bound(bound(r.ALL_M_POLICY_REF)['stages'][stage]['old_config_ref'])
            self.assertNotIn('numeric_revision_ref',c['baseline_unified']);self.assertEqual(s.validate(c),c)
            if stage in ('M_BASE','M_ALL'):self.assertIsNone(c['baseline_unified']['numeric_policies']['ModernTCN-Exchange'])
    def test_old_template_cannot_authorize_the_extended_matrix(self):
        old=json.loads((r.PACKAGE/'start-approval.template.json').read_text())
        exchange_exact=json.loads((r.PACKAGE/'moderntcn-all-m-policy-v1/start-approval.template.json').read_text())
        with attempt(),patch.object(q,'closure',return_value='synthetic-reviewed-HEAD'):
            for a in (old,exchange_exact,q.start_template()):
                a.update(reviewed=True,execution_permitted=True,structure_frozen=True,m6_authorized=True,budget_authorized=True,closure_commit='synthetic-reviewed-HEAD',authorization_basis='synthetic exact permit, no actual authorization file')
                if a is old or a is exchange_exact:
                    with self.assertRaises(PermissionError):q.validate_start(a)
                else:self.assertEqual(q.validate_start(a),a)
    def test_extended_policy_is_bound_in_compact_worker_metadata(self):
        from utils.ch3_type1_execution import metadata_files
        with attempt():
            c=q.configs()['M_ALL'];a=dict(start_authorization_ref=r.SOURCE_REF,probe_recovery_ref=r.REUSE_REF)
            values=metadata_files(c,a)
            for value in (r.ALL_M_POLICY_REF,r.PRIOR_REUSE_REF,r.REUSE_REF,r.POLICY_REF,r.INCLUSION_REF):self.assertEqual(values[value['path']],value['sha256'])
    def test_extended_groups_use_actual_comparator_2e4_boundary(self):
        with attempt(),tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp)
            for stage,c in q.configs().items():
                if stage not in r.CONFIG_REFS:continue
                names=sorted({t['dataset']for t in c['tasks']if t['model']=='ModernTCN'})
                for name in names:
                    t=next(t for t in c['tasks']if t['model']=='ModernTCN'and t['dataset']==name);rule=numeric_probe_policy(c,t)
                    x=trajectory(c,t,root/stage/name/'serial',rule);y=trajectory(c,t,root/stage/name/'parallel',rule)
                    for delta,expected in ((.75*rule['state_atol'],True),(1.05*rule['state_atol'],False)):
                        for point in y['full_numeric_trace']:
                            path=Path(point['data_file']);path.write_bytes(struct.pack('<6d',1.+delta,1.,1.,1.,1.,float(point['step'])));point['data_sha']=ref(path)['sha256']
                        self.assertEqual(runner.compare_probe_trajectories(c,t,x,y)['passed'],expected,(stage,name,delta))
                    y['trajectory'][0]['loss']+=2e-6
                    self.assertFalse(runner.compare_probe_trajectories(c,t,x,y)['passed'])
    def test_Exchange_all_four_H_and_prior_exception_provenance(self):
        extension=json.loads((r.PACKAGE/'moderntcn-all-m-policy-v1/policy-extension.json').read_text())
        self.assertEqual(extension['excluded_dataset'],'Exchange');self.assertEqual(extension['affected_tasks'],44)
        with attempt():
            cs=q.configs();rows=[]
            for stage in ('M_BASE','M_ALL'):
                frozen=bound(bound(r.ALL_M_POLICY_REF)['stages'][stage]['old_config_ref'])
                self.assertIsNone(frozen['baseline_unified']['numeric_policies']['ModernTCN-Exchange'])
                for t in cs[stage]['tasks']:
                    if t['model']=='ModernTCN'and t['dataset']=='Exchange':
                        rule=numeric_probe_policy(cs[stage],t);rows.append((stage,t['h']))
                        self.assertEqual((rule['kind'],rule['state_atol'],rule['loss_atol'],rule['metric_atol']),('full_float_state',2e-4,1e-6,1e-6))
                        self.assertEqual(profile(frozen,t),profile(cs[stage],t));self.assertIn('gradients and Adam moments',rule['floating_state'])
                        self.assertEqual(rule['exact_state'],numeric_probe_policy(cs[stage],next(x for x in cs[stage]['tasks']if x['model']=='ModernTCN'and x['dataset']=='ETTh1'))['exact_state'])
            self.assertEqual(set(rows),{(stage,h)for stage in ('M_BASE','M_ALL')for h in (96,192,336,720)})
    def test_real_original_failure_and_confirmation_are_both_retained(self):
        v=r.evidence();source=bound(r.SOURCE_REF);failure=bound(source['refs']['probe_failure'])
        self.assertIn('numeric gate failure',failure['error']);self.assertEqual(len(failure['decisions']),15)
        self.assertEqual((v['retained_groups'],v['retained_probe_trajectories'],v['independent_confirmation_trajectories']),(16,128,8))
        self.assertFalse(v['original_ModernTCN_comparisons'][0]['passed']);self.assertFalse(v['whole_M_probe_admission'])
        self.assertEqual(v['historical_probe_actual'],dict(adam=768,backward=768,forward=1024));self.assertEqual(v['historical_actual'],dict(adam=816,backward=824,forward=1112))
    def test_old_policy_and_unaffected15_are_not_relabelled(self):
        v=r.evidence();f=bound(bound(r.SOURCE_REF)['refs']['probe_failure'])
        for group,decision in f['decisions'].items():
            adopted=copy.deepcopy(v['decisions'][group]);rows=adopted.pop('numerical_comparisons');original=copy.deepcopy(decision);oldrows=original.pop('numerical_comparisons')
            self.assertEqual(adopted,original);self.assertEqual([x['collection_comparison']for x in rows],oldrows)
            for row in rows:self.assertEqual(row['evaluation_policy_ref'],r.ALL_M_POLICY_REF)
        modern=v['decisions']['ModernTCN-M-ETTh1-four-H']['numerical_comparisons']
        for row in modern:
            self.assertEqual(row['state_atol'],2e-4);self.assertEqual(row['collection_policy_sha'],digest(bound(r.POLICY_REF)['collection_policy']))
            self.assertEqual(row['evaluation_policy_ref'],r.ALL_M_POLICY_REF);self.assertEqual(row['collection_comparison']['evaluation_policy_ref'],r.POLICY_REF);self.assertIn('original_comparison',row)
    def test_wrong_old_permit_or_attempt_cannot_authorize_new(self):
        c=bound(r.CONFIG_REF)
        for value in ({},dict(execution_attempt=r.ATTEMPT,probe_recovery_ref=previous.REUSE_REF)):
            with self.assertRaises(PermissionError):r.validate_permit_link(c,value,True)
    def test_new_public_entry_binding_and_nonexecutable_template(self):
        with attempt():
            self.assertEqual(q.ENTRY,r.ENTRY);self.assertEqual(q.SESSION,r.SESSION)
            t=q.start_template();self.assertFalse(t['reviewed']);self.assertFalse(t['execution_permitted']);self.assertIsNone(t['closure_commit'])
            self.assertEqual(t['upstream_anchors']['policy_revision_ref'],r.POLICY_REF)
            self.assertEqual(t['upstream_anchors']['all_m_policy_ref'],r.ALL_M_POLICY_REF)
            self.assertEqual(t['config_refs']['M_BASE'],r.CONFIG_REF)
            for stage in r.CONFIG_REFS:self.assertEqual(t['config_refs'][stage],r.CONFIG_REFS[stage])
        self.assertIn(str(q.PYTHON),r.WRAPPER.read_text());self.assertNotIn('AMD-m-baselines-v1',r.ENTRY.read_text())
    def test_original_producer_math_is_AST_exact(self):
        old=subprocess.check_output(['git','show',r.BASE+':ch3_runner.py'],cwd=ROOT,text=True)
        def nodes(text):return {n.name:ast.dump(n,include_attributes=False)for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        a,b=nodes(old),nodes(runner.Path(runner.__file__).read_text())
        before=nodes((r.POLICY_PACKAGE/'before/ch3_runner.py').read_text())
        self.assertEqual(a['_compare_full_numeric_files'],before['_compare_full_numeric_files'])
        self.assertNotEqual(a['_compare_full_numeric_files'],b['_compare_full_numeric_files'])
        for name in ('init_training','update','evaluate','compare_probe_trajectories','_probe_validation_metrics','FullNumericStateWriter','save_state','restore_state','formal_worker'):
            self.assertEqual(a[name],b[name],name)
        delta=r.verify_production_inheritance(bound(bound(r.SOURCE_REF)['refs']['producer_permit']))
        self.assertNotIn('update',delta.get('ch3_runner.py',[]));self.assertNotIn('evaluate',delta.get('ch3_runner.py',[]))
    def test_no_runtime_remote_or_formal_probe_replay(self):
        from tests.test_m6_ms_seal_recovery import Dispatch
        Dispatch().test_no_remote_in_runtime_closure_or_worker_source();Dispatch().test_public_preflight_retains_remote_check()
    def test_synthetic_new_scope_owned_tmux_lifecycle(self):
        import shlex,time
        with tempfile.TemporaryDirectory()as tmp,attempt():
            root=Path(tmp);pidfile=root/'pid';session='fixture-etth1-numeric-'+str(os.getpid())+'-'+str(time.time_ns())
            code="import os,time;from pathlib import Path;Path("+repr(str(pidfile))+").write_text(str(os.getpid()));time.sleep(30)"
            command=shlex.join([q.PYTHON,'-B','-c',code,str(q.ENTRY),'start'])
            subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),command],check=True)
            for _ in range(100):
                if pidfile.exists():break
                time.sleep(.02)
            self.assertTrue(pidfile.exists());owner=q.owner(int(pidfile.read_text()));exclusive(root/'controller.json',dict(scope=s.ID,owner=owner))
            with patch.object(q,'CONTROL',root):v=q.safe_stop()
            self.assertTrue(v['supervisor_signal_sent']);self.assertFalse(v['old_chain_signal_sent'])
            for _ in range(100):
                if not q.same(owner):break
                time.sleep(.02)
            self.assertFalse(q.same(owner))
            for _ in range(100):
                if subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:break
                time.sleep(.02)
            self.assertNotEqual(subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode,0)
    def test_no_real_compute_in_unit_tests(self):self.assertNotIn('torch',sys.modules)

class Resume(unittest.TestCase):
    observations=[]
    @contextmanager
    def fixture(self):
        import m5_formal_entry as tool
        old=bound(dict(path=str(r.COLLECTION_FILE),sha256=r.CONFIG_SHA));new=bound(r.CONFIG_REF)
        with tempfile.TemporaryDirectory()as temp,ExitStack()as stack:
            root=Path(temp);probe_old=root/'old/probe';probe_new=root/'new/probe';control=root/'controller';control.mkdir();fixture=root/'fixtures';fixture.mkdir()
            ctx=dict(s.context(old),probe_root=probe_old,control=root/'queue',fixture=fixture,result_root=root/'results');probe_old.mkdir(parents=True);ctx['control'].mkdir()
            binding=dict(commit=r.BASE,protocol_sha=digest(old),code={},environment={},hardware={'cpu_affinity':[]},source_states={})
            permit=dict(binding,type1_scope=s.ID,caps=s.probe_budget(old)['caps']);exclusive(probe_old/'approval.json',permit)
            stack.enter_context(patch.object(s,'context',return_value=ctx));stack.enter_context(patch.object(q,'CONTROL',control));stack.enter_context(patch.object(q,'PROBE_RECOVERY',None));stack.enter_context(patch.object(q,'dynamic',return_value=binding));stack.enter_context(patch.dict(os.environ,{q.SECRET:'synthetic-numeric-secret'}));exclusive(control/'controller.json',dict(owner=q.owner(),scope=s.ID))
            events=[];counter=[10000000]
            def make(c,purpose,out,*,task,approval):
                t=task_by_id(c,task);out=Path(out);phase=out.parent.name;cost=s.worker_counts(c,t);counter[0]+=1;pid=counter[0]
                cfg=dict(task=task,output=str(out),approval=approval,protocol_sha=digest(c),successor_scope=ctx['probe_scope'],successor_phase=phase,prefix_files={},limits=dict(**cost,seconds=1800),purpose=purpose,unified_stage='M_BASE',type1_scope=s.ID,protocol_file=str(s.file('M_BASE')),session_root=str(ctx['probe_root']),artifact_root=str(out),resume=False,kernel_probe=False,device='cuda:0',ids=[task],bound_files={},author_files={},metadata_files={})
                tr=trajectory(c,t,out,numeric_probe_policy(c,t))
                if c==old and t['model']=='ModernTCN'and t['dataset']=='ETTh1'and phase=='q4':
                    for point in tr['full_numeric_trace']:
                        path=Path(point['data_file']);path.write_bytes(struct.pack('<6d',1.00015,1.,1.,1.,1.,float(point['step'])));point['data_sha']=ref(path)['sha256']
                if c==new and phase=='q4':
                    for point in tr['full_numeric_trace']:
                        path=Path(point['data_file']);path.write_bytes(struct.pack('<6d',1.0001,1.,1.,1.,1.,float(point['step'])));point['data_sha']=ref(path)['sha256']
                exclusive(out/'trajectory.json',tr);exclusive(out/'config.json',cfg);exclusive(out/'budget.json',dict(counts=cost,by_pid={str(pid):cost}));exclusive(out/'runtime.json',dict(pid=pid,task=task,error=None));(out/'audit.jsonl').write_text('{"event":"synthetic-no-model"}\n')
                return dict(cfg,budget_file=str(out/'budget.json'),pid=pid)
            def launch(configs,out,monitor):
                out=Path(out);out.mkdir(parents=True);events.extend((v['task'],out.parent.name)for v in configs);peaks={str(v['pid']):1 for v in configs}
                m=dict(failure=None,returncodes=[0]*len(configs),resource_admission=True,elapsed=2. if out.parent.name=='serial'else 1.,exit_transitions_resolved=True,process_attribution='Measured',process_peaks=peaks,cpu_peaks=peaks)
                exclusive(out/'process.json',m);(out/'memory.jsonl').write_text(json.dumps({'owned_pid_metadata':{pid:dict(pid=int(pid),start_ticks='1',namespace='pid:[1]',host_pid=int(pid))for pid in peaks}})+'\n');return m
            stack.enter_context(patch.object(tool,'make_config',side_effect=make));stack.enter_context(patch.object(tool,'run_configs',side_effect=launch))
            with self.assertRaisesRegex(ValueError,'numeric gate failure'):native.run_probe(old,permit)
            failed=bound(ref(probe_old/'failure.json'));self.assertEqual(len(events),128);self.assertEqual(len(failed['decisions']),15)
            proof=copy.deepcopy(r.evidence());groups=s.probe_groups(old);modern=groups[15];comparisons=[]
            for run in modern['representatives']:
                x=bound(failed['artifacts'][str(probe_old/modern['id']/'serial'/run/'trajectory.json')]);y=bound(failed['artifacts'][str(probe_old/modern['id']/'q4'/run/'trajectory.json')])
                orig=native.compare(old,task_by_id(old,run),x,y);self.assertFalse(orig['passed'])
                evaluation=bound(r.POLICY_REF)['evaluation_policy']
                points=[runner._compare_full_numeric_files(evaluation,a,b)for a,b in zip(x['full_numeric_trace'],y['full_numeric_trace'])];self.assertTrue(all(p['passed']for p in points))
                comparisons.append(dict(orig,passed=True,failures=[],exact_residual_state=True,state_atol=2e-4,policy_sha=digest(evaluation)))
            serial=[bound(failed['evidence'][modern['id']+'/serial/'+str(n)]['process'])for n in range(4)];parallel=bound(failed['evidence'][modern['id']+'/q4/0']['process'])
            decisions=copy.deepcopy(failed['decisions']);decisions[modern['id']]=dict(status='Passed',concurrency=4,serial=serial,parallel=[parallel],attempts=[dict(q=4,waves=[parallel],resource_failed=False)],numerical_comparisons=comparisons,coverage=modern['coverage'],makespan_scope='synthetic')
            producer_ref=exclusive(root/'producer.json',permit)
            proof.update(decisions=decisions,evidence=failed['evidence'],artifacts=failed['artifacts'],producer_refs={k:producer_ref for k in failed['evidence']},trajectory_reviews={p:{}for p in failed['artifacts']if p.endswith('trajectory.json')})
            reuse=exclusive(root/'reuse.json',proof);stack.enter_context(patch.object(r,'REUSE_REF',reuse));stack.enter_context(patch.object(r,'ACTIVE',True));stack.enter_context(patch.object(q,'PROBE_RECOVERY',r))
            msids=['MS-'+str(i)for i in range(203)];verification=exclusive(root/'source-verification.json',dict(task_ids=msids,receipts={}))
            ms=exclusive(root/'MS.json',dict(technical_complete=True,training_commit='df6a16403e10d51097db8c88829909c533d15652',task_ids=msids,receipts={},source_verification_ref=verification))
            auth=exclusive(root/'authorization.json',dict(closure_commit=r.BASE));exit_ref=exclusive(root/'exit.json',dict(instances=[]))
            source=exclusive(root/'anchors.json',dict(producer_commit=r.BASE,refs=dict(config=exclusive(root/'config.json',old),producer_permit=producer_ref,
                controller=exclusive(root/'old-controller.json',dict(authorization=auth)),authorization=auth,
                failure=exclusive(root/'old-failure.json',dict(error="RuntimeError('owned child technical failure: M_BASE-probe')")),probe_failure=ref(probe_old/'failure.json'),
                MS_boundary=ms,source_verification=verification,upstream_boundary=exclusive(root/'upstream.json',dict(boundaries={'MS':ms},owned_exited=[])),exit_verification=exit_ref)))
            proof['source_anchors_ref']=source
            previous=copy.deepcopy(proof)
            for key in ('parent_adoption_ref','all_m_policy_ref','candidate_config_refs','policy_inclusion_ref'):previous.pop(key,None)
            previous_ref=exclusive(root/'prior-adoption.json',previous)
            proof['parent_adoption_ref']=previous_ref;stack.enter_context(patch.object(r,'PRIOR_REUSE_REF',previous_ref))
            stack.enter_context(patch.object(r,'SOURCE_REF',source))
            inclusion=r.compile_policy_inclusion(previous,old,new)
            inclusion_ref=exclusive(root/'inclusion.json',inclusion);stack.enter_context(patch.object(r,'INCLUSION_REF',inclusion_ref))
            proof=r.adopt_numeric_defaults(previous,new,inclusion)
            # A separate finalized synthetic proof; preserve every earlier fixture record.
            reuse=exclusive(root/'reuse-final.json',proof);stack.enter_context(patch.object(r,'REUSE_REF',reuse))
            ctx['probe_root']=probe_new;probe_new.mkdir(parents=True);events.clear();fresh=dict(permit,commit='synthetic-current-reviewed',protocol_sha=digest(new),probe_recovery_ref=reuse,execution_attempt=r.ATTEMPT)
            exclusive(probe_new/'approval.json',fresh)
            yield new,fresh,ctx,events,root,stack
    def test_actual_resume40_AUDIT_once21_and_mixed_manifest(self):
        with self.fixture()as(c,a,ctx,events,root,stack):
            audit=stack.enter_context(patch.object(native,'validate_probe_completion',wraps=native.validate_probe_completion))
            report=native.run_probe(c,a);self.assertEqual(audit.call_count,0);self.assertEqual(len(events),40)
            self.assertTrue(all(task_by_id(c,t)['model']in ('ModernTCN','TimeXer')for t,phase in events));self.assertFalse(any(task_by_id(c,t)['model']=='ModernTCN'and task_by_id(c,t)['dataset']=='ETTh1'for t,phase in events))
            self.assertEqual(report['budget']['actual'],dict(adam=1056,backward=1064,forward=1432));self.assertEqual(report['budget']['new_actual'],dict(adam=240,backward=240,forward=320))
            compare=stack.enter_context(patch.object(native,'compare',wraps=native.compare));summary=q.audit_probe(c)
            self.assertEqual(audit.call_count,1);self.assertEqual(compare.call_count,60);self.assertEqual(len(bound(summary)['decisions']),21)
            self.assertTrue(all(row['passed']and not row['bitwise_equal']for g in s.probe_groups(c)[16:]for row in report['decisions'][g['id']]['numerical_comparisons']))
            manifest=bound(bound(summary)['manifest_ref']);scan_manifest(manifest,bound(summary)['complete_ref'],ctx['probe_root'])
            self.observations.append(dict(retained128_not_dispatched=True,new_workers=len(events),probe_tail_audits=0,AUTO_AUDIT=audit.call_count,final_numeric_replays=compare.call_count,groups=21,budget=report['budget']))
    def test_corrupted_old_bytes_rejected_before_new_dispatch(self):
        with self.fixture()as(c,a,ctx,events,root,stack):
            p=next(Path(p)for p in r.evidence()['artifacts']if p.endswith('trajectory.json'));p.write_text('{}')
            with self.assertRaises(ValueError):native.run_probe(c,a)
            self.assertEqual(events,[])
    def test_partial16_not_whole_M_admission(self):
        with self.fixture()as(c,a,ctx,events,root,stack):
            report=dict(purpose='native_successor_probe_complete_v1',scope=ctx['probe_scope'],plan=s.plan(c),execution_complete=True,decisions=r.evidence()['decisions'])
            with self.assertRaises(ValueError):native.validate_probe_completion(c,report)
    def test_wrong_permit_and_nonregistered_failure_still_refused(self):
        with self.fixture()as(c,a,ctx,events,root,stack):
            with self.assertRaises(PermissionError):r.load_seed(c,dict(a,probe_recovery_ref=previous.REUSE_REF))
            anchor=bound(r.SOURCE_REF);anchor['refs']['failure']=exclusive(root/'wrong-failure.json',dict(error='another failure'))
            with patch.object(r,'SOURCE_REF',exclusive(root/'wrong-source.json',anchor)),self.assertRaises(ValueError):r.verify_registered_source()
            self.assertEqual(events,[])
    def test_later_numeric_failure_stops_without_resource_fallback(self):
        import m5_formal_entry as tool
        with self.fixture()as(c,a,ctx,events,root,stack):
            make=tool.make_config.side_effect
            def inject(*args,**kwargs):
                cfg=make(*args,**kwargs)
                if Path(cfg['output']).parent.name=='q4':
                    p=Path(cfg['output'])/'trajectory.json';v=bound(ref(p));v['trajectory'][0]['loss']+=1;p.write_text(json.dumps(v))
                return cfg
            stack.enter_context(patch.object(tool,'make_config',side_effect=inject))
            with self.assertRaisesRegex(ValueError,'numeric gate failure'):native.run_probe(c,a)
            self.assertEqual(len(events),8);self.assertFalse(any(phase=='q2'for _,phase in events))
    def test_MS_reuse_has_zero_training_test_and_no_checkpoint_scan(self):
        from utils import ch3_ms_seal_recovery as ms
        with self.fixture()as(c,a,ctx,events,root,stack),patch.object(ms,'snapshot',side_effect=AssertionError('no MS scan')),patch.object(ms,'verify_source',side_effect=AssertionError('no MS import')),patch.object(q,'closure',return_value='synthetic-current-reviewed'):
            v=bound(r.import_ms());self.assertEqual(v['new_MS_training_calls'],0);self.assertEqual(v['new_MS_test_calls'],0);q.validate_upstream_boundary_light(ref(q.CONTROL/'upstream-technical-boundary.json'))

if __name__=='__main__':unittest.main()
