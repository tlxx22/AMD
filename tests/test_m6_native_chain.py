"""Explicit synthetic gates/owned control, zero model/optimizer/GPU calls."""
import copy
import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from utils import ch3_native_chain as chain
from utils import ch3_native_tasks as scope
from utils import ch3_native_execution as execution
from utils import ch3_m_tasks as source
from utils.ch3_contract import profile,digest
from utils.ch3_m_launch import exclusive_json


class NativeChainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cs=chain.configs()
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=scope.PACKAGE/'fixtures');self.root=Path(self.tmp.name);self.addCleanup(self.tmp.cleanup)

    def state_fixture(self,fail=None,stop_at=None,effect='poor'):
        root=self.root/'control';root.mkdir();calls=[];permits=[]
        def child(c,probe,ref):
            stage=scope.context(c)['stage'];name=stage+('-probe' if probe else '-formal');calls.append(name)
            if name==fail:raise ValueError('synthetic technical failure')
            if name==stop_at:(root/'STOP').touch()
            return dict(technical_complete=True,result_review='pending',effect=effect)
        def permit(c,probe,boundary,replacement=None,admission=None):
            chain.stop_check(root);stage=scope.context(c)['stage'];name=stage+('-probe' if probe else '-formal')
            if stage=='m':self.assertIsNotNone(replacement)
            permits.append(name);return dict(path=str(self.root/(name+'.json')),sha256='synthetic')
        with ExitStack() as stack:
            for target,value in [('ROOT_CONTROL',root),('configs',lambda:self.cs),('binding',lambda c:dict(protocol_sha=digest(c))),('retirement_boundary',lambda:dict(kind='old_MS_user_authorized_retirement_boundary_v1',technical_complete=False,retirement_accepted=True,result_review='pending')),('seal',lambda p,v:dict(path=str(p),sha256='synthetic')),('old_path',lambda:self.root/'old.json'),('replacement_path',lambda:self.root/'replacement.json'),('build_replacement_boundary',lambda c:dict(technical_complete=True,result_review='pending')),('create_permit',permit),('run_owned',child),('audit_probe',lambda c,p:dict(path='synthetic-admission',sha256='synthetic'))]:stack.enter_context(patch.object(chain,target,value))
            if fail or stop_at:
                with self.assertRaises((ValueError,InterruptedError)):chain.run(root,sleep=lambda _:None,interval=0)
            else:chain.run(root,sleep=lambda _:None,interval=0)
        return root,calls,permits

    def test_complete_chain_without_any_manual_permit(self):
        root,calls,permits=self.state_fixture()
        self.assertEqual(calls,['tmark-probe','tmark-formal','m-probe','m-formal']);self.assertEqual(permits,calls)
        complete=json.loads((root/'complete.json').read_text());self.assertTrue(complete['technical_complete']);self.assertEqual(complete['result_review'],'pending');self.assertFalse(complete['manual_review'])
        self.assertEqual([json.loads(x)['state'] for x in (root/'states.jsonl').read_text().splitlines()],list(chain.STATES))
        self.assertFalse(complete['original_old_batch_technical_complete']);self.assertTrue(complete['old_retirement_accepted'])

    def test_tmark_failure_cannot_dispatch_replacement(self):
        root,calls,permits=self.state_fixture(fail='tmark-probe');self.assertEqual(calls,['tmark-probe']);self.assertEqual(permits,['tmark-probe']);self.assertFalse((root/'complete.json').exists());self.assertTrue((root/'failure.json').exists())

    def test_replacement_failure_cannot_dispatch_M(self):
        _,calls,permits=self.state_fixture(fail='tmark-formal');self.assertEqual(calls,['tmark-probe','tmark-formal']);self.assertNotIn('m-probe',permits)

    def test_poor_replacement_effect_still_continues_M(self):
        _,calls,_=self.state_fixture(effect='worse than legacy');self.assertEqual(calls[-2:],['m-probe','m-formal'])

    def test_M_probe_failure_cannot_dispatch_84(self):
        _,calls,permits=self.state_fixture(fail='m-probe');self.assertEqual(calls[-1],'m-probe');self.assertNotIn('m-formal',permits)

    def test_STOP_after_probe_before_permit_prevents_formal(self):
        _,calls,permits=self.state_fixture(stop_at='tmark-probe');self.assertEqual(calls,['tmark-probe']);self.assertEqual(permits,['tmark-probe'])

    def test_wait_old_scope_does_not_query_GPU_or_signal(self):
        root=self.root/'control';root.mkdir();events=[]
        def missing():events.append('receipt-refused');raise ValueError('ordinary STOP without user retirement receipt')
        with patch.object(chain,'ROOT_CONTROL',root),patch.object(chain,'configs',return_value=self.cs),patch.object(chain,'binding',side_effect=lambda c:dict(protocol_sha=digest(c))),patch.object(chain,'retirement_boundary',side_effect=missing),patch.object(chain,'create_permit')as grant,patch.object(chain,'GPULock')as gpu,patch.object(chain.os,'kill')as kill:
            with self.assertRaises(ValueError):chain.run(root,interval=0)
            grant.assert_not_called();gpu.assert_not_called();kill.assert_not_called()
        self.assertEqual(events,['receipt-refused'])

    def test_old_failure_or_identity_change_stops(self):
        root=self.root/'control';root.mkdir()
        with patch.object(chain,'ROOT_CONTROL',root),patch.object(chain,'configs',return_value=self.cs),patch.object(chain,'binding',return_value={}),patch.object(chain,'retirement_boundary',side_effect=ValueError('wrong retirement SHA/identity')),patch.object(chain,'create_permit')as grant:
            with self.assertRaises(ValueError):chain.run(root)
            grant.assert_not_called()

    def test_boundary_exclusive_matching_reuse_and_tamper_reject(self):
        p=self.root/'boundary.json';v=dict(technical_complete=True,result_review='pending');a=chain.seal(p,v);self.assertEqual(chain.seal(p,v),a)
        p.write_text('{}\n')
        with self.assertRaises(ValueError):chain.seal(p,v)
        self.assertEqual(p.read_text(),'{}\n')

    def test_new_scope_ignores_retired_waiter_evidence(self):
        from utils import ch3_m_handoff as old
        self.assertNotEqual(chain.ROOT_CONTROL,old.ROOT);self.assertNotEqual(chain.LOG,source.PACKAGE/'m-handoff-launcher.log')
        self.assertNotEqual(chain.old_path(),source.PACKAGE/'old-queue-technical-boundary.json')
        self.assertEqual(old.OLD_QUEUE_START_ANCHOR['sha256'],'829215cc1fd95dcdf9b6dbdf0e2d44f7974224784a49082ca9dd38cfc0f87ecd')
        for c in self.cs.values():
            self.assertTrue(chain.permit_path(c,True).is_relative_to(scope.PACKAGE));self.assertNotEqual(chain.permit_path(c,True),source.PACKAGE/'auto-probe-permit.json')

    def test_exact_protocol_scope_rejects_cross_file_before_fixture(self):
        c=self.cs['tmark'];s=dict(protocol_file=str(scope.PROFILE_FILE),protocol_sha=digest(c),successor_scope=scope.context(c)['probe_scope'])
        self.assertEqual(execution.read_config(s),c)
        for bad in [dict(s,protocol_file='/tmp/other.json'),dict(s,successor_scope=scope.context(self.cs['m'])['probe_scope']),dict(s,protocol_sha='wrong')]:
            with self.assertRaises((ValueError,PermissionError)):execution.read_config(bad)

    def test_output_replacement_and_M_are_distinct_no_legacy_overwrite(self):
        for c in self.cs.values():
            t=c['tasks'][0];out=scope.result_path(c,t);self.assertEqual(execution.exact_path(c,t,False,None,out),out)
            with self.assertRaises(ValueError):execution.exact_path(c,t,False,None,out.parent/t.get('parent_run_id','wrong'))
        self.assertNotEqual(scope.context(self.cs['m'])['result_root'],source.RESULT)

    def test_template_or_old_machine_permit_cannot_authorize(self):
        c=self.cs['m'];fake=dict(reviewed=True,execution_permitted=True,commit=scope.BASE)
        with patch.object(chain,'binding',return_value={}):
            with self.assertRaises(ValueError):chain.validate_permit(c,fake,False)

    def test_positive_probe_permit_and_binding_mutations(self):
        c=self.cs['tmark'];p=self.root/'probe-permit.json';boundary=dict(path=str(self.root/'boundary.json'),sha256='fixture')
        context=dict(commit='synthetic-closure',code={'fixture':'sha'},protocol_sha=digest(c),environment={'fixture':1},hardware={'fixture':2},plan={'fixture':3},data_binding_artifact={'fixture':4},data_bindings={'fixture':5},source_states={'fixture':6})
        with patch.object(chain,'ROOT_CONTROL',self.root),patch.object(chain,'binding',return_value=context),patch.object(chain,'permit_path',return_value=p),patch.object(chain,'check_boundary'),patch.object(chain,'readiness',side_effect=lambda c,a,probe:execution.authorization_reasons(c,a,probe)),patch.object(execution,'gpu_environment_reasons',return_value=[]):
            ref=chain.create_permit(c,True,boundary);a=source.bound(ref);chain.validate_permit(c,a,True)
            self.assertFalse(a['reviewed']);self.assertFalse(a['manual_review']);self.assertEqual(a['purpose'],'ch3_resource_probe')
            for key in ('commit','protocol_sha','code','environment','hardware','data_bindings','authorized_task_ids','caps','successor_scope'):
                wrong=dict(a);wrong[key]='wrong'
                with self.subTest(key=key),self.assertRaises(ValueError):chain.validate_permit(c,wrong,True)
            with self.assertRaises(FileExistsError):chain.create_permit(c,True,boundary)

    def test_new_M_permit_requires_replacement_boundary(self):
        c=self.cs['m'];context=dict(commit='fixture',protocol_sha=digest(c));p=self.root/'m-probe-permit.json'
        with patch.object(chain,'ROOT_CONTROL',self.root),patch.object(chain,'binding',return_value=context),patch.object(chain,'permit_path',return_value=p),patch.object(chain,'check_boundary'),patch.object(chain,'readiness',side_effect=lambda c,a,probe:execution.authorization_reasons(c,a,probe)),patch.object(execution,'gpu_environment_reasons',return_value=[]):
            with self.assertRaises(PermissionError):chain.create_permit(c,True,dict(path='old',sha256='fixture'))
            self.assertFalse(p.exists())

    def test_launcher_token_exact_inode_and_single_claim(self):
        root=self.root;x={k:root/k for k in ('log','metadata','claim')};x['log'].touch();token='a'*64
        value=dict(kind='handoff',owner=dict(pid=os.getpid(),start_ticks='fixture'),log=str(x['log']),log_identity=chain.file_identity(x['log']),commit='fixture',token_sha256=chain.hashlib.sha256(token.encode()).hexdigest(),permit=None);exclusive_json(x['metadata'],value)
        with patch.object(chain,'launch_paths',return_value=x),patch.object(chain,'git',return_value='fixture'),patch.object(chain,'tmux_view',return_value={}):
            chain.verify_launch('handoff',token)
            for bad in (None,'b'*64):
                with self.assertRaises((ValueError,PermissionError)):chain.verify_launch('handoff',bad)
            with self.assertRaises(ValueError):chain.verify_launch('m-probe',token)
            x['claim'].touch()
            with self.assertRaises(ValueError):chain.verify_launch('handoff',token)

    def test_safe_stop_waiter_signals_only_exact_supervisor(self):
        root=self.root;exclusive_json(root/'controller.json',dict(pid=123,start_ticks='123'))
        with patch.object(chain,'ROOT_CONTROL',root),patch.object(chain,'same',return_value=True),patch.object(chain,'command_has',return_value=True),patch.object(chain.os,'kill')as kill:
            result=chain.safe_stop();kill.assert_called_once_with(123,chain.signal.SIGTERM);self.assertFalse(result['old_MS_signal_sent'])

    def test_signal_owned_child_rejects_foreign_PID_scope(self):
        owner=dict(pid=10,start_ticks='10');v=dict(pid=11,start_ticks='11',kind='m-probe',owner=dict(pid=9,start_ticks='9'))
        with patch.object(chain,'same',return_value=True),patch.object(chain.os,'kill')as kill:
            with self.assertRaises(ValueError):chain.signal_stage(v,owner)
            kill.assert_not_called()

    def test_permits_O_EXCL_no_overwrite(self):
        p=self.root/'permit.json';exclusive_json(p,dict(synthetic=True))
        with self.assertRaises(FileExistsError):exclusive_json(p,dict(synthetic=False))
        self.assertEqual(json.loads(p.read_text()),dict(synthetic=True))

    def test_old_production_anchor_bytes_remain_fixed(self):
        from utils.ch3_m_handoff import OLD_QUEUE_START_ANCHOR
        self.assertEqual(source.sha(OLD_QUEUE_START_ANCHOR['path']),OLD_QUEUE_START_ANCHOR['sha256'])

    def test_fixed_source_proof_accepts_exact_and_rejects_tamper(self):
        c=self.cs['m'];raw=b'synthetic closure file';code={'fixture.py':chain.hashlib.sha256(raw).hexdigest(),'utils/ch3_native_chain.py':'synthetic-policy-code'}
        p=self.root/'proof.json';exclusive_json(p,dict(after_code={'fixture.py':code['fixture.py']},policy_code_sha256=chain.policy_code_sha256()))
        refs=dict(chain.PREPARATION,source_proof=source.ref(p));prior=source.bound(refs['environment_source']);data=source.bound(refs['data_m'])
        def git(*args):
            return {'rev-parse HEAD':'synthetic-closure','branch --show-current':'m6/m-baselines-v1','status --porcelain --untracked-files=all':'','rev-parse HEAD^':scope.BASE}[' '.join(args)]
        with ExitStack() as stack:
            for name,value in [('PREPARATION',refs),('git',git),('code_binding',lambda:code),('environment_binding',lambda:prior['environment']),('hardware_binding',lambda:prior['hardware'])]:stack.enter_context(patch.object(chain,name,value))
            stack.enter_context(patch('utils.ch3_m6.source_states',return_value=data['source_states']))
            stack.enter_context(patch.object(chain.subprocess,'check_output',return_value=raw))
            # Synthetic chain file hash is intentionally replaced for the exact committed-byte check.
            code['utils/ch3_native_chain.py']=code['fixture.py']
            chain.binding(c,worker=True)
            p.write_text('{}\n')
            with self.assertRaises(ValueError):chain.binding(c,worker=True)
            refs['source_proof']=source.ref(p)
            with self.assertRaises((ValueError,KeyError)):chain.binding(c,worker=True)

    def test_policy_fingerprint_exempts_only_recursive_SHA_literal(self):
        text=Path(chain.__file__).read_text();p=self.root/'policy.py';p.write_text(text);expected=chain.policy_code_sha256(p)
        proof_sha=chain.PREPARATION['source_proof']['sha256'];p.write_text(text.replace(proof_sha,'0'*64))
        self.assertEqual(chain.policy_code_sha256(p),expected)
        p.write_text(text.replace("'WAIT_OLD_MS_RETIREMENT'","'OTHER_OLD_MS'"));self.assertNotEqual(chain.policy_code_sha256(p),expected)
        p.write_text(text.replace('source-inheritance.json','unreviewed-source.json'));self.assertNotEqual(chain.policy_code_sha256(p),expected)

    def owned_fixture(self,stop=False):
        import subprocess,sys
        c=self.cs['tmark'];p=self.root/'permit.json';exclusive_json(p,dict(synthetic_fixture=True));paths={k:self.root/k for k in ('log','metadata','claim')};complete=self.root/'stage-complete.json';marker=self.root/'child-ended'
        real_popen=subprocess.Popen;children=[]
        script="import time;from pathlib import Path;time.sleep(.15);Path("+repr(str(marker))+").touch()"
        if stop:script="import time;from pathlib import Path;Path("+repr(str(self.root/'STOP'))+").touch();time.sleep(5)"
        def spawn(*args,**kwargs):
            child=real_popen([sys.executable,'-B','-c',script],**kwargs);children.append(child);return child
        def verify(*args):
            self.assertTrue(marker.exists());self.assertEqual(children[0].poll(),0);exclusive_json(complete,dict(synthetic_fixture=True,technical_complete=True))
        def signal(v,owner):
            self.assertEqual(v['owner'],dict(pid=owner['pid'],start_ticks=owner['start_ticks']));self.assertTrue(chain.same(v));os.kill(v['pid'],chain.signal.SIGTERM);return True
        with ExitStack() as stack:
            for target,value in [('ROOT_CONTROL',self.root),('launch_paths',lambda kind:paths),('readiness',lambda *args:[]),('prepare_launch',lambda *args:'a'*64),('verify_stage',verify),('signal_stage',signal)]:stack.enter_context(patch.object(chain,target,value))
            stack.enter_context(patch.object(chain.subprocess,'Popen',side_effect=spawn));stack.enter_context(patch.object(execution,'roots',return_value=self.root))
            if stop:
                with self.assertRaises(InterruptedError):chain.run_owned(c,True,source.ref(p))
            else:
                self.root.joinpath('complete.json').write_text('{"synthetic_fixture":true}\n')
                result=chain.run_owned(c,True,source.ref(p));self.assertEqual(result['exit_code'],0);self.assertFalse(chain.same(result['child']))
                with self.assertRaises(FileExistsError):chain.run_owned(c,True,source.ref(p))
        self.assertEqual(len(children),1);self.assertIsNotNone(children[0].poll());self.assertFalse(chain.same(chain.identity(children[0].pid)))

    def test_owned_synthetic_child_real_exit_before_handoff_and_duplicate_reject(self):
        self.owned_fixture()

    def test_owned_synthetic_child_STOP_exits_without_formal_dispatch(self):
        self.owned_fixture(stop=True)

    def artifact_namespace_fixture(self):
        c=self.cs['tmark'];model='TimeMixer';ctx=scope.context(c);ctx['result_root']=self.root
        root=self.root/('formal-'+model);root.mkdir()
        tasks=[t for t in c['tasks'] if t['model']==model]
        for t in tasks:(root/t['id']).mkdir()
        return c,model,ctx,root,tasks

    def test_completed_group_exact_29_namespace_and_no_foreign_result(self):
        c,model,ctx,root,tasks=self.artifact_namespace_fixture()
        with patch.object(scope,'context',return_value=ctx):
            self.assertEqual(execution.group_artifact_namespace(c,model),tasks)
            (root/'unregistered-run').mkdir()
            with self.assertRaises(ValueError):execution.group_artifact_namespace(c,model)

    def test_completed_group_missing_or_linked_run_rejected(self):
        c,model,ctx,root,tasks=self.artifact_namespace_fixture();p=root/tasks[0]['id'];p.rmdir()
        with patch.object(scope,'context',return_value=ctx):
            with self.assertRaises(ValueError):execution.group_artifact_namespace(c,model)
            p.symlink_to(root/tasks[1]['id'])
            with self.assertRaises(ValueError):execution.group_artifact_namespace(c,model)

    def test_checkpoint_presence_allows_same_epoch_equal_bytes(self):
        for name in ('best.pt','last.pt'):(self.root/name).write_bytes(b'explicit non-checkpoint synthetic bytes')
        self.assertEqual(execution.checkpoint_files(self.root),[self.root/'best.pt',self.root/'last.pt'])
        (self.root/'last.pt').unlink()
        with self.assertRaises(ValueError):execution.checkpoint_files(self.root)

    def test_checkpoint_shared_link_rejected_without_weight_load(self):
        (self.root/'best.pt').write_bytes(b'explicit synthetic');(self.root/'last.pt').symlink_to(self.root/'best.pt')
        with self.assertRaises(ValueError):execution.checkpoint_files(self.root)
        (self.root/'last.pt').unlink();os.link(self.root/'best.pt',self.root/'last.pt')
        with self.assertRaises(ValueError):execution.checkpoint_files(self.root)


class SavedProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.c=chain.configs()['m']
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=scope.PACKAGE/'fixtures');self.root=Path(self.tmp.name);self.addCleanup(self.tmp.cleanup);self.c=copy.deepcopy(self.c);self.c['groups']=self.c['groups'][:1]
        self.ctx=scope.context(self.c);self.ctx.update(probe_root=self.root/'probe',control=self.root/'formal');self.ctx['probe_root'].mkdir()
        self.a=dict(commit='synthetic',protocol_sha=digest(self.c),code={},environment={},hardware={'cpu_affinity':[]});exclusive_json(self.ctx['probe_root']/'approval.json',self.a)
    def engine(self, failures=None,numeric=True,finite=True):
        import sys
        sys.path.insert(0,str(scope.ROOT/'tools/restricted_regression'));import m5_formal_entry
        attempts=[];failures=failures or {}
        def config(c,purpose,out,task,approval):
            out=Path(out);out.mkdir(parents=True);phase=out.parent.name
            cfg=dict(purpose=purpose,task=task,output=str(out),approval=approval,protocol_sha=digest(c),successor_scope=self.ctx['probe_scope'],successor_phase=phase,prefix_files={},limits=dict(adam=6,forward=8,backward=6,seconds=1800),budget_file=str(out/'budget.json'))
            pid=9000000+c['tasks'].index(next(t for t in c['tasks'] if t['id']==task));counts=dict(adam=6,forward=8,backward=6)
            exclusive_json(out/'config.json',cfg);exclusive_json(out/'budget.json',dict(counts=counts,by_pid={str(pid):counts}))
            from ch3_runner import memory_growth_review
            memory=[dict(allocated=0,rss_before_hash=1024,rss_after_hash=1024) for _ in range(6)]
            exclusive_json(out/'trajectory.json',dict(id=task,profile_sha=digest(profile(c,next(t for t in c['tasks'] if t['id']==task))),finite=finite,time_mark=profile(c,next(t for t in c['tasks'] if t['id']==task)).get('time_mark'),M_full_state_trace=[{}]*6,threads=4,affinity=[],trajectory=[dict(loss=1.)]*6,validation=dict(mse=1.,mae=1.,sse=1.,sae=1.),memory=memory,memory_review=memory_growth_review(memory)))
            exclusive_json(out/'runtime.json',dict(synthetic_fixture=True,pid=pid,task=task,error=None));(out/'audit.jsonl').write_text('');return cfg
        def measured(configs,out,monitor):
            out=Path(out);out.mkdir(parents=True);phase=out.parent.name;attempts.append(phase);failure=failures.get(phase)
            row=dict(failure='synthetic' if failure else None,failure_kind=failure,returncodes=[1 if failure else 0]*len(configs),resource_admission=not bool(failure),elapsed=.1 if phase!='serial' else 1.)
            pids=[str(json.loads((Path(s['output'])/'runtime.json').read_text())['pid']) for s in configs]
            row.update(exit_transitions_resolved=True,process_attribution='Measured',process_peaks={pid:1 for pid in pids},cpu_peaks={pid:1024 for pid in pids})
            exclusive_json(out/'process.json',row);(out/'memory.jsonl').write_text(json.dumps(dict(owned_pid_metadata={pid:dict(start_ticks='0') for pid in pids}))+'\n');return row
        stack=ExitStack();stack.enter_context(patch.object(scope,'context',return_value=self.ctx));stack.enter_context(patch.object(m5_formal_entry,'make_config',side_effect=config));stack.enter_context(patch.object(m5_formal_entry,'run_configs',side_effect=measured));stack.enter_context(patch.object(execution,'compare',return_value=dict(passed=numeric)));return stack,attempts

    def test_saved_probe_positive_replay_and_actual_budget(self):
        stack,_=self.engine()
        with stack:
            r=execution.run_probe(self.c,self.a);execution.validate_probe_completion(self.c,r)
            self.assertEqual(r['budget']['actual'],dict(adam=48,forward=64,backward=48));self.assertEqual(r['budget']['actual'],r['budget']['reserved'])

    def test_resource_q4_failure_q2_pass_keeps_cost(self):
        stack,_=self.engine({'q4':'resource'})
        with stack:
            r=execution.run_probe(self.c,self.a);d=next(iter(r['decisions'].values()));self.assertEqual(d['concurrency'],2);self.assertEqual(r['budget']['reserved'],dict(adam=72,forward=96,backward=72))

    def test_resource_q4_q2_failure_legal_q1(self):
        stack,_=self.engine({'q4':'resource','q2':'resource'})
        with stack:
            r=execution.run_probe(self.c,self.a);self.assertEqual(next(iter(r['decisions'].values()))['concurrency'],1);self.assertEqual(r['budget']['reserved'],dict(adam=60,forward=80,backward=60))

    def test_unknown_or_business_failure_cannot_fallback(self):
        for kind in ('business','unknown'):
            with self.subTest(kind=kind):
                self.setUp();stack,attempts=self.engine({'q4':kind})
                with stack:
                    with self.assertRaises(RuntimeError):execution.run_probe(self.c,self.a)
                    self.assertNotIn('q2',attempts)

    def test_numeric_failure_cannot_dispatch_q2(self):
        stack,attempts=self.engine(numeric=False)
        with stack:
            with self.assertRaises(ValueError):execution.run_probe(self.c,self.a)
            self.assertNotIn('q2',attempts);self.assertTrue((self.ctx['probe_root']/'failure.json').exists())

    def test_actual_q4_numeric_failure_keeps_cost_and_never_fallbacks(self):
        stack,attempts=self.engine()
        with stack,patch.object(execution,'compare',side_effect=lambda c,t,x,y:dict(passed=x is y)):
            with self.assertRaises(ValueError):execution.run_probe(self.c,self.a)
            self.assertIn('q4',attempts);self.assertNotIn('q2',attempts)
            failed=json.loads((self.ctx['probe_root']/'failure.json').read_text())
            self.assertEqual(failed['budget']['actual'],dict(adam=48,forward=64,backward=48));self.assertFalse(failed['budget']['refund'])
            self.assertFalse((self.ctx['probe_root']/'complete.json').exists())

    def test_finite_failure_cannot_be_complete(self):
        stack,_=self.engine(finite=False)
        with stack:
            with self.assertRaises(ValueError):execution.run_probe(self.c,self.a)
            self.assertFalse((self.ctx['probe_root']/'complete.json').exists())

    def test_modified_artifact_SHA_and_budget_rejected(self):
        stack,_=self.engine()
        with stack:
            r=execution.run_probe(self.c,self.a);p=Path(next(iter(r['artifacts'])));original=p.read_bytes();p.write_bytes(original+b' ')
            with self.assertRaises(ValueError):execution.validate_probe_completion(self.c,r)
            p.write_bytes(original);r['budget']['reserved']['adam']=99999
            with self.assertRaises(ValueError):execution.validate_probe_completion(self.c,r)

    def test_STOP_before_dispatch_has_zero_reservation(self):
        (self.ctx['probe_root']/'STOP').touch();stack,attempts=self.engine()
        with stack:
            with self.assertRaises(InterruptedError):execution.run_probe(self.c,self.a)
            self.assertEqual(attempts,[])
