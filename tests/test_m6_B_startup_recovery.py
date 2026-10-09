"""B pre-controller recovery tests: synthetic GPU responses and CPU children only."""
import ast,contextlib,copy,hashlib,json,os,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from utils import ch3_third_round_after_selection as b,ch3_type1_chain as q,ch3_type1_tasks as s,ch3_event_resources as event
from utils.ch3_native_recovery_records import bound,exclusive,ref
from utils.ch3_contract import digest
b.activate()
REAL_CHECK_OUTPUT=subprocess.check_output
REAL_RUNTIME_LOG=b.LOG


def synthetic_gpu(args,**kwargs):
    if args[0]!='nvidia-smi':return REAL_CHECK_OUTPUT(args,**kwargs)
    assert kwargs['timeout']==10
    gpu=event.frozen_hardware()['gpu'];total=float(gpu.split(',')[2]);return gpu+f', 1, {total-1}, 0\n'


@contextlib.contextmanager
def fixture():
    root=Path(tempfile.mkdtemp(prefix='synthetic-startup-',dir=b.PACKAGE/'fixtures'))
    package=root/'technical';package.mkdir();realpackage=b.PACKAGE
    (package/'producer-delta-proof.v11.json').write_bytes((realpackage/'producer-delta-proof.v11.json').read_bytes())
    for stage in b.STAGES:
        name=stage.lower()+'-plan.json';(package/name).write_bytes((realpackage/name).read_bytes())
    cs=q.configs()
    with contextlib.ExitStack()as stack:
        for obj,key,value in [(b,'PACKAGE',package),(b,'LOG',package/'followup-launcher.log'),(q,'LOG',package/'followup-launcher.log'),(s,'PACKAGE',package),(s,'RESULT',root/'synthetic-result'),(q,'CONTROL',root/'synthetic-result/queue/controller'),(q,'QUEUE_LOCK',root/'queue.lock')]:stack.enter_context(patch.object(obj,key,value))
        from utils import ch3_round2_amendment as amend
        stack.enter_context(patch.object(amend,'RESULT',s.RESULT/'round2-amendment'))
        stack.enter_context(patch.object(q,'configs',return_value=cs));stack.enter_context(patch.object(q,'closure',return_value='f'*40))
        stack.enter_context(patch.object(q,'verify_live_remote',return_value='f'*40))
        import ch3_runner as runner
        realgit=runner.git;stack.enter_context(patch.object(runner,'git',side_effect=lambda *args:'f'*40 if args==('rev-parse','HEAD')else realgit(*args)))
        auth=q.start_template()
        for k in('reviewed','execution_permitted','structure_frozen','m6_authorized','budget_authorized'):auth[k]=True
        auth.update(closure_commit='f'*40,authorization_basis='CPU synthetic fixture only; not real executable permission')
        start=exclusive(root/'synthetic-start.json',auth)
        (root/'SYNTHETIC_ONLY.json').write_text('{"synthetic":true,"GPU":0,"training":0}\n')
        event.invalidate_startup();event._PRESTART=None;event._QUERY_AUDIT=None
        yield root,auth,start,stack
        event.invalidate_startup();event._PRESTART=None


def launch_fixture(start,stack):
    b.LOG.write_text('')
    from utils.ch3_m_launch import file_identity
    token='d'*64
    exclusive(str(b.LOG)+'.launch.json',dict(scope=b.ID,approval=start,log=str(b.LOG),log_identity=file_identity(b.LOG),wrapper=q.owner(),token_sha=hashlib.sha256(token.encode()).hexdigest()))
    stack.enter_context(patch('utils.ch3_m_launch.tmux_view',return_value={'synthetic_tmux':True}))
    return token


class BStartupRecovery(unittest.TestCase):
    def test_policy_is_review_bound_B_only_and_old_records_exact(self):
        self.assertEqual(b.startup_query_policy(),dict(max_attempts=3,retry_delay_seconds=2))
        old=b.historical_startup_failure();self.assertEqual(old['state'],'PRE_CONTROLLER_START_FAILED');self.assertEqual(old['formal_completed'],0)
        self.assertEqual(sha_file(old['log_ref']['path']),'3b8f1adc66153453662ab14a32cde82f0c193bea6f98ff384f04c4445ccc8772')
        with patch.object(q,'PROBE_RECOVERY',None):self.assertRaises(PermissionError,b.startup_query_policy)

    def test_no_executable_current_permission_and_old_attempt_rejected(self):
        self.assertFalse((b.PACKAGE/'start-review.json').exists());self.assertFalse(b.RESULT.exists())
        old=bound(b.historical_startup_failure()['authorization_ref'])
        with patch.object(q,'closure',return_value=old['closure_commit']):self.assertRaises(PermissionError,q.validate_start,old)
        from utils import ch3_patchtst_width_reproduction as a
        A=bound(ref(a.PACKAGE/'separate-entrypoints-v1/start-review.json'))
        with patch.object(q,'closure',return_value=A['closure_commit']):self.assertRaises(PermissionError,q.validate_start,A)
        r6=bound(ref(b.parent.EVENT_PACKAGE/'start-review.json'))
        with patch.object(q,'closure',return_value=r6['closure_commit']):self.assertRaises(PermissionError,q.validate_start,r6)

    def test_public_preflight_real_checks_one_synthetic_sample(self):
        with fixture()as(root,auth,start,stack),patch.object(subprocess,'check_output',side_effect=synthetic_gpu)as query:
            report=q.readiness_report(auth)
            self.assertEqual(report['blocked'],[]);self.assertTrue(report['READY_TO_ARM_HANDOFF']);self.assertFalse(report['READY_FOR_GPU_EXECUTION']);self.assertEqual(report['remaining_formal_runs'],371)
            self.assertEqual(sum(call.args[0][0]=='nvidia-smi'for call in query.call_args_list),1)
            self.assertEqual(report['startup_query_audit']['status'],'SAMPLE_ACCEPTED');self.assertFalse(s.RESULT.exists())

    def test_light_wrapper_check_performs_no_GPU_query(self):
        with fixture()as(root,auth,start,stack):
            def query(args,**kwargs):
                if args[0]=='nvidia-smi':raise AssertionError('light arm check must not query GPU')
                return REAL_CHECK_OUTPUT(args,**kwargs)
            with patch.object(subprocess,'check_output',side_effect=query):result=q.launch_check(auth)
            self.assertEqual(result['GPU_queries'],0);self.assertFalse(b.LOG.exists());self.assertFalse(s.RESULT.exists())

    def test_exact_1_2_3_attempts_and_fixed_delay(self):
        for timeouts in(0,1,2):
            with fixture()as(root,auth,start,stack):
                calls=[]
                def query(args,**kwargs):
                    if args[0]!='nvidia-smi':return REAL_CHECK_OUTPUT(args,**kwargs)
                    calls.append(kwargs['timeout'])
                    if len(calls)<=timeouts:raise subprocess.TimeoutExpired(args,10)
                    return synthetic_gpu(args,**kwargs)
                with patch.object(subprocess,'check_output',side_effect=query),patch.object(event.time,'sleep')as sleep:
                    with event.prestart(auth):self.assertIsNotNone(event._PRESTART)
                self.assertEqual(calls,[10]*(timeouts+1));self.assertEqual(sleep.call_count,timeouts);self.assertTrue(all(x.args==(2,)for x in sleep.call_args_list))
                audit=event.query_audit();self.assertEqual(len(audit['attempts']),timeouts+1);self.assertEqual(audit['status'],'SAMPLE_ACCEPTED')
                self.assertTrue(all(x['elapsed']>=0 for x in audit['attempts']));self.assertIsNone(event._PRESTART)

    def test_three_timeouts_refuse_and_clear_previous_grant(self):
        with fixture()as(root,auth,start,stack):
            event._LAST={'stale':True};calls=[]
            def query(args,**kwargs):
                if args[0]!='nvidia-smi':return REAL_CHECK_OUTPUT(args,**kwargs)
                calls.append(kwargs['timeout']);raise subprocess.TimeoutExpired(args,10)
            with patch.object(subprocess,'check_output',side_effect=query),patch.object(event.time,'sleep')as sleep:
                self.assertRaises(subprocess.TimeoutExpired,lambda:run_context(auth))
            self.assertEqual(calls,[10,10,10]);self.assertEqual(sleep.call_count,2);self.assertIsNone(event._LAST);self.assertIsNone(event._PRESTART)
            self.assertEqual(event.query_audit()['status'],'QUERY_FAILED');self.assertFalse(s.RESULT.exists())

    def test_non_timeout_commands_fail_once_without_delay(self):
        for exc in(subprocess.CalledProcessError(1,['nvidia-smi']),FileNotFoundError('nvidia-smi missing'),PermissionError('query denied')):
            with fixture()as(root,auth,start,stack):
                calls=[]
                def query(args,**kwargs):
                    if args[0]!='nvidia-smi':return REAL_CHECK_OUTPUT(args,**kwargs)
                    calls.append(1);raise exc
                with patch.object(subprocess,'check_output',side_effect=query),patch.object(event.time,'sleep')as sleep:
                    self.assertRaises(type(exc),lambda:run_context(auth))
                self.assertEqual(len(calls),1);sleep.assert_not_called();self.assertIsNone(event._LAST)

    def test_sample_identity_margin_finite_and_accounting_fail_without_retry(self):
        gpu=event.frozen_hardware()['gpu'];parts=gpu.split(',');total=float(parts[2])
        samples=[gpu+', 1, 1, 0\n',gpu+f', nan, {total-1}, 0\n',gpu+f', -1, {total+1}, 0\n',gpu+f', {total-1}, 1, 0\n','BAD-'+gpu+f', 1, {total-1}, 0\n',gpu.replace(parts[1],' OTHER',1)+f', 1, {total-1}, 0\n',gpu.rsplit(',',1)[0]+', other-driver'+f', 1, {total-1}, 0\n']
        for sample in samples:
            with fixture()as(root,auth,start,stack):
                calls=[]
                def query(args,**kwargs):
                    if args[0]!='nvidia-smi':return REAL_CHECK_OUTPUT(args,**kwargs)
                    calls.append(1);return sample
                with patch.object(subprocess,'check_output',side_effect=query),patch.object(event.time,'sleep')as sleep:
                    self.assertRaises((ValueError,PermissionError),lambda:run_context(auth))
                self.assertEqual(len(calls),1);sleep.assert_not_called();self.assertIsNone(event._LAST)

    def test_STOP_before_or_during_retry_prevents_more_queries(self):
        with fixture()as(root,auth,start,stack):
            (b.PACKAGE/'STOP').write_text('synthetic STOP\n')
            with patch.object(subprocess,'check_output',side_effect=AssertionError('STOP before GPU')):self.assertRaises(InterruptedError,lambda:run_context(auth))
        with fixture()as(root,auth,start,stack):
            calls=[]
            def query(args,**kwargs):
                if args[0]!='nvidia-smi':return REAL_CHECK_OUTPUT(args,**kwargs)
                calls.append(1);(b.PACKAGE/'STOP').write_text('synthetic STOP\n');raise subprocess.TimeoutExpired(args,10)
            with patch.object(subprocess,'check_output',side_effect=query),patch.object(event.time,'sleep')as sleep:self.assertRaises(InterruptedError,lambda:run_context(auth))
            self.assertEqual(len(calls),1);sleep.assert_not_called();self.assertEqual(event.query_audit()['status'],'RETRY_CANCELLED')

    def test_bad_auth_or_source_rejected_before_GPU(self):
        with fixture()as(root,auth,start,stack):
            bad=copy.deepcopy(auth);bad['scope']='wrong'
            with patch.object(subprocess,'check_output',side_effect=AssertionError('bad permission must not query GPU')):self.assertRaises(PermissionError,lambda:run_context(bad))
            with patch.object(b,'verify_source',side_effect=ValueError('source mismatch')),patch.object(subprocess,'check_output',side_effect=AssertionError('bad source must not query GPU')):self.assertRaises(ValueError,lambda:run_context(auth))

    def test_controller_start_single_sampling_stage_real_MAC_and_no_workers(self):
        with fixture()as(root,auth,start,stack):
            token=launch_fixture(start,stack);calls=[];reached=[]
            def query(args,**kwargs):
                if args[0]=='nvidia-smi':calls.append(1)
                return synthetic_gpu(args,**kwargs)
            stack.enter_context(patch.object(q,'run',side_effect=lambda value:reached.append(value)))
            stack.enter_context(patch.object(q.signal,'signal'))
            with patch.object(subprocess,'check_output',side_effect=query):q.start(start,token)
            self.assertEqual(len(calls),1);self.assertEqual(reached,[start]);self.assertTrue((q.CONTROL/'controller.json').exists())
            grant=event.validate_startup(event.startup_ref(),live=True);self.assertEqual(grant['startup_query_audit']['status'],'SAMPLE_ACCEPTED')
            self.assertEqual(grant['authorization_ref'],start);self.assertEqual(grant['closure_commit'],'f'*40)
            with patch.object(subprocess,'check_output',side_effect=AssertionError('runtime GPU queries forbidden')):
                event.validate_startup(event.startup_ref(),live=True);q.status()
            self.assertRaises(FileExistsError,q.verify_launch,start,token)

    def test_failed_start_snapshot_token_invalidated_and_readonly_status(self):
        with fixture()as(root,auth,start,stack):
            token=launch_fixture(start,stack);calls=[]
            def query(args,**kwargs):
                if args[0]!='nvidia-smi':return REAL_CHECK_OUTPUT(args,**kwargs)
                calls.append(1);raise subprocess.TimeoutExpired(args,10)
            with patch.object(subprocess,'check_output',side_effect=query),patch.object(event.time,'sleep'),patch.object(q,'run',side_effect=AssertionError('no workers after failed startup')):
                self.assertRaises(PermissionError,q.start,start,token)
            failure=bound(ref(b.PACKAGE/'startup-failure.json'));self.assertEqual(len(calls),3);self.assertEqual(failure['error_type'],'subprocess.TimeoutExpired');self.assertEqual(failure['formal_completed'],0)
            self.assertEqual(len(failure['query_audit']['attempts']),3);self.assertTrue(failure['launch_invalidated']);self.assertFalse(s.RESULT.exists());self.assertIsNone(event._LAST)
            b.LOG.write_text('Traceback synthetic trailing output\n')
            files={str(p):sha_file(p)for p in root.rglob('*')if p.is_file()}
            with patch.object(subprocess,'check_output',side_effect=AssertionError('status cannot query GPU')):v=q.status()
            self.assertEqual(v['state'],'PRE_CONTROLLER_START_FAILED');self.assertTrue(v['failure']);self.assertFalse(v['running']);self.assertEqual(v['previous_startup_failures'][0]['formal_completed'],0)
            self.assertEqual(files,{str(p):sha_file(p)for p in root.rglob('*')if p.is_file()})
            self.assertRaises(PermissionError,q.verify_launch,start,token)

    def test_historical_status_distinguishes_new_prepared_attempt(self):
        with fixture()as(root,auth,start,stack):
            v=q.status();self.assertEqual(v['state'],'Prepared');self.assertFalse(v['failure']);self.assertEqual(v['execution_attempt'],b.ATTEMPT)
            old=v['previous_startup_failures'][0];self.assertEqual(old['state'],'PRE_CONTROLLER_START_FAILED');self.assertNotEqual(old['execution_attempt'],b.ATTEMPT)

    def test_no_new_science_configuration_or_result_roots(self):
        cs=q.configs();self.assertEqual([len(cs[k]['tasks'])for k in b.STAGES],[168,35,168]);self.assertEqual(b.selected_main()['retained_cells'],351)
        self.assertFalse(b.RESULT.exists());self.assertFalse(b.LOG.exists());self.assertFalse((b.PACKAGE/'start-review.json').exists())
        old=bound(b.startup_contract()['science_contract_ref']);self.assertEqual(old['selected_main_ref']['sha256'],'e6a5729c0856882f237c9f53c989a6b98557900794f15e4b27deb5fe7cb3c876')

    def test_timeout_child_is_killed_and_reaped_by_real_check_output(self):
        root=Path(tempfile.mkdtemp(prefix='synthetic-timeout-child-',dir=b.PACKAGE/'fixtures'));pidfile=root/'pid.txt'
        source='import os,time;from pathlib import Path;Path('+repr(str(pidfile))+').write_text(str(os.getpid()));time.sleep(30)'
        self.assertRaises(subprocess.TimeoutExpired,REAL_CHECK_OUTPUT,[sys.executable,'-B','-c',source],text=True,timeout=.2)
        pid=int(pidfile.read_text())
        self.assertFalse((Path('/proc')/str(pid)).exists());self.assertRaises(ChildProcessError,os.waitpid,pid,os.WNOHANG)


    def test_actual_shell_arm_to_controller_has_one_GPU_sampling_stage(self):
        with fixture()as(root,auth,start,stack):
            wrapper=root/'scripts/ch3/start_third_round_after_selection.sh';wrapper.parent.mkdir(parents=True)
            bridge=root/'cpu_bridge.py'
            bridge.write_text('import sys;sys.path.insert(0,'+repr(str(q.ROOT))+');from tests.test_m6_B_startup_recovery import shell_fixture_cli;shell_fixture_cli('+repr(str(root))+')\n')
            src=b.WRAPPER.read_text().replace('M6_DEPTH_ENTRY="$M6_DEPTH_ROOT/m6_third_round_after_selection_entry.py"','M6_DEPTH_ENTRY='+repr(str(bridge))).replace(str(real_runtime_log()),str(root/'technical/followup-launcher.log'))
            # Same arm body; only fixture executable/root/log wiring differs.
            wrapper.write_text(src)
            fakebin=root/'bin';fakebin.mkdir();fake=fakebin/'tmux'
            with os.fdopen(os.open(fake,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o700),'w')as executable:executable.write('#!'+sys.executable+'\nimport sys,subprocess\nif sys.argv[1]=="has-session":sys.exit(1)\nassert sys.argv[1]=="new-session"\nsys.exit(subprocess.run(["/bin/bash","-c",sys.argv[-1]]).returncode)\n')
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PATH=str(fakebin)+os.pathsep+os.environ['PATH'])
            result=subprocess.run(['bash',str(wrapper),'arm','--approval',start['path'],'--approval-sha',start['sha256']],env=env,capture_output=True,text=True,timeout=50)
            (root/'shell.stdout').write_text(result.stdout);(root/'shell.stderr').write_text(result.stderr)
            self.assertEqual(result.returncode,0,result.stderr)
            actions=[json.loads(x)['action']for x in(root/'CLI-actions.jsonl').read_text().splitlines()]
            calls=[json.loads(x)for x in(root/'GPU-calls.jsonl').read_text().splitlines()]
            self.assertEqual(actions,['launch-check','prepare-launch','start']);self.assertEqual(calls,[dict(action='start',timeout=10)])
            self.assertTrue((root/'synthetic-controller-reached.json').exists());self.assertTrue((Path(event.startup_ref()['path'])).exists())
            self.assertFalse(b.RESULT.exists());self.assertFalse((b.PACKAGE/'start-review.json').exists())

    def test_wrapper_arm_no_preflight_command_and_new_namespace(self):
        src=b.WRAPPER.read_text();self.assertIn('"$M6_DEPTH_ENTRY" launch-check "$@"',src);self.assertNotIn('"$M6_DEPTH_ENTRY" preflight "$@"',src)
        self.assertIn(b.SESSION,src);self.assertIn(str(b.LOG),src);self.assertNotIn(str(b.SCIENCE_PACKAGE/'followup-launcher.log'),src)


def shell_fixture_cli(root):
    """CPU bridge for the real Shell arm branch; never a server launch entry."""
    root=Path(root);package=root/'technical';cs=q.configs()
    from utils import ch3_round2_amendment as amend
    import ch3_runner as runner
    from m6_type1_followup_entry import cli
    realgit=runner.git
    with contextlib.ExitStack()as stack:
        for obj,key,value in [(b,'PACKAGE',package),(b,'LOG',package/'followup-launcher.log'),(q,'LOG',package/'followup-launcher.log'),(s,'PACKAGE',package),(s,'RESULT',root/'synthetic-result'),(q,'CONTROL',root/'synthetic-result/queue/controller'),(q,'QUEUE_LOCK',root/'queue.lock'),(amend,'RESULT',root/'synthetic-result/round2-amendment')]:stack.enter_context(patch.object(obj,key,value))
        stack.enter_context(patch.object(q,'configs',return_value=cs));stack.enter_context(patch.object(q,'closure',return_value='f'*40))
        stack.enter_context(patch.object(q,'verify_live_remote',return_value='f'*40))
        stack.enter_context(patch.object(runner,'git',side_effect=lambda *args:'f'*40 if args==('rev-parse','HEAD')else realgit(*args)))
        stack.enter_context(patch.object(q,'wrapper_command',return_value=True))
        stack.enter_context(patch('utils.ch3_m_launch.tmux_view',return_value={'synthetic_tmux':True}))
        stack.enter_context(patch.object(q.signal,'signal'))
        def reached(value):
            exclusive(root/'synthetic-controller-reached.json',dict(authorization_ref=value,stages=list(b.STAGES),GPU_workers=0,formal=0))
        stack.enter_context(patch.object(q,'run',side_effect=reached))
        def query(args,**kwargs):
            if args[0]=='nvidia-smi':
                with(root/'GPU-calls.jsonl').open('a')as f:f.write(json.dumps(dict(action=sys.argv[1],timeout=kwargs['timeout']))+'\n')
            return synthetic_gpu(args,**kwargs)
        stack.enter_context(patch.object(subprocess,'check_output',side_effect=query))
        with(root/'CLI-actions.jsonl').open('a')as f:f.write(json.dumps(dict(action=sys.argv[1]))+'\n')
        raise SystemExit(cli())


def real_runtime_log():return str(REAL_RUNTIME_LOG)

def sha_file(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run_context(auth):
    with event.prestart(auth):pass


if __name__=='__main__':unittest.main()
