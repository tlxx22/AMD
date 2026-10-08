"""CPU-only event resource tests; real owned waves/guards/audits, no models."""
import ast
import contextlib
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from tools.restricted_regression import m5_formal_entry as tool,resource_budget
from utils import ch3_event_resources as event,ch3_patchtst_depth_urban6_recovery as r
from utils import ch3_type1_chain as q,ch3_type1_tasks as s,ch3_type1_execution as execution
from utils.ch3_native_recovery_records import bound,exclusive,ref
from utils.ch3_contract import digest,profile,step_arithmetic,numeric_probe_policy

REAL_QUERY=subprocess.check_output


def startup_query(args,**kwargs):
    if args[0]!='nvidia-smi':return REAL_QUERY(args,**kwargs)
    if kwargs.get('timeout')!=10:raise AssertionError('startup query must be bounded')
    if 'query-compute-apps' in str(args):raise AssertionError('no attribution')
    gpu=event.frozen_hardware()['gpu'];parts=[x.strip()for x in gpu.split(',')]
    total=float(parts[2]);return gpu+f', 1, {total-1}, 0\n'


@contextlib.contextmanager
def synthetic_root():
    destination=os.environ.get('M6_EVENT_FIXTURES')
    if destination:
        root=Path(tempfile.mkdtemp(prefix='CPU-synthetic-',dir=destination))
        yield str(root)
    else:
        with tempfile.TemporaryDirectory(prefix='m6-event-synthetic-')as raw:yield raw


@contextlib.contextmanager
def fixture(stage='PATCH_ENC1',synthetic_numeric=False):
    r.activate()
    with synthetic_root()as raw,contextlib.ExitStack()as stack:
        root=Path(raw);validated=q.configs();c=validated[stage];old_context=s.context
        # Reuse actual validated immutable tables as CPU fixture inputs; the
        # comparators, permit/wave/worker validators and audits stay real.
        stack.enter_context(patch.object(q,'configs',return_value=validated))
        ctx=dict(old_context(c),probe_root=root/'probe'/stage,control=root/'queue'/stage,fixture=root/'fixture',result_root=root/stage)
        ctx['fixture'].mkdir()
        stack.enter_context(patch.object(s,'context',side_effect=lambda cc:ctx if cc==c else old_context(cc)))
        stack.enter_context(patch.object(s,'RESULT',root/'execution'))
        stack.enter_context(patch.object(q,'CONTROL',root/'controller'))
        stack.enter_context(patch.object(q,'closure',return_value='f'*40))
        import ch3_runner as runner
        original_git=runner.git
        stack.enter_context(patch.object(runner,'git',side_effect=lambda *args:'f'*40 if args==('rev-parse','HEAD')else original_git(*args)))
        stack.enter_context(patch.dict(os.environ,{q.SECRET:'CPU-synthetic-not-a-real-launch', 'TMPDIR':str(ctx['fixture'])}))
        stack.enter_context(patch.dict(sys.modules,{'m5_formal_entry':tool,'resource_budget':resource_budget}))
        if synthetic_numeric:
            from tests.test_m6_probe_schema_recovery import trajectory
            from utils.ch3_contract import task_by_id
            saved=copy.deepcopy(bound(r.R5_SOURCE_REF))
            for row in saved['accepted'].values():
                t=task_by_id(c,row['task_ids'][0]);out=root/'old-numeric'/t['id']
                tr=trajectory(c,t,out,numeric_probe_policy(c,t));row['trajectory_ref']=exclusive(out/'trajectory.json',tr)
                row['artifacts']={str(p):ref(p)for p in out.iterdir()if p.is_file()}
            saved_ref=exclusive(root/'CPU-synthetic-old-numeric.json',saved)
            stack.enter_context(patch.object(r,'R5_SOURCE_REF',saved_ref));stack.enter_context(patch.object(r,'verify_r5_source',return_value=saved))
        auth=q.start_template();auth.update(reviewed=True,execution_permitted=True,structure_frozen=True,m6_authorized=True,budget_authorized=True,closure_commit='f'*40,authorization_basis='CPU synthetic only; not permission for actual HEAD')
        start=exclusive(root/'synthetic-auth.json',auth)
        queries=[]
        def query(args,**kw):
            if args[0]=='nvidia-smi':queries.append(args)
            return startup_query(args,**kw)
        with patch.object(subprocess,'check_output',side_effect=query),event.prestart(auth):
            assert event.hardware_binding()==event.frozen_hardware()
        exclusive(q.CONTROL/'controller.json',dict(owner=q.owner(),scope=s.ID,authorization=start))
        startup=event.seal_startup(start)
        a=dict(q.dynamic(c),type1_scope=s.ID,successor_scope=ctx['probe_scope'],start_authorization_ref=start,
            probe_recovery_ref=r.REUSE_REF,execution_attempt=r.ATTEMPT,authorized_task_ids=[t['id']for t in c['tasks']])
        yield root,c,ctx,a,startup,stack,queries


def forbidden_query(args,*a,**kw):
    if args[0]=='nvidia-smi':raise AssertionError('runtime GPU query forbidden')
    return REAL_QUERY(args,*a,**kw)


def cpu_wave(c,ctx,a,ids,out,stack,*,formal=False,scripts=None,seconds=8,stop=False):
    phase='serial'if len(ids)==1 else'q'+str(len(ids))
    permit=exclusive(Path(out).parent/'synthetic-permit.json',a)if formal else None
    cfgs=[dict(task=run,purpose='ch3_formal'if formal else'ch3_probe',output=str(ctx['result_root']/'formal-PatchTST'/run if formal else ctx['probe_root']/r.groups(c)[0]['id']/phase/run),
        artifact_root=str(ctx['result_root']/'formal-PatchTST'/run if formal else ctx['probe_root']),
        type1_scope=s.ID,unified_stage=c['baseline_unified']['stage'],probe_schema_recovery_ref=r.REUSE_REF,
        successor_scope=ctx['formal_scope']if formal else ctx['probe_scope'],successor_phase=None if formal else phase,
        limits=dict(seconds=seconds),resource_mode=event.MODE,resource_contract_ref=r.RESOURCE_CONTRACT_REF,startup_hardware_ref=a['startup_hardware_ref'],
        **({'formal_permit_ref':permit,'runtime_admission_ref':ref(q.CONTROL/'controller.json')}if formal else {'approval':a}))for run in ids]
    spawned=[]
    def spawn(cfg):
        d=Path(cfg['output']);d.mkdir(parents=True,exist_ok=True);cfg['audit_log']=str(d/'audit.jsonl');h=(d/'worker.log').open('x')
        script=(scripts or {}).get(cfg['task'],'import time;time.sleep(.35)')
        p=subprocess.Popen([sys.executable,'-B','-c',script],stdout=h,stderr=h);spawned.append(p)
        cfg['audit_log']=str(d/'audit.jsonl')
        return p,h
    for cfg in cfgs:cfg['audit_log']=str(Path(cfg['output'])/'audit.jsonl')
    with contextlib.ExitStack()as local:
        local.enter_context(patch.object(q,'validate_permit',return_value=a))
        local.enter_context(patch.object(tool,'spawn',side_effect=spawn))
        local.enter_context(patch.object(tool,'read_profiles',return_value=dict(execution=dict(evidence=str(ctx['probe_root'])))))
        local.enter_context(patch.object(tool,'execution_profiles',return_value=c))
        local.enter_context(patch.object(subprocess,'check_output',side_effect=forbidden_query))
        local.enter_context(patch.object(tool,'gpu_sample',side_effect=AssertionError('runtime sampler forbidden')))
        local.enter_context(patch.object(tool,'owned_pid_metadata',side_effect=AssertionError('attribution forbidden')))
        local.enter_context(patch.object(tool,'ExitObservation',side_effect=AssertionError('namespace state machine forbidden')))
        if stop:
            def on_spawn(cfg):
                result=spawn(cfg);(q.CONTROL/'STOP').write_text('CPU fixture STOP\n');return result
            local.enter_context(patch.object(tool,'spawn',side_effect=on_spawn))
        result=tool.run_configs(cfgs,out,monitor=True)
    assert all(p.poll()is not None for p in spawned)
    return result,cfgs


class EventResources(unittest.TestCase):
    def test_q1_q2_q4_actual_wave_and_resource_audit_no_runtime_queries(self):
        from utils.ch3_native_execution import wave_resource_identities
        for n in(1,2,4):
            with fixture()as(root,c,ctx,a,hw,stack,queries):
                ids=[t['id']for t in c['tasks'][:n]];out=ctx['probe_root']/r.groups(c)[0]['id']/('serial'if n==1 else'q'+str(n))/'wave-0'
                v,_=cpu_wave(c,ctx,a,ids,out,stack)
                self.assertTrue(v['resource_admission']);self.assertEqual(len(queries),1)
                self.assertEqual(len(wave_resource_identities(v,a,out/'memory.jsonl',ids)),n)
                self.assertFalse((out/'memory.jsonl').exists());self.assertEqual(v['runtime_gpu_queries'],0)
                for k in('whole_card_peak','process_peaks','fresh_post_exit_sample','sample_count','actual_interval_max'):self.assertIsNone(v[k])
                self.assertEqual(v['telemetry'],'not_collected_startup_only')
                with patch.object(subprocess,'check_output',side_effect=forbidden_query):
                    self.assertEqual(q.dynamic(c)['hardware'],a['hardware'])
                bad=copy.deepcopy(v);bad['resource_mode']='exclusive_gpu_whole_card_v1'
                self.assertRaises(ValueError,event.validate_receipt,bad,a,ids)
                bad=copy.deepcopy(v);bad['worker_lifecycles'][0]['returncode']=1
                self.assertRaises(ValueError,event.validate_receipt,bad,a,ids)
                self.assertRaises(FileExistsError,cpu_wave,c,ctx,a,ids,out,stack)

    def test_public_preflight_real_bindings_once_and_running_query_is_blocked(self):
        from utils import ch3_round2_amendment as amend
        with fixture()as(root,c,ctx,a,hw,stack,queries):
            stack.enter_context(patch.object(amend,'RESULT',s.RESULT/'round2-amendment'))
            with patch.object(subprocess,'check_output',side_effect=startup_query),patch.object(q,'verify_live_remote',return_value='f'*40)as remote:
                result=q.readiness_report(bound(a['start_authorization_ref']))
            self.assertEqual(result['blocked'],[]);self.assertTrue(result['READY_TO_ARM_HANDOFF']);self.assertFalse(result['READY_FOR_GPU_EXECUTION']);self.assertEqual(result['remaining_formal_runs'],335);self.assertEqual(remote.call_count,1)
            self.assertFalse(s.RESULT.exists());self.assertFalse((q.CONTROL/'upstream-technical-boundary.json').exists())
            s.RESULT.mkdir()
            with patch.object(subprocess,'check_output',side_effect=forbidden_query):
                denied=q.readiness(bound(a['start_authorization_ref']))
            self.assertTrue(any('no GPU query after' in x for x in denied))

    def test_startup_guard_blocks_query_failure_before_spawn(self):
        with fixture()as(root,c,ctx,a,hw,stack,queries):
            auth=bound(a['start_authorization_ref'])
            for exc in (subprocess.TimeoutExpired('nvidia-smi',10),subprocess.CalledProcessError(1,['nvidia-smi'])):
                with patch.object(subprocess,'check_output',side_effect=exc),self.assertRaises(type(exc)):
                    with event.prestart(auth):raise AssertionError('query failure cannot reach launch')
            self.assertEqual(len(queries),1)

    def test_startup_card_UUID_capacity_finite_accounting_and_margin_reject(self):
        with fixture()as(root,c,ctx,a,hw,stack,queries):
            auth=bound(a['start_authorization_ref']);gpu=event.frozen_hardware()['gpu']
            rows=[gpu+', 1, 1, 0\n',gpu+', nan, 1, 0\n',gpu.replace('GPU-','OTHER-',1)+', 1, 81919, 0\n']
            for raw in rows:
                with patch.object(subprocess,'check_output',return_value=raw),self.assertRaises((ValueError,PermissionError)):
                    with event.prestart(auth):pass
            body=bound(hw);body['hardware']['cpu_affinity']=[];tamper=exclusive(root/'tamper-hw.json',body)
            self.assertRaises((ValueError,PermissionError),event.validate_startup,tamper,a,True)
            bad=copy.deepcopy(a);bad['startup_hardware_ref']=tamper
            self.assertRaises((ValueError,PermissionError),event.validate_startup,hw,bad)

    def test_one_OOM_other_cleanup_classification_and_original_trace(self):
        from utils.ch3_m_execution import resource_fallback
        with fixture()as(root,c,ctx,a,hw,stack,queries):
            ids=[t['id']for t in c['tasks'][:4]];scripts={ids[0]:"import time;time.sleep(.25);raise RuntimeError('CUDA out of memory. original CPU injected fixture')"}
            v,_=cpu_wave(c,ctx,a,ids,ctx['probe_root']/r.groups(c)[0]['id']/'q4/wave-0',stack,scripts=scripts)
            self.assertEqual(v['failure_kind'],'resource');self.assertTrue(resource_fallback(v));self.assertFalse(v['resource_admission'])
            self.assertEqual(len(v['original_failures']),1);self.assertIn('Traceback',v['original_failures'][0]['exception'])
            self.assertGreater(len(v['cleanup_terminations']),0);self.assertTrue(all(row['trigger_tasks']==[ids[0]]for row in v['cleanup_terminations']))

    def test_mixed_nonOOM_and_signals_memoryerrors_never_fallback(self):
        from utils.ch3_m_execution import resource_fallback
        scripts=["raise MemoryError('ordinary CPU')","raise ValueError('numeric failure')","import os,signal;os.kill(os.getpid(),signal.SIGTERM)","raise RuntimeError('CUDA illegal memory access')"]
        for script in scripts:
            with fixture()as(root,c,ctx,a,hw,stack,queries):
                ids=[c['tasks'][0]['id']];v,_=cpu_wave(c,ctx,a,ids,ctx['probe_root']/r.groups(c)[0]['id']/'serial/wave-0',stack,scripts={ids[0]:script})
                self.assertFalse(resource_fallback(v));self.assertFalse(v['resource_admission'])
        with fixture()as(root,c,ctx,a,hw,stack,queries):
            ids=[t['id']for t in c['tasks'][:2]]
            v,_=cpu_wave(c,ctx,a,ids,ctx['probe_root']/r.groups(c)[0]['id']/'q2/wave-0',stack,scripts={ids[0]:"raise RuntimeError('CUDA out of memory')",ids[1]:"raise ValueError('independent data identity failure')"})
            self.assertFalse(resource_fallback(v));self.assertGreaterEqual(len(v['original_failures']),1)

    def test_STOP_timeout_nonzero_and_safe_owned_cleanup(self):
        for stop,seconds in ((True,8),(False,.05)):
            with fixture()as(root,c,ctx,a,hw,stack,queries):
                run=c['tasks'][0]['id'];v,_=cpu_wave(c,ctx,a,[run],ctx['probe_root']/r.groups(c)[0]['id']/'serial/wave-0',stack,stop=stop,seconds=seconds,scripts={run:'import time;time.sleep(5)'})
                self.assertFalse(v['resource_admission']);self.assertEqual(v['failure_kind'],'business');self.assertTrue(v['owned_workers_exited'])

    def test_capture_no_allocator_queries_and_RSS_numeric_protection_still_active(self):
        from ch3_runner import probe_gpu_memory,memory_growth_review
        class GPU:
            def __getattr__(self,name):raise AssertionError('allocator telemetry forbidden')
        self.assertIsNone(probe_gpu_memory(GPU(),'cuda:0','memory_allocated',event.MODE))
        memory=[dict(allocated=None,reserved=None,resource_mode=event.MODE,rss_before_hash=1,rss_after_hash=1)for _ in range(6)]
        self.assertFalse(memory_growth_review(memory,event.MODE)['blocked'])
        memory[0]['rss_before_hash']=float('nan');self.assertRaises(ValueError,memory_growth_review,memory,event.MODE)

    def test_budget_counters_limits_time_and_RSS_without_CUDA_sampling(self):
        with synthetic_root()as raw:
            limits=dict(adam=1,backward=1,forward=1,seconds=10,rss=10**12)
            budget=resource_budget.initialize(Path(raw)/'budget.json','CPU-synthetic',limits,event.MODE)
            class NoGPU:
                def __getattr__(self,name):raise AssertionError('no GPU budget query '+name)
            budget.sample(NoGPU());budget.charge('adam','CPU accounting fixture')
            self.assertRaises(resource_budget.BudgetExceeded,budget.charge,'adam')
            v=json.loads(budget.path.read_text());self.assertEqual(v['counts']['adam'],1);self.assertIsNone(v['cuda_reserved_peak'])
            self.assertRaises(resource_budget.BudgetExceeded,budget.sample,NoGPU())
            for constraints in (dict(limits,seconds=-1),dict(limits,rss=1)):
                item=resource_budget.initialize(Path(raw)/('b'+str(len(list(Path(raw).iterdir())))+'.json'),'CPU-synthetic',constraints,event.MODE)
                self.assertRaises(resource_budget.BudgetExceeded,item.sample,NoGPU())
        self.assertTrue(tool.cuda_oom("OutOfMemoryError('CUDA out of memory.')"))
        for text in ('torch.OutOfMemoryError: CPU allocation failed','RuntimeError: CUDA illegal memory access','RuntimeError: CUDA out of memory\nValueError: independent error','-9','MemoryError: unavailable'):
            self.assertFalse(tool.cuda_oom(text),text)

    def test_old_receipts_still_require_historical_attribution_or_samples(self):
        from utils.ch3_native_execution import wave_resource_identities
        with self.assertRaises(ValueError):wave_resource_identities(dict(returncodes=[0],resource_admission=True),{},'/tmp/not-present',['t'])
        from utils.ch3_native_recovery_records import validate_whole_card_receipt
        old=dict(resource_mode='exclusive_gpu_whole_card_v1',resource_contract_ref=r.LEGACY_RESOURCE_CONTRACT_REF)
        with self.assertRaises(ValueError):validate_whole_card_receipt(dict(old,returncodes=[0]),'/tmp/not-present',old,['t'])

    def test_r5_four_serial_only_numerical_sources_negative_q4_and_exact_cost(self):
        r.activate();v=r.verify_r5_source();c=q.configs()['PATCH_ENC1']
        self.assertEqual(len(v['accepted']),4);self.assertFalse(bound(v['failed_q4_ref'])['resource_admission'])
        self.assertEqual(v['historical_actual'],dict(adam=42,backward=42,forward=56));self.assertEqual(v['timing_supplement']['workers'],4)
        self.assertIsNone(r.serial_check_evidence(c,True))
        self.assertEqual(q.s.probe_budget(c)['caps'],dict(adam=432,backward=432,forward=576))
        self.assertEqual([r.numeric_reference(c,t['id'])is not None for t in c['tasks']],[True]*4+[False]*20)
        self.assertEqual(r.completion_counts()['executed_new_formal_runs'],335)

    def test_generation_static_and_science_numeric_computation_AST_unchanged(self):
        r.activate();configs=q.configs();self.assertEqual(sum(len(c['tasks'])for stage,c in configs.items()if stage not in r.COMPLETED_STAGES),335)
        self.assertEqual(sum(q.s.formal_budget(c)['total']['run_epochs']for stage,c in configs.items()if stage not in r.COMPLETED_STAGES),4550)
        parent='6f525489cff125248cbc0b256e9744acf67590da'
        for name in ('update','evaluate','init_training','formal_worker','compare_probe_trajectories','_compare_full_numeric_files','numeric_probe_snapshot'):
            old=ast.parse(subprocess.check_output(['git','show',parent+':ch3_runner.py'],text=True));new=ast.parse(Path('ch3_runner.py').read_text())
            function=lambda tree:ast.dump(next(n for n in tree.body if getattr(n,'name',None)==name),include_attributes=False)
            self.assertEqual(function(old),function(new),name)
        for filename,names in [('utils/ch3_type1_execution.py',('run_group',)),('tools/restricted_regression/resource_budget.py',('counted','instrument_module_calls'))]:
            old=symbols=ast.parse(subprocess.check_output(['git','show',parent+':'+filename],text=True));new=ast.parse(Path(filename).read_text())
            for name in names:
                function=lambda tree:ast.dump(next(n for n in tree.body if getattr(n,'name',None)==name),include_attributes=False)
                self.assertEqual(function(old),function(new),name)
        self.assertNotIn('torch',sys.modules)


    def test_real_probe_AUTO_AUDIT_permit_runtime_summary_and_formal_no_GPU_queries(self):
        from utils import ch3_native_execution as native,ch3_round2_amendment as amend
        from utils.ch3_contract import task_by_id
        from tests.test_m6_probe_schema_recovery import trajectory
        with fixture(synthetic_numeric=True)as(root,c,ctx,a,hw,stack,queries):
            stack.enter_context(patch.object(amend,'RESULT',s.RESULT/'round2-amendment'))
            stack.enter_context(patch.object(r,'verify_production_inheritance',return_value={}))
            # Saved prefix adoption is real; weights are never read or evaluated.
            adopted=r.adopt_prefix();predecessors={stage:adopted[q.STAGE_STATES[stage][3]]for stage in r.COMPLETED_STAGES}
            pr=q.create_permit(c,a['start_authorization_ref'],True,boundary_ref=predecessors,round2_ref=adopted['SEAL_ROUND2_REVISED_BOUNDARY']);a=bound(pr)
            exclusive(ctx['probe_root']/'approval.json',a)
            saved_ref=r.R5_SOURCE_REF
            calls=[];oom_injected=[]
            def spawn(cfg):
                t=task_by_id(c,cfg['task']);out=Path(cfg['output']);h=(out/'worker.log').open('x')
                script='import time;time.sleep(.25)'
                if cfg['successor_phase']=='q4' and t['id']==c['tasks'][0]['id'] and not oom_injected:
                    script="import time;time.sleep(.05);print('RuntimeError: CUDA out of memory',flush=True);raise SystemExit(1)";oom_injected.append(t['id'])
                p=subprocess.Popen([sys.executable,'-B','-c',script],stdout=h,stderr=h)
                tr=trajectory(c,t,out,numeric_probe_policy(c,t));tr.update(r.resource_binding(),allocated=None,reserved=None)
                for row in tr['memory']:row.update(allocated=None,reserved=None,resource_mode=event.MODE)
                from ch3_runner import memory_growth_review
                tr['affinity']=a['hardware']['cpu_affinity'];tr['memory_review']=memory_growth_review(tr['memory'],event.MODE)
                exclusive(out/'trajectory.json',tr)
                counts=s.worker_counts(c,t);(out/'budget.json').write_text(json.dumps(dict(counts=counts,by_pid={str(p.pid):counts}))+'\n')
                exclusive(out/'runtime.json',dict(pid=p.pid,task=t['id'],error=None));(out/'audit.jsonl').write_text('{"event":"CPU-synthetic-only"}\n')
                calls.append(t['id']);return p,h
            stack.enter_context(patch.object(tool,'spawn',side_effect=spawn))
            stack.enter_context(patch.object(tool,'read_profiles',return_value=dict(execution=dict(evidence=str(ctx['probe_root'])))))
            stack.enter_context(patch.object(subprocess,'check_output',side_effect=forbidden_query))
            stack.enter_context(patch.object(tool,'gpu_sample',side_effect=AssertionError('runtime sampler forbidden')))
            stack.enter_context(patch.object(tool,'ExitObservation',side_effect=AssertionError('no GPU attribution')))
            report=native.run_probe(c,a)
            self.assertEqual(len(calls),52);self.assertEqual(report['budget']['historical_actual'],dict(adam=42,backward=42,forward=56))
            self.assertEqual(report['budget']['new_actual'],dict(adam=312,backward=312,forward=416))
            self.assertEqual(report['numeric_reference_ref'],saved_ref)
            first=report['decisions'][r.groups(c)[0]['id']]
            self.assertEqual(first['concurrency'],2);self.assertEqual([x['q']for x in first['attempts']],[4,2])
            self.assertTrue(first['attempts'][0]['resource_failed']);self.assertFalse(first['attempts'][1]['resource_failed'])
            audit=stack.enter_context(patch.object(native,'validate_probe_completion',wraps=native.validate_probe_completion))
            summary_ref=q.audit_probe(c);self.assertEqual(audit.call_count,1);summary=q.validate_summary_light(c,summary_ref)
            self.assertEqual(summary['startup_hardware_ref'],hw);self.assertEqual(summary['resource_mode'],event.MODE)
            formal=q.create_permit(c,a['start_authorization_ref'],False,summary_ref,predecessors,adopted['SEAL_ROUND2_REVISED_BOUNDARY']);runtime=q.seal_runtime(c,formal);q.validate_runtime(c,runtime,formal)
            formal_calls=[]
            def formal_OOM_spawn(cfg):
                execution.validate_worker(c,cfg);self.assertEqual(cfg['startup_hardware_ref'],hw);formal_calls.append(cfg['task'])
                out=Path(cfg['output']);h=(out/'worker.log').open('x')
                script="import time;time.sleep(.1);print('RuntimeError: CUDA out of memory',flush=True);raise SystemExit(1)"if len(formal_calls)==1 else'import time;time.sleep(2)'
                return subprocess.Popen([sys.executable,'-B','-c',script],stdout=h,stderr=h),h
            with patch.object(tool,'spawn',side_effect=formal_OOM_spawn):
                self.assertRaisesRegex(RuntimeError,'unified formal technical/resource failure',execution.run_group,c,bound(formal),'PatchTST',runtime)
            first_ids=s.formal_waves(c,summary,'PatchTST')[0]
            self.assertEqual(formal_calls,first_ids);self.assertFalse((ctx['control']/'group-PatchTST/complete.json').exists())
            self.assertEqual(audit.call_count,1);self.assertEqual(len(queries),1)
            bad=copy.deepcopy(report);bad['decisions'].pop(next(iter(bad['decisions'])))
            self.assertRaises(ValueError,native.validate_probe_completion,c,bad)

    def test_q1_CUDA_OOM_actual_probe_stops_after_one_owned_wave(self):
        from utils import ch3_native_execution as native,ch3_round2_amendment as amend
        with fixture()as(root,c,ctx,a,hw,stack,queries):
            stack.enter_context(patch.object(amend,'RESULT',s.RESULT/'round2-amendment'))
            adopted=r.adopt_prefix();before={k:adopted[q.STAGE_STATES[k][3]]for k in r.COMPLETED_STAGES}
            permit=q.create_permit(c,a['start_authorization_ref'],True,boundary_ref=before,round2_ref=adopted['SEAL_ROUND2_REVISED_BOUNDARY']);a=bound(permit)
            exclusive(ctx['probe_root']/'approval.json',a);calls=[]
            def spawn(cfg):
                out=Path(cfg['output']);h=(out/'worker.log').open('x');calls.append(cfg['task'])
                return subprocess.Popen([sys.executable,'-B','-c',"import time;time.sleep(.1);print('RuntimeError: CUDA out of memory',flush=True);raise SystemExit(1)"],stdout=h,stderr=h),h
            stack.enter_context(patch.object(tool,'spawn',side_effect=spawn))
            stack.enter_context(patch.object(tool,'read_profiles',return_value=dict(execution=dict(evidence=str(ctx['probe_root'])))))
            stack.enter_context(patch.object(subprocess,'check_output',side_effect=forbidden_query))
            stack.enter_context(patch.object(tool,'gpu_sample',side_effect=AssertionError('runtime sampler forbidden')))
            self.assertRaisesRegex(RuntimeError,'serial technical gate failed; no fallback',native.run_probe,c,a)
            self.assertEqual(calls,[c['tasks'][0]['id']]);self.assertFalse((ctx['probe_root']/'complete.json').exists())
            saved=json.loads((ctx['probe_root']/r.groups(c)[0]['id']/'serial/wave-0/process.json').read_text());self.assertEqual(saved['failure_kind'],'resource');self.assertFalse(saved['resource_admission'])

    def test_frozen_numeric_gates_in_isolated_CPU_interpreter(self):
        with synthetic_root()as raw:
            result=subprocess.run([sys.executable,'-B','-m','unittest','tests.test_m6_numeric_admission_defaults.Comparison','-v'],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''},timeout=120)
            (Path(raw)/'numeric.stdout.txt').write_text(result.stdout);(Path(raw)/'numeric.stderr.txt').write_text(result.stderr);exclusive(Path(raw)/'numeric.exit.json',dict(exit_code=result.returncode,CPU_synthetic_only=True))
            self.assertEqual(result.returncode,0,result.stderr[-5000:]);self.assertIn('Ran 8 tests',result.stderr);self.assertNotIn('skipped',result.stderr)

    def test_formal_actual_wave_saved_result_audit_and_summary_no_memory_log(self):
        from utils import ch3_type1_summary as summary
        with fixture()as(root,c,ctx,a,hw,stack,queries):
            data={}
            for t in c['tasks']:data.setdefault(t['dataset'],{})[t['id']]='CPU-synthetic-data'
            decision=exclusive(root/'synthetic-decisions.json',dict(decisions={g['id']:dict(status='Passed',concurrency=4)for g in r.groups(c)}))
            a=dict(a,data_bindings=data,summary_ref=decision)
            tasks={t['id']:t for t in c['tasks']}
            for i,ids in enumerate(s.formal_waves(c,bound(decision),'PatchTST')):
                out=ctx['control']/'group-PatchTST'/('wave-'+str(i));v,cfgs=cpu_wave(c,ctx,a,ids,out,stack,formal=True)
                self.assertTrue(v['resource_admission'])
                for cfg,worker in zip(cfgs,v['worker_lifecycles']):
                    t=tasks[cfg['task']];p=profile(c,t);ar=step_arithmetic(c,t);dest=Path(cfg['output']);pid=worker['pid']
                    identity=dict(commit=a['commit'],profile_sha=digest(p),protocol_sha=digest(c),data_sha=data[t['dataset']][t['id']],startup_hardware_ref=hw,**r.resource_binding())
                    exclusive(dest/'manifest.json',dict(task=t,profile=p,identity=identity));(dest/'best.pt').write_bytes(b'CPU synthetic checkpoint; never read as weights');(dest/'last.pt').write_bytes(b'CPU synthetic checkpoint')
                    elements=ar['validation_windows']*p['pred_len']*p['C'];epochs=min(p['training']['epochs'],1+(p['training']['patience']or p['training']['epochs']))
                    history=[dict(epoch=n,steps=n*ar['train_batches'],validation=dict(mse=1.,mae=1.,sse=elements,sae=elements,elements=elements),best_epoch=1)for n in range(1,epochs+1)]
                    (dest/'history.jsonl').write_text(''.join(json.dumps(x)+'\n'for x in history));steps=history[-1]['steps'];fwd=epochs*(ar['train_batches']+math.ceil(ar['validation_windows']/p['training']['eval_batch']))+math.ceil(ar['test_windows_arithmetic_only']/p['training']['eval_batch'])
                    counts=dict(adam=steps,backward=steps,forward=fwd);exclusive(dest/'budget.json',dict(counts=counts,by_pid={str(pid):counts}));exclusive(dest/'runtime.json',dict(error=None,task=t['id'],pid=pid))
                    exclusive(dest/'result.json',dict(id=t['id'],commit=a['commit'],protocol_sha=digest(c),profile_sha=digest(p),scientific_protocol=c['baseline_unified']['id'],scheduler_updates=steps,scheduler_sha=digest(p['training']['scheduler']),metric_scope='all_channels',elements=ar['test_windows_arithmetic_only']*p['pred_len']*p['C'],mse=1.,mae=1.,seed=2024,best_epoch=1,final_test=dict(calls=1,selected='best.pt',sha256=ref(dest/'best.pt')['sha256'],epoch=1)))
            with patch.object(subprocess,'check_output',side_effect=forbidden_query):
                receipt=execution.technical_group(c,'PatchTST',a);observed=summary.observed(dest/'result.json')
            self.assertEqual(observed['startup_hardware_ref'],hw);self.assertEqual(receipt['resource_mode'],event.MODE)
            self.assertTrue(all('memory'not in x for x in receipt['resource_waves'].values()));self.assertEqual(len(queries),1)
            bad=copy.deepcopy(v);bad['worker_lifecycles'][0]['task_id']='foreign-task';(out/'process.json').write_text(json.dumps(bad)+'\n')
            self.assertRaises(ValueError,execution.technical_group,c,'PatchTST',a)


if __name__=='__main__':
    unittest.main()
