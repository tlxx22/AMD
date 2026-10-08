"""Exclusive GPU contract: real CPU lifecycles, synthetic card query only."""
import contextlib
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from tools.restricted_regression import m5_formal_entry as tool
from utils import ch3_patchtst_depth_urban6_recovery as recovery
from utils.ch3_native_recovery_records import bound,exclusive,ref,validate_whole_card_receipt

MODE='exclusive_gpu_whole_card_v1'


def gpu_queries(args,**kwargs):
    if args[0]!='nvidia-smi':return ORIGINAL_QUERY(args,**kwargs)
    if any('query-compute-apps'in arg for arg in args):raise AssertionError('GPU process table must never be queried')
    if not 0<kwargs.get('timeout',0)<=10:raise AssertionError('bounded query required')
    return 'GPU-CPU-fixture, 24576, 1024, 23552, 0\n'


ORIGINAL_QUERY=subprocess.check_output
ORIGINAL_READLINK=os.readlink
ORIGINAL_RUN_CONFIGS=tool.run_configs


def control_readlink(path,*args,**kwargs):
    if str(path).endswith('/ns/pid'):raise AssertionError('namespace mapping forbidden')
    return ORIGINAL_READLINK(path,*args,**kwargs)


def probe_fixture_trajectory(c,t,out,rule):
    from tests.test_m6_probe_schema_recovery import trajectory
    from ch3_runner import memory_growth_review
    tr=trajectory(c,t,out,rule);tr.update(recovery.resource_binding(),allocated=None,reserved=None)
    for row in tr['memory']:row.update(allocated=None,reserved=None,resource_mode=MODE)
    tr['memory_review']=memory_growth_review(tr['memory'],MODE)
    return tr


def cpu_wave(configs,out,monitor=True,c=None,query=gpu_queries,child_code='import time; time.sleep(.25)'):
    """Only the compute producer is synthetic; monitor and audits are real."""
    configs=[dict(cfg,**recovery.resource_binding())for cfg in configs]
    for cfg in configs:
        cfg.update(type1_scope='m6-baseline-type1-followup-v3-recovery1',unified_stage='PATCH_ENC1',probe_schema_recovery_ref=recovery.REUSE_REF,
            successor_scope=cfg.get('successor_scope','CPU-fixture'),purpose=cfg.get('purpose','ch3_probe'),artifact_root=cfg['output'],audit_log=str(Path(cfg['output'])/'audit.jsonl'))
        cfg.setdefault('limits',dict(seconds=2))
    if c is None:
        from utils import ch3_type1_chain as q
        c=q.configs()['PATCH_ENC1']
    handles=[]
    def spawn(cfg):
        log=Path(cfg['output'])/'worker.log';log.parent.mkdir(parents=True,exist_ok=True);h=log.open('x');handles.append(h)
        process=subprocess.Popen([sys.executable,'-B','-c',child_code],stdout=h,stderr=h)
        for name in('runtime.json','budget.json'):
            path=Path(cfg['output'])/name
            if path.exists():
                value=json.loads(path.read_text())
                if name=='runtime.json':value['pid']=process.pid
                else:value['by_pid']={str(process.pid):value['counts']}
                path.write_text(json.dumps(value)+'\n')
        return process,h
    with contextlib.ExitStack()as stack:
        stack.enter_context(patch.object(tool,'spawn',side_effect=spawn))
        stack.enter_context(patch.object(tool,'execution_profiles',return_value=c))
        stack.enter_context(patch.object(tool,'read_profiles',return_value=dict(execution=dict(evidence=str(out)))))
        stack.enter_context(patch.object(tool.subprocess,'check_output',side_effect=query))
        for name in('owned_pid_metadata','ExitObservation'):
            stack.enter_context(patch.object(tool,name,side_effect=AssertionError(name+' forbidden in whole-card mode')))
        stack.enter_context(patch.object(os,'readlink',side_effect=control_readlink))
        result=ORIGINAL_RUN_CONFIGS(configs,out,monitor=monitor)
        destination=os.environ.get('M6_WHOLE_FIXTURE_RECORDS')
        if destination:
            saved=Path(destination)/('wave-'+str(time.time_ns()));saved.mkdir(parents=True)
            for name in('process.json','memory.jsonl','observation-failure.json'):
                src=Path(out)/name
                if src.exists():(saved/name).write_bytes(src.read_bytes())
        return result


class WholeCardContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        recovery.activate()

    def fixture(self,n,root):
        from utils import ch3_type1_chain as q
        c=q.configs()['PATCH_ENC1'];tasks=c['tasks'][:n];a=dict(recovery.resource_binding(),probe_recovery_ref=recovery.REUSE_REF,hardware=dict(gpu='GPU-CPU-fixture, fixture only'))
        phase='serial'if n==1 else'q'+str(n);group=q.s.probe_groups(c)[0]['id']
        configs=[dict(task=t['id'],output=str(root/group/phase/t['id']),purpose='ch3_probe',approval=a,limits=dict(seconds=30),successor_phase=phase,successor_scope=q.s.context(c)['probe_scope'])for t in tasks]
        return c,a,configs

    def test_actual_q1_q2_q4_monitor_no_GPU_attribution_calls_and_real_resource_audit(self):
        from utils import ch3_type1_chain as q,ch3_native_execution as native
        for n in(1,2,4):
            with tempfile.TemporaryDirectory(prefix='m6-card-')as raw,contextlib.ExitStack()as stack:
                root=Path(raw);c,a,cfg=self.fixture(n,root)
                stack.enter_context(patch.object(q,'validate_permit',return_value=a))
                ctx=dict(q.s.context(c),probe_root=root,control=root/'queue',fixture=root/'fixture')
                stack.enter_context(patch.object(q.s,'context',return_value=ctx));stack.enter_context(patch.object(q,'CONTROL',root/'controller'))
                q.CONTROL.mkdir()
                wave=Path(cfg[0]['output']).parent/'wave-0'
                result=cpu_wave(cfg,wave,c=c)
                self.assertTrue(result['resource_admission'],result);self.assertTrue(result['owned_workers_exited'])
                identities=validate_whole_card_receipt(result,wave/'memory.jsonl',a,[x['task']for x in cfg])
                self.assertEqual(len(identities),n);self.assertIsNone(result['process_peaks']);self.assertIsNone(result['exit_observation'])
                self.assertIsNone(result['process_attribution']);self.assertIsNone(result['external_occupancy_known'])
                rows=[json.loads(line)for line in(wave/'memory.jsonl').read_text().splitlines()]
                forbidden=('namespace','host_pid','process_gpu','process_peaks','nvml_processes','owned')
                for row in rows:
                    self.assertFalse(any(any(word in key for word in forbidden)for key in row))
                    self.assertIsNone(row['assessment']['unknown_pids']);self.assertIsNone(row['assessment']['external_occupancy_known'])
                bad=copy.deepcopy(result);bad['resource_mode']='shared';self.assertRaises(ValueError,validate_whole_card_receipt,bad,wave/'memory.jsonl',a,[x['task']for x in cfg])
                bad=copy.deepcopy(result);bad['worker_lifecycles'][0]['returncode']=1;self.assertRaises(ValueError,validate_whole_card_receipt,bad,wave/'memory.jsonl',a,[x['task']for x in cfg])

    def test_sampler_does_not_read_proc_or_GPU_process_table(self):
        with patch.object(tool.subprocess,'check_output',side_effect=gpu_queries),patch.object(tool,'owned_pid_metadata',side_effect=AssertionError('not collected')):
            row=tool.gpu_sample([123],resource_mode=MODE)
        self.assertNotIn('raw_processes',row);self.assertNotIn('owned_pid_metadata',row);self.assertTrue(tool.resource_assessment(row,[],resource_mode=MODE)['admission'])

    def test_probe_capture_has_no_allocator_queries_and_RSS_still_checked(self):
        from ch3_runner import probe_gpu_memory,probe_resource_binding,memory_growth_review
        from utils import ch3_type1_chain as q,ch3_native_execution as native
        class ForbiddenCUDA:
            def __getattr__(self,name):raise AssertionError('worker GPU allocator query forbidden: '+name)
        for kind in('memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved'):
            self.assertIsNone(probe_gpu_memory(ForbiddenCUDA(),'cuda:0',kind,MODE))
        c=q.configs()['PATCH_ENC1'];self.assertEqual(probe_resource_binding(c),recovery.resource_binding())
        bad=copy.deepcopy(c);bad['tasks'].pop()
        self.assertRaises(PermissionError,probe_resource_binding,bad)
        with tempfile.TemporaryDirectory(prefix='m6-card-probe-capture-')as raw:
            t=c['tasks'][0]
            from utils.ch3_contract import numeric_probe_policy
            tr=probe_fixture_trajectory(c,t,Path(raw),numeric_probe_policy(c,t));a=recovery.resource_binding()
            self.assertFalse(native.validate_probe_memory(tr,a)['blocked'])
            for field,value in(('allocated',0),('reserved',0),('resource_mode','shared')):
                bad=copy.deepcopy(tr);bad[field]=value
                self.assertRaises(ValueError,native.validate_probe_memory,bad,a)
            bad=copy.deepcopy(tr);bad['memory'][0]['allocated']=0
            self.assertRaises(ValueError,native.validate_probe_memory,bad,a)
            bad=copy.deepcopy(tr);bad['memory'][0]['rss_before_hash']=float('nan')
            self.assertRaises(ValueError,native.validate_probe_memory,bad,a)
            growing=[dict(step=i,allocated=None,reserved=None,resource_mode=MODE,rss_before_hash=i*64*1024**2,rss_after_hash=i*64*1024**2)for i in range(1,25)]
            self.assertTrue(memory_growth_review(growing,MODE)['blocked'])
        legacy=[dict(allocated=i,rss_before_hash=100,rss_after_hash=100)for i in range(6)]
        self.assertIn('allocated',memory_growth_review(legacy)['triggers'])

    def test_card_margin_accounting_finite_UUID_and_query_limits_still_fail_closed(self):
        with patch.object(tool.subprocess,'check_output',side_effect=gpu_queries):base=tool.gpu_sample([],resource_mode=MODE)
        for change in(dict(free=0,used=base['total']),dict(uuid='changed'),dict(device='cuda:1'),dict(total=float('nan')),dict(used=float('inf')),dict(driver_reserved=-1),dict(free=base['free']+10*1024**2),dict(query_elapsed=11)):
            self.assertFalse(tool.resource_assessment(dict(base,**change),[],base,resource_mode=MODE)['admission'])
        for error in(subprocess.TimeoutExpired('nvidia-smi',10),subprocess.CalledProcessError(1,'nvidia-smi')):
            with patch.object(tool.subprocess,'check_output',side_effect=error),self.assertRaises(type(error)):tool.gpu_sample([],resource_mode=MODE)

    def test_contract_and_worker_permit_cannot_be_selected_by_unbound_string(self):
        from utils import ch3_type1_chain as q
        c,a,cfg=self.fixture(1,Path('/tmp/CPU-not-launched'))
        with self.assertRaises((PermissionError,KeyError)):recovery.monitor_binding(cfg)
        cfg[0].update(type1_scope=q.s.ID,probe_schema_recovery_ref=recovery.REUSE_REF,unified_stage='PATCH_ENC1',**recovery.resource_binding())
        cfg[0]['approval']=dict(a,resource_contract_ref=dict(a['resource_contract_ref'],sha256='0'*64))
        with self.assertRaises(PermissionError):recovery.monitor_binding(cfg)
        self.assertFalse(recovery.resource_binding()['resource_contract_ref']==recovery.OBSERVATION_REF)

    def test_real_monitor_rejects_card_query_worker_timeout_STOP_and_retained_wave(self):
        from utils import ch3_type1_chain as q
        cases=('margin','UUID','finite','accounting','query_failure','query_timeout','worker_error','worker_OOM','wall_timeout','STOP','duplicate')
        for case in cases:
            with self.subTest(case=case),tempfile.TemporaryDirectory(prefix='m6-card-negative-')as raw,contextlib.ExitStack()as stack:
                root=Path(raw);c,a,cfg=self.fixture(1,root);wave=Path(cfg[0]['output']).parent/'wave-0'
                stack.enter_context(patch.object(q,'validate_permit',return_value=a))
                stack.enter_context(patch.object(q.s,'context',return_value=dict(q.s.context(c),probe_root=root,control=root/'queue',fixture=root/'fixture')))
                stack.enter_context(patch.object(q,'CONTROL',root/'controller'));q.CONTROL.mkdir()
                count=[0]
                def query(args,**kw):
                    if args[0]!='nvidia-smi':return gpu_queries(args,**kw)
                    count[0]+=1
                    if count[0]==1:return gpu_queries(args,**kw)
                    if case=='query_failure':raise subprocess.CalledProcessError(1,args)
                    if case=='query_timeout':raise subprocess.TimeoutExpired(args,kw['timeout'])
                    if case=='STOP':(q.CONTROL/'STOP').touch()
                    if case=='margin':return 'GPU-CPU-fixture, 24576, 20480, 4096, 0\n'
                    if case=='UUID':return 'GPU-changed, 24576, 1024, 23552, 0\n'
                    if case=='finite':return 'GPU-CPU-fixture, 24576, nan, 23552, 0\n'
                    if case=='accounting':return 'GPU-CPU-fixture, 24576, 1024, 23000, 0\n'
                    return gpu_queries(args,**kw)
                if case=='wall_timeout':cfg[0]['limits']['seconds']=.05
                child_code='import time;time.sleep(.25)'
                if case=='worker_error':child_code+=';raise SystemExit(3)'
                if case=='worker_OOM':child_code+=";print('CUDA out of memory');raise SystemExit(3)"
                result=cpu_wave(cfg,wave,c=c,query=query,child_code=child_code)
                if case=='duplicate':
                    self.assertTrue(result['resource_admission'])
                    with self.assertRaises(FileExistsError):cpu_wave(cfg,wave,c=c)
                    continue
                self.assertFalse(result['resource_admission'],result)
                self.assertEqual(result['failure_kind'],'resource'if case in('margin','worker_OOM')else'business'if case in('worker_error','wall_timeout','STOP')else'observation')
                self.assertRaises(ValueError,validate_whole_card_receipt,result,wave/'memory.jsonl',a,[cfg[0]['task']])
                self.assertTrue(result['owned_workers_exited'])
                self.assertTrue((wave/'observation-failure.json').exists()or case=='worker_error')

    def test_historical_receipts_cannot_bypass_original_attribution_gate(self):
        from utils.ch3_native_execution import wave_resource_identities
        value=dict(returncodes=[0],failure=None,resource_admission=True,exit_transitions_resolved=True,process_attribution=None)
        with self.assertRaises(ValueError):wave_resource_identities(value,{},Path('/tmp/not-read-memory'),['legacy'])

    def test_safe_stop_signals_only_its_reliably_bound_CPU_fixture(self):
        from m6_remaining_entry import identity,signal_owned
        own=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(2)'])
        other=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(.5)'])
        known=identity(own.pid)
        self.assertFalse(signal_owned(dict(known,start_ticks=str(int(known['start_ticks'])+1))))
        self.assertIsNone(own.poll());self.assertTrue(signal_owned(known));self.assertNotEqual(own.wait(timeout=3),0)
        self.assertEqual(other.wait(timeout=3),0)

    def test_formal_wave_and_real_completion_audit_accept_whole_card_lifecycles(self):
        from utils import ch3_type1_chain as q,ch3_type1_execution as execution,ch3_type1_summary as summary
        from utils.ch3_contract import digest,profile,step_arithmetic
        with tempfile.TemporaryDirectory(prefix='m6-card-formal-')as raw,contextlib.ExitStack()as stack:
            root=Path(raw);c=q.configs()['PATCH_ENC1']
            ctx=dict(q.s.context(c),probe_root=root/'probe',control=root/'queue',fixture=root/'fixture',result_root=root/'results')
            stack.enter_context(patch.object(q.s,'context',return_value=ctx));stack.enter_context(patch.object(q,'CONTROL',root/'controller'));q.CONTROL.mkdir()
            decision=exclusive(root/'decisions.json',dict(decisions={g['id']:dict(status='Passed',concurrency=4)for g in q.s.probe_groups(c)}))
            data={}
            for t in c['tasks']:data.setdefault(t['dataset'],{})[t['id']]='synthetic-data-binding'
            a=dict(recovery.resource_binding(),commit='c'*40,data_bindings=data,summary_ref=decision,probe_recovery_ref=recovery.REUSE_REF,hardware=dict(gpu='GPU-CPU-fixture, fixture only'))
            permit=exclusive(root/'synthetic-permit.json',a);runtime_ref=exclusive(root/'synthetic-runtime.json',dict(purpose='CPU-fixture-only'))
            stack.enter_context(patch.object(q,'validate_permit',return_value=a))
            tasks={t['id']:t for t in c['tasks']}
            for i,ids in enumerate(q.s.formal_waves(c,bound(decision),'PatchTST')):
                cfgs=[dict(task=run,output=str(ctx['result_root']/'formal-PatchTST'/run),purpose='ch3_formal',formal_permit_ref=permit,runtime_admission_ref=runtime_ref,limits=dict(seconds=30))for run in ids]
                wave=ctx['control']/'group-PatchTST'/('wave-'+str(i));measured=cpu_wave(cfgs,wave,c=c);self.assertTrue(measured['resource_admission'])
                for cfg,worker in zip(cfgs,measured['worker_lifecycles']):
                    t=tasks[cfg['task']];p=profile(c,t);arithmetic=step_arithmetic(c,t);out=Path(cfg['output']);pid=worker['pid']
                    identity=dict(commit=a['commit'],profile_sha=digest(p),protocol_sha=digest(c),data_sha=a['data_bindings'][t['dataset']][t['id']],**recovery.resource_binding())
                    exclusive(out/'manifest.json',dict(task=t,profile=p,identity=identity));(out/'best.pt').write_bytes(b'CPU synthetic checkpoint identity; never loaded');(out/'last.pt').write_bytes(b'CPU synthetic checkpoint identity; never loaded')
                    elements=arithmetic['validation_windows']*p['pred_len']*p['C'];history=[]
                    epochs=min(p['training']['epochs'],1+(p['training']['patience']or p['training']['epochs']))
                    for epoch in range(1,epochs+1):
                        history.append(dict(epoch=epoch,steps=epoch*arithmetic['train_batches'],validation=dict(mse=1.,mae=1.,sse=elements,sae=elements,elements=elements),best_epoch=1))
                    (out/'history.jsonl').write_text(''.join(json.dumps(row)+'\n'for row in history));steps=history[-1]['steps']
                    import math
                    forward=len(history)*(arithmetic['train_batches']+math.ceil(arithmetic['validation_windows']/p['training']['eval_batch']))+math.ceil(arithmetic['test_windows_arithmetic_only']/p['training']['eval_batch'])
                    counts=dict(adam=steps,backward=steps,forward=forward)
                    exclusive(out/'budget.json',dict(counts=counts,by_pid={str(pid):counts}));exclusive(out/'runtime.json',dict(error=None,task=t['id'],pid=pid))
                    exclusive(out/'result.json',dict(id=t['id'],commit=a['commit'],protocol_sha=digest(c),profile_sha=digest(p),scientific_protocol=c['baseline_unified']['id'],scheduler_updates=steps,scheduler_sha=digest(p['training']['scheduler']),metric_scope='all_channels',elements=arithmetic['test_windows_arithmetic_only']*p['pred_len']*p['C'],mse=1.,mae=1.,seed=2024,best_epoch=1,final_test=dict(calls=1,selected='best.pt',sha256=ref(out/'best.pt')['sha256'],epoch=1)))
            receipt=execution.technical_group(c,'PatchTST',a);self.assertTrue(receipt['technical_complete']);self.assertEqual(receipt['resource_mode'],MODE)
            self.assertEqual(summary.observed(out/'result.json')['resource_mode'],MODE)
            self.assertEqual(len(receipt['resource_waves']),6)
            swapped=copy.deepcopy(measured);w=swapped['worker_lifecycles'];w[0]['pid'],w[1]['pid']=w[1]['pid'],w[0]['pid'];(wave/'process.json').write_text(json.dumps(swapped)+'\n')
            with self.assertRaises(ValueError):execution.technical_group(c,'PatchTST',a)
            incomplete=copy.deepcopy(measured);incomplete['fresh_post_exit_sample']=None;(wave/'process.json').write_text(json.dumps(incomplete)+'\n')
            with self.assertRaises(ValueError):execution.technical_group(c,t['model'],a)
            (wave/'process.json').write_text(json.dumps(measured)+'\n');wrong=copy.deepcopy(a);wrong['resource_contract_ref']['sha256']='0'*64
            with self.assertRaises(ValueError):execution.technical_group(c,t['model'],wrong)


if __name__=='__main__':
    unittest.main()
