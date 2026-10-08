"""No-model regression for the registered enc1 observation failure and r2."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from utils.ch3_native_recovery_records import bound,ref
from utils import ch3_patchtst_depth_urban6_recovery as recovery
from tools.restricted_regression import m5_formal_entry as tool


def sample(now=0,metadata=None,nvml=None):
    return dict(time=now,uuid='CPU-fixture',total=32*1024**3,used=1024**3,free=31*1024**3,driver_reserved=0,
                nvml_processes=nvml or{},process_gpu={},cpu_rss={},owned_host_pids=[],process_table_reliable=True,
                owned_pid_metadata={}if metadata is None else{'10':metadata})


def alive():return dict(pid=10,host_pid=110,start_ticks='100',namespace='pid:[1]')
def gone():return dict(pid=10,host_pid=None,state='exited_during_sample',error_kind='FileNotFoundError',phase='namespace',start_ticks='100')


class IdentityReads(unittest.TestCase):
    def test_real_process_identity_and_natural_exit(self):
        child=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(.15)'])
        row=tool.owned_pid_metadata(child.pid)
        self.assertIsNotNone(row['host_pid']);self.assertTrue(row['start_ticks']);self.assertTrue(row['namespace'])
        self.assertEqual(child.wait(timeout=3),0)
        missing=tool.owned_pid_metadata(child.pid)
        self.assertEqual(missing['state'],'exited_during_sample');self.assertIn('phase',missing)

    def metadata(self,stat_after='100',sched='python (110, #threads: 1)',error=None):
        fields='10 (python) S '+'0 '*18
        values=[fields+'100','Pid:\t10\nVmRSS:\t1',sched,fields+stat_after]
        if error:values[1]=error
        with patch.object(Path,'read_text',side_effect=values),patch.object(tool.os,'readlink',return_value='pid:[1]'):
            return tool.owned_pid_metadata(10)

    def test_parse_failure_is_distinct_from_exit(self):
        row=self.metadata(sched='unparseable')
        self.assertEqual(row['state'],'metadata_unavailable');self.assertEqual(row['error_kind'],'sched_parse')
        self.assertEqual(row['phase'],'sched');self.assertEqual(row['start_ticks'],'100')

    def test_permission_failure_has_phase_and_is_not_absence(self):
        row=self.metadata(error=PermissionError('synthetic permission'))
        self.assertEqual(row['state'],'metadata_unavailable');self.assertEqual(row['error_kind'],'PermissionError');self.assertEqual(row['phase'],'status')

    def test_IO_failure_is_not_absence_and_still_has_a_diagnostic(self):
        row=self.metadata(error=OSError('synthetic read failure'))
        self.assertEqual(row['state'],'metadata_unavailable');self.assertEqual(row['error_kind'],'OSError');self.assertEqual(row['phase'],'status')
        with self.assertRaises(ValueError):tool.ExitObservation(sample()).classify(sample(1,row),[10],[],before=[10])

    def test_changed_start_ticks_is_an_identity_conflict(self):
        row=self.metadata(stat_after='101')
        self.assertEqual(row['error_kind'],'identity_conflict');self.assertEqual(row['start_ticks_before'],'100');self.assertEqual(row['start_ticks_after'],'101')


class StateMachine(unittest.TestCase):
    def observer(self):
        value=tool.ExitObservation(sample());value.classify(sample(1,alive(),{'110':100}),[10],[10]);return value

    def test_original_saved_samples_replay_and_missing_failed_sample_is_not_invented(self):
        source=bound(recovery.OBSERVATION_REF);process=bound(source['refs']['process'])
        rows=[json.loads(line)for line in Path(source['refs']['memory']['path']).read_text().splitlines()]
        obs=tool.ExitObservation(process['baseline']);pid=str(bound(source['refs']['runtime'])['pid'])
        for row in rows:obs.classify(copy.deepcopy(row),[pid],[pid])
        self.assertEqual(len(rows),7);self.assertIs(source['failure_sample_recorded'],False)
        self.assertFalse(obs.final_clean);self.assertEqual(obs.known[pid][1],source['exit_instances'][-1]['start_ticks'])

    def test_active_to_missing_race_defers_until_fresh_confirmed_exit(self):
        obs=self.observer();row=sample(2,gone(),{'110':100})
        self.assertEqual(obs.classify(row,[10],[10],before=[10]),[])
        self.assertFalse(obs.final_clean);self.assertEqual(row['identity_transition_pending'],['10'])
        self.assertEqual(obs.classify(sample(3,gone(),{'110':100}),[10],[],before=[10]),['110'])
        self.assertFalse(obs.final_clean)
        obs.classify(sample(4,gone()),[10],[],before=[]);self.assertTrue(obs.final_clean)

    def test_poll_exit_after_complete_metadata_requires_fresh_sample(self):
        obs=self.observer();obs.classify(sample(2,alive()),[10],[],before=[10]);self.assertFalse(obs.final_clean)
        obs.classify(sample(3,gone()),[10],[],before=[]);self.assertTrue(obs.final_clean)

    def test_before_query_identity_can_prove_short_child(self):
        obs=tool.ExitObservation(sample());row=sample(1,gone());row['owned_before_metadata']={'10':alive()}
        obs.classify(row,[10],[],before=[10]);self.assertFalse(obs.final_clean)
        obs.classify(sample(2,gone()),[10],[],before=[]);self.assertTrue(obs.final_clean)

    def test_no_historical_identity_cannot_authorize_disappearance(self):
        with self.assertRaises(ValueError):tool.ExitObservation(sample()).classify(sample(1,gone()),[10],[],before=[10])

    def test_transition_deadline_does_not_reset_when_NVML_is_empty(self):
        obs=self.observer();obs.classify(sample(2,gone()),[10],[10],before=[10])
        obs.classify(sample(61,gone()),[10],[10],before=[10]);self.assertFalse(obs.final_clean)
        with self.assertRaises(ValueError):obs.classify(sample(63,gone()),[10],[10],before=[10])
        with self.assertRaises(ValueError):obs.query_timeout(63)

    def test_residual_wait_is_bounded_and_never_passes_early(self):
        obs=self.observer();obs.classify(sample(2,gone(),{'110':100}),[10],[],before=[10])
        obs.classify(sample(12,gone(),{'110':100}),[10],[],before=[]);self.assertFalse(obs.final_clean)
        self.assertLessEqual(obs.query_timeout(55),3.5)
        with self.assertRaises(ValueError):obs.classify(sample(63,gone(),{'110':100}),[10],[],before=[])

    def test_permission_parsing_and_unstable_reads_reject_even_after_exit(self):
        for kind in ('PermissionError','sched_parse','identity_conflict'):
            for active in ([10],[]):
                obs=self.observer();row=dict(host_pid=None,state='metadata_unavailable',error_kind=kind,phase='sched')
                with self.assertRaises(ValueError):obs.classify(sample(2,row),[10],active,before=[10])
                self.assertFalse(obs.final_clean)

    def test_PID_reuse_namespace_host_conflict_and_wrong_poll_order_reject(self):
        for key,value in (('start_ticks','101'),('namespace','pid:[2]'),('host_pid',111)):
            obs=self.observer();row=alive();row[key]=value
            with self.assertRaises(ValueError):obs.classify(sample(2,row),[10],[10],before=[10])
        with self.assertRaises(ValueError):self.observer().classify(sample(2,alive()),[10],[10],before=[])

    def test_unknown_occupant_UUID_and_unreliable_card_cannot_admit(self):
        for row in (sample(2,gone(),{'999':100}),dict(sample(2,gone()),uuid='changed')):
            with self.assertRaises(ValueError):self.observer().classify(row,[10],[],before=[10])
        obs=self.observer();obs.classify(dict(sample(2,gone()),process_table_reliable=False),[10],[],before=[10]);self.assertFalse(obs.final_clean)


class RealMonitor(unittest.TestCase):
    def fixture(self,fault=None):
        evidence=os.environ.get('M6_OBSERVATION_FIXTURE_RECORDS')
        if evidence:
            destination=Path(evidence)/(fault or'clean-exit');destination.mkdir(parents=True,exist_ok=False)
            directory=contextlib.nullcontext(str(destination))
        else:directory=tempfile.TemporaryDirectory(prefix='m6-enc1-observation-')
        with directory as name:
            root=Path(name);children=[];calls=[];count=0;host=None
            cfg=dict(purpose='CPU_observation_fixture',output=str(root/'worker'),audit_log=str(root/'audit.jsonl'),limits=dict(seconds=5))
            def spawn(config):
                child=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(.18)'],stdout=subprocess.DEVNULL)
                children.append(child);return child,io.StringIO()
            def gpu(pids,query_timeout=10):
                nonlocal count,host
                calls.append(query_timeout);count+=1
                if not pids:return sample(time.monotonic())
                pid=pids[0]
                if count==2:
                    row=tool.owned_pid_metadata(pid);host=str(row['host_pid'])
                else:
                    time.sleep(.12);self.assertEqual(children[0].wait(timeout=3),0)
                    row=tool.owned_pid_metadata(pid)
                    if fault=='parse':row=dict(pid=pid,host_pid=None,state='metadata_unavailable',error_kind='sched_parse',phase='sched')
                    if fault=='timeout':raise subprocess.TimeoutExpired('synthetic nvidia-smi',query_timeout)
                value=sample(time.monotonic());value['owned_pid_metadata']={str(pid):row}
                if count<=3:value['nvml_processes']={host:100}
                if row.get('host_pid')is not None:value.update(process_gpu={str(pid):100},cpu_rss={str(pid):1000},owned_host_pids=[host])
                return value
            with patch.object(tool,'read_profiles',return_value=dict(execution=dict(evidence=str(root)))),patch.object(tool,'spawn',side_effect=spawn),patch.object(tool,'gpu_sample',side_effect=gpu):
                result=tool.run_configs([cfg],root/'wave',monitor=True)
            self.assertEqual([p.returncode for p in children],[0]);self.assertTrue(all(0<x<=10 for x in calls))
            lines=[json.loads(x)for x in(root/'wave/memory.jsonl').read_text().splitlines()]
            snapshot=bound(result['observation_failure_ref'])if result['observation_failure_ref']else None
            return result,lines,snapshot

    def test_real_CPU_child_slow_query_exit_residual_and_fresh_clean_tail(self):
        result,lines,snapshot=self.fixture()
        self.assertEqual(result['returncodes'],[0]);self.assertTrue(result['resource_admission']);self.assertTrue(result['exit_transitions_resolved'])
        self.assertIsNone(snapshot);self.assertTrue(any(x['admission_deferred']for x in lines));self.assertFalse(lines[-1]['admission_deferred'])

    def test_real_monitor_parse_failure_snapshot_is_bound_and_not_resource_fallback(self):
        result,lines,snapshot=self.fixture('parse')
        self.assertEqual(result['failure_kind'],'observation');self.assertFalse(result['resource_admission'])
        self.assertEqual(snapshot['context']['phase'],'identity_classification')
        self.assertEqual(next(iter(snapshot['sample']['owned_pid_metadata'].values()))['error_kind'],'sched_parse')
        from utils.ch3_native_execution import resource_fallback
        self.assertFalse(resource_fallback(result));self.assertTrue(lines)

    def test_query_timeout_keeps_context_and_rejects(self):
        result,lines,snapshot=self.fixture('timeout')
        self.assertEqual(result['failure_kind'],'observation');self.assertFalse(result['resource_admission'])
        self.assertEqual(snapshot['error_type'],'TimeoutExpired');self.assertEqual(snapshot['context']['phase'],'gpu_query');self.assertTrue(lines)


class RecoveryBinding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        recovery.activate()
        from utils import ch3_type1_chain as q
        cls.q=q;cls.cs=q.configs()

    def test_exact_failure_source_unchanged_and_original_instances_exited(self):
        value=recovery.verify_observation_source()
        self.assertEqual(value['retained_actual'],dict(adam=6,backward=6,forward=8))
        self.assertFalse(bound(value['refs']['process'])['resource_admission'])

    def test_seed_keeps_failed_cost_without_Passed_evidence_or_old_worker_reuse(self):
        c=self.cs['PATCH_ENC1'];a=dict(execution_attempt=recovery.ATTEMPT,probe_recovery_ref=recovery.REUSE_REF)
        value=recovery.load_seed(c,a)
        self.assertEqual(value['decisions'],{});self.assertEqual(value['evidence'],{});self.assertEqual(value['artifacts'],{})
        self.assertEqual(value['budget']['historical_actual'],dict(adam=6,backward=6,forward=8))
        self.assertEqual(value['budget']['diagnostic_actual'],value['budget']['actual'])
        self.assertEqual(value['budget']['caps'],bound(recovery.OBSERVATION_REF)['old_caps'])

    def test_old_attempt_permit_cannot_authorize_new_attempt(self):
        with self.assertRaises(PermissionError):recovery.validate_permit_link(self.cs['PATCH_ENC1'],dict(execution_attempt='PATCHTST-depth-Urban6-exit-r1',probe_recovery_ref=recovery.REUSE_REF),True)
        self.assertEqual(recovery.ATTEMPT,'PATCHTST-depth-Urban6-exit-r2');self.assertTrue(str(recovery.RESULT).endswith('-r2'))
        self.assertEqual(recovery.anchors()['parent_technical_failure_ref'],recovery.OBSERVATION_REF)

    def test_all_profiles_numeric_policies_tasks_and_caps_stay_frozen(self):
        from utils import ch3_type1_tasks as s
        for stage,value in recovery.config_refs().items():self.assertEqual(self.cs[stage],bound(value))
        self.assertEqual(sum(len(self.cs[k]['tasks'])for k in s.STAGES[2:]),335)
        self.assertEqual(self.q.completion_counts()['adopted_new_formal_runs'],196)
        self.assertEqual(s.probe_budget(self.cs['PATCH_ENC1'])['caps'],bound(recovery.OBSERVATION_REF)['old_caps'])
        self.assertEqual(recovery.extra_actual(dict(scope=s.context(self.cs['PATCH_ENC1'])['probe_scope'],probe_recovery_ref=recovery.REUSE_REF)),dict(adam=6,backward=6,forward=8))

    def test_light_preflight_source_and_bundle_no_torch_no_new_outputs(self):
        value=recovery.status()
        self.assertFalse(value['READY_FOR_GPU_EXECUTION']);self.assertEqual(value['remaining_formal_runs'],335)
        self.assertNotIn('torch',sys.modules);self.assertFalse(recovery.RESULT.exists());self.assertFalse(recovery.LOG.exists())


if __name__=='__main__':unittest.main()
