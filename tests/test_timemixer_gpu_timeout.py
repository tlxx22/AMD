"""GPU management timeout tests: no model computation or real GPU query."""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from timemixer_dev import launcher, worker, training_state
from timemixer_dev.training_config import ROOT

UUID = 'GPU-3d365efd-300b-f527-e8fe-703fb0cfb738'
OUTPUT = '0, '+UUID+', 81920, 81222\n'

def tree(folder):
    if not folder.exists():
        return None
    return {str(p.relative_to(folder)):(hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_mode&0o777)
            for p in folder.rglob('*') if p.is_file()}

class ManagementTimeout(unittest.TestCase):
    def test_success_parsing_passes_timeout60(self):
        with patch.object(launcher.subprocess,'check_output',return_value='2, GPU-other, 81920, 10000\n'+OUTPUT) as query:
            result=launcher.gpu_startup({'gpu_uuid':UUID})
        self.assertEqual(result,dict(index='0',uuid=UUID,total_MiB=81920.,free_MiB=81222.))
        query.assert_called_once_with(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.free','--format=csv,noheader,nounits'],text=True,timeout=60)

    def test_UUID_mismatch_rejected_without_retry(self):
        with patch.object(launcher.subprocess,'check_output',return_value=OUTPUT) as query,patch.object(launcher.time,'sleep') as wait:
            with self.assertRaises(PermissionError):launcher.gpu_startup({'gpu_uuid':'GPU-wrong'})
        self.assertEqual(query.call_count,1);wait.assert_not_called()

    def test_memory_minimum_and_total_fraction_retained(self):
        for total,free in [(81920,8191),(131072,9000)]:
            with self.subTest(total=total,free=free),patch.object(launcher.subprocess,'check_output',return_value=f'0, {UUID}, {total}, {free}\n') as query:
                with self.assertRaises(PermissionError):launcher.gpu_startup({'gpu_uuid':UUID})
                self.assertEqual(query.call_count,1)

    def test_memory_exact_boundary_passes(self):
        with patch.object(launcher.subprocess,'check_output',return_value=f'0, {UUID}, 81920, 8192\n'):
            self.assertEqual(launcher.gpu_startup({'gpu_uuid':UUID})['free_MiB'],8192.)

    def test_timeout_retry_then_success_is_bounded(self):
        failures=[subprocess.TimeoutExpired('synthetic-nvidia-smi',60) for _ in range(2)]
        with patch.object(launcher.subprocess,'check_output',side_effect=failures+[OUTPUT]) as query,patch.object(launcher.time,'sleep') as wait:
            self.assertEqual(launcher.gpu_startup({'gpu_uuid':UUID})['uuid'],UUID)
        self.assertEqual(query.call_count,3);self.assertEqual([c.kwargs['timeout'] for c in query.call_args_list],[60]*3)
        self.assertEqual([c.args for c in wait.call_args_list],[(2,),(2,)])

    def test_three_timeouts_raise_final_failure(self):
        failures=[subprocess.TimeoutExpired('synthetic-nvidia-smi',60) for _ in range(3)]
        with patch.object(launcher.subprocess,'check_output',side_effect=failures) as query,patch.object(launcher.time,'sleep') as wait:
            with self.assertRaises(subprocess.TimeoutExpired) as caught:launcher.gpu_startup({'gpu_uuid':UUID})
        self.assertIs(caught.exception,failures[-1]);self.assertEqual(query.call_count,3)
        self.assertEqual([c.kwargs['timeout'] for c in query.call_args_list],[60]*3);self.assertEqual(wait.call_count,2)

    def test_non_timeout_error_not_retried(self):
        failure=subprocess.CalledProcessError(1,['synthetic-nvidia-smi'])
        with patch.object(launcher.subprocess,'check_output',side_effect=failure) as query,patch.object(launcher.time,'sleep') as wait:
            with self.assertRaises(subprocess.CalledProcessError):launcher.gpu_startup({'gpu_uuid':UUID})
        self.assertEqual(query.call_count,1);wait.assert_not_called()

    def supervise_failure(self,existing,mode):
        # Artificial namespace/lock/control records only. No real launch grant,
        # data loading, production owner or process environment is altered.
        with tempfile.TemporaryDirectory(prefix='.uv-timeout-',dir=ROOT) as temp:
            base=Path(temp);root=base/'namespace';root.mkdir();attempt='fixture-gpu-failure';folder=root/attempt
            if existing:
                folder.mkdir();(folder/'owner.json').write_text(json.dumps({'pid':999999999,'start':'0'}))
                (folder/'STOP').write_text('historical STOP\n');(folder/'failure.json').write_text('{"historical":true}\n')
            before=tree(folder)
            config=json.loads((ROOT/'configs/timemixer_uv_round2_resolved_v1.json').read_text());config['global_lock']=str(base/'synthetic.lock')
            cp=base/'metadata-config.json';cp.write_text(json.dumps(config));auth=base/'invalid-real-grant.json';auth.write_text(json.dumps({'gpu_uuid':UUID,'review_approved':False}))
            id=config['tasks'][0]['id']
            output={'uuid':OUTPUT.replace(UUID,'GPU-other'),'memory':f'0, {UUID}, 81920, 1\n'}.get(mode)
            effects=[subprocess.TimeoutExpired('synthetic-nvidia-smi',60)]*3 if mode=='timeout' else subprocess.CalledProcessError(1,['synthetic-nvidia-smi']) if mode=='command' else None
            query_args={'side_effect':effects} if effects is not None else {'return_value':output}
            # Only the launcher's read of a mapping is mocked; actual process
            # CUDA_VISIBLE_DEVICES remains empty throughout the CPU suite.
            fake_os=SimpleNamespace(environ={'CUDA_VISIBLE_DEVICES':'0'},devnull=os.devnull,getpid=os.getpid)
            with patch.object(launcher,'os',fake_os),patch.object(worker,'RUN_NAMESPACE',root),patch.object(launcher,'validate',return_value=config),patch.object(launcher,'validate_authorization'),patch.object(launcher,'prepare_resume',return_value={}) as prepare,patch.object(launcher.subprocess,'check_output',**query_args),patch.object(launcher.time,'sleep'),patch.object(launcher.subprocess,'Popen') as spawn,patch.object(worker,'run_worker') as run,patch.object(worker.adapter,'build') as build,patch.object(training_state,'atomic_json') as write:
                with self.assertRaises((PermissionError,subprocess.TimeoutExpired,subprocess.CalledProcessError)):
                    launcher.supervise(cp,[id],root,attempt,auth,resume=existing)
                spawn.assert_not_called();run.assert_not_called();build.assert_not_called();write.assert_not_called()
                if existing:self.assertEqual(prepare.call_args.kwargs['archive'],False)
                else:prepare.assert_not_called()
            self.assertEqual(tree(folder),before)
            with (base/'synthetic.lock').open('r') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_failed_query_leaves_existing_attempt_unchanged(self):
        for mode in ('timeout','uuid','memory','command'):
            with self.subTest(mode=mode):self.supervise_failure(True,mode)

    def test_failed_query_does_not_create_fresh_attempt_or_worker(self):
        for mode in ('timeout','uuid','memory','command'):
            with self.subTest(mode=mode):self.supervise_failure(False,mode)

if __name__=='__main__':unittest.main(verbosity=2)
