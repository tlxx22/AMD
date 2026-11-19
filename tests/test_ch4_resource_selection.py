import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from ch4 import queue,resources
from ch4.protocol import validate_plan,dataset_protocol

W=Path(__file__).resolve().parents[1]
PLAN=W/'ch4/comparison-plan.json'

class ResourceSelection(unittest.TestCase):
    def setUp(self):self.plan=json.loads(PLAN.read_text())

    def test_q1_t2_exact_scope_and_budgets(self):
        p=self.plan;self.assertTrue(validate_plan(p))
        self.assertEqual(p['resource_plan']['waves'],[[t['id']] for t in p['tasks']])
        self.assertEqual(sum(t['max_updates'] for t in p['tasks']),6667200)
        self.assertEqual(sum(t['training']['max_epochs'] for t in p['tasks']),16200)
        self.assertFalse(p['execution_authorized']);self.assertTrue(p['resource_frozen'])
        for t in p['tasks']:
            self.assertEqual(t['training']['threads'],2)
            self.assertEqual(t['training']['patience'],40 if t['training']['max_epochs']==200 else 20)
            self.assertEqual(dataset_protocol(t['domain'])['threads'],2)

    def test_wrong_resources_and_threads_rejected(self):
        for which in ('threads','q','waves','batch','interop','wall_cap'):
            p=copy.deepcopy(self.plan)
            if which=='threads':p['tasks'][0]['training']['threads']=1
            elif which=='q':p['resource_plan']['concurrency']=2
            elif which=='waves':p['resource_plan']['waves'][0]+=p['resource_plan']['waves'][1]
            elif which=='batch':p['resource_plan']['domain_batches']['AMS']=[8,8]
            elif which=='interop':p['resource_plan']['interop_threads']=2
            else:p['resource_plan']['formal_wall_clock_cap']=38280
            with self.subTest(which=which),self.assertRaises(ValueError):validate_plan(p)

    def test_fresh_cpu_process_formal_module_entry_train_test(self):
        for phase in ('train','test'):
            env=resources.worker_environment(dict(os.environ,CUDA_VISIBLE_DEVICES=''))
            env.pop('PYTHONPATH',None)
            result=subprocess.run([sys.executable,'-B','-m','tests.ch4_thread_fixture',str(PLAN),phase],cwd=W,env=env,text=True,capture_output=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            d=json.loads(result.stdout);self.assertEqual((d['intra_op'],d['inter_op'],d['loader_workers']),(2,1,0))
            self.assertEqual(d['environment'],resources.THREAD_ENV);self.assertFalse(d['models_imported'])
        env['OMP_NUM_THREADS']='4'
        result=subprocess.run([sys.executable,'-B','-m','tests.ch4_thread_fixture',str(PLAN),'train'],cwd=W,env=env,text=True,capture_output=True,timeout=30)
        self.assertNotEqual(result.returncode,0);self.assertIn('worker thread environment mismatch',result.stderr)

    def test_missing_or_unreviewed_auth_rejected_before_gpu(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(queue,'verify_gpu_allocation',side_effect=AssertionError('must not query GPU')):
            auth=Path(tmp)/'AUTH.json'
            with self.assertRaises(FileNotFoundError):queue.admission(PLAN,auth)
            auth.write_text(json.dumps(dict(scope=queue.SCOPE,review_approved=False,commit=None)))
            with self.assertRaises(PermissionError):queue.admission(PLAN,auth)

    def test_allocation_validation_synthetic_no_gpu_query(self):
        r=self.plan['resource_plan'];uuid=r['expected_gpu_uuid']
        with tempfile.TemporaryDirectory() as tmp:
            info=Path(tmp)/'information';info.write_text('GPU UUID: '+uuid+'\nDevice Minor: 0\n')
            def query(command,**kwargs):
                if '--query-gpu=uuid' in command:return uuid+'\n'
                if '--query-compute-apps=pid' in command:return ''
                return '80000\n'
            with patch.dict(os.environ,dict(NVIDIA_VISIBLE_DEVICES='0',CUDA_VISIBLE_DEVICES=uuid)),patch.object(Path,'glob',return_value=iter([info])),patch.object(resources.subprocess,'check_output',side_effect=query):
                self.assertEqual(resources.verify_gpu_allocation(r)['uuid'],uuid)
            with patch.dict(os.environ,dict(NVIDIA_VISIBLE_DEVICES='all',CUDA_VISIBLE_DEVICES=uuid)),patch.object(resources.subprocess,'check_output',side_effect=AssertionError('no query')):
                with self.assertRaises(PermissionError):resources.verify_gpu_allocation(r)
            with patch.dict(os.environ,dict(NVIDIA_VISIBLE_DEVICES='0',CUDA_VISIBLE_DEVICES=uuid)),patch.object(Path,'glob',return_value=iter([info])),patch.object(resources.subprocess,'check_output',side_effect=[uuid+'\n','1234\n']):
                with self.assertRaises(PermissionError):resources.verify_gpu_allocation(r)

    def test_supervisor_env_and_original_freeze_then_test_path(self):
        from tests.test_ch4_supervisor import Supervisor
        helper=Supervisor();original=resources.worker_environment;calls=[]
        def env(base):
            result=original(base);calls.append(result);return result
        with patch.object(queue,'worker_environment',side_effect=env):
            helper.test_success_full_supervisor_then_seal_and_claim_refusal()
        self.assertEqual(len(calls),8)
        self.assertTrue(all(all(e[k]==v for k,v in resources.THREAD_ENV.items()) for e in calls))

if __name__=='__main__':unittest.main()
