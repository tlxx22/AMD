"""CPU preparation guard: synthetic processes, no dataset/model access."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from ch4 import pattern_batch


class PatternBatchTests(unittest.TestCase):
    def test_first_failure_preserves_receipt_and_stops_dispatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);plan=root/'plan.json';folder=root/'run'
            plan.write_text(json.dumps(dict(kind='m7-pdformer-train-only-cpu-cache',threads=2,
                python=sys.executable,environment={},bindings={},worktree=str(root),
                tasks=[dict(domain=name,output=str(root/name),fixed_graph_identity={'path':'unused'},candidate_identity={'path':'bound-candidates'})
                       for name in ['first','never-dispatched']])))
            real=subprocess.Popen;calls=[]
            def synthetic(command,**kwargs):
                calls.append(command)
                return real([sys.executable,'-B','-c','raise SystemExit(7)'],**kwargs)
            with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=''),patch('ch4.queue.verify_environment'),patch.object(pattern_batch.subprocess,'Popen',synthetic):
                pattern_batch.start(plan,folder)
            result=json.loads((folder/'results.json').read_text())
            self.assertEqual(len(calls),1)
            self.assertIn('--candidate-identity',calls[0])
            self.assertIn('bound-candidates',calls[0])
            self.assertEqual([x['status'] for x in result['outcomes']],['failed','not_executed'])
            self.assertTrue((folder/'first-receipt.json').exists())
            finish=json.loads((folder/'preparation-finished.json').read_text())
            self.assertFalse(finish['all_caches_valid'])
            self.assertFalse(finish['scientific_or_engineering_review_passed'])
            self.assertFalse((folder/'complete.json').exists())

    def test_non_cpu_invocation_rejected_before_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);plan=root/'plan.json'
            plan.write_text(json.dumps(dict(kind='m7-pdformer-train-only-cpu-cache',threads=2)))
            with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0'),self.assertRaises(PermissionError):
                pattern_batch.start(plan,root/'run')
            self.assertFalse((root/'run').exists())


if __name__=='__main__':unittest.main()
