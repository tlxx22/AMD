"""Zero-update delivery checks: stored identities, historical protection, CLI."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from tests.timemixer_v4_fixtures import ROOT, EVIDENCE, STORED, small_profile
from timemixer_dev_v4 import adapter
from timemixer_dev_v4.launcher import batch, validate_authorization
from timemixer_dev_v4.worker import source_binding


class Delivery(unittest.TestCase):
    def test_exact_first_batch_order_is_mapped_from_actual_v3_list(self):
        old=ROOT.parent/'amd-execution-evidence/m6/M6-Timemixer-v3-implementation-r1/launch-preparation/task-ids.json'
        expected=[x.replace('timemixer-uv-dev-v3','timemixer-uv-dev-v4',1) for x in json.loads(old.read_text())]
        actual=json.loads((EVIDENCE/'launch-preparation/task-ids.json').read_text())
        self.assertEqual(actual,expected);waves=batch(STORED,actual);self.assertEqual(len(waves),6)
        for wave in waves:
            ts=[next(t for t in STORED['tasks'] if t['id']==i) for i in wave]
            self.assertEqual(len({(t['fold'],t['arm']) for t in ts}),1)
            self.assertEqual([t['h'] for t in ts],[3,6,9,12])

    def test_v1_v2_grants_and_v4_draft_refused_before_resource_or_worker(self):
        launch=EVIDENCE/'launch-preparation';ids=json.loads((launch/'task-ids.json').read_text());draft=json.loads((launch/'launch-binding.template.json').read_text())
        self.assertIsNone(draft['commit']);self.assertFalse(draft['review_approved'])
        self.assertEqual(draft['source_binding'],source_binding());self.assertEqual(STORED['source_binding'],source_binding())
        for candidate in ('timemixer-uv-dev-v1','timemixer-uv-dev-v2','timemixer-uv-dev-v3','timemixer-uv-dev-v4'):
            grant=copy.deepcopy(draft);grant['candidate_id']=candidate
            if candidate!='timemixer-uv-dev-v4':grant['review_approved']=True
            with patch('timemixer_dev_v4.launcher.git',side_effect=AssertionError('reject before network/Git resource checks')):
                with self.assertRaises(PermissionError):validate_authorization(STORED,ROOT/'configs/timemixer_uv_round2_resolved_v4.json',ids,grant)

    def test_original_sources_frozen_and_existing_SOFTS_APT_reused(self):
        before=json.loads((EVIDENCE/'before/snapshot.json').read_text())
        for name,ref in before['files'].items():
            if name.startswith('docs/'):continue
            path=ROOT/name;self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),ref['sha256'],name)
            self.assertEqual(oct(path.stat().st_mode&0o777),ref['mode'])
        src=json.loads((ROOT/'timemixer_dev_v4/source_binding.json').read_text())
        self.assertEqual(src['APT']['commit'],'98a4c9c017666207b02029f842eff92818f0eab8')
        self.assertNotIn('Pathformer',json.dumps(src));self.assertNotIn('pathformer',json.dumps(src))
        self.assertEqual(src['SOFTS']['commit'],'f5d35fd7c3e716b6383ce6d3cc42c131e32c3c44')

    def test_future_decoder_data_never_enters_v4_branches(self):
        p=small_profile();m=adapter.build(p,use_u=True,use_v=True).eval();x=torch.randn(3,12,11);marks=torch.randn(3,12,4)
        with torch.no_grad():
            m.u.out.weight.normal_(0,.1);m.v.final.weight.normal_(0,.1)
            a=m(x,marks,None,None)
            b=m(x,marks,torch.randn(3,9,11)*1e6,torch.randn(3,9,4)*1e6)
        self.assertTrue(torch.equal(a,b))


if __name__=='__main__':unittest.main(verbosity=2)
