"""Finite CPU checks for scheme B; no real KShape fit or dataset access."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from ch4.pattern_candidates import candidate_indices,build_candidates,candidate_spec,candidate_identity,validate_candidate_identity
from ch4.native_graphs import fit_patterns
from ch4.imports import sha


class CandidateCoverageTests(unittest.TestCase):
    def test_integer_formula_boundaries_and_determinism(self):
        for legal,budget in [(1,336),(3,336),(11,4),(3501,336),(10172,4032),(10690,6048)]:
            first=candidate_indices(legal,budget);np.random.seed(9)
            self.assertTrue(np.array_equal(first,candidate_indices(legal,budget)))
            self.assertEqual(len(first),min(legal,budget));self.assertEqual(first[0],0)
            self.assertEqual(first[-1],legal-1);self.assertTrue(np.all(np.diff(first)>0))
            if len(first)>1:self.assertEqual(first.tolist(),[k*(legal-1)//(len(first)-1) for k in range(len(first))])
        for legal,budget in [(0,336),(5,0)]:
            with self.assertRaises(ValueError):candidate_indices(legal,budget)

    def test_domains_common_urban_range_and_pems_frequency(self):
        cases=[('UrbanEV',3475,3452,336,60),('AMS',3513,3501,336,60),
               ('PEMS04',10195,10172,4032,5),('PEMS08',10713,10690,6048,5)]
        for domain,end,legal,count,minutes in cases:
            meta=dict(endpoints=[end,end+100,end+200],nodes=2,node_order=['b','a'])
            times=np.datetime64('2023-01-01')+np.arange(end)*np.timedelta64(minutes,'m')
            s=candidate_spec(domain,meta,times)
            self.assertEqual((s['legal_train_windows'],s['candidate_windows']),(legal,count))
            self.assertEqual(s['latest_label'],s['train_last'])
            self.assertLess(max(s['indices'])+11,end)
            if domain=='UrbanEV':
                for h in (3,6,9,12):self.assertLess(max(s['indices'])+11+h,end)
            bad=times.copy();bad[20]+=np.timedelta64(1,'m')
            with self.assertRaisesRegex(ValueError,'time axis'):candidate_spec(domain,meta,bad)

    def test_actual_fit_receives_uniform_candidates_without_filtering(self):
        target=np.arange(60*3,dtype=float).reshape(60,3);target[:,1]=0
        indices=candidate_indices(48,24);expected=build_candidates(target,indices)
        self.assertEqual(expected.shape,(72,3,1))
        for k,i in enumerate(indices):
            for node in range(3):np.testing.assert_array_equal(expected[k*3+node,:,0],target[i:i+3,node])
        with patch('ch4.kshape_rollback.RollbackKShape') as constructor,patch('ch4.pattern_initialization.select_initial_centers',return_value=(expected[:16],{})),patch('ch4.native_graphs.validate_pattern_fit',return_value={'synthetic':True}):
            model=constructor.return_value;model.fit.return_value=model;model.cluster_centers_=np.ones((16,3,1))
            fit_patterns(target,48,24,cand_key_days=1)
            self.assertEqual(constructor.call_count,1)
            self.assertEqual({k:v for k,v in constructor.call_args.kwargs.items() if k!='init'},dict(n_clusters=16,max_iter=5,random_state=2024))
            np.testing.assert_array_equal(constructor.call_args.kwargs['init'],expected[:16])
            np.testing.assert_array_equal(model.fit.call_args.args[0],expected)
        # Node order is preserved, not sorted or filtered for true zeros.
        perm=[2,0,1];permuted=build_candidates(target[:,perm],indices).reshape(len(indices),3,3)
        np.testing.assert_array_equal(permuted,expected.reshape(len(indices),3,3)[:,perm])

    def test_identity_rejects_tampering_and_old_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);times=np.datetime64('2023-01-01')+np.arange(400)*np.timedelta64(1,'h')
            np.save(root/'times.npy',times)
            meta=dict(endpoints=[400],nodes=2,node_order=['b','a'],files={'times.npy':sha(root/'times.npy'),'y.npy':'synthetic-y','valid.npy':'synthetic-mask'})
            (root/'metadata.json').write_text(json.dumps(meta));spec=candidate_identity('AMS',root/'metadata.json')
            p=root/'candidate.json';p.write_text(json.dumps(spec));ref=dict(path=str(p),sha256=sha(p))
            self.assertEqual(validate_candidate_identity(ref,'AMS',root/'metadata.json'),spec['selection'])
            spec['selection']['indices']=list(range(336));p.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError,'bytes changed'):validate_candidate_identity(ref,'AMS',root/'metadata.json')
            ref['sha256']=sha(p)
            with self.assertRaisesRegex(ValueError,'contract/source'):validate_candidate_identity(ref,'AMS',root/'metadata.json')


if __name__=='__main__':unittest.main()
