import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from ch4.imports import sha
from ch4.runtime_io import write_new
from ch4.pattern_identity import validate_pattern_identity,validate_policy_provenance
from ch4 import kshape_rollback
from ch4.prepare_native_patterns import reuse_dtw
from ch4.pattern_candidates import candidate_identity,build_candidates
from ch4.pattern_initialization import select_initial_centers


class PatternIdentity(unittest.TestCase):
    def test_adoption_requires_success_and_full_bound_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'producer-source';source.mkdir()
            (source/'kernel.py').write_text('# synthetic producer\n')
            ref=lambda p:dict(path=str(p),sha256=sha(p))
            y=np.sin(np.arange(800).reshape(400,2)**2)
            for key,array in dict(y=y,valid=np.ones((400,2),dtype=bool),times=np.datetime64('2023-04-01')+np.arange(400)*np.timedelta64(1,'h'),
                                  dtw_matrix=np.zeros((2,2)),pattern_keys=np.ones((16,3,1))).items():
                np.save(root/(key+'.npy'),array,allow_pickle=False)
            metadata=dict(endpoints=[400,450,500],node_order=['a','b'],nodes=2,files={key+'.npy':sha(root/(key+'.npy')) for key in ('y','valid','times')})
            write_new(root/'metadata.json',metadata);write_new(root/'fixed.json',{'synthetic':True});fixed=ref(root/'fixed.json')
            candidate=candidate_identity('JHB',root/'metadata.json');write_new(root/'candidate.json',candidate)
            centers,details=select_initial_centers(build_candidates(y,np.array(candidate['selection']['indices'])))
            np.save(root/'initial_centers.npy',centers)
            write_new(root/'initialization.json',dict(domain='JHB',candidate_identity=ref(root/'candidate.json'),initialization=details,centers=ref(root/'initial_centers.npy')))
            record=dict(domain='JHB',train_rows=400,node_order=['a','b'],metadata_sha256=sha(root/'metadata.json'),
                train_only=True,validation_or_test_fitting=False,dtw_radius=6,pattern_length=3,clusters=16,max_iter=5,seed=2024,
                cand_key_days=14,steps_per_day=24,candidate_windows=336,fixed_graph_identity=fixed,
                legal_train_windows=388,candidate_identity=ref(root/'candidate.json'),candidate_selection=candidate['selection'],
                initialization_identity=ref(root/'initialization.json'),
                fit=dict(successful_fit=True,inertia=.1,n_iter=2,cluster_counts=[42]*16,parameters=dict(n_clusters=16,max_iter=5,random_state=2024,init='explicit-candidate-centers',init_array_sha256=details['centers_array_sha256'])),
                code_sha256={'/synthetic/kernel.py':sha(source/'kernel.py')},
                input_files={key:ref(root/(key+'.npy')) for key in ('y','valid','times')},
                artifacts={key:ref(root/(key+'.npy')) for key in ('dtw_matrix','pattern_keys')})
            # Identity fixture, not a claim of real fitting: bind synthetic final
            # metadata and output through the new policy sidecar.
            final=dict(cluster_counts=[42]*16,inertia=.1,n_iter=2,centers=kshape_rollback.digest(np.ones((16,3,1))),
                       centers_finite=True,norms_finite=True,inertia_finite=True)
            policy=dict(version=kshape_rollback.POLICY,adapter_sha256=sha(kshape_rollback.__file__),
                native_source_sha256=kshape_rollback.NATIVE_SHA,final_state=final,
                attempts=[dict(legal_updates_completed=2,attempted_updates=2,retained_legal_updates=1,
                    triggered=False,early_terminated=False,stop_reason='native_inertia_rollback')])
            write_new(root/'policy.json',policy)
            record.update(rollback_policy_version=kshape_rollback.POLICY,rollback_policy_identity=ref(root/'policy.json'))
            record['fit']['rollback_policy']=policy
            write_new(root/'identity.json',record);identity=ref(root/'identity.json')
            validate_pattern_identity(identity,'JHB',root/'metadata.json',fixed)
            for mutation in ('rule','indices','pool'):
                bad_init=json.loads((root/'initialization.json').read_text())
                if mutation=='rule':bad_init['initialization']['rule']='random-old-rule'
                if mutation=='indices':bad_init['initialization']['selected_indices'][0]=999999
                if mutation=='pool':bad_init['initialization']['selected_indices'][0]=next(i for i in range(672) if i not in details['selected_indices'])
                init_path=root/('bad-init-'+mutation+'.json');write_new(init_path,bad_init)
                bad=copy.deepcopy(record);bad['initialization_identity']=ref(init_path)
                path=root/('bad-pattern-'+mutation+'.json');write_new(path,bad)
                with self.assertRaises(ValueError):validate_pattern_identity(ref(path),'JHB',root/'metadata.json',fixed)
            for mutate in ('fit','scope','parameter','prefix','init','policy','sidecar'):
                bad=copy.deepcopy(record)
                if mutate=='fit':bad['fit']['successful_fit']=False
                if mutate=='scope':bad['train_rows']=450
                if mutate=='parameter':bad['fit']['parameters']['max_iter']=6
                if mutate=='prefix':bad['candidate_selection']['indices']=list(range(336))
                if mutate=='init':bad['fit']['parameters']['init_array_sha256']='unbound'
                if mutate=='policy':bad['rollback_policy_version']='unapproved'
                if mutate=='sidecar':bad['fit']['rollback_policy']['attempts'][0]['legal_updates_completed']=3
                path=root/('bad-'+mutate+'.json');write_new(path,bad)
                with self.assertRaises(ValueError):validate_pattern_identity(ref(path),'JHB',root/'metadata.json',fixed)
            np.save(root/'pattern_keys.npy',np.zeros((16,3,1)),allow_pickle=False)
            with self.assertRaisesRegex(ValueError,'reference changed'):validate_pattern_identity(identity,'JHB',root/'metadata.json',fixed)
            (root/'identity.json').unlink()
            with self.assertRaises(FileNotFoundError):validate_pattern_identity(identity,'JHB',root/'metadata.json',fixed)

    def test_historical_policy_is_exact_identity_only_and_not_relabelled(self):
        record=dict(fit={'successful_fit':True});original=copy.deepcopy(record)
        # Temporary fixture allowlist only; production four identities are immutable.
        with patch.dict('ch4.pattern_identity.LEGACY_SUCCESS_IDS',{'JHB':'fixture-sha'},clear=True):
            self.assertEqual(validate_policy_provenance({'sha256':'fixture-sha'},record,'JHB'),
                             'historical-native-update-no-new-policy')
            self.assertEqual(record,original)
            with self.assertRaisesRegex(ValueError,'unrecognized historical'):
                validate_policy_provenance({'sha256':'different-sha'},record,'JHB')
            record['fit']['rollback_policy']={}
            with self.assertRaisesRegex(ValueError,'unrecognized historical'):
                validate_policy_provenance({'sha256':'fixture-sha'},record,'JHB')

    def test_reuse_checks_source_parameters_and_node_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ref=lambda p:dict(path=str(p),sha256=sha(p))
            np.save(root/'dtw.npy',np.zeros((2,2)));(root/'source.py').write_text('# synthetic DTW producer\n')
            record=dict(domain='UrbanEV',train_rows=100,metadata_sha256='synthetic-meta',node_order=['a','b'],steps_per_day=24,
                        complete_train_days=4,dtw_radius=6,source={'sha256':'synthetic-data'},code_sha256={'/old/source.py':sha(root/'source.py')})
            write_new(root/'start.json',record)
            reuse=dict(dtw=ref(root/'dtw.npy'),producer_start=ref(root/'start.json'),producer_source_snapshots={'/old/source.py':str(root/'source.py')})
            write_new(root/'reuse.json',reuse);reference=ref(root/'reuse.json')
            self.assertEqual(reuse_dtw(reference,record,2),reuse['dtw'])
            bad=copy.deepcopy(record);bad['node_order'].reverse()
            with self.assertRaisesRegex(ValueError,'node_order'):reuse_dtw(reference,bad,2)
            bad=copy.deepcopy(record);bad['dtw_radius']=5
            with self.assertRaisesRegex(ValueError,'dtw_radius'):reuse_dtw(reference,bad,2)
            (root/'source.py').write_text('# changed temporary source\n')
            with self.assertRaisesRegex(ValueError,'source snapshot'):reuse_dtw(reference,record,2)


if __name__=='__main__':unittest.main()
