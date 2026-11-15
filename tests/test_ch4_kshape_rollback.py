"""Small synthetic state-machine tests; never load a real candidate pool."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from tslearn.clustering import KShape
from tslearn.clustering.utils import EmptyClusterError
from ch4.kshape_rollback import RollbackKShape,validate_rollback_record
from ch4.native_graphs import validate_pattern_fit


def fixture():
    angle=np.arange(16)*2*np.pi/16
    centers=np.cos(angle[:,None]+np.arange(3)[None,:]*2*np.pi/3)[:,:,None]
    return np.repeat(centers,2,axis=0),centers


class ControlledUpdate:
    """Fault injection only: native assignment/distance/post-fit remain active."""
    fault_at=2
    fault='empty'

    def _update_centroids(self,X):
        self.updates=getattr(self,'updates',0)+1
        if self.updates==self.fault_at:
            if self.fault=='empty':self.cluster_centers_[7]=self.cluster_centers_[5]
            elif self.fault=='nonfinite':self.cluster_centers_[7]=np.nan
            elif self.fault=='inertia':self.inertia_=np.nan;self.cluster_centers_[7]=self.cluster_centers_[5]
            elif self.fault=='other':raise RuntimeError('synthetic extraction error')
            elif self.fault=='update_empty':raise EmptyClusterError('from update, not assignment')
        self.norms_centroids_=np.linalg.norm(self.cluster_centers_,axis=(1,2))


class Controlled(ControlledUpdate,RollbackKShape):pass
class NativeControlled(ControlledUpdate,KShape):pass


def model(cls=Controlled):
    _,centers=fixture()
    return cls(n_clusters=16,max_iter=5,random_state=2024,init=centers.copy())


class RollbackTests(unittest.TestCase):
    def test_normal_native_loop_inertia_rollback_exact(self):
        x,_=fixture();native=model(NativeControlled);adapted=model()
        native.fault_at=adapted.fault_at=None
        native.fit(x);adapted.fit(x)
        for key in ('cluster_centers_','norms_centroids_','labels_','inertia_','_iter','n_iter_'):
            np.testing.assert_array_equal(getattr(native,key),getattr(adapted,key))
        fit=validate_pattern_fit(adapted);record=fit['rollback_policy']['attempts'][-1]
        self.assertFalse(record['triggered']);self.assertEqual(record['stop_reason'],'native_inertia_rollback')
        self.assertEqual((record['legal_updates_completed'],record['retained_legal_updates']),(2,1))

    def test_real_native_update_formula_normal_path(self):
        # Two clearly different synthetic shapes; exercise inherited extraction,
        # normalization and original convergence without injected updates.
        x=np.array([[1.,-1.,0.],[-1.,1.,0.]])[:,:,None]
        x=np.repeat(x,4,axis=0);init=x[[0,4]].copy()
        a=KShape(n_clusters=2,max_iter=5,random_state=2024,init=init).fit(x)
        b=RollbackKShape(n_clusters=2,max_iter=5,random_state=2024,init=init).fit(x)
        self.assertIsNotNone(a._X_fit);self.assertIsNotNone(b._X_fit)
        for key in ('cluster_centers_','norms_centroids_','labels_','inertia_','_iter','n_iter_'):
            np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
        self.assertFalse(b.policy_attempts_[-1]['triggered'])
        for method in ('_shape_extraction','_update_centroids','_assign','_post_fit'):
            self.assertIs(getattr(RollbackKShape,method),getattr(KShape,method))

    def test_second_update_empty_restores_complete_legal_state(self):
        x,_=fixture();m=model().fit(x);fit=validate_pattern_fit(m)
        r=fit['rollback_policy']['attempts'][-1]
        self.assertEqual(r['empty_clusters'],[7]);self.assertEqual(r['trigger_iteration'],2)
        self.assertEqual(r['before'],r['restored'])
        self.assertEqual(r['restored']['cluster_counts'],[2]*16)
        self.assertEqual((m.n_iter_,m._iter,r['legal_updates_completed'],r['retained_legal_updates']),(1,1,1,1))
        self.assertEqual(r['stop_reason'],'post_update_empty_rollback');self.assertTrue(r['early_terminated'])
        expected=model();expected.fault_at=None;expected.max_iter=1;expected.fit(x)
        for key in ('cluster_centers_','norms_centroids_','labels_','inertia_'):
            np.testing.assert_array_equal(getattr(m,key),getattr(expected,key))

    def test_deterministic_results_and_reason(self):
        x,_=fixture();a=model().fit(x);b=model().fit(x)
        self.assertEqual(a.rollback_record(),b.rollback_record())

    def test_first_update_empty_is_failure_not_initialization_success(self):
        x,_=fixture()
        class AlwaysEmpty(Controlled):
            def _update_centroids(self,X):
                self.updates=0;self.fault_at=1;super()._update_centroids(X)
        m=model(AlwaysEmpty).fit(x)
        self.assertIsNone(m._X_fit);self.assertEqual(m.n_iter_,0)
        self.assertEqual(len(m.policy_attempts_),10)  # inherited, unchanged retry limit
        for r in m.policy_attempts_:
            self.assertEqual(r['legal_updates_completed'],0);self.assertFalse(r['early_terminated'])
            self.assertEqual(r['stop_reason'],'first_update_empty_no_legal_update')
        with self.assertRaisesRegex(ValueError,'no accepted successful fit'):validate_pattern_fit(m)

    def test_initial_assignment_failure_is_not_rollback(self):
        x,_=fixture();m=model();m.init[7]=m.init[5];m.fit(x)
        self.assertIsNone(m._X_fit)
        for r in m.policy_attempts_:
            self.assertFalse(r['triggered']);self.assertEqual(r['attempted_updates'],0)

    def test_nonfinite_and_other_update_errors_are_not_hidden(self):
        x,_=fixture()
        for fault,error in [('nonfinite',FloatingPointError),('inertia',FloatingPointError),('other',RuntimeError)]:
            m=model();m.fault=fault
            with self.assertRaises(error):m.fit(x)
            self.assertFalse(m.policy_attempts_[-1]['triggered'])
        m=model();m.fault='update_empty';m.norms_=np.linalg.norm(x,axis=(1,2));m.n_iter_=0
        with self.assertRaises(EmptyClusterError):m._fit_one_init(x,np.random.RandomState(2024))
        self.assertFalse(m.policy_attempts_[-1]['triggered'])

    def test_nonfinite_distance_and_source_identity_rejected(self):
        x,_=fixture()
        with patch.object(KShape,'_cross_dists',return_value=np.full((32,16),np.nan)):
            with self.assertRaises(FloatingPointError):model().fit(x)
        with patch('ch4.kshape_rollback.NATIVE_SHA','unreviewed'):
            with self.assertRaisesRegex(ValueError,'unreviewed'):model().fit(x)

    def test_native_inertia_rollback_assignment_error_not_new_policy(self):
        x,_=fixture()
        class InertiaFailure(Controlled):
            fault_at=None
            def _assign(self,X):
                self.assignments=getattr(self,'assignments',0)+1
                if self.assignments==4:raise EmptyClusterError('native inertia rollback assignment')
                return super()._assign(X)
        m=model(InertiaFailure);m.norms_=np.linalg.norm(x,axis=(1,2));m.n_iter_=0
        with self.assertRaises(EmptyClusterError):m._fit_one_init(x,np.random.RandomState(2024))
        self.assertFalse(m.policy_attempts_[-1]['triggered'])

    def test_normal_max_iter_path_preserved(self):
        x,_=fixture();a=model(NativeControlled);b=model()
        for m in (a,b):m.fault_at=None;m.max_iter=1;m.fit(x)
        for key in ('cluster_centers_','norms_centroids_','labels_','inertia_','_iter','n_iter_'):
            np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
        self.assertEqual(b.policy_attempts_[-1]['stop_reason'],'native_max_iter')
        self.assertFalse(b.policy_attempts_[-1]['triggered'])

    def test_record_rejects_failed_update_count_and_false_native_convergence(self):
        x,_=fixture();fit=validate_pattern_fit(model().fit(x));p=fit['rollback_policy']
        for key,value in [('legal_updates_completed',2),('stop_reason','native_inertia_rollback'),('restored',{})]:
            bad=copy.deepcopy(p);bad['attempts'][-1][key]=value
            with self.assertRaises(ValueError):validate_rollback_record(bad,fit)


if __name__=='__main__':unittest.main()
