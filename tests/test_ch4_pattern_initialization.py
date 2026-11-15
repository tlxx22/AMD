"""Only small synthetic distances/assignments; no KShape.fit."""
import unittest
import numpy as np
from ch4.pattern_initialization import select_initial_centers,distances_to_center,native_witness,TOL


class InitializationTests(unittest.TestCase):
    def test_pool_members_deterministic_and_native_witness(self):
        x=np.sin(np.arange(600).reshape(200,3,1)**2)
        x=np.concatenate((np.zeros((40,3,1)),np.ones((20,3,1)),x,x))
        before=x.copy();a,ia=select_initial_centers(x);b,ib=select_initial_centers(x)
        np.testing.assert_array_equal(x,before);np.testing.assert_array_equal(a,b)
        self.assertEqual(ia,ib);np.testing.assert_array_equal(a,x[ia['selected_indices']])
        self.assertEqual(ia['witness']['counts'],[1]*16)
        self.assertFalse(ia['witness']['full_fit_success'])
        self.assertTrue(all(i>=60 for i in ia['selected_indices']))
        # Each exact duplicate resolves to the earliest original index.
        self.assertTrue(all(i<260 for i in ia['selected_indices']))

    def test_vectorized_distance_matches_actual_tslearn(self):
        from tslearn.metrics.cycc import cdist_normalized_cc
        x=np.array([[0,0,0],[2,2,2],[1,0,0],[0,1,0],[0,0,1],[2,0,0],[-1,2,3],[1e-8,0,0]],dtype=float)[:,:,None]
        norms=np.linalg.norm(x,axis=(1,2))
        native=1-cdist_normalized_cc(x,x,norms.copy(),norms.copy(),False)
        actual=np.column_stack([distances_to_center(x,c,norms) for c in x])
        np.testing.assert_allclose(actual,native,rtol=0,atol=TOL)
        self.assertLessEqual(abs(native[2,3]),TOL)  # shifted impulse equivalence
        self.assertLessEqual(abs(native[2,5]),TOL)  # positive amplitude equivalence
        self.assertEqual(native[0,0],1.)  # native zero-norm convention, not self-distance zero

    def test_degenerate_pools_and_ambiguous_centers_rejected(self):
        for x in (np.zeros((40,3,1)),np.ones((40,3,1)),np.full((40,3,1),np.nan),np.tile([1.,0.,0.],(40,1))[:,:,None],
                  np.array([[1e-8,0,0]]*40)[:,:,None]):
            with self.assertRaises(ValueError):select_initial_centers(x)
        with self.assertRaisesRegex(ValueError,'equivalent|ambiguous'):native_witness(np.ones((16,3,1)))


if __name__=='__main__':unittest.main()
