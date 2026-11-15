import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from ch4.native_graphs import (charged_adjacency,threshold_weights,doubletransition,d2_doubletransition,
                              stdn_binary,topology_diagnostics,source_hops,save_fixed_artifacts)
from ch4.prepare_plan import structure


class GraphContract(unittest.TestCase):
    def test_gaussian_population_and_invalid_identity(self):
        d=np.array([[0,0,2],[0,0,4],[2,4,0]],dtype=float);ids=['b','a','c']
        a,rule=charged_adjacency(d,ids,ids,ids)
        expected=np.exp(-(d/d.std(ddof=0))**2);expected[expected<.1]=0
        np.testing.assert_array_equal(a,expected.astype('float32'))
        self.assertEqual(rule['sigma'],d.std(ddof=0));self.assertEqual(a[0,1],1)
        np.testing.assert_array_equal(np.diag(a),np.ones(3))
        for bad in (d+np.eye(3),np.where(d==2,np.nan,d),-d,np.zeros_like(d)):
            with self.assertRaises(ValueError):charged_adjacency(bad,ids,ids,ids)
        with self.assertRaises(ValueError):charged_adjacency(d,ids,ids[::-1],ids)
        bad=d.copy();bad[0,2]=3
        with self.assertRaises(ValueError):charged_adjacency(bad,ids,ids,ids)

    def test_threshold_and_distinct_native_directions(self):
        x=np.array([np.nextafter(.1,0),.1,np.nextafter(.1,1)])
        np.testing.assert_array_equal(threshold_weights(x),np.array([0,.1,.1],dtype='float32'))
        for a in (np.array([[1,2,0],[0,1,4],[0,0,0.]]),np.array([[1,2,0],[2,1,4],[0,4,1.]])):
            expected=[]
            for matrix in (a,a.T):
                result=np.zeros_like(matrix)
                for i,row in enumerate(matrix):
                    if row.sum():result[i]=row/row.sum()
                expected.append(result)
            np.testing.assert_allclose(doubletransition(a),expected,rtol=1e-7,atol=1e-8)
            np.testing.assert_allclose(d2_doubletransition(a),np.asarray(expected).transpose(0,2,1),rtol=1e-7,atol=1e-8)
            self.assertFalse(np.array_equal(doubletransition(a),d2_doubletransition(a)))
            self.assertTrue(np.isfinite(d2_doubletransition(a)).all())

    def test_consumer_files_self_loops_no_regaussianization(self):
        n=40;a=np.full((n,n),.2,dtype='float32');np.fill_diagonal(a,1)
        identity=dict(reviewed_migration='synthetic accepted rule',source_sha256='synthetic-only')
        with tempfile.TemporaryDirectory() as parent:
            root=Path(parent)/'graphs';refs=save_fixed_artifacts(root,a,identity,non_pems=True)
            np.testing.assert_array_equal(np.load(refs['adj_mx']['path']),a)
            np.testing.assert_array_equal(np.diag(np.load(refs['stdn_adjacency']['path'])),np.zeros(n))
            self.assertNotIn('supports',refs)
            for key,dim in [('laplacian8',8),('laplacian32',32)]:
                value=np.load(refs[key]['path']);self.assertEqual(value.shape,(n,dim));self.assertTrue(np.isfinite(value).all())
            self.assertEqual(json.loads((root/'identity.json').read_text())['failures'],{})
        disconnected=np.eye(3);diag=topology_diagnostics(disconnected)
        self.assertEqual((diag['external_directed_edges'],diag['isolated_nodes'],diag['weak_components']),(0,3,3))
        self.assertEqual(source_hops(disconnected)[0,1],511)
        self.assertEqual(source_hops(disconnected)[0,0],0)
        np.testing.assert_array_equal(stdn_binary(disconnected,non_pems=True),np.zeros((3,3)))

    def test_seven_temporal_sources_are_not_fallback_profiles(self):
        from ch4.prepare_plan import ROOT
        from ch4.protocol import MODELS
        adopted=json.loads((ROOT/'ch4/temporal_sources.json').read_text())
        for name in MODELS[:7]:
            for domain in ('UrbanEV','AMS','PEMS08'):
                value=structure(name,domain,3,12,7,1)
                self.assertEqual(value['values'],adopted[name]['profile']['structure'])
                self.assertNotIn('pending',value['selection'])
        self.assertEqual(adopted['PatchTST']['profile']['structure']['e_layers'],2)


if __name__=='__main__':unittest.main()
