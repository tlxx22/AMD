import copy
import unittest
import numpy as np
import torch
from ch4.model import Chapter4Model
from ch4.spatial import RelationBranch, SpatialResponse
from ch4.graphs import geographic_neighbors, semantic_neighbors
from ch4.data import GraphWindows, masked_mse, MetricTotals, fit_scaler, causal_fill


def dense_reference(branch, z, state, embedding, neighbors, available):
    q, key, value = branch.projections(z, state)
    b, k, n, _ = z.shape
    message = torch.zeros_like(z)
    for bi in range(b):
        for ki in range(k):
            for ni in range(n):
                if not available[bi, ki, ni]:
                    continue
                pairs = [(j, ki-d, d) for j in range(n) if j in neighbors[ni].tolist()
                         for d in branch.offsets if ki >= d and available[bi, ki-d, j]]
                if not pairs:
                    continue
                heads = []
                for h in range(4):
                    scores = torch.stack([(q[bi, ki, ni, h]*key[bi, t, j, h]).sum()/4
                                          + (branch.offset_bias[h] if d else 0) for j, t, d in pairs])
                    weights = torch.softmax(scores, 0)
                    heads.append(sum(weights[a]*value[bi, t, j, h] for a, (j,t,d) in enumerate(pairs)))
                message[bi, ki, ni] = torch.cat(heads)
    context = torch.cat((z, embedding[None,None].expand(b,k,-1,-1), state[:,None].expand(-1,k,-1,-1)), -1)
    return branch.w0(message) + branch.right(branch.left(message)*branch.condition(context))


class Core(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_temporal_init_rng_and_import_binding(self):
        import moderntcn_dev_v10.model as v10
        from pathlib import Path
        self.assertEqual(Path(v10.__file__).resolve().parents[1], Path(__file__).resolve().parents[1])
        torch.manual_seed(2024); a = Chapter4Model(3, 12, 3, spatial=False); rng = torch.get_rng_state()
        torch.manual_seed(2024); b = Chapter4Model(3, 12, 3, spatial=True)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in a.temporal.state_dict().items():
            self.assertTrue(torch.equal(value, b.temporal.state_dict()[key]), key)

    def test_split_execution_and_zero_delta_exact(self):
        for p in (1, 12, 24):
            with self.subTest(prediction=p):
                model = Chapter4Model(3, p, 3, target_idx=1).eval()
                x = torch.randn(2,12,3,3); support = torch.arange(3).repeat(3,1)
                reference = copy.deepcopy(model.temporal).eval()
                expected = reference(x.permute(0,2,1,3).reshape(6,12,3))[:,:,1].reshape(2,3,p).permute(0,2,1)
                counts = dict(head=0,u=0,norm=0,denorm=0,space=0)
                captured = {}
                def head_hook(m, args):
                    counts['head'] += 1; captured['head'] = args[0]
                def u_hook(m,args):
                    counts['u'] += 1; captured['u'] = args[0]
                def revin_hook(m,args): counts[args[1]] += 1
                handles = [model.temporal.model.head.register_forward_pre_hook(head_hook),
                           model.temporal.u.register_forward_pre_hook(u_hook),
                           model.temporal.model.revin_layer.register_forward_pre_hook(revin_hook),
                           model.spatial.register_forward_hook(lambda *args: counts.__setitem__('space',counts['space']+1))]
                actual, d = model(x,support,support,torch.ones(2,12,3,dtype=torch.bool),return_diagnostics=True)
                self.assertTrue(torch.equal(expected,actual))
                self.assertEqual(counts,dict(head=1,u=1,norm=1,denorm=1,space=1))
                self.assertTrue(torch.equal(captured['u'], captured['head'][:,1].reshape(6,-1)))
                self.assertTrue(torch.equal(d['H_before'],d['H_after']))
                for handle in handles: handle.remove()
                disabled = model(x,spatial_disabled=True)
                self.assertTrue(torch.equal(expected,disabled))

    def test_sparse_dense_boundaries_and_batch(self):
        torch.manual_seed(10)
        for k in (1,3):
            branch = RelationBranch((0,1))
            z=torch.randn(2,k,3,64);s=torch.randn(2,3,2);e=torch.randn(3,16)
            neighbors=torch.tensor([[0,2,-1],[1,-1,-1],[0,2,-1]])
            valid=torch.ones(2,k,3,dtype=torch.bool);valid[0,:,1]=False;valid[1,0,2]=False
            actual, flag=branch(z,s,e,neighbors,valid)
            expected=dense_reference(branch,z,s,e,neighbors,valid)
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
            self.assertTrue(torch.equal(actual[0,:,1],torch.zeros_like(actual[0,:,1])))
            self.assertFalse(flag[0,:,1].any())
            alone,_=branch(z[:1],s[:1],e,neighbors,valid[:1])
            torch.testing.assert_close(actual[:1],alone,atol=1e-6,rtol=1e-5)

    def test_node_permutation_and_non_target_unchanged(self):
        model=Chapter4Model(3,1,3).eval()
        torch.nn.init.xavier_uniform_(model.spatial.output.weight)
        x=torch.randn(2,12,3,3);g=torch.tensor([[0,1,-1],[0,1,2],[1,2,-1]])
        s=torch.tensor([[0,2],[1,-1],[2,0]]);v=torch.ones(2,12,3,dtype=torch.bool)
        y,d=model(x,g,s,v,return_diagnostics=True)
        self.assertTrue(torch.equal(d['H_before'][:,1:], d['H_after'][:,1:]))
        perm=torch.tensor([2,0,1]);inv=torch.argsort(perm)
        def remap(a):return torch.where(a[perm]>=0,inv[a[perm].clamp_min(0)],-1)
        yp=model(x[:,:,perm],remap(g),remap(s),v[:,:,perm],node_indices=perm)
        torch.testing.assert_close(yp,y[:,:,perm],atol=1e-6,rtol=1e-5)

    def test_both_readout_input_gradients_with_frozen_time(self):
        model=Chapter4Model(3,12,3).eval()
        torch.nn.init.xavier_uniform_(model.spatial.output.weight)
        torch.nn.init.normal_(model.temporal.u.coef,0,.02)
        model.temporal.requires_grad_(False)
        captured={}
        hooks=[model.temporal.model.head.register_forward_hook(lambda m,a,o:captured.__setitem__('head',o)),
               model.temporal.u.register_forward_hook(lambda m,a,o:captured.__setitem__('u',o))]
        x=torch.randn(2,12,3,3);g=torch.arange(3).repeat(3,1)
        model(x,g,g,torch.ones(2,12,3,dtype=torch.bool))
        for name in ['head','u']:
            grad=torch.autograd.grad(captured[name].square().sum(),model.spatial.output.weight,retain_graph=True)[0]
            self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.abs().sum()),0)
        for h in hooks:h.remove()

    def test_graph_support_constant_ties_and_missing_hour(self):
        dist=np.array([[0,1,1,np.inf],[1,0,2,np.inf],[1,2,0,np.inf],[np.inf,np.inf,np.inf,0]])
        g=geographic_neighbors(dist,1)
        self.assertEqual(set(g[3][g[3]>=0]),{3})
        self.assertIn(0,g[2]);self.assertIn(2,g[0])
        t=np.arange('2023-01-01','2023-01-03',dtype='datetime64[h]')
        wave=np.tile(np.arange(24),2);y=np.stack((wave,wave,np.zeros(48)),1)
        s,info=semantic_neighbors(y,np.ones_like(y,dtype=bool),t)
        self.assertEqual(set(s[2][s[2]>=0]),{2});self.assertNotIn(2,s[0])
        valid=np.ones_like(y,dtype=bool);valid[::24,0]=False
        with self.assertRaisesRegex(ValueError,'coverage missing'):semantic_neighbors(y,valid,t)

    def test_window_calendar_scaler_and_zero_mask(self):
        t=np.arange('2023-01-01','2023-01-05',dtype='datetime64[h]');x=np.zeros((96,2,3),np.float32)
        y=np.arange(96)[:,None].repeat(2,1);mask=np.ones_like(y,dtype=bool)
        urban=GraphWindows(x,y,mask,t,20,60,offset=6,history_within_split=True)
        self.assertEqual(len(urban),23);sample=urban[0]
        self.assertEqual(sample['origin'],32);self.assertEqual(float(sample['y'][0,0]),37)
        self.assertAlmostEqual(float(sample['future_calendar'][0,0]),13/24,places=6)
        generic=GraphWindows(x,y,mask,t,60,80,prediction=12)
        self.assertEqual(len(generic),9);self.assertEqual(generic.origins[0],60)
        mean,scale=fit_scaler(x[:50],np.ones_like(x[:50],dtype=bool))
        np.testing.assert_array_equal(scale,np.ones(3))
        broken=x.copy();valid=np.ones_like(x,dtype=bool);broken[0,0,0]=np.nan;valid[0,0,0]=False
        self.assertEqual(causal_fill(broken,valid,mean)[0,0,0],0)
        pred=torch.tensor([0.,2.,3.],requires_grad=True);truth=torch.tensor([0.,0.,float('nan')]);mask=torch.tensor([True,True,False])
        loss=masked_mse(pred,truth,mask);loss.backward();self.assertEqual(float(loss),2.)
        self.assertTrue(torch.isfinite(pred.grad).all())
        totals=MetricTotals();totals.update(pred.detach(),truth,mask)
        self.assertEqual(totals.result()['count'],2);self.assertEqual(totals.result()['mae'],1.)


if __name__ == '__main__':
    unittest.main()
