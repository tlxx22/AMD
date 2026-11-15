import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
import torch
from ch4.dependencies import import_closure,verify_closure,execution_seeds
from ch4.native_graphs import validate_pattern_fit
from ch4.model import Chapter4Model
from ch4.optimization import optimizer_and_scheduler
from ch4.protocol import resolved_optimization


class Repair(unittest.TestCase):
    def test_dependency_missing_and_modified_temporary_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'pkg').mkdir()
            (root/'main.py').write_text('from pkg.model import f\n')
            (root/'pkg/__init__.py').write_text('from . import side_effect\n')
            (root/'pkg/model.py').write_text('from .helper import f\n')
            (root/'pkg/helper.py').write_text('def f(): return 1\n')
            (root/'pkg/side_effect.py').write_text('value=7\n')
            seeds=[(root/'main.py',root)];files=import_closure(seeds)['files']
            self.assertEqual(len(files),5);verify_closure(files,seeds)
            omitted=dict(files);del omitted[str(root/'pkg/side_effect.py')]
            with self.assertRaisesRegex(PermissionError,'unbound'):verify_closure(omitted,seeds)
            (root/'pkg/helper.py').write_text('def f(): return 2\n')
            with self.assertRaisesRegex(PermissionError,'changed'):verify_closure(files,seeds)

    def test_actual_project_closure_includes_initializers_not_launcher(self):
        from ch4.prepare_plan import ROOT,E
        temporal=json.loads((ROOT/'ch4/temporal_sources.json').read_text())
        files=import_closure(execution_seeds(ROOT,E/'sources',temporal))['files']
        for name in ('utils/feature_schema.py','models/modules/__init__.py','models/modules/target_exogenous_bridge.py'):
            self.assertIn(str(ROOT/name),files)
        self.assertNotIn(str(ROOT/'moderntcn_input_ablation_v10/launcher.py'),files)
        self.assertNotIn(str(ROOT/'moderntcn_input_ablation_v10/gate_policy.py'),files)

    def test_pattern_success_and_failure_decision(self):
        class Fit(SimpleNamespace):
            def get_params(self):return self.parameters
        valid=Fit(_X_fit=np.zeros((32,3,1)),inertia_=.1,n_iter_=3,cluster_centers_=np.ones((16,3,1)),
                  labels_=np.tile(np.arange(16),2),parameters=dict(n_clusters=16,max_iter=5,random_state=2024,init=np.ones((16,3,1))))
        self.assertTrue(validate_pattern_fit(valid)['successful_fit'])
        for key,value in [('_X_fit',None),('inertia_',np.inf),('n_iter_',0),('cluster_centers_',np.ones((15,3,1))),
                          ('cluster_centers_',np.full((16,3,1),np.nan)),('labels_',np.zeros(32,dtype=int))]:
            invalid=copy.deepcopy(valid);setattr(invalid,key,value)
            with self.subTest(key=key),self.assertRaises(ValueError):validate_pattern_fit(invalid)
        invalid=copy.deepcopy(valid);invalid.parameters['max_iter']=6
        with self.assertRaises(ValueError):validate_pattern_fit(invalid)

    def test_one_optimizer_joint_temporal_v_u_spatial_paths(self):
        torch.set_num_threads(1)
        for prediction in (1,12):
            torch.manual_seed(2024);model=Chapter4Model(3,prediction,3).train()
            self.assertTrue(all(p.requires_grad for p in model.parameters()))
            cfg=resolved_optimization('ch4_v10_pdformer_him_v1','UrbanEV')
            optimizer,_=optimizer_and_scheduler(model,cfg,2)
            ids=[id(p) for group in optimizer.param_groups for p in group['params']]
            self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{id(p) for p in model.parameters()})
            # Controlled nonzero output projection checks internal spatial paths;
            # zero-init first-step internal zero gradients are not a failure.
            torch.nn.init.xavier_uniform_(model.spatial.output.weight)
            x=torch.randn(2,12,3,3);neighbors=torch.arange(3).repeat(3,1)
            output=model(x,neighbors,neighbors,torch.ones(2,12,3,dtype=torch.bool))
            loss=(output-torch.randn_like(output)).square().mean();loss.backward()
            groups={'backbone':'temporal.model.','V3':'temporal.v.','U':'temporal.u.','space':'spatial.'}
            for name,prefix in groups.items():
                params=[p for key,p in model.named_parameters() if key.startswith(prefix)]
                self.assertTrue(params,name)
                grads=[p.grad for p in params if p.grad is not None]
                self.assertTrue(grads,name);self.assertTrue(all(torch.isfinite(g).all() for g in grads),name)
                self.assertGreater(sum(float(g.abs().sum()) for g in grads),0,name)
            before={name:p.detach().clone() for name,p in model.named_parameters()}
            optimizer.step()
            for name,prefix in groups.items():
                self.assertTrue(any(not torch.equal(before[key],p) for key,p in model.named_parameters() if key.startswith(prefix)),name)


if __name__=='__main__':unittest.main()
