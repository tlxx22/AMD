"""Metadata/fixture tests only: no model, data loader, optimizer or GPU."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from moderntcn_dev_v1.training_config import ROOT, validate, arithmetic
from moderntcn_dev_v1.extension_scope import (SCOPE, derive, expected_tasks, expected_waves,
                                             reference, validate_launch_scope)
from moderntcn_dev_v1.launcher import batch, validate_authorization, prepare_resume
from moderntcn_dev_v1.summary import aggregate
from moderntcn_dev_v1.worker import successful_identity, source_binding
from moderntcn_dev_v1.training_state import restore
from moderntcn_dev_v1.spec import StructuralSpec

class ExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = json.loads((ROOT/'configs/moderntcn_uv_round2_resolved_v1.json').read_text())
        cls.new = json.loads((ROOT/'configs/moderntcn_uv_extension_f6_epf_3arm_v1.json').read_text())

    def test_01_original53_unchanged_and_valid(self):
        self.assertEqual(hashlib.sha256((ROOT/'configs/moderntcn_uv_round2_resolved_v1.json').read_bytes()).hexdigest(),
                         '8f39253f53ddb88cb5763d4f08d50f1139250a2f90029eaedcfa22b9c04ded68')
        validate(self.old);self.assertEqual(len(self.old['tasks']),53)

    def test_02_exact_derivation_sources(self):
        self.assertEqual(derive(),self.new)
        self.assertEqual(self.new['source_binding'],source_binding())
        self.assertEqual(self.new['tasks'],expected_tasks())

    def test_03_nine_waves_and_order(self):
        ids=[t['id'] for t in self.new['tasks']]
        self.assertEqual(batch(self.new,ids),expected_waves())
        self.assertEqual(list(map(len,batch(self.new,ids))),[4,4,1]*3)
        for bad in [ids[::-1],ids+[ids[0]],['foreign']]:
            with self.assertRaises(ValueError):batch(self.new,bad)
        for bad in [ids[:-1],ids[:4]]:
            with self.assertRaises(ValueError):validate_launch_scope(self.new,bad,dict(task_scope=SCOPE))

    def test_04_budgets_and_original_fold6_endpoints(self):
        tasks=self.new['tasks'];urban=[t for t in tasks if t['dataset']=='UrbanEV']
        for t in urban:
            self.assertEqual(t['data_config']['urban_folds'][5],[3475,3909,4344])
            n=(3475-12-t['h']+1)*275
            self.assertEqual(t['arithmetic']['windows']['train'],n)
            self.assertEqual(t['structural_source_profile']['training']['scheduler']['steps_per_epoch'],n//128)
            self.assertEqual(t['arithmetic'],arithmetic(t['data_config'],t['structural_source_profile'],6))
        self.assertEqual(sum(t['arithmetic']['max_optimizer_steps'] for t in urban),1782120)
        self.assertEqual(sum(t['arithmetic']['max_optimizer_steps'] for t in tasks if t not in urban),85500)
        self.assertEqual(sum(t['arithmetic']['max_epochs'] for t in tasks),540)

    def test_05_epf_schema_and_original_five_UV(self):
        old={t['id']:t for t in self.old['tasks']}
        for t in self.new['tasks']:
            if t['dataset']=='UrbanEV':continue
            p=t['structural_source_profile']
            self.assertEqual((p['C'],p['target_idx'],p['aux_idx'],p['T'],p['pred_len']),(3,2,[0,1],96,24))
            if t['arm']=='UV':self.assertEqual(t,old[t['id']])
        self.assertEqual(len({t['id'] for t in self.new['tasks']}-set(old)),22)

    def test_06_nine_B_and_unrounded_values(self):
        self.assertEqual(len(self.new['existing_B']),9)
        for k,b in self.new['existing_B'].items():
            p=Path(b['result_ref']['path']);self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),b['result_ref']['sha256'])
            v=json.loads(p.read_text());self.assertEqual((v['mse'],v['mae'],v['profile_sha']),(b['mse'],b['mae'],b['profile_sha']))
            self.assertEqual(v['metric_space'],'train-standardized');self.assertEqual(v['task'],'MS')

    def test_07_scope_corruption_rejected(self):
        for key,value in [('task_scope','generic'),('max_optimizer_steps',1867621),('max_run_epochs',541),
                          ('B_ref','paper_C1_r8_v1'),('F4_reuse',self.old['F4_reuse'])]:
            c=copy.deepcopy(self.new);c[key]=value
            with self.assertRaises(ValueError):validate(c)
        for mode in ['missing','duplicate','order']:
            c=copy.deepcopy(self.new)
            if mode=='missing':c['tasks'].pop()
            elif mode=='duplicate':c['tasks'].append(c['tasks'][0])
            else:c['tasks'].reverse()
            with self.assertRaises(ValueError):validate(c)

    def test_08_science_mutations_rejected(self):
        for key,value in [('fold',4),('arm','B'),('goal',1),('baseline_pair_cell_id','foreign')]:
            c=copy.deepcopy(self.new);c['tasks'][0][key]=value
            with self.assertRaises(ValueError):validate(c)
        c=copy.deepcopy(self.new);c['existing_B'][next(iter(c['existing_B']))]['mse']=0
        with self.assertRaises(ValueError):validate(c)

    def records(self,ratio=.9):
        return [(t,dict(task_id=t['id'],**{m:self.new['existing_B'][t['baseline_pair_cell_id']][m]*ratio for m in ['mse','mae']})) for t in self.new['tasks']]

    def test_09_extension_summary_does_not_replace_goals(self):
        s=aggregate(self.new,self.records());self.assertEqual(s['engineering_completion'],'Complete')
        self.assertEqual(s['completed'],27);self.assertEqual(s['goals']['goal2']['winning_cells'],10)
        for k in ['goal1','goal3']:self.assertEqual(s['goals'][k]['numerical_status'],'Not evaluated')
        self.assertFalse(s['fold6']['replaces_goal1']);self.assertEqual(len(s['fold6']['per_horizon']),4)

    def test_10_missing_UV_does_not_shrink_denominator(self):
        rec=self.records()[:-1];s=aggregate(self.new,rec)
        self.assertEqual(s['goals']['goal2']['numerical_status'],'Not evaluated')
        self.assertEqual(s['goals']['goal2']['denominator'],10)
        s=aggregate(self.new,[(t,m) for t,m in self.records() if t['arm']=='UV'])
        self.assertEqual(s['goals']['goal2']['winning_cells'],10)
        self.assertEqual(s['engineering_completion'],'Partial')

    def test_11_ties_and_seven_of_ten(self):
        s=aggregate(self.new,self.records(1.));self.assertEqual(s['goals']['goal2']['winning_cells'],0)
        rec=self.records();uv=[m for t,m in rec if t['dataset']!='UrbanEV' and t['arm']=='UV']
        # All five MSE + only two MAE wins; no separate 70% MAE gate.
        for m in uv[2:]:m['mae']/=.9
        s=aggregate(self.new,rec);self.assertEqual(s['goals']['goal2']['winning_cells'],7)
        self.assertTrue(s['goals']['goal2']['numerical_conditions_met'])

    def test_12_duplicate_and_foreign_summary_rejected(self):
        rec=self.records()
        with self.assertRaises(ValueError):aggregate(self.new,rec+[rec[0]])
        bad=copy.deepcopy(rec);bad[0][0]['fold']=1
        with self.assertRaises(ValueError):aggregate(self.new,bad)

    def test_13_unapproved_and_old_grants_rejected_before_git(self):
        ids=[t['id'] for t in self.new['tasks']]
        with self.assertRaises(ValueError):validate_launch_scope(self.new,ids,{})
        with patch('moderntcn_dev_v1.launcher.git',side_effect=AssertionError('must reject before Git')):
            with self.assertRaises(PermissionError):validate_authorization(self.new,'unused',ids,dict(task_scope=SCOPE,review_approved=False))

    def test_14_cross_revision_restore_before_any_read(self):
        t=self.new['tasks'][0];spec=StructuralSpec.from_profile(t['structural_source_profile'],use_u=True,use_v=False,fold=6).value
        live=SimpleNamespace(spec=SimpleNamespace(value=copy.deepcopy(spec)))
        identity=dict(candidate_id='moderntcn-uv-dev-v1',structural_spec=copy.deepcopy(spec));identity['structural_spec']['structural_revision']['u']='foreign'
        with patch('timemixer_dev.training_state.read',side_effect=AssertionError('no read allowed')):
            with self.assertRaises(ValueError):restore('unused',live,None,None,None,identity)
        self.assertEqual(live.spec.value,spec)

    def test_15_success_key_preserves_EPF_identity(self):
        t=next(t for t in self.new['tasks'] if t['dataset']=='PJM' and t['arm']=='UV')
        spec=StructuralSpec.from_profile(t['structural_source_profile'],use_u=True,use_v=True,fold=1).value
        identity=dict(candidate_id='moderntcn-uv-dev-v1',task=t,profile=t['structural_source_profile'],
                      structural_spec=spec,data_sha='fixed-fixture',purpose='development',
                      reference_original=False,training_protocol='round2_complete_onecycle_v3')
        old=copy.deepcopy(identity)
        old['task']=next(x for x in self.old['tasks'] if x['id']==t['id'])
        old['structural_spec']['implementation']['training_config.py']='old queue bytes'
        old['structural_spec']['implementation'].pop('extension_scope.py',None)
        old['structural_spec']['implementation'].pop('extension_reference.json',None)
        self.assertEqual(successful_identity(old),successful_identity(identity))
        other=copy.deepcopy(identity);other['task']['arm']='U'
        self.assertNotEqual(successful_identity(other),successful_identity(identity))

    def test_16_successful_attempt_and_unapproved_resume_refused(self):
        root=ROOT.parent/'amd-execution-evidence/m6/M6-ModernTCN-UV-v1-extension-f6-epf-implementation-r1/metadata-fixtures'
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            folder=Path(tmp);(folder/'complete.json').write_text('{}')
            with self.assertRaises(PermissionError):prepare_resume(folder,[],{},config=self.new)
            self.assertEqual((folder/'complete.json').read_text(),'{}')
        ids=[t['id'] for t in self.new['tasks']][4:9]
        with self.assertRaises(ValueError):validate_launch_scope(self.new,ids,dict(task_scope=SCOPE))
        grant=dict(task_scope=SCOPE,resume_review_approved=True,resume_task_ids=ids,previous_runtime_ref={'path':'fixture'})
        validate_launch_scope(self.new,ids,grant)
        self.assertEqual(list(map(len,batch(self.new,ids))),[4,1])

    def test_17_old_scope_and_gate_unchanged(self):
        self.assertEqual(self.old['gate_policy'],self.new['gate_policy'])
        ids=[t['id'] for arm in ('U','V','UV') for fold in (1,4) for h in (3,6,9,12)
             for t in self.old['tasks'] if t['goal']==1 and (t['arm'],t['fold'],t['h'])==(arm,fold,h)]
        self.assertEqual(list(map(len,batch(self.old,ids))),[4]*6)
        s=aggregate(self.old,[])
        self.assertEqual(s['required'],53)
        self.assertTrue(all(v['numerical_status']=='Not evaluated' for v in s['goals'].values()))

if __name__=='__main__':
    unittest.main()
