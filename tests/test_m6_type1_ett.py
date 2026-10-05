"""Incremental ETT source/M/231-boundary fixtures; zero real model computation."""
import ast,copy,importlib.util,json,math,unittest
from pathlib import Path
from unittest.mock import patch
from utils import ch3_type1_tasks as s,ch3_type1_ett as e,ch3_type1_chain as q,ch3_type1_upstream as u
from utils.ch3_contract import ROOT,digest,profile,step_arithmetic
from utils.ch3_native_recovery_records import bound,ref,sha

class ETTExtension(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cs=q.configs();cls.c=cls.cs['M_ALL'];cls.old=e.template()
    def test_exact_28_35_168_231(self):
        self.assertEqual([len(c['tasks'])for c in self.cs.values()],[28,35,168]);self.assertEqual(sum(len(c['tasks'])for c in self.cs.values()),231)
    def test_exact_new_84_matrix(self):
        self.assertEqual({(t['model'],t['dataset'],t['h'])for t in self.c['tasks']if t['dataset']in e.NEW_DATASETS},{(m,d,h)for m in s.MODELS for d in e.NEW_DATASETS for h in (96,192,336,720)})
    def test_order_old_then_new_per_model(self):
        for m in s.MODELS:
            ts=[t for t in self.c['tasks']if t['model']==m]
            self.assertEqual([t['dataset']for t in ts],[d for d in e.M_DATASETS for _ in range(4)])
            self.assertEqual([t for t in ts if t['dataset']not in e.NEW_DATASETS],[t for t in self.old['tasks']if t['model']==m])
    def test_exclusions_and_seed(self):
        for c in self.cs.values():
            self.assertFalse(any(t['model']in ('J','N','S')or t['dataset']=='ECL'or t['seed']!=2024 for t in c['tasks']))
    def test_original_147_profile_exact(self):
        proof=bound(ref(s.PACKAGE/'original-147-invariance.json'));self.assertEqual(proof['count'],147)
        self.assertEqual(proof['profile_changes'],0);self.assertTrue(all(r['old_sha']==r['new_sha']for r in proof['rows']))
        for stage,c in self.cs.items():
            old=json.loads((s.PACKAGE/'reviewed-147-candidate/workspace-bytes'/s.file(stage).relative_to(ROOT)).read_text())
            for t in old['tasks']:self.assertEqual(profile(c,t),old['resolved_profiles'][t['id']])
    def test_Urban_EPF_bytes_unchanged(self):
        for stage in s.STAGES[:2]:self.assertEqual(sha(s.file(stage)),sha(s.PACKAGE/'reviewed-147-candidate/workspace-bytes'/s.file(stage).relative_to(ROOT)))
    def test_each_own_model_H_template(self):
        for t in self.c['tasks']:
            if t['dataset']not in e.NEW_DATASETS:continue
            before=copy.deepcopy(self.old['resolved_profiles'][e.source_task(t)['id']]);after=profile(self.c,t)
            self.assertEqual(e.source_task(t)['model'],t['model']);self.assertEqual(e.source_task(t)['h'],t['h'])
            for p in (before,after):
                p.pop('dataset');p.pop('time_mark',None);p['structure'].pop('freq',None);p['training']['scheduler'].pop('steps_per_epoch')
            self.assertEqual(before,after)
    def test_new_training_contract(self):
        for t in self.c['tasks']:
            p=profile(self.c,t);tr=p['training'];self.assertEqual((p['T'],p['pred_len'],p['C']),(96,t['h'],len(p['features'])))
            self.assertEqual(tuple(tr[k]for k in ('batch','eval_batch','epochs','patience','accumulation','seed')),(128,128,10,None,1,2024))
            self.assertFalse(tr['eval_drop_last']);self.assertTrue(tr['train_drop_last']);self.assertTrue(tr['train_shuffle'])
    def test_Table7_bound_PDF_rows(self):
        p=bound(e.PAPER_REF);self.assertEqual(p['pdf_ref']['sha256'],'a599b338e44af70d8e9c87be3c5417bde7864b2c92074e1346703f3e2b641e3d')
        self.assertEqual(sha(p['pdf_ref']['path']),p['pdf_ref']['sha256']);self.assertEqual((p['page_number'],p['table']),(14,7))
        for d in e.NEW_DATASETS:self.assertEqual(p['rows'][d],dict(train_batch=128,epochs=10))
    def test_Table7_is_not_eval_or_patience_source(self):
        p=bound(e.PAPER_REF);self.assertEqual(p['project_inherited']['eval_batch'],128);self.assertIsNone(p['project_inherited']['patience']);self.assertEqual(p['Weather'],dict(paper_epochs=20,project_epochs=10,unchanged=True))
    def test_new_independent_data_SHA(self):
        rows=e.sources()['datasets'];hashes=set()
        for d,r in rows.items():
            a=self.c['datasets'][d]['source_admission'];h=sha(self.c['datasets'][d]['path']);self.assertEqual(h,r['object']['sha256']);self.assertEqual(h,sha(r['object']['official_path']));self.assertEqual(h,a['local_sha256_inherited']);self.assertNotEqual(h,self.old['datasets']['ETTh1']['source_admission']['local_sha256_inherited']);hashes.add(h)
            self.assertIn('not an inherited M5 admission',a['accepted_scope'])
        self.assertEqual(len(hashes),3)
    def test_hour_minute_reader_split_frequency(self):
        for d,r in e.sources()['datasets'].items():
            minute=d.startswith('ETTm');factor=4 if minute else 1
            self.assertEqual(r['endpoints'],[8640*factor,11520*factor,14400*factor]);self.assertEqual(r['interval_seconds'],900 if minute else 3600)
            self.assertEqual(r['reader'],'Dataset_ETT_minute'if minute else'Dataset_ETT_hour');self.assertEqual(r['native_mark_dimensions'],5 if minute else 4)
    def test_source_state_current(self):
        from utils.ch3_m6 import source_states
        self.assertEqual(source_states(self.c),bound(self.c['baseline_unified']['data_ref'])['source_states'])
    def test_new_prefix_scaler_no_test(self):
        from utils.ch3_data import load
        data=bound(self.c['baseline_unified']['data_ref'])
        for d in e.NEW_DATASETS:
            t=next(t for t in self.c['tasks']if t['dataset']==d and t['model']=='TimeMixer'and t['h']==96);splits,m=load(self.c,t)
            self.assertEqual(set(splits),{'train','validation'});self.assertFalse(m['test_observations_accessed']);self.assertEqual(m['scaler_fit_records'],self.c['datasets'][d]['endpoints'][0]);self.assertEqual(digest(m),data['data_bindings'][d][t['id']]);self.assertNotEqual(m['scaler'],data['metadata']['ETTh1'][e.source_task(t)['id']]['scaler'])
            x,y,mark=splits['train'][0];self.assertEqual(tuple(x.shape),(96,7));self.assertEqual(tuple(y.shape),(96,7));self.assertEqual(tuple(mark.shape),(96,5 if d.startswith('ETTm')else 4))
    def test_all_84_metadata_windows(self):
        data=bound(self.c['baseline_unified']['data_ref'])
        for t in self.c['tasks']:
            if t['dataset']not in e.NEW_DATASETS:continue
            m=data['metadata'][t['dataset']][t['id']];a=step_arithmetic(self.c,t)
            self.assertEqual(m['window_counts'],dict(train=a['train_windows'],validation=a['validation_windows']));self.assertEqual(digest(m),data['data_bindings'][t['dataset']][t['id']]);self.assertFalse(m['test_observations_accessed'])
    def test_real_minute_timeF_matches_author(self):
        import numpy as np,pandas as pd
        from utils.ch3_time_marks import marks
        source=e.sources()['author_timeF_ref'];self.assertEqual(sha(source['path']),source['sha256']);spec=importlib.util.spec_from_file_location('fixture_bound_timeF',source['path']);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        clock=pd.date_range('2016-07-01',periods=96,freq='15min');expected=module.time_features(clock,freq='t').T.astype('float32');self.assertTrue(np.array_equal(marks(clock,'t'),expected));self.assertEqual(expected.shape,(96,5))
    def test_hourly_marks_unchanged(self):
        import numpy as np,pandas as pd
        from utils.ch3_time_marks import marks
        clock=pd.date_range('2016-07-01',periods=96,freq='h');old=np.stack((clock.hour/23.-.5,clock.dayofweek/6.-.5,(clock.day-1)/30.-.5,(clock.dayofyear-1)/365.-.5),axis=1).astype('float32');self.assertTrue(np.array_equal(marks(clock,'h'),old))
    def test_minute_synthetic_clock_and_K(self):
        import torch
        from utils.ch3_time_marks import synthetic,policy
        for name in ('iTransformer','TimeMixer','TimeXer'):
            t=next(t for t in self.c['tasks']if t['dataset']=='ETTm1'and t['model']==name);p=profile(self.c,t);self.assertEqual(p['structure']['freq'],'t');self.assertEqual(p['time_mark'],policy(p));a=synthetic(p,2);self.assertEqual(tuple(a.shape),(2,96,5));self.assertTrue(torch.equal(a[0],a[1]));self.assertEqual(a[0,4,0].item(),-.5)
    def test_actual_native_adapter_M_7_channels_K5(self):
        import torch
        from models.ch3_adapter import target_prediction
        from utils.ch3_time_marks import synthetic
        class ForwardFixture:
            def __call__(self,x,m,y,ym):self.received=m;return torch.zeros(x.shape[0],96,7)
        for model in ('iTransformer','TimeMixer','TimeXer'):
            t=next(t for t in self.c['tasks']if t['dataset']=='ETTm1'and t['model']==model and t['h']==96);p=profile(self.c,t);fixture=ForwardFixture();x=torch.zeros(2,96,7);mark=synthetic(p,2);prediction,aux=target_prediction(fixture,x,p,mark);self.assertEqual(tuple(prediction.shape),(2,96,7));self.assertIs(fixture.received,mark)
            with self.assertRaises(ValueError):target_prediction(fixture,x,p,mark[:,:,:4])
    def test_native_adapter_math_AST_guard_only(self):
        import subprocess
        old=ast.parse(subprocess.check_output(['git','show',s.BASE+':models/ch3_adapter.py'],cwd=ROOT,text=True));new=ast.parse((ROOT/'models/ch3_adapter.py').read_text())
        def replace_guard(tree):
            for n in ast.walk(tree):
                if isinstance(n,ast.Tuple)and len(n.elts)==3 and isinstance(n.elts[2],ast.Subscript)and ast.unparse(n.elts[2])=="p['time_mark']['K']":n.elts[2]=ast.Constant(value=4)
            return ast.dump(tree,include_attributes=False)
        self.assertEqual(ast.dump(old,include_attributes=False),replace_guard(new))
    def test_original_numeric_scopes_exact(self):
        for stage,c in self.cs.items():
            old=json.loads((s.PACKAGE/'reviewed-147-candidate/workspace-bytes'/s.file(stage).relative_to(ROOT)).read_text())
            for key,value in old['baseline_unified']['numeric_policies'].items():self.assertEqual(c['baseline_unified']['numeric_policies'][key],value)
    def test_new_21_policy_template_migration(self):
        proof=bound(ref(s.PACKAGE/'ett-numeric-policy-migration.json'));self.assertEqual(proof['new_entry_count'],21);self.assertEqual(proof['old_scope_changes'],[])
        for r in proof['new_scopes']:
            a=copy.deepcopy(r['template_policy']);b=copy.deepcopy(r['policy'])
            if a is not None:
                for x in (a,b):x.pop('id');x.pop('dataset')
            self.assertEqual(a,b)
    def test_new_M_never_uses_EPFnumeric(self):
        for d in e.NEW_DATASETS:self.assertEqual(self.c['baseline_unified']['numeric_policies']['ModernTCN-'+d]['state_atol'],1e-4)
        self.assertEqual(self.cs['EPF_ALL']['baseline_unified']['numeric_policies']['ModernTCN-PJM']['state_atol'],5e-4)
    def test_group_57_representative_231(self):
        groups=[s.probe_groups(c)for c in self.cs.values()];self.assertEqual([len(g)for g in groups],[7,8,42]);self.assertEqual([sum(len(g['representatives'])for g in bank)for bank in groups],[28,35,168]);self.assertEqual(sum(map(len,groups)),57)
    def test_M_42_independent_identity_groups(self):
        groups=s.probe_groups(self.c);self.assertEqual({(g['model'],next(t['dataset']for t in self.c['tasks']if t['id']==g['representatives'][0]))for g in groups},{(m,d)for m in s.MODELS for d in e.M_DATASETS})
        self.assertTrue(all(g['planned_q']==4 and len(g['representatives'])==4 and len(g['identities'])==4 for g in groups))
    def test_budget_increment_and_total(self):
        b=bound(ref(s.PACKAGE/'budget.json'));self.assertEqual(b['new_84_increment'],dict(runs=84,run_epochs=840,adam=166880,backward=166880,forward=227325));self.assertEqual(b['total'],dict(runs=231,run_epochs=2590,adam=1702400,backward=1702400,forward=1967175));self.assertEqual(s.formal_budget(self.c)['total'],b['formal']['M_ALL'])
    def test_per_H_window_batch_tail_arithmetic(self):
        for t in self.c['tasks']:
            if t['dataset']not in e.NEW_DATASETS:continue
            end,val,test=self.c['datasets'][t['dataset']]['endpoints'];a=step_arithmetic(self.c,t)
            self.assertEqual(a['train_windows'],end-96-t['h']+1);self.assertEqual(a['train_batches'],a['train_windows']//128);self.assertEqual(a['max_optimizer_steps'],a['train_batches']*10)
            counts=s.formal_budget(self.c)['tasks'][t['id']];self.assertEqual(counts['forward'],10*(a['train_batches']+math.ceil((val-end-t['h']+1)/128))+math.ceil((test-val-t['h']+1)/128))
    def test_probe_new_caps_independent(self):
        b=s.probe_budget(self.c);self.assertEqual(b['nominal'],dict(adam=2016,backward=2016,forward=2688));self.assertEqual(b['caps'],dict(adam=3024,backward=3024,forward=4032));self.assertIn('resource-only',b['fallback'])
    def test_scheduler_formula_unchanged(self):
        from utils.ch3_type1_scaled import configuration,lr_used
        for t in self.c['tasks']:
            p=profile(self.c,t);self.assertEqual(p['training']['scheduler'],configuration(10,step_arithmetic(self.c,t)['train_batches']))
        self.assertEqual([lr_used(e,10)for e in (1,2,3,10)],[1e-4,1e-4,5e-5,3.90625e-7])
    def test_upstream_287_new_231_distinct(self):self.assertEqual(u.anchors()['expected_runs'],dict(MS=203,M=84,total=287));self.assertEqual(sum(len(c['tasks'])for c in self.cs.values()),231)
    def test_old_M84_cannot_seal_new_complete(self):
        value=dict(purpose='baseline_type1_M_ALL_boundary_v1',scope=s.ID,technical_complete=True,result_review='pending',task_ids=[t['id']for t in self.old['tasks']],protocol_sha=digest(self.c))
        with patch.object(q,'bound',return_value=value):
            with self.assertRaises(ValueError):q.validate_boundary_light(dict(path=str(s.context(self.c)['control']/'technical-boundary.json'),sha256='fixture'),'M_ALL')
    def test_old_147_template_cannot_authorize_231(self):
        a=json.loads((s.PACKAGE/'reviewed-147-candidate/start-approval.template.json').read_text());a.update(reviewed=True,execution_permitted=True,structure_frozen=True,m6_authorized=True,budget_authorized=True,closure_commit='fixture',authorization_basis='fixture authorized')
        with patch.object(q,'closure',return_value='fixture'):
            with self.assertRaises(PermissionError):q.validate_start(a)
    def test_source_refs_worker_allowlist(self):
        from utils.ch3_type1_execution import metadata_files
        values=metadata_files(self.c,dict(start_authorization_ref=ref(s.PACKAGE/'start-approval.template.json')))
        for value in (e.SOURCE_REF,e.PAPER_REF,e.TEMPLATE_REF):self.assertEqual(values[value['path']],value['sha256'])
        self.assertFalse(any(p.endswith('.bin')or '/probe/'in p for p in values))
    def test_new_output_and_auth_not_materialized(self):self.assertFalse(s.RESULT.exists());self.assertFalse((s.PACKAGE/'start-review.json').exists());self.assertTrue(all(q.start_template()[k]is False for k in ('reviewed','execution_permitted','budget_authorized')))
    def test_training_test_save_math_unchanged_AST(self):
        import subprocess
        old=ast.parse(subprocess.check_output(['git','show',s.BASE+':ch3_runner.py'],cwd=ROOT,text=True));new=ast.parse((ROOT/'ch3_runner.py').read_text())
        for name in ('formal_worker','evaluate','save_state','restore_state','audit_resume'):
            get=lambda t:ast.dump(next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name==name),include_attributes=False)
            self.assertEqual(get(old),get(new))
