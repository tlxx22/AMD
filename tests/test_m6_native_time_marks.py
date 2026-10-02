"""Synthetic timestamp/adapter/contract checks. No model construction or training."""
import copy
import hashlib
import importlib.util
import json
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from utils.ch3_contract import read_profiles,profile,digest,step_arithmetic,validate_manifest
from utils import ch3_time_marks as marks
from utils import ch3_native_tasks as scope
from utils import ch3_m_tasks as m
from models.ch3_adapter import target_prediction,native_options


class NativeTimeMarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=validate_manifest(read_profiles(scope.PROFILE_FILE));cls.mc=validate_manifest(read_profiles())

    def test_three_locked_hourly_author_functions(self):
        times=pd.DatetimeIndex(['2022-01-01 00:00:00','2024-02-29 23:00:00','2026-12-31 15:00:00'])
        proof=m.bound(m.ref(scope.PACKAGE/'author-timefeatures.json'))
        for row in proof['files']:
            with self.subTest(model=row['model']):
                self.assertEqual(m.sha(row['path']),row['sha256'])
                spec=importlib.util.spec_from_file_location('locked_timefeatures_'+row['model'],row['path']);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                np.testing.assert_array_equal(marks.marks(times),module.time_features(times,freq='h').T.astype('float32'))
        self.assertEqual(marks.marks(times).shape,(3,4));self.assertEqual(marks.marks(times).dtype,np.float32)

    def test_historical_window_no_future_leak(self):
        from utils.ch3_data import Windows
        x=np.arange(30,dtype='float32').reshape(10,3);y=x[:,2:3]
        times=pd.date_range('2022-09-01',periods=10,freq='h');a=marks.marks(times)
        altered=list(times);altered[5:]=pd.date_range('2026-01-01',periods=5,freq='h');b=marks.marks(altered)
        first=Windows(x,y,0,10,4,2,marks=a)[0];second=Windows(x,y,0,10,4,2,marks=b)[0]
        np.testing.assert_array_equal(first[2].numpy(),second[2].numpy());np.testing.assert_array_equal(first[2].numpy(),a[:4])
        np.testing.assert_array_equal(first[0].numpy(),x[:4]);self.assertEqual(first[1].shape,(2,1))

    def test_native_MS_calls_and_TimeXer_channel_reorder(self):
        for model in scope.MODELS:
            t=next(t for t in self.c['tasks'] if t['model']==model and t['dataset']=='UrbanEV');p=profile(self.c,t);x=np.arange(2*p['T']*p['C']).reshape(2,p['T'],p['C']);mark=np.zeros((2,p['T'],4),dtype='float32');seen=[]
            def spy(*args):
                seen.append(args)
                return np.zeros((2,p['pred_len'],1 if model=='TimeXer' else p['C']))
            pred,_=target_prediction(spy,x,p,mark)
            self.assertIs(seen[0][1],mark);self.assertEqual(seen[0][2:],(None,None));self.assertEqual(pred.shape,(2,1,1))
            np.testing.assert_array_equal(seen[0][0],x[:,:,p['aux_idx']+[p['target_idx']]] if model=='TimeXer' else x)

    def test_M_full_output_original_column_order(self):
        for model in scope.MODELS:
            t=next(t for t in self.mc['tasks'] if t['model']==model);p=profile(self.mc,t);x=np.zeros((2,p['T'],p['C']));mark=np.zeros((2,p['T'],4));seen=[]
            def spy(*args):seen.append(args);return np.zeros((2,p['pred_len'],p['C']))
            y,_=target_prediction(spy,x,p,mark);self.assertEqual(y.shape,(2,p['pred_len'],p['C']));self.assertIs(seen[0][0],x);self.assertIs(seen[0][1],mark)
            self.assertEqual(native_options(p)['features'],'M');self.assertEqual(native_options(p)['enc_in'],p['C'])

    def test_other_model_calls_and_marks_rejection(self):
        for model in ('AMD','DLinear','PatchTST','ModernTCN'):
            t=next(t for t in self.mc['tasks'] if t['model']==model);p=profile(self.mc,t);x=np.zeros((2,p['T'],p['C']));seen=[]
            def spy(*args):seen.append(args);return np.zeros((2,p['pred_len'],p['C']))
            target_prediction(spy,x,p);self.assertEqual(len(seen[0]),1);self.assertNotIn('time_mark',p)
            with self.assertRaises(ValueError):target_prediction(spy,x,p,np.zeros((2,p['T'],4)))

    def test_missing_wrong_mark_scope_shape_rejected(self):
        t=self.c['tasks'][0];p=profile(self.c,t);x=np.zeros((2,p['T'],p['C']))
        def never(*args):raise AssertionError('must reject before native call')
        for value in (None,np.zeros((2,p['T']+1,4)),np.zeros((2,p['T'],5))):
            with self.assertRaises(ValueError):target_prediction(never,x,p,value)
        with self.assertRaises(ValueError):marks.marks(pd.date_range('2022',periods=2),freq='d')

    def test_exact_87_budget_and_parent_profile_inheritance(self):
        self.assertEqual(len(self.c['tasks']),87);self.assertEqual(sum(t['dataset']=='UrbanEV' for t in self.c['tasks']),72)
        self.assertEqual(sum(profile(self.c,t)['training']['epochs'] for t in self.c['tasks']),1020)
        self.assertEqual(sum(step_arithmetic(self.c,t)['max_optimizer_steps'] for t in self.c['tasks']),3687530)
        for t in self.c['tasks']:
            p=profile(self.c,t);base={k:v for k,v in p.items() if k not in ('time_mark','parent_run_id','parent_profile_sha','revision')}
            self.assertEqual(base,self.c['native_replacement']['parents'][t['id']]['profile']);self.assertNotEqual(t['id'],t['parent_run_id'])
            self.assertEqual(p['structure'].get('use_future_temporal_feature',0),0)

    def test_replacement_order_inherits_existing_parent_per_model(self):
        parent=scope.parent_config()['tasks'];expected=[]
        for model in scope.MODELS:
            expected.extend(t['id'] for t in parent if t['model']==model and t['dataset'] in scope.DOMAINS and (t['dataset']!='UrbanEV' or t['input_variant']=='F4'))
        self.assertEqual([t['parent_run_id'] for t in self.c['tasks']],expected)
        mixer=[t for t in self.c['tasks'] if t['model']=='TimeMixer']
        self.assertEqual(list(dict.fromkeys(t['dataset'] for t in mixer)),['NP','BE','FR','DE','PJM','UrbanEV'])

    def test_Urban_F4_business_columns_and_fold_alignment(self):
        from utils.ch3_data import Windows
        t=self.c['tasks'][0];p=profile(self.c,t)
        self.assertEqual(p['C'],11)
        for name in ('hour_sin','hour_cos','weekday_sin','weekday_cos','is_weekend'):self.assertIn(name,p['features'])
        x=np.zeros((8,2,11),dtype='float32');y=np.zeros((8,2),dtype='float32');a=marks.marks(pd.date_range('2022-09-01',periods=8,freq='h'));ds=Windows(x,y,0,8,3,1,urban=True,label_horizon=2,marks=a)
        np.testing.assert_array_equal(ds[2][2].numpy(),a[1:4]);self.assertEqual(ds[2][0].shape,(3,11))

    def test_M_science_and_freq_still_frozen(self):
        import subprocess
        before=json.loads(subprocess.check_output(['git','show',scope.BASE+':configs/ch3_formal_profiles.json']))
        self.assertEqual(len(self.mc['tasks']),84);self.assertEqual(sum(step_arithmetic(self.mc,t)['max_optimizer_steps'] for t in self.mc['tasks']),294790)
        self.assertEqual(digest(before),digest(self.mc))
        for t in self.mc['tasks']:
            old=m.resolved(before,t);new=profile(self.mc,t)
            self.assertEqual(new,old)
            if new.get('time_mark'):self.assertEqual(new['time_mark']['freq'],old['structure']['freq'])

    def test_computational_identity_minimal_coverage_and_caps(self):
        gs=scope.probe_groups(self.c);self.assertEqual(len(gs),7);self.assertEqual(sum(len(g['representatives']) for g in gs),25)
        covered=[r for g in gs for ids in g['coverage'].values() for r in ids];self.assertEqual(len(covered),87);self.assertEqual(set(covered),{t['id'] for t in self.c['tasks']})
        for g in gs:
            for rep,ids in g['coverage'].items():
                identity=scope.computational_identity(self.c,next(t for t in self.c['tasks'] if t['id']==rep))
                for r in ids:self.assertEqual(identity,scope.computational_identity(self.c,next(t for t in self.c['tasks'] if t['id']==r)))
        self.assertEqual(scope.plan(self.c)['nominal_workers'],50);self.assertEqual(scope.context(self.c)['caps'],dict(adam=438,forward=608,backward=438))
        self.assertEqual(scope.plan(self.mc)['max_workers'],252);self.assertEqual(scope.context(self.mc)['caps'],m.CAPS)

    def test_fixed_result_map_no_performance_choice_or_legacy_write(self):
        mapping=scope.effective_results(self.c)
        self.assertEqual(len(mapping['replacement']),87)
        for t in self.c['tasks']:
            path=scope.result_path(self.c,t);self.assertTrue(path.is_relative_to(scope.REPLACEMENT));self.assertNotEqual(path.name,t['parent_run_id']);self.assertEqual(mapping['replacement'][t['parent_run_id']]['run_id'],t['id'])

    def test_data_derivations_only_add_native_metadata(self):
        data=m.bound(self.c['native_time_mark']['data_binding_artifact'])
        for t in self.c['tasks']:
            value=copy.deepcopy(data['metadata'][t['dataset']][t['id']]);row=next(r for r in data['derivations'] if r['run']==t['id'])
            for key in ('time_mark','time_mark_shape'):value.pop(key)
            self.assertEqual(digest(value),row['parent_data_sha']);self.assertEqual(data['data_bindings'][t['dataset']][t['id']],row['new_data_sha'])
