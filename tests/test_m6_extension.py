"""Exact-case, synthetic-only M6 EPF extension regression; no old checkpoint/test."""
import copy,csv,datetime,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from utils.ch3_contract import read_profiles,profile,digest,step_arithmetic
from utils.ch3_extension import *

class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.c=read_profiles();self.tmp=tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']);self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
    def selected(self,model='AMD',domain='NP'):
        return next(t for t in self.c['tasks']if(t['model'],t['dataset'])==(model,domain))
    def fixture(self):
        c=copy.deepcopy(self.c);c['datasets']['NP']['endpoints']=[700,800,1000];c['datasets']['NP']['declared_rows']=1000
        p=self.root/'synthetic.csv'
        with p.open('w',newline='')as f:
            w=csv.writer(f);w.writerow(['date']+c['datasets']['NP']['features'])
            for i in range(1000):w.writerow([str(datetime.datetime(2020,1,1)+datetime.timedelta(hours=i))]+([i,3*i,2*i+5]if i<800 else ['TEST_FORBIDDEN']*3))
        return c,p
    def test_counts_and_epoch_caps(self):
        validate_extension(self.c);self.assertEqual(len(self.c['tasks']),552);self.assertEqual(sum(profile(self.c,t)['training']['epochs']for t in self.c['tasks']),6240)
    def test_exact_57_partition(self):
        x=self.c['extension'];self.assertEqual(len(x['new_ids']),57);self.assertEqual(len(x['catchup_ids']),41);self.assertEqual(len(set(sum(x['append_ids'].values(),[]))),16);self.assertFalse(set(x['catchup_ids'])&set(sum(x['append_ids'].values(),[])))
    def test_original_profiles_bound(self):
        for t in self.c['tasks'][:495]:
            with self.subTest(run=t['id']):
                value=profile(self.c,t)
                if t['id'] in self.c.get('timemixer_revision',{}).get('task_ids',[]) and t['dataset']!='Exchange':
                    self.assertEqual(value['training']['lr'],.001)
                    value['training']['lr']=.01
                self.assertEqual(digest(value),self.c['extension']['original_profile_hashes'][t['id']])
    def test_prohibited_scope(self):
        for t in self.c['tasks']:
            with self.subTest(run=t['id']):
                self.assertNotIn(t['input_variant'],('F0','TargetOnly'));self.assertNotEqual(t['model'],'TiDE')
                if t['model']in('N','S'):self.assertEqual((t['dataset'],t['input_variant']),('UrbanEV','F4'))
    def test_queue_order_and_append(self):
        ids=queue_ids(self.c);self.assertEqual(ids[:4],[task('AMD',m,1,24)['id']for m in MARKETS]);self.assertEqual(ids[12:17],[task('TimeMixer',m,1,24)['id']for m in MARKETS+('PJM',)])
        for m in APPEND:
            with self.subTest(model=m):self.assertEqual(queue_ids(self.c,m),[t['id']for t in self.c['tasks'][:495]if t['model']==m]+[task(m,d,1,24)['id']for d in MARKETS])
    def test_fixed_waves_and_epf_single_fit(self):
        ds={g['id']:dict(status='Passed',concurrency=g['q'],representatives=g['representatives'])for g in self.c['groups']}
        waves=fixed_waves(self.c,queue_ids(self.c),ds);self.assertEqual(sum(map(len,waves)),41);self.assertEqual(len(waves),23)
        g='AMD-NP-MS';ds[g]['concurrency']=4
        with self.assertRaises(ValueError):fixed_waves(self.c,queue_ids(self.c),ds)
    def test_prefix_scaler_and_target(self):
        from utils.ch3_data import load
        c,p=self.fixture();ds,m=load(c,self.selected(),path_override=p)
        self.assertEqual(m['parsed_records'],800);self.assertEqual(m['scaler_fit_records'],700);self.assertFalse(m['test_observations_accessed']);self.assertEqual(set(ds),{'train','validation'});self.assertAlmostEqual(m['scaler']['mean'][2],704.)
        x,y=ds['validation'][0];self.assertEqual(tuple(x.shape),(168,3));self.assertEqual(tuple(y.shape),(24,1));self.assertTrue((y.numpy()==ds['validation'].y[700:724]).all())
    def test_windows_tail_arithmetic(self):
        a=step_arithmetic(self.c,self.selected());self.assertEqual((a['train_windows'],a['train_batches'],a['train_dropped']),(36500,285,20));self.assertEqual((a['validation_windows'],a['validation_tail']),(5219,99))
        t=self.selected('TimeXer');a=step_arithmetic(self.c,t);self.assertEqual(a['train_batches'],9125);self.assertEqual(a['validation_tail'],3)
    def test_time_reversal_rejected(self):
        import pandas as pd
        from utils.ch3_data import validate_times
        with self.assertRaises(ValueError):validate_times(pd.DatetimeIndex(['2020-01-02','2020-01-01']),'NP',self.c['datasets']['NP'])
    def test_no_unauthorized_training_changes(self):
        a=profile(self.c,self.selected('AMD'));j=profile(self.c,self.selected('J'));self.assertEqual(a['training'],j['training']);self.assertEqual(a['structure'],j['structure'])
        for m in PRIMARY:
            with self.subTest(model=m):
                p=profile(self.c,self.selected(m));self.assertEqual((p['T'],p['pred_len'],p['training']['epochs'],p['training']['patience']),(168,24,20,5))
    def test_index_preserves_paths_and_unreviewed(self):
        rows=joined_index(self.c);self.assertEqual(len(rows),552);self.assertTrue(all(r['status']=='unverified'for r in rows));self.assertEqual(sum(r['batch']=='extension'for r in rows),57)
        old=rows[0];new=next(r for r in rows if r['batch']=='extension');self.assertNotIn('/supplements/',old['path']);self.assertIn('/supplements/epf4-timemixer-v1/',new['path']);self.assertTrue(all('mse'not in v for v in summarized_index(rows)))
    def test_index_duplicate_old_protocol_rejected(self):
        t=self.selected();r=dict(id=t['id'],batch='extension',profile_sha=digest(profile(self.c,t)),status='success',reviewed=True,commit='synthetic',protocol_sha=self.c['extension']['parent_protocol_sha'],data_sha='synthetic',result_sha256='synthetic',mse=1.,mae=.5)
        with self.assertRaises(ValueError):joined_index(self.c,[r])
        r['protocol_sha']=digest(self.c)
        with self.assertRaises(ValueError):joined_index(self.c,[r,r])
        r['reviewed']=False
        with self.assertRaises(ValueError):joined_index(self.c,[r])
    def test_failure_stops_next_wave(self):
        calls=[]
        def failure(*args):calls.append(args);raise RuntimeError('synthetic worker failure')
        with self.assertRaises(RuntimeError):execute_waves(self.c,[[task('AMD','NP',1,24)['id']],[task('AMD','BE',1,24)['id']]],self.root,{},failure)
        self.assertEqual(len(calls),1)
    def test_duplicate_start_and_stop_before_next(self):
        with self.assertRaises(FileExistsError):reject_existing(self.c,queue_ids(self.c),self.root)
        (self.root/'STOP').touch();calls=[]
        with self.assertRaises(InterruptedError):execute_waves(self.c,[[task('AMD','NP',1,24)['id']]],self.root,{},lambda *a:calls.append(a))
        self.assertFalse(calls)
    def test_safe_stop_pid_identity(self):
        (self.root/'controller.json').write_text(json.dumps(dict(pid=99999999,start_ticks='1')))
        with patch('utils.ch3_extension.controller_live',return_value=False),patch('utils.ch3_extension.os.kill')as k:
            with self.assertRaises(RuntimeError):safe_stop(self.root)
            k.assert_not_called()
        with patch('utils.ch3_extension.controller_live',return_value=True),patch('utils.ch3_extension.os.kill')as k:safe_stop(self.root);k.assert_called_once_with(99999999,signal.SIGTERM)
    def test_absent_legacy_approval_rejected(self):
        self.assertTrue(extension_reasons(self.c,None));self.assertTrue(extension_reasons(self.c,dict(purpose='ch3_formal',protocol_sha=self.c['extension']['parent_protocol_sha'])))

class ExtensionGuardTests(unittest.TestCase):
    def test_resource_rules_retained(self):
        from utils.ch3_contract import validate_manifest
        c=read_profiles();validate_manifest(c)
        c['execution']['probe']['rss_policy']={}
        with self.assertRaises(ValueError):validate_manifest(c)
    def test_receipt_cannot_override_identity(self):
        c=read_profiles();t=c['tasks'][495]
        r=dict(id=t['id'],batch='extension',profile_sha=digest(profile(c,t)),status='unverified')
        for key,value in [('path','/foreign'),('model','J'),('dataset','ECL'),('h',96),('fold',2),('input_variant','F0')]:
            with self.subTest(field=key):
                with self.assertRaises(ValueError):joined_index(c,[dict(r,**{key:value})])
    def test_active_parent_blocks_absent_approval(self):
        c=read_profiles()
        with patch('utils.ch3_extension.Path.glob',return_value=[Path('/synthetic/formal-TimeMixer/controller.json')]),patch('utils.ch3_extension.controller_live',return_value=True):
            self.assertIn('original formal group live: formal-TimeMixer',extension_reasons(c,None))

class TimeMixerCPUShapeTests(unittest.TestCase):
    def test_urban_all_four_labels(self):
        import torch
        from models.ch3_adapter import build,target_prediction,native_options
        c=read_profiles();tasks=[t for t in c['tasks'][495:]if t['dataset']=='UrbanEV'and t['fold']==1];model=build(c,tasks[0]);model.eval();x=torch.zeros(2,12,11)
        self.assertEqual(native_options(profile(c,tasks[0]))['c_out'],11)
        with torch.no_grad():
            for t in tasks:
                with self.subTest(h=t['h']):
                    p=profile(c,t);out,_=target_prediction(model,x,p);self.assertEqual(tuple(out.shape),(2,1,1));self.assertTrue(torch.isfinite(out).all());self.assertEqual(p['structure']['channel_independence'],0)
        self.assertEqual([12//2**i for i in range(3)],[12,6,3])
    def test_epf_ci0_all_market_binding(self):
        import torch
        from models.ch3_adapter import build,target_prediction,native_options
        c=read_profiles();tasks=[t for t in c['tasks'][495:]if t['model']=='TimeMixer'and t['dataset']in MARKETS+('PJM',)];model=build(c,tasks[0]);model.eval()
        for t in tasks:
            with self.subTest(market=t['dataset']):
                p=profile(c,t);self.assertEqual(p['structure'],profile(c,tasks[0])['structure']);self.assertEqual((native_options(p)['enc_in'],native_options(p)['dec_in'],native_options(p)['c_out']),(3,3,3))
        with torch.no_grad():out,_=target_prediction(model,torch.zeros(2,168,3),profile(c,tasks[0]))
        self.assertEqual(tuple(out.shape),(2,24,1));self.assertTrue(torch.isfinite(out).all());self.assertEqual([168//2**i for i in range(4)],[168,84,42,21])
