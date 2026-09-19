"""M6 preparation exact-ID CPU checks. No model/GPU or observation fixtures."""
import copy,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from collections import Counter
from utils.ch3_contract import read_profiles,profile,digest,task_by_id
from utils.ch3_m6 import metadata_for_horizon,verify_result,approval_reasons,run_formal_waves,source_states

class M6PreparationTests(unittest.TestCase):
    def setUp(self):
        self.c=read_profiles();self.root=Path(tempfile.mkdtemp(prefix='m6-check-',dir=self.c['execution']['fixture']))
    def result(self,run):
        from utils.ch3_contract import step_arithmetic
        t=task_by_id(self.c,run);n=step_arithmetic(self.c,t)['test_windows_arithmetic_only']*profile(self.c,t)['pred_len']
        return dict(id=run,purpose='ch3_formal',protocol_sha=digest(self.c),input_variant=t['input_variant'],seed=2024,
                    metric_space='train-standardized',best_epoch=1,mse=1.,mae=.5,sse=float(n),sae=.5*n,elements=n)
    def test_freeze_and_stage(self):
        self.assertEqual(self.c['structure_freeze']['id'],'el-amd-s2-thls-v1');self.assertEqual(self.c['structure_freeze']['variant'],'J');self.assertEqual(self.c['execution']['phase'],'M6')
    def test_all_scientific_profiles_unchanged(self):
        old=json.loads((Path(self.c['execution']['evidence'])/'before/configs/ch3_formal_profiles.json').read_text())
        self.assertEqual(self.c['tasks'],old['tasks']);self.assertEqual(self.c['groups'],old['groups'])
        for t in self.c['tasks']:
            with self.subTest(run=t['id']):self.assertEqual(profile(self.c,t),profile(old,t))
    def test_model_task_counts(self):
        self.assertEqual(dict(Counter(t['model'] for t in self.c['tasks'])),dict(AMD=113,J=113,DLinear=41,PatchTST=41,iTransformer=41,TimeMixer=16,ModernTCN=41,TimeXer=41,N=24,S=24))
    def test_wave_coverage_and_q(self):
        from ch3_runner import verified_waves
        for g in self.c['groups']:
            q=min(2,g['q']);d=dict(concurrency=q,representatives=g['representatives'],status='Passed')
            waves=verified_waves(g,d)
            self.assertEqual([r for w in waves for r in w],[r for w in g['waves'] for r in w]);self.assertTrue(all(len(w)<=q for w in waves))
    def test_no_cross_model_or_duplicate_tasks(self):
        ids=[]
        for g in self.c['groups']:
            for wave in g['waves']:
                for run in wave:self.assertEqual(task_by_id(self.c,run)['model'],g['model']);ids.append(run)
        self.assertEqual(len(ids),495);self.assertEqual(len(set(ids)),495)
    def test_one_coordinator_per_wave(self):
        runs=[t['id'] for t in self.c['tasks'] if t['model']=='J'][:4];calls=[]
        def make(c,purpose,out,**kw):
            self.assertEqual(purpose,'ch3_formal');out.mkdir(parents=True);return dict(output=str(out),task=kw['task'])
        def execute(configs,out,monitor):
            calls.append(configs);self.assertTrue(monitor)
            for cfg in configs:(Path(cfg['output'])/'result.json').write_text(json.dumps(self.result(cfg['task'])))
            return dict(failure=None,returncodes=[0]*len(configs),resource_admission=True)
        with patch('m5_formal_entry.make_config',side_effect=make),patch('m5_formal_entry.run_configs',side_effect=execute):run_formal_waves(self.c,[runs],self.root,self.root,{},None)
        self.assertEqual(len(calls),1);self.assertEqual(len(calls[0]),4)
    def test_wave_failure_stops_next(self):
        runs=[t['id'] for t in self.c['tasks'] if t['model']=='AMD'][:2]
        with patch('m5_formal_entry.make_config',return_value={}),patch('m5_formal_entry.run_configs',return_value=dict(failure='failed',returncodes=[1],resource_admission=False)) as execute:
            with self.assertRaises(RuntimeError):run_formal_waves(self.c,[[runs[0]],[runs[1]]],self.root,self.root,{})
            self.assertEqual(execute.call_count,1)
    def test_stop_before_launch(self):
        (self.root/'STOP').write_text('stop')
        with patch('m5_formal_entry.make_config') as make:
            with self.assertRaises(InterruptedError):run_formal_waves(self.c,[['unused']],self.root,self.root,{})
            make.assert_not_called()
    def test_horizon_metadata_generic(self):
        import numpy as np
        from utils.ch3_data import Windows
        t=next(t for t in self.c['tasks'] if t['model']=='AMD' and t['dataset']=='ETTh1' and t['h']==192)
        x=np.zeros((2000,7),dtype='float32');y=np.zeros((2000,1),dtype='float32')
        w={'train':Windows(x,y,0,1000,512,96),'validation':Windows(x,y,488,1500,512,96)}
        base={'window_counts':{k:len(v) for k,v in w.items()},'target':'OT'}
        result=metadata_for_horizon(self.c,t,base,w);self.assertEqual(result['window_counts'],dict(train=297,validation=309));self.assertEqual(base['window_counts']['train'],393)
    def test_horizon_metadata_urban(self):
        import numpy as np
        from utils.ch3_data import Windows
        t=next(t for t in self.c['tasks'] if t['model']=='J' and t['dataset']=='UrbanEV' and t['h']==12)
        x=np.zeros((100,2,11),dtype='float32');y=np.zeros((100,2),dtype='float32')
        w={'train':Windows(x,y,0,60,12,1,urban=True,label_horizon=3),'validation':Windows(x,y,60,100,12,1,urban=True,label_horizon=3)}
        v=metadata_for_horizon(self.c,t,{'window_counts':{}},w);self.assertEqual(v['window_counts'],dict(train=74,validation=34))
    def test_result_identity_rejected(self):
        run=self.c['tasks'][0]['id'];r=self.result(run);verify_result(self.c,run,r);r['protocol_sha']='old'
        with self.assertRaises(ValueError):verify_result(self.c,run,r)
        r=self.result(run);r['elements']-=1
        with self.assertRaises(ValueError):verify_result(self.c,run,r)
    def test_nonfinite_result_rejected(self):
        run=self.c['tasks'][0]['id'];r=self.result(run);r['mse']=float('nan')
        with self.assertRaises(ValueError):verify_result(self.c,run,r)
    def test_approval_requires_every_run_binding(self):
        with patch('utils.ch3_m6.source_states',return_value={}):
            reasons=approval_reasons(self.c,'AMD',dict(freeze_id='el-amd-s2-thls-v1',models=['AMD'],source_states={},data_bindings={}))
        self.assertEqual(sum(x.startswith('missing task data binding') for x in reasons),113)
    def test_binding_capability_only_validation(self):
        import m5_formal_entry as entry
        c=copy.deepcopy(self.c);c['execution']['evidence']=str(self.root)
        cfg=entry.make_config(c,'ch3_formal_bindings',self.root/'binding')
        for name,d in c['datasets'].items():
            if name!='UrbanEV':self.assertEqual(cfg['prefix_files'][d['path']],d['endpoints'][1])
        self.assertEqual(tuple(cfg['limits'][x] for x in ('adam','forward','backward')),(0,0,0));self.assertEqual(cfg['device'],'cpu')
    def test_formal_monitor_streams(self):
        import m5_formal_entry as entry
        class P:
            pid=987654;returncode=0
            def __init__(self):self.n=0
            def poll(self):self.n+=1;return 0 if self.n>9 else None
            def wait(self):return 0
            def terminate(self):self.returncode=-15
        p=P();tick=[0]
        def sample(pids):
            tick[0]+=1;return dict(time=float(tick[0]),uuid='gpu',used=100,free=1000,process_gpu={str(p.pid):50},cpu_rss={str(p.pid):80})
        class X:
            pending=[]
            def __init__(self,*a):pass
            def classify(self,*a):return []
        cfg=dict(purpose='ch3_formal',artifact_root=str(self.root/'run'),output=str(self.root),audit_log=str(self.root/'audit.jsonl'),limits={'seconds':None})
        good=dict(admission=True,unknown_pids=[],card_reliable=True,reserve=1,external_occupancy_known=True)
        with patch.object(entry,'spawn',return_value=(p,io.StringIO())),patch.object(entry,'gpu_sample',side_effect=sample),patch.object(entry,'ExitObservation',X),patch.object(entry,'resource_assessment',return_value=good),patch.object(entry.time,'sleep'):
            r=entry.run_configs([cfg],self.root/'process',monitor=True)
        self.assertTrue(r['resource_admission']);self.assertEqual(r['process_peaks'][str(p.pid)],50);self.assertGreater(r['sample_count'],0)
    def test_formal_failfast_owned_workers(self):
        import m5_formal_entry as entry
        class P:
            pid=987655;returncode=1
            def poll(self):return 1
            def wait(self):return 1
            def terminate(self):raise AssertionError('already terminated')
        # Two workers: one failed and one still alive. Only the owned live worker is stopped.
        class Q:
            pid=987656;returncode=None
            def poll(self):return self.returncode
            def wait(self):return self.returncode
            def terminate(self):self.returncode=-15
        q=Q();(self.root/'worker.log').write_text('synthetic failure')
        cfg=dict(purpose='ch3_formal',artifact_root=str(self.root/'run'),output=str(self.root),audit_log=str(self.root/'audit.jsonl'),limits={'seconds':None})
        with patch.object(entry,'spawn',side_effect=[(P(),io.StringIO()),(q,io.StringIO())]):r=entry.run_configs([cfg,cfg],self.root/'process',monitor=False)
        self.assertEqual(q.returncode,-15);self.assertIn('formal worker failed',r['failure'])
