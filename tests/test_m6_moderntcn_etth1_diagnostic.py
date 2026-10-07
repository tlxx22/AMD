"""No business models: exact short-diagnostic capability and candidate bounds."""
import copy,json,os,struct,tempfile,unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from utils import ch3_moderntcn_etth1_diagnostic as d
from utils.ch3_contract import profile,numeric_probe_policy,digest
from utils.ch3_native_recovery_records import ref
from tests.test_m6_probe_schema_recovery import trajectory
from utils.ch3_urban_confirmation import cached_endpoint

class Diagnostic(unittest.TestCase):
    def setUp(self):
        self.c=d.read_profile();self.t=d.tasks(self.c)[0]
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
    def pair(self,delta=0.):
        r=numeric_probe_policy(self.c,self.t)
        x=trajectory(self.c,self.t,self.root/'serial',r);y=trajectory(self.c,self.t,self.root/'q4',r)
        for side in (x,y):
            b=side['full_numeric_trace'];side['M_full_state_trace']=b
            snap=dict(rng=side['final_rng'],model=side['final'],optimizer=side['trajectory'][-1]['optimizer'])
            side['M_confirmation']=dict(evaluations={'2':dict(metrics=side['validation'],batch_ids=['full','tail']),
                '6':dict(metrics=copy.deepcopy(side['validation']),batch_ids=['full','tail'],before=snap,after=snap.copy(),mode_restored=True)})
        if delta:
            for row in y['full_numeric_trace']:
                p=Path(row['data_file']);p.write_bytes(struct.pack('<6d',1.+delta,1.,1.,1.,1.,float(row['step'])));row['data_sha']=ref(p)['sha256']
        return x,y
    def worker_fixture(self):
        stack=ExitStack();self.addCleanup(stack.close)
        for name,v in dict(OUTPUT=self.root/'out',PLAN=self.root/'plan.json').items():stack.enter_context(patch.object(d,name,v))
        d.OUTPUT.mkdir();(d.OUTPUT/'fixtures').mkdir();d.PLAN.write_text('{}')
        stack.enter_context(patch.dict(os.environ,{'TMPDIR':str(d.OUTPUT/'fixtures')}))
        ids=[t['id']for t in d.tasks(self.c)]
        v=dict(waves=[dict(phase='serial',tasks=[i])for i in ids]+[dict(phase='q4',tasks=ids)])
        stack.enter_context(patch.object(d,'plan',return_value=v))
        return d.make_config(self.c,self.t,'serial'),v
    def test_fixed_four_h_and_original_policy(self):
        self.assertEqual([t['h']for t in d.tasks(self.c)],[96,192,336,720])
        self.assertTrue(all(numeric_probe_policy(self.c,t)['state_atol']==1e-4 for t in d.tasks(self.c)))
    def test_foreign_model_dataset_or_M_identity(self):
        for model,dataset in [('TimeMixer','ETTh1'),('ModernTCN','Weather'),('ModernTCN','Exchange')]:
            t=next(t for t in self.c['tasks']if t['model']==model and t['dataset']==dataset)
            with self.assertRaises(ValueError):d.target_profile(self.c,t)
        with self.assertRaises(ValueError):d.target_profile(self.c,dict(self.t,task='MS'))
    def test_exact_eight_worker_caps(self):
        total={k:0 for k in d.CAPS}
        for phase in ('serial','q4'):
            for t in d.tasks(self.c):
                for k in total:total[k]+=d.limits(t,phase)[k]
        self.assertEqual(total,dict(adam=48,backward=56,forward=88))
    def test_actual_worker_guard_route_and_counting(self):
        import m5_formal_entry as entry
        s,_=self.worker_fixture()
        entry.validate_config(s)
        self.assertEqual(entry.execution_profiles(s),self.c)
        self.assertFalse(s['approval']);self.assertFalse(s['prefix_files'])
    def test_wrong_worker_path_and_capability_rejected(self):
        s,_=self.worker_fixture()
        for key,value in [('output',str(self.root/'escape')),('limits',dict(s['limits'],adam=7)),('device','cpu'),('resume',True),('approval',{'reviewed':True}),('kernel_probe',False),('protocol_sha','wrong')]:
            with self.subTest(key=key),self.assertRaises(ValueError):d.validate_worker(self.c,dict(s,**{key:value}))
    def test_actual_wave_scope_rejects_fallback_and_duplicate(self):
        s,_=self.worker_fixture();out=d.OUTPUT/'serial'/('wave-'+self.t['id'])
        self.assertEqual(d.validate_wave(self.c,[s],out),d.OUTPUT/'STOP')
        for configs,where in [([s,s],out),([dict(s,phase='q2')],out),([s],self.root/'outside')]:
            with self.assertRaises(ValueError):d.validate_wave(self.c,configs,where)
    def test_existing_output_prevents_repeat_claim(self):
        with patch.object(d,'OUTPUT',self.root),patch.object(d,'plan',return_value={}),patch.object(d.subprocess,'check_output',return_value=d.BASE+'\n'):
            with self.assertRaises(FileExistsError):d.run()
    def test_old_gate_rejects_predeclared_new_bound_accepts(self):
        x,y=self.pair(1.5e-4);r=d.candidate_comparison(self.c,self.t,x,y)
        self.assertFalse(r['original_comparison']['passed']);self.assertTrue(r['passed'])
        self.assertEqual(numeric_probe_policy(self.c,self.t)['state_atol'],1e-4)
    def test_candidate_over_2e4_rejected(self):
        x,y=self.pair(2.01e-4);self.assertFalse(d.candidate_comparison(self.c,self.t,x,y)['passed'])
    def test_loss_bound_unchanged(self):
        x,y=self.pair();y['trajectory'][0]['loss']+=2e-6
        self.assertFalse(d.candidate_comparison(self.c,self.t,x,y)['passed'])
    def test_step6_metrics_bound_unchanged(self):
        x,y=self.pair();v=y['M_confirmation']['evaluations']['6']['metrics'];v['channels'][0]['sse']+=v['channels'][0]['elements']*2e-5
        row=v['channels'][0];row['mse']=row['sse']/row['elements'];v['sse']=sum(z['sse']for z in v['channels']);v['mse']=v['sse']/v['elements']
        self.assertFalse(d.candidate_comparison(self.c,self.t,x,y)['passed'])
    def test_exact_scheduler_RNG_and_policy_rejected(self):
        for key,value in [('scheduler_trace',[]),('initial_rng','different'),('numeric_policy_sha','foreign')]:
            x,y=self.pair()
            y[key]=value
            with self.assertRaises(ValueError):d.candidate_comparison(self.c,self.t,x,y)
            # Each synthetic pair gets a fresh temporary directory.
            self.root=self.root/(key+'-next');self.root.mkdir()
    def test_nan_sidecar_rejected(self):
        x,y=self.pair();row=y['full_numeric_trace'][0];p=Path(row['data_file']);p.write_bytes(struct.pack('<6d',float('nan'),1.,1.,1.,1.,1.));row['data_sha']=ref(p)['sha256']
        with self.assertRaises(ValueError):d.candidate_comparison(self.c,self.t,x,y)
    def test_endpoint_change_or_extra_schema_rejected(self):
        x,y=self.pair();y['M_confirmation']['evaluations']['6']['after']={'rng':'different'}
        with self.assertRaises(ValueError):d.candidate_comparison(self.c,self.t,x,y)
        y['M_confirmation']['evaluations']['6']['after']=y['M_confirmation']['evaluations']['6']['before'];y['validation']['extra']=0
        with self.assertRaises(ValueError):d.candidate_comparison(self.c,self.t,x,y)
    def test_cached_endpoint_restores_modes_without_RNG_or_updates(self):
        class Mode:
            training=True
            def modules(self):return [self]
        m=Mode();b=[['full'],['tail']];calls=[]
        def evaluate(bs):calls.append(bs);m.training=False;return {'elements':2}
        r=cached_endpoint(m,b,evaluate,lambda:{'rng':'unchanged','model':'unchanged','optimizer':'unchanged'},digest)
        self.assertTrue(m.training);self.assertEqual(calls,[b]);self.assertEqual(r['before'],r['after'])
    def test_no_M_endpoint_in_ordinary_production_default(self):
        import inspect,ch3_runner
        self.assertFalse(inspect.signature(ch3_runner.probe_worker).parameters['m_confirmation'].default)
    def test_no_science_or_formal_permit_generation(self):
        self.assertEqual(d.CANDIDATE,dict(state_atol=2e-4,loss_atol=1e-6,metric_atol=1e-6,rtol=0,loss_rtol=0,equal_nan=False))
        self.assertEqual(d.CONFIG_SHA,'7f4720a723e31e3f5b291466c5b483021d480e0152c4317871f8d8097b4b14a5')

if __name__=='__main__':unittest.main()
