"""Exact no-model tests. Fixtures never construct a model, optimizer or torch tensor."""
import ast,copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from utils import ch3_urban_diagnostic as scope
from utils import ch3_urban_capture as capture
from utils.ch3_contract import read_profiles,digest
from utils.ch3_extension_probe import Budget

class UrbanDiagnosticTests(unittest.TestCase):
 def setUp(self):
  self.c=read_profiles();self.spec=scope.specification(self.c)
  self.fixture=Path(tempfile.mkdtemp(prefix='case-',dir=scope.PACKAGE_ROOT/'fixture'))
 def test_scope_counts(self):
  self.assertEqual(len(self.spec['task_ids']),4)
  workers=sum(len(w['task_ids'])for w in self.spec['waves']);self.assertEqual(workers,9)
  self.assertEqual({k:workers*v for k,v in scope.WORKER.items()},scope.CAPS)
  self.assertEqual(self.spec['waves'][4],dict(stage='repeat-h3',task_ids=[self.spec['task_ids'][0]]))
 def test_exact_paths(self):
  for w in self.spec['waves']:
   for run in w['task_ids']:
    with self.subTest(stage=w['stage'],run=run):self.assertEqual(scope.worker_path(self.c,run,scope.ROOT/w['stage']/run),w['stage'])
  for stage,run in [('repeat-h3',self.spec['task_ids'][1]),('q2',self.spec['task_ids'][0]),('serial','TimeMixer-PJM-MS-f1-h24-s2024')]:
   with self.subTest(stage=stage,run=run),self.assertRaises(ValueError):scope.worker_path(self.c,run,scope.ROOT/stage/run)
 def test_template_cross_scope_and_budget(self):
  a=json.loads((scope.PACKAGE_ROOT/'approval.template.json').read_text())
  self.assertTrue(scope.authorization_reasons(self.c,a))
  for other in ['extension-probe-v1','timemixer-revision-numeric-v1']:
   with self.subTest(scope=other):
    bad=dict(a,probe_scope=other);self.assertIn('foreign purpose/scope',scope.authorization_reasons(self.c,bad))
  bad=dict(a,reviewed=True,execution_permitted=True,budget_authorized=False)
  self.assertIn('template/fixture/unapproved diagnostic budget',scope.authorization_reasons(self.c,bad))
 def test_plan_tampering(self):
  original=scope.read(scope.PLAN)
  for key in ('caps','profile_shas','waves'):
   with self.subTest(field=key):
    bad=copy.deepcopy(original);bad[key]={}
    with patch.object(scope,'read',return_value=bad),self.assertRaises(ValueError):scope.specification(self.c)
 def test_existing_output_rejected(self):
  with patch.object(scope,'ROOT',self.fixture),patch.object(scope,'authorization_reasons',return_value=[]):
   self.assertIn('retained diagnostic output; no restart',scope.readiness(self.c,{}))
 def test_fixed_order_stop_and_failure(self):
  called=[]
  def launch(runs,directory):
   called.append((directory.name,runs))
   return dict(failure='synthetic failure',returncodes=[1],resource_admission=False),[]
  with self.assertRaises(RuntimeError):scope.run_waves(self.c,launch,lambda:False)
  self.assertEqual(len(called),1)
  with self.assertRaises(InterruptedError):scope.run_waves(self.c,launch,lambda:True)
  self.assertEqual(len(called),1)
 def test_success_orchestration_no_admission(self):
  calls=[]
  def launch(runs,directory):
   calls.append(dict(stage=directory.name,task_ids=runs))
   return dict(failure=None,returncodes=[0]*len(runs),resource_admission=True),[
    dict(id=r,profile_sha=self.spec['profile_shas'][r],steps=6,finite=True,memory_review=dict(blocked=False,needs_long_window=False),urban_diagnostic_trace=[dict(step=i)for i in range(7)])for r in runs]
  traces,measure=scope.run_waves(self.c,launch,lambda:False)
  self.assertEqual(calls,self.spec['waves']);self.assertEqual(len(traces),9);self.assertEqual(len(measure),6)
 def test_budget_no_refund(self):
  b=Budget(scope.CAPS,scope.ZERO);configs=[dict(output='synthetic',limits=scope.WORKER)]
  b.reserve(configs);self.assertEqual(b.counts,scope.WORKER)
  for i in range(8):b.reserve(configs)
  before=copy.deepcopy(b.counts)
  with self.assertRaises(RuntimeError):b.reserve(configs)
  self.assertEqual(before,b.counts);self.assertEqual(b.counts,scope.CAPS)
 def test_worker_no_prefix_and_exact_id(self):
  run=self.spec['task_ids'][0];out=str(scope.ROOT/'serial'/run)
  s=dict(probe_scope=scope.SCOPE,task=run,output=out,session_root=str(scope.ROOT),fixture_root=str(scope.ROOT/'fixture'),prefix_files={},resume=False,artifact_root=out,limits=dict(scope.WORKER,seconds=1800),ids=[run],approval={})
  with patch.object(scope,'authorization_reasons',return_value=[]):
   scope.validate_worker(self.c,s)
   for key,value in [('prefix_files',{'forbidden.csv':1}),('ids',['other']),('probe_scope','extension-probe-v1'),('resume',True)]:
    with self.subTest(field=key),self.assertRaises(ValueError):scope.validate_worker(self.c,dict(s,**{key:value}))
 def test_wave_q4_no_q2(self):
  runs=self.spec['task_ids'];configs=[dict(task=r,output=str(scope.ROOT/'q4'/r))for r in runs]
  with patch.object(scope,'validate_worker'):
   self.assertEqual(scope.validate_wave(self.c,configs,scope.ROOT/'q4'/('monitor-'+runs[0])),scope.ROOT/'STOP')
   with self.assertRaises(ValueError):scope.validate_wave(self.c,configs[:2],scope.ROOT/'q4'/('monitor-'+runs[0]))
 def test_capture_no_compute_or_rng(self):
  import ch3_runner
  calls=[]
  class PlainState:
   def named_parameters(self):return []
  class Writer:
   def __init__(self,*args):pass
   def capture(self,step):calls.append(step);return dict(step=step)
  state=PlainState()
  with patch.object(ch3_runner,'FullNumericStateWriter',Writer),patch.object(capture,'rng_value',return_value={'synthetic_rng':[1,2]}):
   obj=capture.DiagnosticCapture(state,object(),self.fixture,None,None)
   for step in range(7):obj.capture(step)
   self.assertEqual(calls,list(range(7)));self.assertEqual(len(obj.rows),7)
  tree=ast.parse(Path(capture.__file__).read_text())
  forbidden={'forward','backward','step','zero_grad','rand','randn','manual_seed'}
  self.assertFalse([n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr in forbidden])
 def test_tensor_metrics_float_nearzero(self):
  a=np.array([0.,1e-14,2.],dtype=np.float32);b=np.array([1e-15,2e-14,2.0001],dtype=np.float32)
  m=capture.tensor_metrics(a,b)
  self.assertEqual(m['different_elements'],3);self.assertEqual(m['near_zero_elements'],2);self.assertEqual(m['near_zero_changed'],2)
  self.assertAlmostEqual(m['max_relative_diff'],float(abs(float(b[2])-2)/float(b[2])))
  z=capture.tensor_metrics(a[:2],b[:2]);self.assertIsNone(z['max_relative_diff'])
 def test_nonfloat_and_nonfinite(self):
  a=np.array([2**60],dtype=np.int64);b=np.array([2**60+1],dtype=np.int64)
  m=capture.tensor_metrics(a,b,exact=True);self.assertEqual(m['max_abs_diff'],1);self.assertFalse(m['exact_equal'])
  self.assertFalse(capture.tensor_metrics(np.array([np.nan]),np.array([0.]))['finite'])
  with self.assertRaises(ValueError):capture.tensor_metrics(np.zeros(2),np.zeros(3))
 def test_worker_hook_and_no_policy_change(self):
  root=Path(__file__).parents[1]
  tree=ast.parse((root/'ch3_runner.py').read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='probe_worker')
  self.assertEqual(sum(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='update'for n in ast.walk(f)),1)
  self.assertEqual(sum(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='evaluate'for n in ast.walk(f)),1)
  src=(root/'tools/restricted_regression/m5_formal_entry.py').read_text()
  self.assertIn("urban_diagnostic=s.get('probe_scope')=='urban-numeric-diagnostic-v1'",src)
  self.assertNotIn('numeric_probe_policy',Path(scope.__file__).read_text())
 def test_payload_comparison_binding(self):
  def make(name,value):
   folder=self.fixture/name;folder.mkdir();data=folder/'data.bin';data.write_bytes(np.array([value],dtype=np.float32).tobytes())
   schema=folder/'schema.json';schema.write_text(json.dumps(dict(entries=[dict(path='model/parameter/x',dtype='torch.float32',shape=[1],mode='bounded',offset=0,nbytes=4,numel=1)],optimizer_groups=[],optimizer_non_tensor_state={},total_bytes=4)))
   meta=folder/'meta.json';meta.write_text(json.dumps(dict(rng=[2024],absent_gradients=[],empty_optimizer_state=[])))
   return dict(step=1,schema_file=str(schema),schema_sha=capture.sha(schema),data_file=str(data),data_sha=capture.sha(data),bytes=4,meta_file=str(meta),meta_sha=capture.sha(meta))
  x,y=make('reference',1.),make('observed',1.0001)
  result=capture.compare_point(x,y,self.fixture);self.assertFalse(result['identical']);self.assertEqual(result['tensors'][0]['different_elements'],1)
  bad=dict(y,data_sha='0'*64)
  with self.assertRaises(ValueError):capture.compare_point(x,bad,self.fixture)
