"""No-model scoped confirmation tests. Synthetic state only; no torch import."""
import ast,copy,json,unittest,tempfile
from pathlib import Path
from unittest.mock import patch
from utils import ch3_urban_confirmation as s
from utils import ch3_admission_merge as merge
from utils.ch3_contract import read_profiles,profile,task_by_id,digest,numeric_probe_policy
from utils.ch3_extension_probe import Budget,scope_module

class UrbanConfirmationTests(unittest.TestCase):
 def setUp(self):self.c=read_profiles();self.spec=s.specification(self.c)
 def pair(self):
  t=task_by_id(self.c,self.spec['task_ids'][0]);p=profile(self.c,t)
  metrics=dict(mse=.5,mae=.5,sse=5.,sae=5.,elements=10);state=dict(rng='r6',model='m6',optimizer='o6')
  target=dict(index=p['target_idx'],pred_len=p['pred_len'],C=p['C'],metric_space='train-standardized target-only')
  evaluation=dict(metrics=metrics,batch_ids=['full','tail'],target=target)
  x=dict(id=t['id'],profile_sha=digest(p),initial='init',initial_rng='r0',batch_ids=list(range(6)),steps=6,final_rng='r6',final='m6',validation_tail=1,threads=4,affinity=[0],finite=True,validation=copy.deepcopy(metrics),trajectory=[dict(step=i,loss=.2,optimizer='o'+str(i))for i in range(1,7)],urban_confirmation=dict(policy_sha=digest(s.POLICY),evaluations={'2':copy.deepcopy(evaluation),'6':dict(copy.deepcopy(evaluation),before=state,after=copy.deepcopy(state),mode_restored=True)}))
  detail=dict(points=[dict(step=i,metadata_equal=True,rng_equal=True,identical=True,tensors=[dict(name='model/parameter/a',structure_equal=True,finite=True,max_abs_diff=0.,exact_required=False,byte_equal=True),dict(name='optimizer/a/step',structure_equal=True,finite=True,max_abs_diff=0.,exact_required=True,byte_equal=True)])for i in range(7)])
  return t,x,copy.deepcopy(x),detail
 def test_policy_scope(self):
  matched=[t for t in self.c['tasks']if s.policy(self.c,t)]
  self.assertEqual(len(matched),24);self.assertEqual({t['h']for t in matched},{3,6,9,12})
  self.assertTrue(all((t['model'],t['dataset'],t['input_variant'])==('TimeMixer','UrbanEV','F4')for t in matched))
  for t in matched:self.assertIsNone(numeric_probe_policy(self.c,t))
 def test_exact_budget_and_waves(self):
  self.assertEqual([len(w['task_ids'])for w in self.spec['waves']],[1,1,1,1,4]);self.assertEqual({k:8*v for k,v in s.WORKER.items()},s.CAPS)
  self.assertEqual({k:s.DEBIT[k]+s.CAPS[k]for k in s.CAPS},dict(adam=192,forward=290,backward=192))
  self.assertEqual({k:s.PARENT_CAPS[k]-s.DEBIT[k]-s.CAPS[k]for k in s.CAPS},dict(adam=168,forward=222,backward=168))
 def test_plan_and_template_reject(self):
  a=s.read(s.PACKAGE_ROOT/'approval.template.json');self.assertTrue(s.authorization_reasons(self.c,a));plan=s.read(s.PLAN)
  for key in ('caps','profile_shas','waves','policy','carry_forward_sha256'):
   with self.subTest(field=key),patch.object(s,'read',return_value=dict(plan,**{key:None})),self.assertRaises(ValueError):s.specification(self.c)
  for name in ('extension-probe-v1','urban-numeric-diagnostic-v1','timemixer-revision-numeric-v1'):
   with self.subTest(scope=name):self.assertIn('foreign purpose/scope',s.authorization_reasons(self.c,dict(a,probe_scope=name)))
 def test_worker_paths_and_caps(self):
  r=self.spec['task_ids'][0];out=str(s.ROOT/'serial'/r)
  cfg=dict(probe_scope=s.SCOPE,task=r,output=out,session_root=str(s.ROOT),fixture_root=str(s.ROOT/'fixture'),prefix_files={},resume=False,artifact_root=out,limits=dict(s.WORKER,seconds=1800),ids=[r],approval={})
  with patch.object(s,'authorization_reasons',return_value=[]):
   s.validate_worker(self.c,cfg)
   for k,v in [('output',str(s.ROOT.parent/'escape')),('limits',dict(adam=6,forward=8,backward=6,seconds=1800)),('prefix_files',{'real.csv':1}),('resume',True),('probe_scope','extension-probe-v1')]:
    with self.subTest(field=k),self.assertRaises(ValueError):s.validate_worker(self.c,dict(cfg,**{k:v}))
  for name in (s.SCOPE,'urban-numeric-diagnostic-v1','extension-probe-v1','timemixer-revision-numeric-v1'):
   with self.subTest(router=name):self.assertEqual(scope_module(dict(probe_scope=name)).__name__.endswith('ch3_urban_confirmation'),name==s.SCOPE)
 def test_stop_failure_no_refund(self):
  b=Budget(s.CAPS,s.ZERO);calls=[]
  def launch(runs,out):
   b.reserve([dict(output=str(out/r),limits=s.WORKER)for r in runs]);calls.append(runs)
   return dict(failure='unknown',returncodes=[1],resource_admission=False),[]
  with self.assertRaises(RuntimeError):s.run_waves(self.c,launch,lambda:False)
  self.assertEqual(calls,[self.spec['task_ids'][:1]]);self.assertEqual(b.counts,s.WORKER)
  with self.assertRaises(InterruptedError):s.run_waves(self.c,launch,lambda:True)
  self.assertEqual(len(calls),1)
  for _ in range(7):b.reserve([dict(output='synthetic',limits=s.WORKER)])
  with self.assertRaises(RuntimeError):b.reserve([dict(output='denied',limits=s.WORKER)])
  self.assertEqual(b.counts,s.CAPS)
 def test_duplicate_and_wave_reject(self):
  with patch.object(s,'ROOT',s.PACKAGE_ROOT/'fixture'),patch.object(s,'authorization_reasons',return_value=[]):self.assertIn('retained diagnostic output; no restart',s.readiness(self.c,{}))
  runs=self.spec['task_ids'];configs=[dict(task=r,output=str(s.ROOT/'q4'/r))for r in runs]
  with patch.object(s,'validate_worker'):
   s.validate_wave(self.c,configs,s.ROOT/'q4'/('monitor-'+runs[0]))
   with self.assertRaises(ValueError):s.validate_wave(self.c,configs[:2],s.ROOT/'q4'/('monitor-'+runs[0]))
 def test_endpoint_cached_no_sampling(self):
  class PlainMode:
   training=True
   def modules(self):return [self]
  mode=PlainMode();batches=[['full'],['tail']];calls=[]
  def evaluate(bs):self.assertIs(bs,batches);calls.extend(bs);mode.training=False;return dict(elements=2)
  value=s.cached_endpoint(mode,batches,evaluate,lambda:dict(rng='fixed',model='same',optimizer='same'),digest)
  self.assertEqual(calls,batches);self.assertTrue(mode.training);self.assertEqual(value['before'],value['after'])
  with self.assertRaises(ValueError):s.cached_endpoint(mode,batches,evaluate,iter([{'rng':0},{'rng':1}]).__next__,digest)
  def fail(bs):mode.training=False;raise RuntimeError('fixture')
  with self.assertRaises(RuntimeError):s.cached_endpoint(mode,batches,fail,lambda:{},digest)
  self.assertTrue(mode.training)
 def test_comparison_at_bound(self):
  t,x,y,d=self.pair();d['points'][1]['tensors'][0].update(max_abs_diff=1e-4,byte_equal=False)
  r=s.compare_measured(self.c,t,x,y,d);self.assertTrue(r['passed']);self.assertEqual(r['state_max_abs'],1e-4)
 def test_state_rejection(self):
  for kind in ('float','exact','metadata','initial','finite'):
   with self.subTest(kind=kind):
    t,x,y,d=self.pair();point=d['points'][1]
    if kind=='float':point['tensors'][0]['max_abs_diff']=.00010001
    elif kind=='exact':point['tensors'][1]['byte_equal']=False
    elif kind=='metadata':point['metadata_equal']=False
    elif kind=='initial':d['points'][0]['identical']=False
    else:point['tensors'][0]['finite']=False
    self.assertFalse(s.compare_measured(self.c,t,x,y,d)['passed'])
 def test_loss_and_two_metrics(self):
  for kind in ('loss','2','6','sse','sae'):
   with self.subTest(kind=kind):
    t,x,y,d=self.pair()
    if kind=='loss':y['trajectory'][0]['loss']+=2e-6
    elif kind in ('2','6'):
     y['urban_confirmation']['evaluations'][kind]['metrics']['mse']+=2e-6
     if kind=='2':y['validation']=copy.deepcopy(y['urban_confirmation']['evaluations']['2']['metrics'])
    else:y['urban_confirmation']['evaluations']['6']['metrics'][kind]+=2e-5
    self.assertFalse(s.compare_measured(self.c,t,x,y,d)['passed'])
 def test_identity_endpoint_and_metric_finite(self):
  for kind in ('rng','batch','count','endpoint','finite'):
   with self.subTest(kind=kind):
    t,x,y,d=self.pair()
    if kind=='rng':y['initial_rng']='other'
    elif kind=='batch':y['urban_confirmation']['evaluations']['6']['batch_ids']=['other','tail']
    elif kind=='count':y['urban_confirmation']['evaluations']['6']['metrics']['elements']=11
    elif kind=='endpoint':y['urban_confirmation']['evaluations']['6']['after']={'rng':'changed'}
    else:y['urban_confirmation']['evaluations']['6']['metrics']['mae']=float('nan')
    with self.assertRaises(ValueError):s.compare_measured(self.c,t,x,y,d)
 def test_real_capture_comparison_no_model(self):
  import numpy as np
  from utils.ch3_urban_capture import compare_traces
  fixture=Path(tempfile.mkdtemp(dir=s.PACKAGE_ROOT/'fixture',prefix='numeric-'))
  def points(side,value):
   rows=[]
   for i in range(7):
    base=fixture/f'{side}-{i}';base.mkdir();(base/'data.bin').write_bytes(np.array([0. if i==0 else value],dtype='<f4').tobytes())
    schema=dict(entries=[dict(path='model/parameter/a',dtype='torch.float32',shape=[1],numel=1,nbytes=4,mode='bounded',offset=0)],total_bytes=4,optimizer_groups=[],optimizer_non_tensor_state={})
    (base/'schema.json').write_text(json.dumps(schema));(base/'meta.json').write_text(json.dumps(dict(step=i,rng=i)))
    row=dict(step=i,bytes=4)
    for key,file in [('data','data.bin'),('schema','schema.json'),('meta','meta.json')]:row[key+'_file']=str(base/file);row[key+'_sha']=s.sha(base/file)
    rows.append(row)
   return rows
  t,x,y,d=self.pair();x['urban_diagnostic_trace']=points('a',.1);y['urban_diagnostic_trace']=points('b',.10001)
  d=compare_traces(x,y,fixture);self.assertTrue(s.compare_measured(self.c,t,x,y,d)['passed'])
 def test_worker_static_scope_isolation(self):
  import ch3_runner
  src=Path(ch3_runner.__file__).read_text();tree=ast.parse(src);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='probe_worker');self.assertFalse(fn.args.defaults[-1].value)
  self.assertIn("else:validation=evaluate(model,[batch(v),batch(tail or v)],p,device)",src)
  self.assertIn('validation_batches=[batch(v),batch(tail or v)]',src);self.assertLess(src.index('final_rng=rng(),memory=memory'),src.index('endpoint=cached_endpoint'))
  entry=Path('tools/restricted_regression/m5_formal_entry.py').read_text();self.assertIn("s.get('probe_scope')=='urban-numeric-confirmation-v1':ceilings[purpose]=(6,10,6)",entry);self.assertIn("'ch3_probe':(6,8,6)",entry)
 def test_87_source_partition(self):
  actual=merge.inherited_sources(self.c);self.assertEqual(len(actual['decisions']),87);self.assertNotIn('TimeMixer-UrbanEV-F4',actual['decisions']);counts={}
  for v in actual['lineage'].values():counts[v['kind']]=counts.get(v['kind'],0)+1
  self.assertEqual(counts,dict(original_unchanged_51=51,reviewed_transfer_26=26,reviewed_q1_7=7,reviewed_revised_lr_3=3))
 def test_merged_review_and_tampering(self):
  fixture=dict(purpose=merge.PURPOSE,synthetic_fixture=False,confirmation={'path':'synthetic'},reviewed=False,decisions={'88th':'fixture'})
  with patch.object(merge,'build_merged',return_value=fixture):
   merge.validate_merged(self.c,fixture)
   with self.assertRaises(ValueError):merge.validate_merged(self.c,fixture,require_review=True)
   with self.assertRaises(ValueError):merge.validate_merged(self.c,dict(fixture,decisions={'88th':'forged'}))
   with self.assertRaises(ValueError):merge.validate_merged(self.c,dict(fixture,reviewed=True),require_review=True)
 def test_profiles_and_queues_preserved(self):
  old=s.read(s.PACKAGE_ROOT/'start.json');self.assertEqual({t['id']:digest(profile(self.c,t))for t in self.c['tasks']},old['before_profiles']);self.assertEqual(self.c['groups'],old['groups'])
  from utils.ch3_revision import waves
  self.assertEqual([len(w)for w in waves(self.c)],[4,4,2,2,4])
  for name,h in old['protected'].items():self.assertEqual(s.sha(Path(name)),h)

 def test_completed_report_replay_lifecycle(self):
  a=s.read(s.PACKAGE_ROOT/'approval.template.json');a.update(reviewed=True,execution_permitted=True,budget_authorized=True,commit='synthetic actual-byte fixture',code={'synthetic':True},environment={'synthetic':True},hardware={'synthetic':True})
  with patch.object(s,'controller_live',return_value=True):
   self.assertIn('conflicting controller live',s.authorization_reasons(self.c,a))
   self.assertEqual(s.authorization_reasons(self.c,a,check_live=False),[])
  run=self.spec['task_ids'][0];out=str(s.ROOT/'serial'/run)
  cfg=dict(probe_scope=s.SCOPE,task=run,output=out,session_root=str(s.ROOT),fixture_root=str(s.ROOT/'fixture'),prefix_files={},resume=False,artifact_root=out,limits=dict(s.WORKER,seconds=1800),ids=[run],approval=a)
  with patch.object(s,'authorization_reasons',return_value=[])as auth:
   s.validate_worker(self.c,cfg,replay=True);self.assertFalse(auth.call_args.kwargs['check_live'])
   s.validate_worker(self.c,cfg);self.assertTrue(auth.call_args.kwargs['check_live'])

 def test_cli_complete_requires_merged(self):
  import contextlib,io,m6_urban_confirmation_entry as entry
  def load(path):return {'scope_sha':digest(self.spec)}if path.name=='complete.json'else {'synthetic_fixture':True}
  with patch.object(Path,'exists',return_value=True),patch.object(entry,'read',side_effect=load),patch.object(entry,'validate_completion')as checked,patch.object(merge,'validate_merged')as combined,contextlib.redirect_stdout(io.StringIO()):
   self.assertEqual(entry.cli(['complete']),0);checked.assert_called_once();combined.assert_called_once()
  def missing(path):
   if path.name=='merged-admission.json':raise FileNotFoundError('synthetic missing merge')
   return {'scope_sha':digest(self.spec)}
  with patch.object(Path,'exists',return_value=True),patch.object(entry,'read',side_effect=missing),patch.object(entry,'validate_completion'),self.assertRaises(FileNotFoundError):entry.cli(['complete'])
