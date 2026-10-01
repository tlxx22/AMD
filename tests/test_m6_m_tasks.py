import unittest,copy,os,json,tempfile,subprocess,sys,time
from pathlib import Path
from unittest.mock import patch
from utils.ch3_contract import read_profiles,validate_manifest,profile,step_arithmetic,BestState,digest
from utils import ch3_m_tasks as s
from utils import ch3_m_execution as e

class MContractTests(unittest.TestCase):
 def setUp(self):self.c=validate_manifest(read_profiles())
 def test_exact_matrix_and_budget(self):
  self.assertEqual(len(self.c['tasks']),84);self.assertEqual(e.plan(self.c)['run_budget'],dict(runs=84,run_epochs=840));self.assertEqual(e.plan(self.c)['order'],list(s.MODELS));self.assertEqual(len(self.c['groups']),21)
 def test_parent_profiles_only_authorized_changes(self):
  for t in self.c['tasks']:
   with self.subTest(run=t['id']):
    p=profile(self.c,t);old=self.c['m_experiment']['parent_profiles'][t['id']]['profile']
    for k in ('T','H','pred_len','features','C','target_idx','training','structure'):self.assertEqual(p[k],old[k])
    self.assertEqual(p['metric_scope'],'all_channels');self.assertEqual(p['parent_MS_profile']['sha256'],digest(old))
 def test_fixed_lr_and_no_ecl_or_J(self):
  expected={'ETTh1':.001,'Weather':.001,'Exchange':.0003}
  for t in self.c['tasks']:
   self.assertNotIn(t['model'],('J','N','S'));self.assertNotEqual(t['dataset'],'ECL');p=profile(self.c,t);self.assertIsNone(p['training']['scheduler'])
   if t['model']=='TimeMixer':self.assertEqual(p['training']['lr'],expected[t['dataset']])
 def test_endpoints_target_columns_and_tail(self):
  for t in self.c['tasks']:
   with self.subTest(run=t['id']):
    p=profile(self.c,t);T,C,(a,z,n),idx=s.DATA[t['dataset']];self.assertEqual((p['T'],p['C'],p['target_idx']),(T,C,idx));ar=step_arithmetic(self.c,t)
    self.assertEqual(ar['train_windows'],a-T-t['h']+1);self.assertEqual(ar['validation_windows'],z-a-t['h']+1);self.assertEqual(ar['test_windows_arithmetic_only'],n-z-t['h']+1)
    self.assertEqual(ar['max_optimizer_steps'],ar['train_batches']*10)
 def test_full_channel_windows_and_context(self):
  import numpy as np
  from utils.ch3_data import Windows
  x=np.arange(60,dtype='float32').reshape(20,3);w=Windows(x,x,4,15,4,2);a,b=w[0]
  self.assertEqual(tuple(a.shape),(4,3));self.assertEqual(tuple(b.shape),(2,3));self.assertEqual(b.tolist(),x[8:10].tolist())
 def test_bounded_loader_train_scaler_all_labels(self):
  from utils.ch3_data import load
  p=copy.deepcopy(profile(self.c,self.c['tasks'][0]));p.update(T=4,H=2,pred_len=2,C=3,features=['a','b','OT'],target_idx=2)
  d=copy.deepcopy(self.c['datasets']['ETTh1']);d.update(features=p['features'],target='OT',endpoints=[8,12,16]);c=dict(self.c,datasets={'ETTh1':d})
  with tempfile.TemporaryDirectory(dir=s.PACKAGE/'fixtures')as tmp:
   f=Path(tmp)/'synthetic.csv';f.write_text('date,a,b,OT\n'+''.join(f'2020-01-01 {i:02d}:00:00,{i},{i*i},{100-i}\n'for i in range(12))+ '2020-01-01 12:00:00,DO_NOT_PARSE,DO_NOT_PARSE,DO_NOT_PARSE\n')
   with patch('utils.ch3_data.profile',return_value=p):windows,m=load(c,self.c['tasks'][0],path_override=f)
   self.assertEqual(set(windows),{'train','validation'});self.assertEqual(m['scaler_fit_records'],8);self.assertFalse(m['test_observations_accessed']);self.assertEqual(m['scaler']['mean'],[3.5,17.5,96.5]);self.assertEqual(tuple(windows['validation'][0][1].shape),(2,3))
   import numpy as np
   labels=windows['validation'][0][1].numpy();raw=labels*np.array(m['scaler']['scale'])+np.array(m['scaler']['mean'])
   np.testing.assert_allclose(raw,[[8,64,92],[9,81,91]],rtol=0,atol=1e-5)
 def test_weather_duplicates_reverse_and_other_domain(self):
  import pandas as pd
  from utils.ch3_data import validate_times
  dup=pd.DatetimeIndex(['2020-01-01','2020-01-01','2020-01-02']);self.assertEqual(validate_times(dup,'Weather',self.c['datasets']['Weather'])['duplicate_timestamps'],1)
  with self.assertRaises(ValueError):validate_times(dup[::-1],'Weather',self.c['datasets']['Weather'])
  with self.assertRaises(ValueError):validate_times(dup,'Exchange',self.c['datasets']['Exchange'])
 def test_weighted_metric_and_channel_diagnostic(self):
  import numpy as np
  rows=s.metric_totals([np.ones((2,2,3)),np.full((1,2,3),3)],['x','temp','OT'],1)
  self.assertAlmostEqual(rows['mse'],11/3);self.assertEqual(rows['elements'],18);self.assertEqual(rows['MS_target_diagnostic']['name'],'temp')
 def test_fake_evaluate_same_pass_and_no_broadcast(self):
  import torch
  from ch3_runner import evaluate
  class Fake:
   def eval(self):pass
  p=profile(self.c,self.c['tasks'][0]);p=dict(p,C=3,features=['x','temp','OT'],target_idx=1,pred_len=2)
  pairs=[(torch.ones(2,2,3),torch.zeros(2,2,3)),(torch.ones(1,2,3)*3,torch.zeros(1,2,3))];calls=[]
  def native(m,x,p):calls.append(tuple(x.shape));return x,None
  with patch('models.ch3_adapter.target_prediction',side_effect=native):r=evaluate(Fake(),pairs,p,'cpu')
  self.assertEqual(len(calls),2);self.assertEqual(r['elements'],18);self.assertAlmostEqual(r['mse'],11/3)
 def test_native_TimeXer_M_order_and_options(self):
  import torch
  from models.ch3_adapter import native_options,target_prediction
  t=next(t for t in self.c['tasks']if t['model']=='TimeXer'and t['dataset']=='Weather');p=profile(self.c,t);opts=native_options(p)
  self.assertEqual(opts['features'],'M');self.assertEqual(opts['enc_in'],21);x=torch.arange(21.).reshape(1,1,21).repeat(1,p['T'],1)
  class Native:
   def __call__(self,x,*args):self.seen=x.clone();return x[:,:1].repeat(1,p['pred_len'],1)
  m=Native();out,_=target_prediction(m,x,p);self.assertEqual(m.seen[0,0].tolist(),list(range(21)));self.assertEqual(out.shape[-1],21)
 def test_MS_adapter_route_preserved(self):
  import torch
  from models.ch3_adapter import target_prediction
  p=dict(profile(self.c,next(t for t in self.c['tasks']if t['model']=='DLinear')),task='MS',input_variant='MS');x=torch.arange(7.).reshape(1,1,7).repeat(1,p['T'],1)
  class Native:
   def __call__(self,x):return x[:,:1].repeat(1,p['pred_len'],1)
  out,_=target_prediction(Native(),x,p);self.assertEqual(out.shape[-1],1);self.assertTrue((out==6).all())
 def test_bad_output_replication_and_metrics_rejected(self):
  import torch
  from models.ch3_adapter import target_prediction
  t=self.c['tasks'][0];p=profile(self.c,t)
  class Wrong:
   def __call__(self,x):return torch.ones(x.shape[0],p['pred_len'],1)
  with self.assertRaises(ValueError):target_prediction(Wrong(),torch.ones(2,p['T'],p['C']),p)
 def test_best_mse_tie_rule(self):
  best=BestState();self.assertTrue(best.update(.2,1));self.assertFalse(best.update(.2,2));self.assertEqual(best.epoch,1);self.assertTrue(best.update(.1,3))
 def test_M_and_MS_ids_never_deduplicate(self):
  ids=[t['id']for t in self.c['tasks']];self.assertEqual(len(set(ids)),84);self.assertTrue(all('-M-f1-'in r for r in ids));self.assertFalse(any('-MS-'in r for r in ids))
 def test_independent_probe_caps_and_no_refund(self):
  b=dict(reserved=dict(adam=0,forward=0,backward=0));e.debit(b,168);self.assertEqual(b['reserved'],dict(adam=1008,forward=1344,backward=1008));e.debit(b,84);self.assertEqual(b['reserved'],s.CAPS)
  with self.assertRaises(ValueError):e.debit(b,1)
  self.assertEqual(b['reserved'],s.CAPS)
 def test_resource_only_q2_failure_boundary(self):
  for kind in ['resource',None,'business','numeric','identity','guard','unknown']:
   with self.subTest(kind=kind):self.assertEqual(e.resource_fallback(dict(failure_kind=kind)),kind=='resource')
 def test_exact_worker_paths_and_wrong_scope(self):
  t=self.c['tasks'][0];self.assertEqual(e.exact_path(self.c,t,'ch3_formal',None,s.result_path(t)),s.result_path(t))
  with self.assertRaises(ValueError):e.exact_path(self.c,t,'ch3_formal',None,s.RESULT.parent/'formal-AMD'/t['id'])
  with self.assertRaises(ValueError):e.exact_path(self.c,t,'ch3_probe','q4',e.PROBE_ROOT/t['group']/'serial'/t['id'])
 def test_numeric_policies_specific_scope(self):
  self.assertIsNone(s.numeric_policy(self.c,self.c['tasks'][0]))
  for t in self.c['tasks']:
   if t['model']=='TimeMixer':self.assertEqual(s.numeric_policy(self.c,t)['task'],'M')
  with self.assertRaises(ValueError):s.numeric_policy(self.c,dict(self.c['tasks'][0],input_variant='MS'))
 def test_template_and_old_permit_rejection(self):
  from ch3_runner import code_binding
  a=dict(m_scope='extension-probe-v1',purpose='ch3_resource_probe',reviewed=False,execution_permitted=False,budget_authorized=False)
  reasons=e.authorization_reasons(self.c,a,probe=True);self.assertIn('M scope mismatch',reasons);self.assertIn('M template not executable',reasons);self.assertIn('M independent budget approval',reasons)
  self.assertIn('old MS queue including TimeXer/N/S completion boundary required',reasons)
 def test_valid_authorization_positive_and_missing_boundary(self):
  from ch3_runner import code_binding
  fake_env={};fake_hw={}
  with patch('ch3_runner.environment_binding',return_value=fake_env),patch('ch3_runner.hardware_binding',return_value=fake_hw):a=e.approval_template(self.c,True)
  a.update(reviewed=True,execution_permitted=True,budget_authorized=True,commit='future-fixture-only',old_boundary={'path':'fixture','sha256':'fixture'},plan={'path':'plan','sha256':'fixture'})
  b=s.bound(self.c['m_experiment']['data_binding_artifact']);boundary=dict(kind='old_MS_queue_technical_complete');real_bound=s.bound
  def load(r):return e.plan(self.c)if r.get('path')=='plan'else boundary if r.get('path')=='fixture'else real_bound(r)
  def git(*args):return 'm6/m-baselines-v1'if args[0]=='branch'else''if args[0]=='status'else'future-fixture-only'
  with patch.object(e,'gpu_environment_reasons',return_value=[]),patch('ch3_runner.git',side_effect=git),patch('ch3_runner.environment_binding',return_value=fake_env),patch('ch3_runner.hardware_binding',return_value=fake_hw),patch.object(s,'bound',side_effect=load),patch.object(e,'old_boundary',return_value=boundary):self.assertEqual(e.authorization_reasons(self.c,a,probe=True),[])
 def test_empty_summary_no_fabrication_or_MS_ranking(self):
  from utils.ch3_m_summary import summarize
  with tempfile.TemporaryDirectory(dir=s.PACKAGE/'fixtures')as tmp,patch.object(s,'RESULT',Path(tmp)):
   r=summarize(self.c);self.assertEqual(len(r['missing']),84);self.assertEqual(len(r['index']),84);self.assertFalse(r['MS_ranking_included']);self.assertIsNone(r['cross_dataset_mean'])
 def test_summary_cost_source_and_failed_not_ranked(self):
  from utils.ch3_m_summary import summarize
  with tempfile.TemporaryDirectory(dir=s.PACKAGE/'fixtures')as tmp,patch.object(s,'RESULT',Path(tmp)):
   t=self.c['tasks'][0];out=s.result_path(t);out.mkdir(parents=True);(out/'budget.json').write_text(json.dumps(dict(counts=dict(adam=3,forward=3,backward=3))));(out/'runtime.json').write_text(json.dumps(dict(elapsed=1.5,error='synthetic retained failure')))
   r=summarize(self.c);self.assertEqual(r['index'][0]['status'],'failed');self.assertEqual(r['execution'][0]['actual']['budget']['counts']['adam'],3);self.assertEqual(r['execution'][0]['actual']['runtime']['elapsed'],1.5);self.assertEqual(r['execution'][0]['parent_MS_profile'],profile(self.c,t)['parent_MS_profile']);self.assertEqual(r['baseline_ranking'],{});self.assertEqual(r['execution'][0]['sources']['runtime'],s.ref(out/'runtime.json'))
 def test_fixed_waves_q4_q2_q1(self):
  ids=self.c['groups'][0]['task_ids'];self.assertEqual(e.wave_ids(ids,4),[ids]);self.assertEqual(e.wave_ids(ids,2),[ids[:2],ids[2:]]);self.assertEqual(len(e.wave_ids(ids,1)),4)
 def test_synchronous_handoff_and_failure_stops(self):
  from m6_remaining_entry import sequence
  with tempfile.TemporaryDirectory(dir=s.PACKAGE/'fixtures')as tmp:
   root=Path(tmp);p=dict(groups=[dict(model='A'),dict(model='B')],task_ids=['a','b']);cmd={m:[sys.executable,'-B','-c','import time;time.sleep(.15)']for m in ['A','B']};events=[]
   def before(g):events.append(('before',g['model']))
   def check(g):events.append(('after',g['model']));return dict(status='technical-complete',result_review='pending')
   sequence(p,root,cmd,before,check,queue_id=s.ID);self.assertEqual(events,[('before','A'),('after','A'),('before','B'),('after','B')])
   self.assertEqual(json.loads((root/'complete.json').read_text())['queue_id'],s.ID)
 def test_failed_handoff_and_STOP_no_next_group(self):
  from m6_remaining_entry import sequence
  for stop in (False,True):
   with self.subTest(stop=stop),tempfile.TemporaryDirectory(dir=s.PACKAGE/'fixtures')as tmp:
    root=Path(tmp);p=dict(groups=[dict(model='A'),dict(model='B')],task_ids=['a','b']);events=[];cmd={m:[sys.executable,'-B','-c','import time;time.sleep(.1)']for m in ['A','B']}
    def check(g):
     if stop:(root/'STOP').write_text('stop')
     else:raise ValueError('missing result/identity')
     return dict(status='technical-complete',result_review='pending')
    with self.assertRaises((ValueError,InterruptedError)):sequence(p,root,cmd,lambda g:events.append(g['model']),check,queue_id=s.ID)
    self.assertEqual(events,['A']);self.assertFalse((root/'complete.json').exists());self.assertTrue((root/'failure.json').exists())

class MExecutionTests(unittest.TestCase):
 def setUp(self):self.c=validate_manifest(read_profiles())
 def test_policy_or_source_mutation_rejected(self):
  for field in ('numeric_policies','sources'):
   with self.subTest(field=field):
    c=copy.deepcopy(self.c)
    if field=='numeric_policies':c['m_experiment'][field][c['groups'][0]['id']]['rule']={'state_atol':999}
    else:c[field]['TimeXer']['commit']='wrong'
    with self.assertRaises(ValueError):validate_manifest(c)
 def test_worker_positive_and_cross_scope_before_fixture(self):
  from contextlib import ExitStack
  sys.path.insert(0,str(s.ROOT/'tools/restricted_regression'));import m5_formal_entry as worker
  c=self.c;t=c['tasks'][0];p=e.PROBE_ROOT/t['group']/'serial'/t['id'];a={'m_scope':s.PROBE}
  cfg=dict(purpose='ch3_probe',resume=False,task=t['id'],approval=a,m_scope=s.PROBE,protocol_sha=digest(c),bound_files={},ids=[t['id']],output=str(p),artifact_root=str(p),m_phase='serial',session_root=str(e.PROBE_ROOT),fixture_root=str(s.PACKAGE/'fixtures'),limits=dict(adam=6,forward=8,backward=6,seconds=1800),prefix_files={},author_files={},metadata_files={})
  cfg['metadata_files']=e.metadata_files(c,a)
  cfg['device']='cuda:0';cfg['kernel_probe']=False
  with patch.object(worker,'repository_files',return_value={}),patch.object(e,'authorization_reasons',return_value=[]),patch.dict(os.environ,TMPDIR=cfg['fixture_root']):
   e.validate_worker(c,cfg)
   for key,value in [('task',t['id'].replace('-M-','-MS-')),('m_scope','extension-probe-v1'),('prefix_files',{'CSV':1}),('output',str(s.RESULT/'wrong')),('limits',dict(adam=7,forward=8,backward=6,seconds=1800))]:
    with self.subTest(key=key),self.assertRaises((ValueError,KeyError)):e.validate_worker(c,dict(cfg,**{key:value}))
 def test_wave_positive_stop_and_mixed_group_rejection(self):
  g=self.c['groups'][0];a={'m_scope':s.PROBE};cfg=[dict(task=r,purpose='ch3_probe',m_phase='q4',approval=a,output=str(e.PROBE_ROOT/g['id']/'q4'/r))for r in g['task_ids']]
  self.assertEqual(e.validate_wave(self.c,cfg,e.PROBE_ROOT/g['id']/'q4/wave-0'),e.PROBE_ROOT/'STOP')
  with self.assertRaises(ValueError):e.validate_wave(self.c,cfg[:3],e.PROBE_ROOT/g['id']/'q4/wave-0')
  wrong=dict(cfg[-1],task=self.c['groups'][1]['task_ids'][0])
  with self.assertRaises(ValueError):e.validate_wave(self.c,cfg[:-1]+[wrong],e.PROBE_ROOT/g['id']/'q4/wave-0')
 def test_symlink_output_parent_refused(self):
  with tempfile.TemporaryDirectory(dir=s.PACKAGE/'fixtures')as tmp:
   root=Path(tmp);(root/'actual').mkdir();(root/'alias').symlink_to(root/'actual',target_is_directory=True)
   with patch.object(e,'PROBE_ROOT',root/'alias'),self.assertRaises(ValueError):e.exact_path(self.c,self.c['tasks'][0],'ch3_probe','serial',root/'alias'/self.c['tasks'][0]['group']/'serial'/self.c['tasks'][0]['id'])
 def test_probe_producer_success_resource_only_fallback_and_stop(self):
  sys.path.insert(0,str(s.ROOT/'tools/restricted_regression'));import m5_formal_entry as worker
  from ch3_runner import dump
  for kind in ('success','resource',None,'numeric','guard','business'):
   with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir=s.PACKAGE/'fixtures')as tmp:
    root=Path(tmp);g=self.c['groups'][0];c=dict(self.c,groups=[g]);calls=[]
    def config(c,purpose,out,task,approval):
     out=Path(out);out.mkdir(parents=True);dump(out/'budget.json',dict(counts=dict(adam=0,forward=0,backward=0)));return dict(task=task,output=str(out),budget_file=str(out/'budget.json'))
    def run(cfg,out,monitor):
     out=Path(out);out.mkdir();phase=out.parent.name;calls.append((phase,len(cfg)))
     fail=phase!='serial'and kind!='success';row=dict(failure='fixture failure'if fail else None,failure_kind=kind if fail else None,returncodes=[1 if fail else 0]*len(cfg),resource_admission=not fail,elapsed=1.)
     dump(out/'process.json',row)
     for x in cfg:
      dump(Path(x['budget_file']),dict(counts=dict(adam=6,forward=8,backward=6)));dump(Path(x['output'])/'trajectory.json',dict(fixture=True,profile_sha='fixture'))
     return row
    a=dict(commit='synthetic-only',code={},environment={},hardware={})
    dump(root/'approval.json',dict(a,synthetic_fixture=True))
    with patch.object(e,'PROBE_ROOT',root),patch.object(worker,'make_config',side_effect=config),patch.object(worker,'run_configs',side_effect=run),patch.object(e,'compare',return_value={'passed':True}),patch.object(e,'validate_probe_completion',return_value=True):
     if kind in ('success','resource'):r=e.run_probe(c,a,root);self.assertEqual(r['decisions'][g['id']]['concurrency'],4 if kind=='success'else 1)
     else:
      with self.assertRaises(RuntimeError):e.run_probe(c,a,root)
      self.assertFalse((root/'complete.json').exists())
    b=json.loads((root/'budget.json').read_text());self.assertEqual(b['actual'],b['reserved']);self.assertFalse(b['refund']);self.assertEqual(calls[:4],[('serial',1)]*4)
    if kind=='resource':self.assertEqual(calls[4:],[('q4',4),('q2',2)])
    elif kind!='success':self.assertEqual(calls[4:],[('q4',4)])
 def test_probe_STOP_before_precharge(self):
  sys.path.insert(0,str(s.ROOT/'tools/restricted_regression'));import m5_formal_entry as worker
  with tempfile.TemporaryDirectory(dir=s.PACKAGE/'fixtures')as tmp:
   root=Path(tmp);(root/'STOP').write_text('stop')
   with patch.object(worker,'make_config')as mk,self.assertRaises(InterruptedError):e.run_probe(dict(self.c,groups=self.c['groups'][:1]),{},root)
   mk.assert_not_called();self.assertEqual(json.loads((root/'failure.json').read_text())['budget']['reserved'],dict(adam=0,forward=0,backward=0))
 def test_lock_conflict_and_double_start_preserved(self):
  from m6_remaining_entry import lock
  with tempfile.TemporaryDirectory(dir=s.PACKAGE/'fixtures')as tmp:
   root=Path(tmp);p=root/'queue.lock'
   with lock(p):
    with self.assertRaises(BlockingIOError):
     with lock(p):pass
    with lock(root/'gpu.lock'):pass
   (root/'complete.json').write_text('{}');self.assertTrue((root/'complete.json').exists())
 def test_allchannel_result_identity_and_summary_reject_MS(self):
  from utils.ch3_m_summary import verify_result
  t=self.c['tasks'][0];p=profile(self.c,t);n=step_arithmetic(self.c,t)['test_windows_arithmetic_only']*p['pred_len'];channels=[dict(index=i,name=f,mse=1.,mae=1.,sse=float(n),sae=float(n),elements=n)for i,f in enumerate(p['features'])]
  r=dict(id=t['id'],task='M',metric_scope='all_channels',input_variant='M',purpose='ch3_formal',protocol_sha=digest(self.c),profile_sha=digest(p),parent_MS_profile=p['parent_MS_profile'],from_scratch=True,seed=2024,metric_space='train-standardized',best_epoch=1,mse=1.,mae=1.,sse=float(n*p['C']),sae=float(n*p['C']),elements=n*p['C'],channels=channels,MS_target_diagnostic=channels[p['target_idx']],final_test=dict(calls=1,selected='best.pt',epoch=1))
  verify_result(self.c,t,r)
  for k,v in [('task','MS'),('elements',n),('MS_target_diagnostic',channels[0]),('final_test',dict(calls=2,selected='last.pt',epoch=1))]:
   with self.subTest(field=k),self.assertRaises(ValueError):verify_result(self.c,t,dict(r,**{k:v}))

class MCloseoutTests(unittest.TestCase):
 def setUp(self):self.c=validate_manifest(read_profiles())
 def test_repeat_start_completed_failed_staging_and_output_refused(self):
  with tempfile.TemporaryDirectory(dir=s.PACKAGE/'fixtures')as tmp:
   root=Path(tmp);control=root/'queue';log=root/'launch.log'
   with patch.object(e,'CONTROL',control),patch.object(e,'launcher_log',return_value=log),patch.object(s,'RESULT',root/'results'):
    self.assertEqual(e.fresh_conflicts(self.c),[])
    control.mkdir()
    for n in ('complete.json','failure.json','staging'):
     with self.subTest(state=n):
      (control/n).write_text('{}');self.assertTrue(e.fresh_conflicts(self.c));(control/n).unlink()
    control.rmdir();s.result_path(self.c['tasks'][0]).mkdir(parents=True)
    self.assertIn('M task output conflict',e.fresh_conflicts(self.c))
 def test_formal_positive_requires_new_M_report_not_MS(self):
  code='fixture-future-closure';bound=s.bound;boundary=dict(kind='old_MS_queue_technical_complete');report=dict(purpose='M_resource_admission',reviewed=True)
  def source(r):return e.plan(self.c)if r['path']=='plan-fixture'else boundary if r['path']=='boundary-fixture'else report if r['path']=='report-fixture'else bound(r)
  def git(*args):return'm6/m-baselines-v1'if args[0]=='branch'else''if args[0]=='status'else code
  with patch('ch3_runner.environment_binding',return_value={}),patch('ch3_runner.hardware_binding',return_value={}):a=e.approval_template(self.c,False)
  a.update(commit=code,reviewed=True,execution_permitted=True,budget_authorized=True,structure_frozen=True,m6_authorized=True,plan=dict(path='plan-fixture'),old_boundary=dict(path='boundary-fixture'),resource_report=dict(path='report-fixture'))
  with patch.object(e,'gpu_environment_reasons',return_value=[]),patch.object(s,'bound',side_effect=source),patch.object(e,'old_boundary',return_value=boundary),patch.object(e,'validate_probe_report',return_value=True),patch('ch3_runner.git',side_effect=git),patch('ch3_runner.environment_binding',return_value={}),patch('ch3_runner.hardware_binding',return_value={}):
   self.assertEqual(e.authorization_reasons(self.c,a),[])
   self.assertIn('existing M6 structure/stage authorization required',e.authorization_reasons(self.c,dict(a,structure_frozen=False)))
  with self.assertRaises(ValueError):e.validate_probe_report(self.c,dict(purpose='m6_merged_resource_admission_v1',reviewed=True))
 def test_metadata_binding_guard_and_scope(self):
  import m5_formal_entry as worker
  with tempfile.TemporaryDirectory(dir=s.PACKAGE/'fixtures')as tmp:
   f=Path(tmp)/'metadata.json';f.write_text('{}');state=dict(m_scope=s.PROBE,metadata_files={str(f):s.sha(f)})
   self.assertTrue(worker.check_access(str(f),False,state,lambda *a:(_ for _ in()).throw(PermissionError()),None))
   f.write_text('{"changed":1}')
   with self.assertRaises(PermissionError):worker.check_access(str(f),False,state,lambda *a:(_ for _ in()).throw(PermissionError()),None)
 def test_future_M_probe_counts_no_extra_forward_sites(self):
  import ast,inspect,ch3_runner
  tree=ast.parse(inspect.getsource(ch3_runner.probe_worker));calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)]
  self.assertEqual(sum(isinstance(n.func,ast.Name)and n.func.id=='update'for n in calls),1)
  self.assertEqual(sum(isinstance(n.func,ast.Name)and n.func.id=='init_training'for n in calls),1)
  self.assertEqual(e.plan(self.c)['nominal'],dict(workers=168,adam=1008,forward=1344,backward=1008))
  self.assertEqual(e.plan(self.c)['all_q2_extra'],dict(workers=84,adam=504,forward=672,backward=504))

class MWorkerCleanupTests(unittest.TestCase):
 def setUp(self):self.c=validate_manifest(read_profiles())
 def test_partial_spawn_failure_reaps_only_own_child(self):
  sys.path.insert(0,str(s.ROOT/'tools/restricted_regression'));import m5_formal_entry as worker
  with tempfile.TemporaryDirectory(dir=s.PACKAGE/'fixtures')as tmp:
   root=Path(tmp);g=self.c['groups'][0];kids=[]
   cfg=[dict(task=r,purpose='ch3_probe',m_phase='q4',approval={},output=str(root/g['id']/'q4'/r),artifact_root=str(root/g['id']/'q4'/r),limits=dict(seconds=3))for r in g['task_ids']]
   def spawn(c):
    if kids:raise OSError('synthetic second-spawn failure')
    log=(root/'own-child.log').open('x');p=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(20)'],stdout=log,stderr=subprocess.STDOUT);kids.append(p);return p,log
   with patch.object(e,'PROBE_ROOT',root),patch.object(worker,'spawn',side_effect=spawn),self.assertRaises(OSError):worker.run_configs(cfg,root/'wave',monitor=False)
   self.assertEqual(len(kids),1);self.assertIsNotNone(kids[0].poll())
 def test_wave_STOP_reaps_all_owned_synthetic_children(self):
  sys.path.insert(0,str(s.ROOT/'tools/restricted_regression'));import m5_formal_entry as worker
  with tempfile.TemporaryDirectory(dir=s.PACKAGE/'fixtures')as tmp:
   root=Path(tmp);g=self.c['groups'][0];kids=[]
   cfg=[dict(task=r,purpose='ch3_probe',m_phase='q4',approval={},output=str(root/g['id']/'q4'/r),artifact_root=str(root/g['id']/'q4'/r),limits=dict(seconds=3))for r in g['task_ids']]
   for item in cfg:item['audit_log']=str(Path(item['output'])/'audit.jsonl')
   def spawn(c):
    d=Path(c['output']);d.mkdir(parents=True);log=(d/'worker.log').open('x');code="import time;from pathlib import Path;time.sleep(.2);Path("+repr(str(root/'STOP'))+").write_text('synthetic STOP');time.sleep(20)";p=subprocess.Popen([sys.executable,'-B','-c',code],stdout=log,stderr=subprocess.STDOUT);kids.append(p);return p,log
   with patch.object(e,'PROBE_ROOT',root),patch.object(worker,'spawn',side_effect=spawn):r=worker.run_configs(cfg,root/'wave',monitor=False)
   self.assertIn('safe-stop',r['failure']);self.assertEqual(len(kids),4);self.assertTrue(all(p.poll()is not None for p in kids));self.assertFalse(e.resource_fallback(r))

class MEnvironmentAndStopTests(unittest.TestCase):
 def setUp(self):self.c=validate_manifest(read_profiles())
 def test_GPU_masks_explicitly_reject_CPU_environment(self):
  for value in ('','1','0,1'):
   with self.subTest(value=value):self.assertTrue(e.gpu_environment_reasons({'CUDA_VISIBLE_DEVICES':value}))
  self.assertEqual(e.gpu_environment_reasons({'CUDA_VISIBLE_DEVICES':'0'}),[]);self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'),'')
 def test_stop_cannot_signal_unrelated_process(self):
  import m6_m_entry as entry
  from m6_remaining_entry import identity
  with patch.object(entry,'signal_owned')as send,self.assertRaises(ValueError):entry.signal_M_owned(identity(os.getpid()))
  send.assert_not_called()
 def test_result_namespace_symlink_refused_before_read(self):
  with tempfile.TemporaryDirectory(dir=s.PACKAGE/'fixtures')as tmp:
   root=Path(tmp);(root/'real').mkdir();(root/'alias').symlink_to(root/'real',target_is_directory=True)
   with patch.object(s,'RESULT',root/'alias'),self.assertRaises(ValueError):s.result_path(self.c['tasks'][0])

class MCPUShapeTests(unittest.TestCase):
 def test_native_all_channel_output(self):
  import torch
  from restricted_io_guard import require_installed
  from models.ch3_adapter import build,target_prediction
  from resource_budget import INSTANCE
  config=require_installed();c=validate_manifest(read_profiles());model_name=config['case']
  for run in config['cpu_tasks']:
   with self.subTest(run=run):
    t=next(t for t in c['tasks']if t['id']==run);p=profile(c,t);model=build(c,t).cpu().eval()
    if model_name=='TimeXer':self.assertEqual(model.n_vars,p['C']);self.assertEqual(model.features,'M');self.assertEqual(model.en_embedding.glb_token.shape[1],p['C'])
    if model_name=='AMD':self.assertFalse(model.use_sonnet_mvca);self.assertFalse(model.use_target_history_local_shape);self.assertEqual(model.task_mode,'parallel_multivariate')
    x=torch.randn(2,p['T'],p['C'],generator=torch.Generator().manual_seed(2024))
    INSTANCE.sample(torch)
    with torch.no_grad():prediction,aux=target_prediction(model,x,p)
    self.assertEqual(tuple(prediction.shape),(2,p['pred_len'],p['C']));self.assertTrue(torch.isfinite(prediction).all())
    if aux is not None:self.assertTrue(torch.isfinite(aux).all())
    self.assertFalse(torch.cuda.is_initialized());del model,prediction,aux,x
    INSTANCE.sample(torch)
