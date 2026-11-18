"""CPU lifecycle and test-access accounting; no model or checkpoint loading."""
import contextlib,json,os,signal,subprocess,sys,time,unittest
from pathlib import Path
from unittest.mock import patch
from ch4 import queue,lifecycle,runtime_io
from ch4.runtime_io import read,write_new
from tests import test_ch4_supervisor as supervisor_tests

class Boundaries(unittest.TestCase):
 def setUp(self):self.helper=supervisor_tests.Supervisor('runTest')
 def timing(self,stack):
  # Accelerate only synthetic exception cleanup; production constants stay10/5/5.
  for key,value in [('COOPERATIVE_STOP_SECONDS',.25),('TERM_STOP_SECONDS',.25),('KILL_REAP_SECONDS',.5)]:stack.enter_context(patch.object(runtime_io,key,value))
 def test_failure_uncooperative_peer_and_unrelated_process(self):
  with self.helper.temporary() as d,contextlib.ExitStack() as stack:
   self.timing(stack);base=Path(d);c,a=self.helper.fixture(base)
   c['tasks'][0]['failure_mode']='exception';c['tasks'][1]['failure_mode']='ignore_term'
   a['resource_plan']['waves']=[['fixture-H3','fixture-H6'],['fixture-H9'],['fixture-H12']]
   outsider=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],start_new_session=True)
   try:
    started=time.monotonic()
    with self.assertRaisesRegex(RuntimeError,'technical failure'):self.helper.execute(base,c,a)
    self.assertLess(time.monotonic()-started,5)
    f=read(base/'attempt/failure.json');self.assertTrue(f['all_workers_exit_confirmed']);self.assertIsNone(outsider.poll())
    peer=read(base/'attempt/failures/fixture-H6-train.json');self.assertEqual(peer['returncode'],-signal.SIGKILL)
    self.assertEqual(peer['completed_updates'],3);self.assertFalse((base/'attempt/complete.json').exists())
    self.assertTrue(any(e['signal']==signal.SIGKILL and e['sent'] for e in f['cleanup']['events']))
   finally:outsider.terminate();outsider.wait(timeout=2)
 def test_user_stop_and_termination_signal(self):
  for mode in ['stop','supervisor_signal']:
   with self.subTest(mode=mode),self.helper.temporary() as d,contextlib.ExitStack() as stack:
    self.timing(stack);base=Path(d);c,a=self.helper.fixture(base);c['tasks'][0]['failure_mode']=mode
    with self.assertRaises((InterruptedError,RuntimeError)):self.helper.execute(base,c,a)
    f=read(base/'attempt/failure.json');self.assertTrue(f['all_workers_exit_confirmed']);self.assertFalse((base/'attempt/complete.json').exists())
    if mode=='supervisor_signal':self.assertEqual(f['signals'][0]['signal'],signal.SIGTERM)
    self.assertEqual(len(f['failure_receipts']),1)
 def test_supervisor_original_exception_survives_accounting_failure(self):
  with self.helper.temporary() as d,contextlib.ExitStack() as stack:
   self.timing(stack);base=Path(d);c,a=self.helper.fixture(base);c['tasks'][0]['failure_mode']='ignore_term'
   original=lifecycle.attach
   def attach(*args):
    original(*args)
    deadline=time.monotonic()+2
    while not (Path(args[0])/'fixture-H3-ready.json').exists():
     if time.monotonic()>deadline:raise TimeoutError('fixture readiness')
     time.sleep(.01)
    raise ValueError('original supervisor exception')
   stack.enter_context(patch.object(lifecycle,'attach',side_effect=attach))
   stack.enter_context(patch.object(lifecycle,'finish',side_effect=OSError('fixture accounting unavailable')))
   with self.assertRaisesRegex(ValueError,'original supervisor exception'):self.helper.execute(base,c,a)
   f=read(base/'attempt/failure.json');self.assertEqual(f['failure_type'],'builtins.ValueError');self.assertTrue(f['finalization_errors']);self.assertTrue(f['all_workers_exit_confirmed'])
 def test_identity_conflict_never_kills_and_unconfirmed_receipt(self):
  with self.helper.temporary() as d,contextlib.ExitStack() as stack:
   self.timing(stack);base=Path(d);write_new(base/'config.json',{});write_new(base/'runtime.json',{});(base/'logs').mkdir()
   (base/'logs/a-train.log').write_text('fixture')
   child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],start_new_session=True)
   record=lifecycle.begin(base,'a','train');lifecycle.attach(base,record,child);record['proc_start']='wrong'
   try:
    out=runtime_io.bounded_shutdown([dict(child=child,record=record)])
    self.assertFalse(out['all_exited']);self.assertIsNone(child.poll());self.assertTrue(all(not e['sent'] for e in out['events']))
    ref=lifecycle.finish(base,record,None);v=read(ref['path']);self.assertIsNone(v['returncode']);self.assertFalse(v['exit_confirmed']);self.assertFalse(v['accounting_complete'])
   finally:child.terminate();child.wait(timeout=2)
 def test_test_partial_and_corrupt_claim_retained(self):
  for mode in ['partial','corrupt']:
   with self.subTest(mode=mode),self.helper.temporary() as d:
    base=Path(d);c,a=self.helper.fixture(base);c['tasks'][0]['test_failure_mode']=mode
    with self.assertRaisesRegex(RuntimeError,'technical failure'):self.helper.execute(base,c,a)
    folder=base/'attempt';v=read(folder/'failures/fixture-H3-test.json');p=v['test_progress']
    self.assertTrue(v['access_log_path'].endswith('test-access.jsonl'));self.assertEqual((p['started_batches'],p['completed_batches'],p['started_samples'],p['completed_samples']),(2,1,3,2))
    self.assertEqual(p['unconfirmed_batches'],[1]);self.assertEqual(p['unconfirmed_samples'],1);self.assertFalse(v['accounting_complete']);self.assertIsNone(v['completed_updates'])
    self.assertTrue((folder/'claims/fixture-H3.json').exists())
    with self.assertRaises(FileExistsError):queue.claim_test(folder,'fixture-H3')
    self.assertEqual(len(list((folder/'claims').glob('*.json'))),1);self.assertFalse((folder/'complete.json').exists())
 def test_test_missing_truncated_offset_and_sequence_semantics(self):
  with self.helper.temporary() as d:
   path=Path(d)/'test.jsonl';start=dict(offset=0,inode=None)
   self.assertIn('missing',lifecycle.test_progress(path,start)['errors'][0])
   path.write_text('old bytes\n');start=lifecycle.log_position(path)
   rows=[dict(event='before_forward',phase='test',batch=7,origins=[10,11]),dict(event='after_forward',phase='test',batch=8,samples=2)]
   with path.open('a') as f:
    for row in rows:f.write(json.dumps(row)+'\n')
   v=lifecycle.test_progress(path,start);self.assertEqual(v['completed_batches'],1);self.assertEqual(v['errors'],[])
   with path.open('a') as f:f.write('{"event":')
   v=lifecycle.test_progress(path,start);self.assertEqual(v['completed_samples'],2);self.assertTrue(v['errors'])
   path.write_text(json.dumps(rows[0])+'\n'+json.dumps(dict(rows[1],batch=7))+'\n')
   v=lifecycle.test_progress(path,dict(offset=0,inode=None));self.assertEqual(v['completed_batches'],0);self.assertTrue(v['errors'])
 def test_test_success_totals_missing_and_mismatch(self):
  for mode in ['complete','missing','mismatch']:
   with self.subTest(mode=mode),self.helper.temporary() as d:
    base=Path(d);write_new(base/'config.json',{});write_new(base/'runtime.json',{});(base/'logs').mkdir();(base/'a').mkdir()
    (base/'logs/a-test.log').write_text('fixture');record=lifecycle.begin(base,'a','test')
    if mode!='missing':
     (base/'a/test-access.jsonl').write_text('{"event":"before_forward","phase":"test","batch":0,"origins":[12]}\n{"event":"after_forward","phase":"test","batch":1,"samples":1}\n')
    write_new(base/'receipts/a-test.json',dict(task_id='a',phase='test',files={},result=dict(batches=2 if mode=='mismatch' else 1,origins=1)))
    with patch.object(lifecycle,'progress',side_effect=AssertionError('train progress must not be read for test')):reference=lifecycle.finish(base,record,0)
    if mode=='complete':self.assertIsNone(reference)
    else:self.assertEqual(read(reference['path'])['failure_reason'],'test_accounting_incomplete')
    end=read(base/'lifecycles/a-test-end.json');self.assertEqual(end['accounting_complete'],mode=='complete')
    self.assertIsNone(end['completed_updates'])
 def tearDown(self):self.assertNotIn('torch',sys.modules)

if __name__=='__main__':unittest.main()
