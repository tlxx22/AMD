"""No-model M launcher and waiting supervisor regressions."""
import contextlib,copy,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from utils import ch3_m_tasks as s
from utils import ch3_m_execution as e
from utils import ch3_m_launch as l
from utils import ch3_m_handoff as h
from utils.ch3_contract import read_profiles,validate_manifest,digest,profile

class MRepairTests(unittest.TestCase):
 def setUp(self):
  self.c=validate_manifest(read_profiles());self.tmp=tempfile.TemporaryDirectory(dir=s.PACKAGE/'fixtures');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.descriptions={kind:l.spec(kind)for kind in ('probe','formal','handoff')};self.real_run=subprocess.run
  def spec(kind):
   x=self.descriptions[kind];return dict(x,log=self.root/(kind+'.log'),root=self.root/(kind+'-root'))
  self.spec=spec;self.stack=contextlib.ExitStack();self.addCleanup(self.stack.close);self.stack.enter_context(patch.object(l,'spec',side_effect=spec));self.stack.enter_context(patch.object(e,'PROBE_ROOT',spec('probe')['root']));self.stack.enter_context(patch.object(e,'CONTROL',spec('formal')['root']));self.stack.enter_context(patch.object(e,'launcher_log',side_effect=lambda probe:spec('probe'if probe else'formal')['log']));self.stack.enter_context(patch.object(s,'RESULT',self.root/'results'))
  self.app=self.root/'approval.json';self.app.write_text('{}');self.boundary=dict(kind='old_MS_queue_technical_complete',result_review='pending',task_ids=['old'])
 def mint(self,kind='probe',driver='tmux'):
  x=self.spec(kind);x['log'].open('x').close()
  with patch.object(l,'verify_creator'):return l.prepare(self.c,kind,os.getpid(),driver=driver,approval=self.app if kind!='handoff'else None)
 def view(self,name):return dict(session=name,session_id='$fixture',pane=dict(pid=os.getpid(),start_ticks='fixture'),socket='fixture')
 def test_stale_log_preflight_and_start_refuse(self):
  self.spec('probe')['log'].write_text('stale')
  self.assertTrue(e.fresh_conflicts(self.c,True));self.assertTrue(e.fresh_conflicts(self.c,True,launch_token='x'*64,approval=self.app))
 def test_matching_current_log_passes_both_scope_freshness(self):
  for kind in ('probe','formal'):
   with self.subTest(scope=kind):
    token=self.mint(kind)
    with patch.object(l,'tmux_view',side_effect=self.view):self.assertEqual(e.fresh_conflicts(self.c,kind=='probe',launch_token=token,approval=self.app),[])
 def test_missing_wrong_and_cross_scope_tokens_refuse(self):
  probe=self.mint('probe');formal=self.mint('formal')
  with patch.object(l,'tmux_view',side_effect=self.view):
   for token in (None,'z'*64,formal):
    with self.subTest(token_kind='missing'if token is None else'wrong'):self.assertTrue(e.fresh_conflicts(self.c,True,launch_token=token,approval=self.app))
   self.assertTrue(e.fresh_conflicts(self.c,False,launch_token=probe,approval=self.app))
 def test_retained_controller_and_result_refuse_current_token(self):
  token=self.mint('formal');self.spec('formal')['root'].mkdir();s.result_path(self.c['tasks'][0]).mkdir(parents=True)
  with patch.object(l,'tmux_view',side_effect=self.view):
   reasons=e.fresh_conflicts(self.c,False,launch_token=token,approval=self.app);self.assertIn('M task output conflict',reasons);self.assertTrue(any('controller'in x for x in reasons))
 def test_duplicate_noclobber_and_single_claim_refuse(self):
  token=self.mint()
  with self.assertRaises(FileExistsError):self.spec('probe')['log'].open('x')
  with patch.object(l,'verify_creator'),self.assertRaises(FileExistsError):l.prepare(self.c,'probe',os.getpid(),approval=self.app)
  with patch.object(l,'tmux_view',side_effect=self.view):
   l.claim(self.c,'probe',token,approval=self.app)
   with self.assertRaises(FileExistsError):l.claim(self.c,'probe',token,approval=self.app)
 def test_log_inode_replacement_refuse(self):
  token=self.mint();p=self.spec('probe')['log'];p.rename(self.root/'retained-original.log');p.write_text('replacement')
  with patch.object(l,'tmux_view',side_effect=self.view),self.assertRaises(ValueError):l.verify(self.c,'probe',token,approval=self.app)
 def test_creator_must_be_fixed_wrapper(self):
  with self.assertRaises(ValueError):l.verify_creator('probe','tmux',l.identity(os.getpid()))
 def test_internal_gate_accepts_own_session_but_rejects_bare_start(self):
  import m6_m_entry as entry
  token=self.mint()
  with patch.object(e,'authorization_reasons',return_value=[]),patch.object(l,'tmux_view',side_effect=self.view),patch('subprocess.run',side_effect=lambda args,**kw:subprocess.CompletedProcess(args,0)if args[0]=='tmux'else self.real_run(args,**kw)),patch.object(entry,'GPULock',return_value=contextlib.nullcontext()),patch('m5_formal_entry.gpu_sample',return_value={}),patch('m5_formal_entry.resource_assessment',return_value=dict(admission=True)):
   self.assertEqual(entry.readiness(self.c,{},str(self.app),True,starting=True,token=token),[])
   self.assertTrue(entry.readiness(self.c,{},str(self.app),True,starting=True,token=None))
   self.assertTrue(entry.readiness(self.c,{},str(self.app),True))
 def test_handoff_child_cannot_exempt_retained_M_session(self):
  import m6_m_entry as entry
  token=self.mint(driver='handoff')
  with patch.object(e,'authorization_reasons',return_value=[]),patch.object(l,'tmux_view',side_effect=self.view),patch.object(l,'same',return_value=True),patch.object(l,'command_has',return_value=True),patch('os.getppid',return_value=os.getpid()),patch('subprocess.run',side_effect=lambda args,**kw:subprocess.CompletedProcess(args,0)if args[0]=='tmux'else self.real_run(args,**kw)):
   self.assertIn('M retained tmux session',entry.readiness(self.c,{},str(self.app),True,starting=True,token=token))
 def old_fixture(self):
  old=self.root/'old';old.mkdir();controller=dict(pid=987654,start_ticks='42',queue_id='m6-remaining-models-v1');(old/'controller.json').write_text(json.dumps(controller));c=copy.deepcopy(self.c);c['m_experiment']['old_queue']=str(old);return c,old,dict(content=controller,sha256=s.sha(old/'controller.json'))
 def anchor_fixture(self):
  c,old,row=self.old_fixture();p=self.root/'production-start.json';p.write_text(json.dumps(dict(commit=s.BASE,queue={'controller.json':row})));return c,old,row,p,s.ref(p)
 def test_fixed_reviewed_anchor_and_code_binding(self):
  from ch3_runner import code_binding
  self.assertEqual(h.OLD_QUEUE_START_ANCHOR,dict(path=str(s.PACKAGE/'production-start.json'),sha256='829215cc1fd95dcdf9b6dbdf0e2d44f7974224784a49082ca9dd38cfc0f87ecd'))
  with patch.object(s,'ref',side_effect=AssertionError('anchor must not self-sign')):
   row=h.anchor(self.c)
  self.assertEqual(row['sha256'],'a7ed5ce00d4eded6bf6012353babda599ff35a6ab7ca631085aa4f5fc92f2157');self.assertEqual(row['content']['pid'],55852);self.assertEqual(int(row['content']['start_ticks']),147796501)
  self.assertEqual(code_binding()['utils/ch3_m_handoff.py'],s.sha(s.ROOT/'utils/ch3_m_handoff.py'))
 def test_fixture_exact_anchor_SHA_accepts(self):
  c,old,row,p,ref=self.anchor_fixture()
  with patch.object(h,'OLD_QUEUE_START_ANCHOR',ref),patch.object(s,'ref',side_effect=AssertionError('no current-byte signature')):self.assertEqual(h.anchor(c),row)
 def test_anchor_same_path_tamper_rejected(self):
  c,old,row,p,ref=self.anchor_fixture();data=json.loads(p.read_text());data['commit']='0'*40;p.write_text(json.dumps(data))
  with patch.object(h,'OLD_QUEUE_START_ANCHOR',ref),self.assertRaisesRegex(ValueError,'source path/SHA mismatch'):h.anchor(c)
 def test_anchor_wrong_fixed_SHA_rejected(self):
  c,old,row,p,ref=self.anchor_fixture();wrong=dict(ref,sha256='0'*64)
  with patch.object(h,'OLD_QUEUE_START_ANCHOR',wrong),self.assertRaisesRegex(ValueError,'source path/SHA mismatch'):h.anchor(c)
 def test_valid_looking_replacement_rejected_before_controller_read(self):
  c,old,row,p,ref=self.anchor_fixture();new=dict(row['content'],pid=987655,start_ticks='43');controller=old/'controller.json';controller.write_text(json.dumps(new));p.write_text(json.dumps(dict(commit=s.BASE,queue={'controller.json':dict(content=new,sha256=s.sha(controller))})));real_sha=s.sha
  def guarded_sha(path):
   self.assertNotEqual(Path(path),controller,'controller must not be read before anchor SHA succeeds');return real_sha(path)
  with patch.object(h,'OLD_QUEUE_START_ANCHOR',ref),patch.object(s,'sha',side_effect=guarded_sha),patch.object(h,'same')as live,patch.object(e,'old_boundary')as boundary,patch.object(Path,'read_text',side_effect=AssertionError('no JSON/controller read on mismatched anchor')),self.assertRaisesRegex(ValueError,'source path/SHA mismatch'):h.old_queue_state(c)
  live.assert_not_called();boundary.assert_not_called()
 def test_fixed_anchor_old_wait_still_checks_controller(self):
  c,old,row,p,ref=self.anchor_fixture()
  with patch.object(h,'OLD_QUEUE_START_ANCHOR',ref),patch.object(h,'same',return_value=True)as live,patch('m5_formal_entry.gpu_sample',side_effect=AssertionError('no GPU while waiting')):
   self.assertEqual(h.old_queue_state(c),'wait');live.assert_called_once_with(row['content'])
 def test_old_wait_live_only_and_no_GPU_calls(self):
  c,old,row=self.old_fixture()
  with patch.object(h,'anchor',return_value=row),patch.object(h,'same',return_value=True),patch('m5_formal_entry.gpu_sample',side_effect=AssertionError('no GPU while waiting')):self.assertEqual(h.old_queue_state(c),'wait')
 def test_old_failure_STOP_identity_and_disappearance_stop(self):
  c,old,row=self.old_fixture()
  with patch.object(h,'anchor',return_value=row),patch.object(h,'same',return_value=False):
   with self.assertRaises(ValueError):h.old_queue_state(c)
   for name in ('failure.json','STOP'):
    with self.subTest(marker=name):
     (old/name).write_text('fixture')
     with self.assertRaises(ValueError):h.old_queue_state(c)
     (old/name).unlink()
   (old/'controller.json').write_text('{}')
   with self.assertRaises(ValueError):h.old_queue_state(c)
 def test_old_complete_scope_uses_existing_gate(self):
  c,old,row=self.old_fixture();(old/'complete.json').write_text('{}')
  with patch.object(h,'anchor',return_value=row),patch.object(e,'old_boundary',side_effect=ValueError('wrong complete scope')),self.assertRaises(ValueError):h.old_queue_state(c)
  with patch.object(h,'anchor',return_value=row),patch.object(e,'old_boundary',return_value=self.boundary):self.assertEqual(h.old_queue_state(c),'complete')
 def test_boundary_exclusive_identical_reuse_conflict_no_overwrite(self):
  with patch.object(s,'PACKAGE',self.root),patch.object(e,'old_boundary',return_value=self.boundary):
   first=h.seal_boundary(self.c);self.assertEqual(first,h.seal_boundary(self.c));p=Path(first['path']);p.write_text('{}');before=p.read_bytes()
   with self.assertRaises(ValueError):h.seal_boundary(self.c)
   self.assertEqual(before,p.read_bytes())
 def test_missing_permits_do_not_call_GPU(self):
  with patch.object(s,'PACKAGE',self.root),patch('m5_formal_entry.gpu_sample',side_effect=AssertionError('no GPU')):self.assertIsNone(h.permit(self.c,True));self.assertIsNone(h.permit(self.c,False))
 def test_bad_permit_stops_not_ignored(self):
  with patch.object(s,'PACKAGE',self.root):
   (self.root/'probe-review.json').write_text('{}')
   with patch('m6_m_entry.readiness',return_value=['bad scope']),self.assertRaises(PermissionError):h.permit(self.c,True)
 def run_fixture(self,permits,*,stop_after=None):
  root=self.root/'supervisor';root.mkdir();events=[];sleep_calls=[]
  def read(c,probe):
   events.append(('permit',probe));return dict(path='synthetic-probe'if probe else'synthetic-formal',sha256='fixture',value={})if permits[probe]else None
  def owned(c,r,probe,a):events.append(('child',probe));return dict(technical_complete=True,reviewed=False,kind='probe'if probe else'formal',synthetic_fixture=True)
  def wait(n):
   sleep_calls.append(n)
   if stop_after is not None and len(sleep_calls)>=stop_after:(root/'STOP').write_text('fixture stop')
  with patch.object(h,'static_binding',return_value=dict(commit='fixture')),patch.object(h,'old_queue_state',return_value='complete'),patch.object(h,'seal_boundary',return_value=dict(path='fixture-boundary',sha256='fixture')),patch.object(h,'permit',side_effect=read),patch.object(h,'run_owned',side_effect=owned):
   try:h.run(self.c,root,sleep=wait,interval=.01)
   except InterruptedError:pass
  return root,events,sleep_calls
 def test_probe_completes_but_missing_formal_review_waits(self):
  root,events,waits=self.run_fixture({True:True,False:False},stop_after=1)
  self.assertEqual([x for x in events if x[0]=='child'],[('child',True)]);self.assertFalse((root/'complete.json').exists());self.assertTrue((root/'failure.json').exists());self.assertEqual(json.loads((root/'progress.json').read_text())['state'],'WAIT_FORMAL_REVIEW')
 def test_valid_fixed_state_chain_only_expected_children(self):
  root,events,waits=self.run_fixture({True:True,False:True});self.assertEqual([x for x in events if x[0]=='child'],[('child',True),('child',False)]);self.assertEqual(waits,[]);self.assertEqual([json.loads(x)['state']for x in(root/'states.jsonl').read_text().splitlines()],list(h.STATES));self.assertEqual(json.loads((root/'complete.json').read_text())['result_review'],'pending')
 def test_STOP_in_old_wait_does_not_signal_old_queue(self):
  root=self.root/'supervisor';root.mkdir()
  def wait(n):(root/'STOP').write_text('fixture stop')
  with patch.object(h,'static_binding',return_value={}),patch.object(h,'old_queue_state',return_value='wait'),patch.object(h,'signal_child')as signal,patch.object(h,'run_owned')as child,self.assertRaises(InterruptedError):h.run(self.c,root,sleep=wait,interval=.01)
  signal.assert_not_called();child.assert_not_called()
 def test_STOP_in_approval_wait_does_not_generate_permits(self):
  root,events,waits=self.run_fixture({True:False,False:False},stop_after=1);self.assertFalse(any(x[0]=='child'for x in events));self.assertFalse((self.root/'probe-review.json').exists());self.assertTrue((root/'STOP').exists())
 def test_child_failure_prevents_formal(self):
  root=self.root/'supervisor';root.mkdir();calls=[]
  with patch.object(h,'static_binding',return_value={}),patch.object(h,'old_queue_state',return_value='complete'),patch.object(h,'seal_boundary',return_value={}),patch.object(h,'permit',return_value=dict(path='fixture',sha256='fixture',value={})),patch.object(h,'run_owned',side_effect=RuntimeError('synthetic worker failed'))as child,self.assertRaises(RuntimeError):h.run(self.c,root)
  self.assertEqual(child.call_count,1);self.assertEqual(json.loads((root/'failure.json').read_text())['state'],'PROBE_RUNNING');self.assertFalse((root/'complete.json').exists())
 def test_duplicate_handoff_preflight_retained_root(self):
  root=self.root/'handoff';root.mkdir()
  with patch.object(h,'ROOT',root),patch.object(h,'static_reasons',return_value=[]),patch.object(h,'old_queue_state',return_value='wait'),patch('subprocess.run',return_value=subprocess.CompletedProcess([],1)):self.assertTrue(h.preflight(self.c))
 def test_safe_stop_waiter_only_and_owned_child_fallback(self):
  root=self.root/'handoff';root.mkdir();(root/'controller.json').write_text(json.dumps(dict(pid=123,start_ticks='1')));(root/'current.json').write_text(json.dumps(dict(child=dict(pid=456,start_ticks='2'))))
  with patch.object(h,'ROOT',root),patch.object(h,'signal_supervisor',return_value=True)as sup,patch.object(h,'signal_child')as child:
   h.safe_stop();sup.assert_called_once();child.assert_not_called();self.assertTrue((root/'STOP').exists())
  with patch.object(h,'ROOT',root),patch.object(h,'signal_supervisor',return_value=False),patch.object(h,'signal_child',return_value=True)as child:h.safe_stop();child.assert_called_once_with(dict(pid=456,start_ticks='2'),owner=dict(pid=123,start_ticks='1'))
 def test_signal_supervisor_refuses_unrelated_PID(self):
  with patch.object(h,'same',return_value=True),patch.object(l,'command_has',return_value=False),patch('os.kill')as kill,self.assertRaises(ValueError):h.signal_supervisor(dict(pid=os.getpid(),start_ticks='fixture'))
  kill.assert_not_called()
 def test_signal_child_refuses_foreign_supervisor_identity(self):
  with patch.object(h,'same',return_value=True),patch('m6_m_entry.signal_M_owned')as signal,self.assertRaises(ValueError):h.signal_child(dict(pid=os.getpid(),start_ticks='fixture',handoff_owner=dict(pid=999999,start_ticks='wrong'),m_kind='probe'))
  signal.assert_not_called()
 def test_report_table_values_from_all_H_profiles(self):
  expected={('DLinear','Exchange'):[8,8,32,32],('ModernTCN','Weather'):[256,256,512,512],('ModernTCN','Exchange'):[128,128,512,512],('TimeXer','ETTh1'):[32,4,32,128]}
  for (model,dataset),values in expected.items():
   rows=[profile(self.c,t)for t in self.c['tasks']if t['model']==model and t['dataset']==dataset];self.assertEqual([p['training']['batch']for p in rows],values);self.assertEqual(len(rows),4)
 def test_valid_permits_require_exact_boundary_and_reviewed_probe_source(self):
  e.PROBE_ROOT.mkdir();(e.PROBE_ROOT/'complete.json').write_text('{"synthetic_fixture":true}')
  with patch.object(s,'PACKAGE',self.root):
   boundary=self.root/'old-queue-technical-boundary.json';boundary.write_text(json.dumps(self.boundary));report=self.root/'reviewed-M-report.json';report.write_text(json.dumps(dict(review=dict(original_report=s.ref(e.PROBE_ROOT/'complete.json')),reviewed=True,synthetic_fixture=True)))
   probe=dict(old_boundary=s.ref(boundary),synthetic_fixture=True);formal=dict(probe,resource_report=s.ref(report))
   (self.root/'probe-review.json').write_text(json.dumps(probe));(self.root/'formal-review.json').write_text(json.dumps(formal))
   with patch('m6_m_entry.readiness',return_value=[]):
    self.assertEqual(h.permit(self.c,True)['value'],probe);self.assertEqual(h.permit(self.c,False)['value'],formal)
    report.write_text(json.dumps(dict(review=dict(original_report=dict(path='wrong-old-MS-report',sha256='fixture')))));(self.root/'formal-review.json').write_text(json.dumps(dict(formal,resource_report=s.ref(report))))
    with self.assertRaises(ValueError):h.permit(self.c,False)
 def test_owned_child_is_synchronous_and_complete_is_checked(self):
  root=self.root/'supervisor';root.mkdir();original=subprocess.Popen;commands=[]
  def spawn(args,**kw):
   if args[0]!=self.c['execution']['python']:return original(args,**kw)
   commands.append(args);return original([sys.executable,'-B','-c','import time;time.sleep(.2)'],**kw)
  def completed(c,path,a,probe):path.mkdir();(path/'complete.json').write_text('{"synthetic_fixture":true}');return dict(synthetic_fixture=True)
  with patch.object(l,'verify_creator'),patch('subprocess.Popen',side_effect=spawn),patch('m6_m_entry.verify_completion',side_effect=completed)as check:
   r=h.run_owned(self.c,root,True,dict(path=str(self.app),sha256=s.sha(self.app),value=dict(synthetic_fixture=True)))
  self.assertEqual(r['exit_code'],0);self.assertFalse(l.same(r['child']));self.assertEqual(commands[0][2],str(s.ROOT/'m6_m_entry.py'));self.assertIn('--probe',commands[0]);check.assert_called_once();self.assertFalse(r['reviewed'])
 def test_owned_formal_child_creates_only_exact_missing_launcher_parent(self):
  root=self.root/'supervisor';root.mkdir();original=subprocess.Popen;commands=[];log=self.root/'new-M-root/queues/formal.log'
  def descriptions(kind):return dict(self.spec(kind),log=log)if kind=='formal'else self.spec(kind)
  def spawn(args,**kw):
   if args[0]!=self.c['execution']['python']:return original(args,**kw)
   commands.append(args);return original([sys.executable,'-B','-c','import time;time.sleep(.2)'],**kw)
  def completed(c,path,a,probe):self.assertFalse(probe);path.mkdir();(path/'complete.json').write_text('{"synthetic_fixture":true}');return dict(synthetic_fixture=True)
  self.assertFalse(log.parent.exists())
  with patch.object(l,'spec',side_effect=descriptions),patch.object(l,'verify_creator'),patch('subprocess.Popen',side_effect=spawn),patch('m6_m_entry.verify_completion',side_effect=completed)as check:
   r=h.run_owned(self.c,root,False,dict(path=str(self.app),sha256=s.sha(self.app),value=dict(synthetic_fixture=True)))
  self.assertTrue(log.is_file());self.assertEqual(r['exit_code'],0);self.assertNotIn('--probe',commands[0]);self.assertFalse(l.same(r['child']));check.assert_called_once();self.assertFalse(r['reviewed'])
 def test_owned_child_STOP_cleanup_no_success(self):
  root=self.root/'supervisor';root.mkdir();original=subprocess.Popen;owned=[];signals=[]
  def spawn(args,**kw):
   if args[0]!=self.c['execution']['python']:return original(args,**kw)
   child=original([sys.executable,'-B','-c','import time;time.sleep(30)'],**kw);owned.append(l.identity(child.pid));(root/'STOP').write_text('synthetic stop');return child
  def signal(v):
   self.assertEqual(v['pid'],owned[0]['pid']);self.assertEqual(v['start_ticks'],owned[0]['start_ticks']);signals.append(v)
   if l.same(v):os.kill(v['pid'],__import__('signal').SIGTERM);return True
   return False
  with patch.object(l,'verify_creator'),patch('subprocess.Popen',side_effect=spawn),patch.object(h,'signal_child',side_effect=signal),patch('m6_m_entry.verify_completion')as check,self.assertRaises(RuntimeError):h.run_owned(self.c,root,True,dict(path=str(self.app),sha256=s.sha(self.app),value=dict(synthetic_fixture=True)))
  self.assertEqual(len(signals),1);self.assertFalse(l.same(owned[0]));check.assert_not_called();self.assertFalse((root/'complete.json').exists())
 def test_STOP_at_probe_handoff_prevents_formal_dispatch(self):
  root=self.root/'supervisor';root.mkdir();children=[]
  def complete(c,path,probe,a):children.append(probe);(root/'STOP').write_text('synthetic handoff stop');return dict(synthetic_fixture=True,reviewed=False)
  with patch.object(h,'static_binding',return_value={}),patch.object(h,'old_queue_state',return_value='complete'),patch.object(h,'seal_boundary',return_value={}),patch.object(h,'permit',return_value=dict(value={})),patch.object(h,'run_owned',side_effect=complete),self.assertRaises(InterruptedError):h.run(self.c,root)
  self.assertEqual(children,[True]);self.assertFalse((root/'complete.json').exists())
