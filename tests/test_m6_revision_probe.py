"""Synthetic, model-free wiring fixtures only; NEVER execution permits/reports."""
import copy,json,os,tempfile,unittest,sys
from pathlib import Path
from unittest.mock import patch
from utils.ch3_contract import read_profiles,profile,task_by_id,digest,validate_manifest
from utils import ch3_revision_probe as rp,ch3_extension_probe as ep
from utils.ch3_revision import ids,revision_report_reasons
import ch3_runner as runner
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/restricted_regression'))
import m5_formal_entry as tool

class RevisionProbeTests(unittest.TestCase):
 def setUp(self):
  self.c=read_profiles();self.tmp=tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']);self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name)
  self.root=self.base/'numeric';self.plan=self.base/'plan.json';self.plan.write_bytes(rp.PLAN.read_bytes())
  for target,value in [('ROOT',self.root),('PLAN',self.plan)]:
   p=patch.object(rp,target,value);p.start();self.addCleanup(p.stop)
  self.spec=rp.specification(self.c)
  # These values are deliberately unusable outside mocked closure tests.
  self.a=dict(purpose='ch3_resource_probe',probe_scope=rp.SCOPE,reviewed=True,execution_permitted=True,budget_authorized=True,synthetic_fixture=False,
   commit='SYNTHETIC-NOT-A-COMMIT',code={'synthetic_fixture':'source'},environment={'synthetic_fixture':'environment'},hardware={'synthetic_fixture':'hardware'},
   scope_sha=digest(self.spec),protocol_sha=digest(self.c),plan_sha256=self.spec['plan_sha256'],caps=rp.CAPS,already_charged=rp.ZERO,authorized_task_ids=self.spec['task_ids'],
   safe_boundary=dict(kind='user_authorized_retirement',receipt_path=str(ep.PACKAGE/'timemixer-retire-v2/deletion-receipt.json'),receipt_sha256=ep.sha(ep.PACKAGE/'timemixer-retire-v2/deletion-receipt.json')))
 def trace(self,r):
  return dict(id=r,profile_sha=digest(profile(self.c,task_by_id(self.c,r))),steps=6,finite=True,memory_review=dict(blocked=False,needs_long_window=False))
 def launch(self,records,budget=None,write=False):
  def f(runs,directory):
   records.append((list(runs),str(directory)))
   if budget:budget.reserve([dict(output=str(directory/r),limits=rp.WORKER)for r in runs])
   m=dict(failure=None,failure_kind=None,returncodes=[0]*len(runs),resource_admission=True,elapsed=1)
   traces=[self.trace(r)for r in runs]
   if write:
    for r,t in zip(runs,traces):
     p=directory/r;p.mkdir(parents=True);(p/'trajectory.json').write_text(json.dumps(t));(p/'budget.json').write_text(json.dumps(dict(counts={k:rp.WORKER[k]for k in rp.CAPS})));(p/'config.json').write_text(json.dumps(dict(synthetic_fixture=True,task=r)))
    p=directory/('monitor-'+runs[0]);p.mkdir();(p/'process.json').write_text(json.dumps(m))
   return m,traces
  return f
 def test_scopes_partition_and_budget_disjoint(self):
  old=ep.specification(self.c);parts=[old['original_groups'],old['proposed_inheritance'],old['new_measurements'],old['excluded_original_groups']]
  self.assertEqual(list(map(len,parts)),[51,26,8,3]);self.assertEqual(len(set(sum(parts,[]))),88)
  self.assertEqual(len(self.spec['task_ids']),12);self.assertTrue(set(self.spec['task_ids']).isdisjoint(sum(old['representatives'].values(),[])))
  self.assertNotEqual(self.spec['root'],old['root']);self.assertEqual(rp.CAPS,dict(adam=144,forward=192,backward=144));self.assertEqual(ep.PREPARATION['forward'],18)
 def test_no_self_report_dependency(self):
  self.assertNotIn('revision_resource_report',self.a)
  with patch('utils.ch3_revision.revision_report_reasons',side_effect=AssertionError('self report')),patch.object(runner,'preflight',return_value=[])as p:
   self.assertEqual(rp.readiness(self.c,self.a),[]);p.assert_called_once_with(self.c,'TimeMixer',self.a,probe=True)
 def test_cross_scope_permits_rejected(self):
  ext=json.loads((ep.PACKAGE/'workspace-integration-v1/extension-probe-review.template.json').read_text())
  self.assertTrue(rp.authorization_reasons(self.c,ext));self.assertTrue(ep.authorization_reasons(self.c,self.a))
  self.assertIs(ep.scope_module(self.a),rp);self.assertIs(ep.scope_module(ext),ep)
  with self.assertRaises(ValueError):ep.scope_module(dict(probe_scope='unknown'))
 def test_template_budget_plan_profile_and_fixture_rejected(self):
  for k,v in [('budget_authorized',False),('reviewed',False),('execution_permitted',False),('synthetic_fixture',True),('plan_sha256','bad'),('scope_sha','bad'),('protocol_sha','old'),('caps',ep.CAPS),('already_charged',ep.PREPARATION),('authorized_task_ids',ids()),('commit',None)]:
   with self.subTest(field=k):
    a=copy.deepcopy(self.a);a[k]=v;self.assertTrue(rp.authorization_reasons(self.c,a))
  c=copy.deepcopy(self.c);c['timemixer_revision']['effective_profile_hashes'][ids()[0]]='old LR'
  with self.assertRaises(ValueError):rp.specification(c)
 def test_make_config_and_worker_exact_binding(self):
  self.root.mkdir();(self.root/'fixture').mkdir();r=ids()[0];out=self.root/self.spec['groups'][0]['id']/'serial'/r
  with patch.object(tool,'preflight',return_value=[]):s=tool.make_config(self.c,'ch3_probe',out,task=r,approval=self.a)
  self.assertEqual(s['probe_scope'],rp.SCOPE);self.assertEqual(s['session_root'],str(self.root));rp.validate_worker(self.c,s)
  for key,v in [('prefix_files',{'fake.csv':1}),('session_root',str(ep.ROOT)),('probe_scope','extension-probe-v1'),('resume',True),('task',ids()[-1])]:
   with self.subTest(field=key):
    x=copy.deepcopy(s);x[key]=v
    with self.assertRaises((PermissionError,ValueError)):rp.validate_worker(self.c,x)
 def test_old_source_dirty_or_path_refuses_before_mkdir(self):
  r=ids()[0];out=self.root/self.spec['groups'][0]['id']/'serial'/r
  for reason in ('approval source mismatch','reviewed clean closure required','approval configuration mismatch'):
   with self.subTest(reason=reason),patch.object(tool,'preflight',return_value=[reason]):
    with self.assertRaises(PermissionError):tool.make_config(self.c,'ch3_probe',out,task=r,approval=self.a)
    self.assertFalse(out.exists())
  with self.assertRaises((PermissionError,ValueError)):tool.make_config(self.c,'ch3_probe',ep.ROOT/'foreign',task=r,approval=self.a)
 def test_common_preflight_dispatches_numeric_only(self):
  with patch.object(rp,'authorization_reasons',return_value=[] )as a,patch.object(ep,'authorization_reasons',side_effect=AssertionError('wrong scope')),patch.object(runner,'kernel_admission_reasons',return_value=[]),patch('utils.ch3_data.verify_source_state'),patch.object(runner,'git',return_value='fixture'),patch.object(runner,'code_binding',return_value=self.a['code']),patch.object(runner,'environment_binding',return_value=self.a['environment']),patch.object(runner,'hardware_binding',return_value=self.a['hardware']),patch.object(runner.subprocess,'check_output',return_value='fixture'):
   reasons=runner.preflight(self.c,'TimeMixer',self.a,probe=True);a.assert_called_once();self.assertIn('reviewed clean closure required',reasons)
 def test_STOP_before_monitor_or_spawn(self):
  self.root.mkdir();(self.root/'fixture').mkdir();r=ids()[0];out=self.root/self.spec['groups'][0]['id']/'serial'/r
  with patch.object(tool,'preflight',return_value=[]):s=tool.make_config(self.c,'ch3_probe',out,task=r,approval=self.a)
  (self.root/'STOP').write_text('synthetic stop')
  with patch.object(tool,'spawn',side_effect=AssertionError('worker')),patch.object(tool,'gpu_sample',side_effect=AssertionError('GPU')):
   with self.assertRaises(InterruptedError):tool.run_configs([s],out.parent/('monitor-'+r),monitor=True)
 def test_fixed_serial_parallel_budget_and_no_exchange(self):
  calls=[];budget=ep.Budget(rp.CAPS,rp.ZERO)
  d=rp.run_fixed_groups(self.c,self.launch(calls,budget),lambda *a:dict(passed=True,synthetic_fixture=True),lambda:False)
  self.assertEqual(len(calls),16);self.assertEqual(budget.counts,rp.CAPS);self.assertEqual([len(r)for r,p in calls],[1]*4+[4]+[1]*4+[4]+[1]*4+[2,2]);self.assertEqual(list(d),[g['id']for g in self.spec['groups']])
 def test_failure_stops_suffix_and_never_refunds(self):
  for kind in (None,'resource','business','numeric','identity','guard'):
   with self.subTest(kind=kind):
    calls=[];b=ep.Budget(rp.CAPS,rp.ZERO);ok=self.launch(calls,b)
    def fail(runs,directory):
     m,t=ok(runs,directory);m.update(failure='fixture failure',failure_kind=kind);return m,t
    with self.assertRaises(RuntimeError):rp.run_fixed_groups(self.c,fail,lambda *a:self.fail('comparison after failure'),lambda:False)
    self.assertEqual(len(calls),1);self.assertEqual(b.counts,dict(adam=6,forward=8,backward=6))
 def test_numerical_failure_never_dispatches_next_group(self):
  calls=[]
  with self.assertRaises(ValueError):rp.run_fixed_groups(self.c,self.launch(calls),lambda *a:dict(passed=False),lambda:False)
  self.assertEqual(len(calls),5)
 def test_rss_unresolved_and_STOP_fail_closed(self):
  calls=[];ok=self.launch(calls)
  def growth(runs,directory):
   m,t=ok(runs,directory);t[0]['memory_review']['needs_long_window']=True;return m,t
  with self.assertRaises(ValueError):rp.run_fixed_groups(self.c,growth,lambda *a:self.fail('comparison'),lambda:False)
  self.assertEqual(len(calls),1)
  with self.assertRaises(InterruptedError):rp.run_fixed_groups(self.c,self.launch(calls),lambda *a:dict(passed=True),lambda:True)
  self.assertEqual(len(calls),1)
 def test_budget_reservation_no_retry_balance(self):
  b=ep.Budget(rp.CAPS,rp.ZERO);items=[dict(output='synthetic',limits=rp.WORKER)]*24;b.reserve(items);self.assertEqual(b.counts,rp.CAPS)
  with self.assertRaises(RuntimeError):b.reserve(items[:1])
  self.assertEqual(b.counts,rp.CAPS)
 def test_measured_report_pending_review_and_exchange_merge(self):
  self.root.mkdir();calls=[];b=ep.Budget(rp.CAPS,rp.ZERO);compare=lambda *a:dict(passed=True,synthetic_fixture=True)
  d=rp.run_fixed_groups(self.c,self.launch(calls,b,True),compare,lambda:False)
  exchange,binding=rp.exchange_inheritance(self.c);d[binding['group']]=exchange
  r=dict(purpose='revision_numeric_report',probe_scope=rp.SCOPE,scope_sha=digest(self.spec),scope=self.spec,synthetic_fixture=False,execution_complete=True,execution_revision=rp.REVISION,protocol_sha=digest(self.c),task_ids=ids(),profile_shas=self.c['timemixer_revision']['effective_profile_hashes'],budget=vars(b),decisions=d,inheritance={binding['group']:binding},reviewed=False,**{k:self.a[k]for k in ('code','environment','hardware')})
  r['evidence']={str(p.relative_to(self.root)):ep.sha(p)for p in self.root.rglob('*.json')}
  p=self.base/'SYNTHETIC-report.json'
  with patch.object(runner,'compare_probe_trajectories',side_effect=compare):
   rp.validate_completion(self.c,r);p.write_text(json.dumps(r));ref=dict(path=str(p),sha256=ep.sha(p));self.assertEqual(revision_report_reasons(self.c,ref,self.a),['revision resource evidence not reviewed'])
   r['reviewed']=True;p.write_text(json.dumps(r));ref['sha256']=ep.sha(p);self.assertEqual(revision_report_reasons(self.c,ref,self.a),[])
   r['decisions'][self.spec['groups'][0]['id']]['concurrency']=1
   with self.assertRaises(ValueError):rp.validate_completion(self.c,r)
  with self.assertRaises(ValueError):rp.validate_completion(self.c,dict(r,synthetic_fixture=True))
 def test_formal_queues_and_267_source_index_unchanged(self):
  from utils.ch3_extension import joined_index
  from utils.ch3_result_index import DEFAULT_RECEIPTS,load_receipts
  from utils.ch3_revision import waves
  validate_manifest(self.c);self.assertEqual(list(map(len,waves(self.c))),[4,4,2,2,4]);self.assertEqual(len(self.c['extension']['catchup_ids']),41);self.assertEqual(len(sum(self.c['extension']['append_ids'].values(),[])),16)
  old=json.loads((ep.PACKAGE/'workspace-integration-v1/unified-index-final.json').read_text())
  # Source identity assertions use the accepted preview; no new historical artifact audit.
  index=json.loads((ep.PACKAGE/'evidence-index.json').read_text());binding=next(x for x in index['files']if x['relative_path']=='workspace-integration-v1/unified-index-final.json')
  self.assertEqual(ep.sha(ep.PACKAGE/'workspace-integration-v1/unified-index-final.json'),binding['sha256'])
  rows=joined_index(self.c,load_receipts(self.c,DEFAULT_RECEIPTS))
  if isinstance(old,dict):old=old.get('rows',old.get('index',old))
  self.assertEqual(rows,old);self.assertEqual(len(rows),552)
