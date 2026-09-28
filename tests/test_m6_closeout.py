"""Exact, model-free extension wiring/receipt negative tests. No live artifacts."""
import copy,hashlib,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from utils.ch3_contract import read_profiles,digest,profile,task_by_id,numeric_probe_policy
from utils import ch3_extension_probe as ep
from utils.ch3_result_index import VerifiedReceipts,validate_receipt_document,accepted_files
from utils.ch3_extension import joined_index

def revision_fixture(c,root):
 from utils.ch3_revision import REVISION,ids,DOMAINS
 paths=[str(Path(c['execution']['evidence'])/'formal-TimeMixer'),str(Path(c['execution']['evidence'])/'model-TimeMixer-controller-1790524409463376147.log'),str(Path(c['execution']['evidence']).parent/'m6-timemixer-launch-poaodauo')]
 receipt=root/'retired.json';receipt.write_text(json.dumps(dict(status='deleted',remaining_paths=[],authorized_paths=paths,boundary_kind='user_authorized_retirement',checkpoint_integrity_audited=False,backup_created=False)))
 runtime=dict(code={'fixture':'source'},environment={'fixture':'environment'},hardware={'fixture':'hardware'})
 report=root/'resource.json';report.write_text(json.dumps(dict(**runtime,execution_revision=REVISION,protocol_sha=digest(c),profile_shas=c['timemixer_revision']['effective_profile_hashes'],task_ids=ids(),reviewed=True,decisions={'TimeMixer-'+d+'-MS':dict(status='Passed',concurrency=q,numerical_protocol_sha=digest(c))for d,q in zip(DOMAINS,[4,4,2,4])})))
 return dict(**runtime,safe_boundary=dict(kind='user_authorized_retirement',receipt_path=str(receipt),receipt_sha256=ep.sha(receipt)),revision_resource_report=dict(path=str(report),sha256=ep.sha(report))),set(paths)

class CloseoutTests(unittest.TestCase):
 def setUp(self):
  self.c=read_profiles();self.tmp=tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']);self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
 def approval(self):
  s=ep.specification(self.c);f=self.root/'safe.json';f.write_text(json.dumps(dict(safe_for_version_switch=True,TimeMixer_completed=16)))
  a=dict(purpose='ch3_resource_probe',reviewed=True,execution_permitted=True,extension_batch=ep.BATCH,scope_sha=digest(s),plan_sha256=s['plan_sha256'],protocol_sha=digest(self.c),parent_protocol_sha=self.c['extension']['parent_protocol_sha'],parent_report_sha=s['parent_report_sha'],caps=ep.CAPS,already_charged=ep.PREPARATION,carry_forward_mode='reviewed_26',inherited_group_ids=s['proposed_inheritance'],execute_group_ids=s['new_measurements'],authorized_task_ids=[r for g in s['new_measurements']for r in s['representatives'][g]],safe_boundary=dict(audited_complete=True,original_TimeMixer_16_complete=True,audit_path=str(f),audit_sha256=ep.sha(f)),meter_receipt_sha=ep.sha(ep.PACKAGE/'meter-closeout-v1/meter-ledger.json'))
  extra,self.retired_paths=revision_fixture(self.c,self.root);a.update(extra);return a

 def reasons(self,a):
  exists=Path.exists
  with patch('pathlib.Path.glob',return_value=[]),patch.object(Path,'exists',lambda p:False if str(p)in getattr(self,'retired_paths',set()) else exists(p)):return ep.authorization_reasons(self.c,a)
 def test_exact_scope_and_debit(self):
  s=ep.specification(self.c);self.assertEqual([len(s[k])for k in ('original_groups','new_groups','proposed_inheritance','new_measurements')],[51,34,26,8]);self.assertEqual(sum(map(len,s['representatives'].values())),37);self.assertEqual(s['already_charged']['forward'],18)
 def test_template_old_absent_rejected(self):
  for a in (None,{},dict(purpose='ch3_resource_probe',reviewed=True)):
   with self.subTest(approval=a):self.assertTrue(self.reasons(a))
 def test_scope_and_ids_rejected(self):
  a=self.approval();self.assertEqual(self.reasons(a),[])
  for key,value in [('scope_sha','0'*64),('authorized_task_ids',a['authorized_task_ids'][:-1]),('execute_group_ids',a['execute_group_ids'][::-1])]:
   with self.subTest(field=key):b=copy.deepcopy(a);b[key]=value;self.assertTrue(self.reasons(b))
 def test_safe_end_and_budget_rejected(self):
  for key,value in [('safe_boundary',{}),('already_charged',dict(adam=0,forward=0,backward=0)),('caps',dict(adam=999,forward=999,backward=999))]:
   with self.subTest(field=key):a=self.approval();a[key]=value;self.assertTrue(self.reasons(a))
 def test_candidate_no_load_even_good_template(self):
  with patch('ch3_runner.ROOT',self.root/'candidate'),patch('ch3_runner.hardware_binding',side_effect=AssertionError('hardware should not run')),patch('pathlib.Path.glob',return_value=[]):
   self.assertIn('external candidate is not deployed reviewed production',ep.readiness(self.c,self.approval()))
 def test_budget_reserve_and_missing_ledger_no_refund(self):
  b=ep.Budget(charged=dict(adam=350,forward=500,backward=350));c=dict(output=str(self.root/'w'),budget_file=str(self.root/'missing'),limits=dict(adam=6,forward=8,backward=6));b.reserve([c]);b.record([c]);self.assertEqual(b.counts,dict(adam=356,forward=508,backward=356));self.assertIsNone(b.attempts[0]['actual'][0]['counts'])
  with self.assertRaises(RuntimeError):b.reserve([c])
 def test_worker_prefix_and_escape_rejected(self):
  r=ep.specification(self.c)['representatives']['TimeMixer-PJM-MS'][0];s=dict(task=r,approval=dict(authorized_task_ids=[r]),session_root=str(ep.ROOT),fixture_root=str(ep.ROOT/'fixture'),output=str(ep.ROOT/'w'),prefix_files={});ep.validate_worker(self.c,s)
  for key,value in [('output',str(self.root/'escape')),('prefix_files',{'real.csv':99}),('task','old-run')]:
   with self.subTest(field=key):b=copy.deepcopy(s);b[key]=value
   with self.assertRaises(ValueError):ep.validate_worker(self.c,b)
 def test_factory_binds_separate_root_and_rejects_before_write(self):
  import sys
  tool=Path(__file__).resolve().parents[1]/'tools/restricted_regression';sys.path.insert(0,str(tool))
  from m5_formal_entry import make_config
  run=ep.specification(self.c)['representatives']['TimeMixer-PJM-MS'][0];approval=dict(authorized_task_ids=[run]);root=self.root/'probe'
  with patch.object(ep,'ROOT',root):
   s=make_config(self.c,'ch3_probe',root/'worker',task=run,approval=approval);self.assertEqual(s['session_root'],str(root));self.assertEqual(s['limits']['forward'],8);self.assertFalse(s['prefix_files'])
   forbidden=self.root/'outside'
   with self.assertRaises(PermissionError):make_config(self.c,'ch3_probe',forbidden,task=run,approval=approval)
   self.assertFalse(forbidden.exists())
 def test_old_probe_controller_rejected(self):
  import sys
  sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/restricted_regression'))
  from m5_formal_entry import probe_all
  with self.assertRaises(PermissionError):probe_all(self.c,None)
 def test_numeric_exceptions_not_extended(self):
  for domain in ('NP','BE','FR','DE','PJM','UrbanEV'):
   with self.subTest(dataset=domain):self.assertIsNone(numeric_probe_policy(self.c,dict(model='TimeMixer',dataset=domain,h=24 if domain!='UrbanEV'else 3)))
 def test_unbound_and_mutated_receipts_rejected(self):
  with self.assertRaises(ValueError):joined_index(self.c,[dict(id='invented',reviewed=True,status='success')])
  expected=dict(records=[dict(id='fixture',mse=1.)]);changed=copy.deepcopy(expected);changed['records'][0]['mse']=0.
  with self.assertRaises(ValueError):validate_receipt_document(self.c,changed,expected)
 def test_partial_identity_preserved_and_duplicates_rejected(self):
  t=next(t for t in self.c['tasks']if t['model']=='J');r=dict(id=t['id'],batch='original',status='accepted-complete-binding-incomplete',accepted_completed=True,profile_sha=None,data_sha=None,commit=None,protocol_sha=None,expected_profile_sha=digest(profile(self.c,t)),missing_actual_bindings=['profile_sha','data_sha','commit','protocol_sha'])
  rows=joined_index(self.c,VerifiedReceipts([r]));v=next(v for v in rows if v['id']==t['id']);self.assertIsNone(v['profile_sha']);self.assertEqual(sum(v['status']=='not-run'for v in rows),57)
  with self.assertRaises(ValueError):joined_index(self.c,VerifiedReceipts([r,r]))
 def test_accepted_source_bytes_cannot_change(self):
  from utils import ch3_result_index as ri
  folder=self.root/'audit';folder.mkdir();f=folder/'result-summary.json';f.write_text('{}');index=folder/'evidence-index.json';index.write_text(json.dumps({'result-summary.json':ep.sha(f)}));anchors={'AMD':('audit',ep.sha(index))};f.write_text('{"forged":true}')
  with patch.object(ri,'BASE',self.root),patch.object(ri,'ANCHORS',anchors):
   with self.assertRaises(ValueError):accepted_files('AMD',['result-summary.json'])
