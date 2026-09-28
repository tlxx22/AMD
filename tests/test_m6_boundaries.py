"""Model-free proof-binding, fail-closed scheduling and manifest identity tests."""
import copy,hashlib,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from utils import ch3_extension_probe as ep
from utils.ch3_contract import read_profiles,digest
from utils.ch3_result_index import validate_manifest_identity,validate_identity_collection,validate_receipt_document

def revision_fixture(c,root):
 from utils.ch3_revision import REVISION,ids,DOMAINS
 paths=[str(Path(c['execution']['evidence'])/'formal-TimeMixer'),str(Path(c['execution']['evidence'])/'model-TimeMixer-controller-1790524409463376147.log'),str(Path(c['execution']['evidence']).parent/'m6-timemixer-launch-poaodauo')]
 receipt=root/'retired.json';receipt.write_text(json.dumps(dict(status='deleted',remaining_paths=[],authorized_paths=paths,boundary_kind='user_authorized_retirement',checkpoint_integrity_audited=False,backup_created=False)))
 runtime=dict(code={'fixture':'source'},environment={'fixture':'environment'},hardware={'fixture':'hardware'})
 report=root/'resource.json';report.write_text(json.dumps(dict(**runtime,execution_revision=REVISION,protocol_sha=digest(c),profile_shas=c['timemixer_revision']['effective_profile_hashes'],task_ids=ids(),reviewed=True,decisions={'TimeMixer-'+d+'-MS':dict(status='Passed',concurrency=q,numerical_protocol_sha=digest(c))for d,q in zip(DOMAINS,[4,4,2,4])})))
 return dict(**runtime,safe_boundary=dict(kind='user_authorized_retirement',receipt_path=str(receipt),receipt_sha256=ep.sha(receipt)),revision_resource_report=dict(path=str(report),sha256=ep.sha(report))),set(paths)

class BoundaryTests(unittest.TestCase):
 def setUp(self):
  self.c=read_profiles();self.temp=tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']);self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
 def license_fixture(self):
  s=ep.specification(self.c);p=self.root/'safe.json';p.write_text(json.dumps(dict(safe_for_version_switch=True,TimeMixer_completed=16)))
  a=dict(purpose='ch3_resource_probe',reviewed=True,execution_permitted=True,extension_batch=ep.BATCH,scope_sha=digest(s),plan_sha256=s['plan_sha256'],protocol_sha=digest(self.c),parent_protocol_sha=self.c['extension']['parent_protocol_sha'],parent_report_sha=s['parent_report_sha'],caps=ep.CAPS,already_charged=ep.PREPARATION,carry_forward_mode='reviewed_26',inherited_group_ids=s['proposed_inheritance'],execute_group_ids=s['new_measurements'],authorized_task_ids=[r for g in s['new_measurements']for r in s['representatives'][g]],safe_boundary=dict(audited_complete=True,original_TimeMixer_16_complete=True,audit_path=str(p),audit_sha256=ep.sha(p)),meter_receipt_sha=ep.sha(ep.PACKAGE/'meter-closeout-v1/meter-ledger.json'))
  extra,self.retired_paths=revision_fixture(self.c,self.root);a.update(extra);return a

 def reasons(self,a):
  exists=Path.exists
  with patch('pathlib.Path.glob',return_value=[]),patch.object(Path,'exists',lambda p:False if str(p)in getattr(self,'retired_paths',set()) else exists(p)):return ep.authorization_reasons(self.c,a)
 def test_legal_proof_bound(self):
  a=self.license_fixture();self.assertEqual(self.reasons(a),[]);self.assertEqual(a['plan_sha256'],ep.sha(ep.PLAN));s=ep.specification(self.c);self.assertEqual([len(s[k])for k in ['original_groups','proposed_inheritance','new_measurements']],[51,26,8])
 def test_changed_parent_or_differences_reject_old_license(self):
  a=self.license_fixture();original=json.loads(ep.PLAN.read_text())
  for field,value in [('parent_group','foreign-parent'),('profile_differences',['dataset','features','training'])]:
   with self.subTest(field=field):
    changed=copy.deepcopy(original);item=next(x for x in changed['extension_groups']if x['status']=='eligible_for_reviewed_q1_carry_forward');item[field]=value;p=self.root/(field+'.json');p.write_text(json.dumps(changed))
    with patch.object(ep,'PLAN',p):self.assertIn('complete probe PLAN SHA mismatch',self.reasons(a));self.assertIn('extension scope/protocol binding mismatch',self.reasons(a))
 def test_wrong_plan_sha_rejected(self):
  a=self.license_fixture();a['plan_sha256']='0'*64;self.assertIn('complete probe PLAN SHA mismatch',self.reasons(a))
 def test_resource_only_fixed_fallback_order(self):
  calls=[];qs=[]
  def launch(ids,directory):
   calls.append(list(ids));bad=len(ids)==4
   return dict(failure=bad,returncodes=[1]if bad else[0],resource_admission=not bad,failure_kind='resource'if bad else None),[]if bad else ids
  for q in (4,2):
   qs.append(q);m,t,failed=ep.parallel_wave(launch,list('abcd'),self.root,q)
   if not failed:break
  self.assertEqual(qs,[4,2]);self.assertEqual(calls,[list('abcd'),list('ab'),list('cd')]);self.assertEqual(t,list('abcd'))
 def test_unknown_business_numeric_guard_stop_no_next_wave(self):
  for kind in [None,'unknown','business','numeric','identity','guard','measurement']:
   with self.subTest(kind=kind):
    calls=[]
    def launch(ids,directory):calls.append(list(ids));return dict(failure=True,returncodes=[1],resource_admission=False,failure_kind=kind),[]
    with self.assertRaises(RuntimeError):
     for q in (4,2):ep.parallel_wave(launch,list('abcd'),self.root,q)
    self.assertEqual(calls,[list('abcd')])
 def test_failure_reservation_and_denial_no_refund(self):
  for kind in [None,'resource']:
   with self.subTest(kind=kind):
    budget=ep.Budget(charged=dict(adam=350,forward=500,backward=350));config=dict(output=str(self.root/'w'),budget_file=str(self.root/'missing'),limits=dict(adam=6,forward=8,backward=6));budget.reserve([config]);budget.record([config]);before=copy.deepcopy(budget.counts)
    def launch(ids,directory):return dict(failure=True,returncodes=[1],resource_admission=False,failure_kind=kind),[]
    if kind is None:
     with self.assertRaises(RuntimeError):ep.parallel_wave(launch,list('abcd'),self.root,4)
    else:self.assertTrue(ep.parallel_wave(launch,list('abcd'),self.root,4)[2])
    with self.assertRaises(RuntimeError):budget.reserve([config])
    self.assertEqual(budget.counts,before);self.assertEqual(before,dict(adam=356,forward=508,backward=356));self.assertIsNone(budget.attempts[0]['actual'][0]['counts'])
 def manifest_fixture(self):
  task=dict(id='J-fixture',model='J',dataset='fixture',input_variant='MS');p=dict(T=12,C=3);metadata=dict(train_end=50,scaler=dict(fit_count=50));approval=dict(commit='1'*40,protocol_sha='2'*64,code={'fixture.py':'3'*64},data_bindings={'fixture':{'J-fixture':digest(metadata)}})
  identity=dict(purpose='ch3_formal',run_id=task['id'],input_variant='MS',profile_sha=digest(p),data_sha=digest(metadata),commit=approval['commit'],protocol_sha=approval['protocol_sha'],source=approval['code'])
  return dict(identity=identity,task=task,profile=p,metadata=metadata),task,p,approval
 def verify_fixture(self,doc,task,p,a,want=None):
  raw=json.dumps(doc).encode();return validate_manifest_identity(raw,want or hashlib.sha256(raw).hexdigest(),task,p,a,a['protocol_sha'])
 def test_legal_actual_identity_import(self):
  doc,t,p,a=self.manifest_fixture();got=self.verify_fixture(doc,t,p,a);self.assertEqual(got,{k:doc['identity'][k]for k in ['run_id','profile_sha','data_sha','commit','protocol_sha']})
 def test_manifest_sha_mismatch(self):
  doc,t,p,a=self.manifest_fixture()
  with self.assertRaisesRegex(ValueError,'SHA mismatch'):self.verify_fixture(doc,t,p,a,'0'*64)
 def test_four_actual_fields_missing(self):
  for k in ['profile_sha','data_sha','commit','protocol_sha']:
   with self.subTest(field=k):
    doc,t,p,a=self.manifest_fixture();del doc['identity'][k]
    with self.assertRaisesRegex(ValueError,'missing actual'):self.verify_fixture(doc,t,p,a)
 def test_actual_expected_conflicts(self):
  for k in ['profile_sha','data_sha','commit','protocol_sha','source','run_id']:
   with self.subTest(field=k):
    doc,t,p,a=self.manifest_fixture();doc['identity'][k]='wrong'
    with self.assertRaises(ValueError):self.verify_fixture(doc,t,p,a)
 def test_duplicate_run_rejected(self):
  with self.assertRaisesRegex(ValueError,'duplicate'):validate_identity_collection([dict(run_id='J-fixture')]*2,['J-fixture'])
 def test_modified_metrics_cannot_upgrade_receipt(self):
  expected=dict(records=[dict(id='J-fixture',mse=1.2,best_epoch=3)]);changed=copy.deepcopy(expected)
  for key,value in [('mse',0.),('best_epoch',1)]:
   with self.subTest(field=key):
    changed=copy.deepcopy(expected);changed['records'][0][key]=value
    with self.assertRaises(ValueError):validate_receipt_document(self.c,changed,expected)
