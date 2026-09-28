"""No model execution: exact LR revision, retirement gate, index and fixed dispatch."""
import copy,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from utils.ch3_contract import read_profiles,validate_manifest,profile,digest
from utils import ch3_revision as rev
from utils.ch3_extension import queue_ids,joined_index,execute_waves,safe_stop
from utils.ch3_result_index import load_receipts,VerifiedReceipts,PACKAGE
class RevisionTests(unittest.TestCase):
 def setUp(self):
  self.c=read_profiles();self.tmp=tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']);self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
 def test_only_twelve_LR_changes_and_540_equal(self):
  validate_manifest(self.c);diff=json.loads((PACKAGE/'timemixer-retire-v2/profile-diff.json').read_text());self.assertEqual(len(diff['changed']),12);self.assertEqual(len(diff['unchanged_ids']),540);self.assertEqual(diff['original_unchanged'],483)
  for row in diff['changed']:
   with self.subTest(run=row['id']):
    t=next(t for t in self.c['tasks']if t['id']==row['id']);p=profile(self.c,t);self.assertEqual(p['training']['lr'],.001);old=copy.deepcopy(p);old['training']['lr']=.01;self.assertEqual(digest(old),row['old_sha']);self.assertEqual(digest(p),row['new_sha'])
  for t in self.c['tasks']:
   if t['id']not in rev.ids():continue
   if t['dataset']=='Exchange':self.assertEqual((profile(self.c,t)['T'],profile(self.c,t)['training']['lr']),(96,.0003))
 def test_unapproved_field_mutation_rejected(self):
  c=copy.deepcopy(self.c);c['baseline_training_overrides']['TimeMixer']['ETTh1']['96']['values']['batch']=64
  with self.assertRaises(ValueError):validate_manifest(c)
 def test_exact_repeat_budget_and_queue(self):
  self.assertEqual(len(rev.ids()),16);self.assertEqual([len(w)for w in rev.waves(self.c)],[4,4,2,2,4]);self.assertEqual(sum(rev.waves(self.c),[]),rev.ids());self.assertEqual(sum(profile(self.c,t)['training']['epochs']for t in self.c['tasks']if t['id']in rev.ids()),200);self.assertEqual(self.c['timemixer_revision']['attempt_runs'],568);self.assertEqual(self.c['timemixer_revision']['attempt_epoch_cap'],6440)
 def test_extension_57_disjoint_no_duplicate(self):
  extra=self.c['extension']['new_ids'];self.assertEqual(len(extra),57);self.assertFalse(set(extra)&set(rev.ids()));self.assertEqual(len(queue_ids(self.c)),41);self.assertEqual(sum(len(v)for v in self.c['extension']['append_ids'].values()),16)
 def test_old_permit_and_old_artifact_path_reject(self):
  a=dict(purpose='ch3_formal',reviewed=True,protocol_sha=self.c['extension']['parent_protocol_sha'])
  self.assertIn('old/foreign revision authorization',rev.authorization_reasons(self.c,a))
  with self.assertRaises(PermissionError):rev.validate_spawn(self.c,rev.ids()[0],str(self.root/'old'),str(self.root/'old'),a)
  self.assertFalse((self.root/'old').exists())
 def test_retirement_not_integrity_or_missing_path_success(self):
  self.assertTrue(rev.boundary_reasons(self.c,dict(audited_complete=True,original_TimeMixer_16_complete=True,audit_sha256='x')))
  self.assertTrue(rev.boundary_reasons(self.c,dict(kind='user_authorized_retirement',receipt_path=str(self.root/'absent'),receipt_sha256='0'*64)))
 def test_duplicate_start_retained_staging_reject(self):
  out=self.root/'formal-TimeMixer';out.mkdir()
  with patch.object(rev,'root',return_value=out):
   with self.assertRaises(FileExistsError):rev.reject_existing(self.c)
  self.assertTrue(out.exists())
 def test_failed_wave_stops_suffix(self):
  calls=[]
  def fail(*args):calls.append(args[1]);raise RuntimeError('fixture worker failure')
  with self.assertRaises(RuntimeError):execute_waves(self.c,rev.waves(self.c),self.root,{},execute=fail)
  self.assertEqual(len(calls),1)
 def test_reused_pid_never_signalled(self):
  (self.root/'controller.json').write_text(json.dumps(dict(pid=123,start_ticks='wrong')))
  with patch('utils.ch3_extension.controller_live',return_value=False),patch('os.kill')as kill:
   with self.assertRaises(RuntimeError):safe_stop(self.root)
   kill.assert_not_called()
 def test_index_other_267_preserved_and_TimeMixer_pending(self):
  records=load_receipts(self.c);rows=joined_index(self.c,records);old=json.loads((PACKAGE/'boundary-index-v1/unified-index-preview.json').read_text())['rows'];old={r['id']:r for r in old}
  for row in rows:
   if row['status']=='success':self.assertEqual(row,old[row['id']])
   if row['id']in rev.ids():self.assertEqual(row['status'],'not-run');self.assertNotIn('mse',row);self.assertIn('/revisions/timemixer-fixedlr-v2/',row['path'])
  self.assertEqual(sum(r['status']=='success'for r in rows),267)
  with self.assertRaises(ValueError):joined_index(self.c,VerifiedReceipts([dict(id=rev.ids()[0],status='success')]))
 def test_three_LR_groups_excluded_from_original_inheritance(self):
  from utils.ch3_extension_probe import specification
  s=specification(self.c);self.assertEqual(len(s['original_groups']),51);self.assertEqual(s['excluded_original_groups'],rev.changed_groups());self.assertFalse(set(s['original_groups'])&set(rev.changed_groups()));self.assertEqual(len(s['proposed_inheritance']),26);self.assertEqual(len(s['new_measurements']),8)

class WorkspaceIntegrationTests(unittest.TestCase):
 def setUp(self):
  self.c=read_profiles();self.tmp=tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']);self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
 def audit_fixture(self):
  out=self.root/'revision';out.mkdir();tasks={t['id']:t for t in self.c['tasks']};commit='a'*40
  complete=dict(execution_revision=rev.REVISION,attempt=2,protocol_sha=digest(self.c),task_ids=rev.ids())
  (out/'complete.json').write_text(json.dumps(complete));runs={}
  for run in rev.ids():
   d=out/run;d.mkdir();data='b'*64
   (d/'manifest.json').write_text(json.dumps(dict(identity=dict(run_id=run,execution_revision=rev.REVISION,attempt=2,protocol_sha=digest(self.c),profile_sha=digest(profile(self.c,tasks[run])),data_sha=data,commit=commit))))
   (d/'result.json').write_text(json.dumps(dict(synthetic_fixture=True)))
   runs[run]=dict(data_sha=data,manifest=dict(path=str(d/'manifest.json'),sha256=rev.sha(d/'manifest.json')),result=dict(path=str(d/'result.json'),sha256=rev.sha(d/'result.json')))
  a=dict(purpose='m6_revision_completion_audit',reviewed=True,execution_revision=rev.REVISION,attempt=2,protocol_sha=digest(self.c),task_ids=rev.ids(),commit=commit,complete=dict(path=str(out/'complete.json'),sha256=rev.sha(out/'complete.json')),runs=runs)
  path=self.root/'audit.json';path.write_text(json.dumps(a));return out,path,dict(path=str(path),sha256=rev.sha(path))
 def test_running_imports_and_code_binding_use_R(self):
  import ch3_runner,utils.ch3_m6,utils.ch3_extension,utils.ch3_revision,utils.ch3_extension_probe,utils.ch3_result_index
  repo=Path('/public/home/yueweiting/大论文/AMD')
  for module in [ch3_runner,utils.ch3_m6,utils.ch3_extension,utils.ch3_revision,utils.ch3_extension_probe,utils.ch3_result_index]:
   with self.subTest(module=module.__name__):self.assertTrue(Path(module.__file__).resolve().is_relative_to(repo))
  self.assertEqual(ch3_runner.ROOT,repo);bound=ch3_runner.code_binding()
  for rel in ['m6_revision_entry.py','m6_extension_entry.py','utils/ch3_revision.py','utils/ch3_extension.py','tests/test_m6_revision.py']:
   with self.subTest(path=rel):self.assertEqual(bound[rel],rev.sha(repo/rel))
 def test_actual_retirement_receipt_no_old_artifacts(self):
  receipt=PACKAGE/'timemixer-retire-v2/deletion-receipt.json'
  b=dict(kind='user_authorized_retirement',receipt_path=str(receipt),receipt_sha256='aa0fa6775b9201fb5af6c9c21bd2d3f3ccc8d8a97c81db5f3e154e79eab487a9')
  self.assertEqual(rev.boundary_reasons(self.c,b),[])
  self.assertTrue(rev.boundary_reasons(self.c,dict(b,receipt_sha256='0'*64)))
  self.assertFalse(rev.root(self.c).exists());self.assertFalse((Path(self.c['execution']['evidence'])/'formal-TimeMixer').exists())
 def test_41_requires_audited_revision_not_retirement_only(self):
  from utils.ch3_extension import extension_reasons
  reasons=extension_reasons(self.c,None)
  self.assertIn('TimeMixer revision completion audit required before 41-run supplement',reasons)
  self.assertNotIn('TimeMixer revision completion audit required before 41-run supplement',extension_reasons(self.c,None,'DLinear'))
  with patch.object(rev,'execute',side_effect=AssertionError('must not start revision')):
   self.assertTrue(rev.completion_audit_reasons(self.c,None))
 def test_completion_audit_source_bound_and_tamper_rejected(self):
  out,path,ref=self.audit_fixture()
  with patch.object(rev,'root',return_value=out):
   self.assertEqual(rev.completion_audit_reasons(self.c,ref),[])
   with self.subTest(case='audit SHA'):self.assertTrue(rev.completion_audit_reasons(self.c,dict(ref,sha256='0'*64)))
   with self.subTest(case='result source mutation'):
    (out/rev.ids()[0]/'result.json').write_text('{}');self.assertTrue(rev.completion_audit_reasons(self.c,ref))
 def test_completion_wrong_attempt_or_unreviewed_rejected(self):
  out,path,ref=self.audit_fixture();a=json.loads(path.read_text())
  for key,value in [('attempt',1),('reviewed',False),('task_ids',rev.ids()[:-1])]:
   with self.subTest(field=key),patch.object(rev,'root',return_value=out):
    bad=dict(a);bad[key]=value;path.write_text(json.dumps(bad));self.assertTrue(rev.completion_audit_reasons(self.c,dict(path=str(path),sha256=rev.sha(path))))
 def test_effective_formal_authorization_scopes_independent(self):
  from utils.ch3_m6 import scoped_model_tasks
  catch=dict(extension_batch=self.c['extension']['id'],authorized_task_ids=queue_ids(self.c))
  repeat=dict(execution_revision=rev.REVISION,authorized_task_ids=rev.ids())
  a={t['id']for t in scoped_model_tasks(self.c,'TimeMixer',repeat)};b={t['id']for t in scoped_model_tasks(self.c,'TimeMixer',catch)}
  self.assertEqual(len(a),16);self.assertEqual(len(b),29);self.assertFalse(a&b)
  self.assertEqual(len(scoped_model_tasks(self.c,'AMD',catch)),4)
  appended=dict(extension_batch=self.c['extension']['id'],authorized_task_ids=queue_ids(self.c,'DLinear'));self.assertEqual(len(scoped_model_tasks(self.c,'DLinear',appended)),45)
 def test_admission_partition_and_budget_separation(self):
  from utils.ch3_extension_probe import specification
  s=specification(self.c);parts=[s['original_groups'],s['proposed_inheritance'],s['new_measurements'],s['excluded_original_groups']]
  self.assertEqual(list(map(len,parts)),[51,26,8,3]);self.assertEqual(len(set(sum(parts,[]))),88)
  self.assertEqual(s['caps'],dict(adam=360,forward=512,backward=360));self.assertEqual(s['already_charged'],dict(adam=0,forward=18,backward=0))
  plan=json.loads((PACKAGE/'timemixer-retire-v2/admission-plan.json').read_text());self.assertIsInstance(plan,dict)

 def test_revision_resource_runtime_binding_rejects_foreign(self):
  from test_m6_closeout import revision_fixture
  a,_=revision_fixture(self.c,self.root)
  self.assertEqual(rev.revision_report_reasons(self.c,a['revision_resource_report'],a),[])
  for key in ('code','environment','hardware'):
   with self.subTest(field=key):
    other=copy.deepcopy(a);other[key]={'foreign':True};self.assertTrue(rev.revision_report_reasons(self.c,a['revision_resource_report'],other))
 def test_templates_refuse_before_hardware_or_business(self):
  import m6_extension_entry,m6_revision_entry
  template=self.root/'template.json';template.write_text(json.dumps(dict(reviewed=False,execution_permitted=False)))
  with patch('ch3_runner.hardware_binding',side_effect=AssertionError('no hardware query before template rejection')),patch('ch3_runner.preflight',side_effect=AssertionError('template must reject first')),patch('sys.stdout'):
   self.assertEqual(m6_extension_entry.cli(['preflight','--approval',str(template)]),2)
   self.assertEqual(m6_revision_entry.cli(['preflight','--approval',str(template)]),2)

 def test_revision_pending_index_exposes_planned_attempt(self):
  rows=joined_index(self.c);pending=[r for r in rows if r['id'] in rev.ids()]
  self.assertEqual(len(pending),16)
  for row in pending:
   with self.subTest(run=row['id']):
    self.assertEqual((row['status'],row['execution_revision'],row['attempt']),('not-run',rev.REVISION,2));self.assertNotIn('mse',row);self.assertNotIn('old_execution_retirement',row)
