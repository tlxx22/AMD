"""v5 scope, fixed sources, non-executable operations and old-version protection."""
import copy,hashlib,json,subprocess,unittest
from pathlib import Path
from unittest.mock import patch
from tests.timemixer_v5_fixtures import ROOT,STORED,EVIDENCE
from timemixer_dev_v5.source import binding,verify_training_materials
from timemixer_dev_v5.worker import source_binding
from timemixer_dev_v5.spec import REVISION,MODULE_CONSTANTS
from timemixer_dev_v5.launcher import validate_authorization,batch
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
PREPARATION=EVIDENCE/'source-completion-r1/launch-preparation'

def damaged_paper_bindings():
 """Only copied metadata is damaged; neither original PDF is ever written."""
 for field,value in [('status','Pending official PDF reading (HTTP403)'),('path',None),('sha256',None),('sha256','0'*64)]:
  fake=copy.deepcopy(binding());paper=fake['Crossformer']['paper']
  if value is None:paper.pop(field)
  else:paper[field]=value
  yield field,value,fake

class Delivery(unittest.TestCase):
 def test_actual_v4_first_batch_order_exact_mapping_and_non_executable_template(self):
  old=json.loads((EVIDENCE.parent/'M6-Timemixer-v4-implementation-r1/launch-preparation/task-ids.json').read_text());new=json.loads((PREPARATION/'task-ids.json').read_text());self.assertEqual(new,[i.replace('timemixer-uv-dev-v4','timemixer-uv-dev-v5',1)for i in old]);a=json.loads((PREPARATION/'launch-binding.template.json').read_text());self.assertFalse(a['review_approved']);self.assertIsNone(a['commit']);self.assertFalse(a['resume_review_approved']);self.assertEqual(a['source_binding'],source_binding());self.assertEqual(a['config_sha256'],sha(ROOT/'configs/timemixer_uv_round2_resolved_v5.json'));self.assertEqual(a['task_ids'],new);self.assertEqual(a['resource_plan']['waves'],batch(STORED,new));self.assertEqual(len(new),24);self.assertEqual(len(a['resource_plan']['waves']),6)
 def test_fixed_sources_verified_papers_and_damaged_bindings_refused(self):
  b=binding();self.assertIn('Crossformer',b);self.assertIn('ModernTCN',b);self.assertFalse(set(b)&{'SOFTS','APT','Pathformer'})
  for name,commit,license in [('Crossformer','c10c8eadb153d1dd9798250967747ca3ebb81383','Apache-2.0'),('ModernTCN','56a9a2c018385cd5acef015378cae7f084d1b11c','MIT')]:
   s=b[name];self.assertEqual(subprocess.check_output(['git','-C',s['root'],'rev-parse','HEAD'],text=True).strip(),commit);self.assertEqual(s['license'],license)
   for p,h in s['files'].items():self.assertEqual(sha(p),h)
  self.assertEqual(b['ModernTCN']['paper']['status'],'method_3.1_3.2_appendix_G_read')
  paper=b['Crossformer']['paper'];self.assertEqual(paper['status'],'verified_method_read');self.assertEqual(paper['pages'],21);self.assertEqual(sha(paper['path']),paper['sha256'])
  verify_training_materials()
  for field,value,fake in damaged_paper_bindings():
   with self.subTest(field=field,value=value),patch('timemixer_dev_v5.source.binding',return_value=fake),self.assertRaisesRegex(PermissionError,'PDF'):
    verify_training_materials()
  for field,value in [('status','Pending'),('sha256','0'*64)]:
   fake=copy.deepcopy(b);fake['ModernTCN']['paper'][field]=value
   with self.subTest(modern=field),patch('timemixer_dev_v5.source.binding',return_value=fake),self.assertRaisesRegex(PermissionError,'ModernTCN'):
    verify_training_materials()
  self.assertNotIn('kappa_U',MODULE_CONSTANTS);self.assertNotIn('prototype_count',MODULE_CONSTANTS);self.assertIn('internal',REVISION['fusion'])
 def test_old_grants_and_damaged_papers_refused_without_GPU_or_worker(self):
  ids=json.loads((PREPARATION/'task-ids.json').read_text());auth=json.loads((PREPARATION/'launch-binding.template.json').read_text())
  for version in(1,2,3,4):
   wrong=dict(auth,review_approved=True,candidate_id='timemixer-uv-dev-v'+str(version))
   with self.assertRaises(PermissionError):validate_authorization(STORED,'unused',ids,wrong)
  with self.assertRaises(PermissionError):validate_authorization(STORED,'unused',ids,auth)
  fake=copy.deepcopy(auth);fake.update(review_approved=True,commit='metadata-test-only')
  def git(*args):return ''if args[0]=='status'else'metadata-test-only'
  with patch('timemixer_dev_v5.launcher.gpu_startup')as gpu,patch('timemixer_dev_v5.launcher.supervise')as supervisor,patch('timemixer_dev_v5.worker.run_worker')as worker:
   for field,value,damaged in damaged_paper_bindings():
    with self.subTest(field=field,value=value),patch('timemixer_dev_v5.launcher.git',side_effect=git),patch('timemixer_dev_v5.source.binding',return_value=damaged),self.assertRaisesRegex(PermissionError,'PDF'):
     validate_authorization(STORED,'unused',ids,fake)
   gpu.assert_not_called();supervisor.assert_not_called();worker.assert_not_called()
 def test_old_tracked_sources_frozen_and_operations_syntax(self):
  before=json.loads((EVIDENCE/'before/snapshot.json').read_text())
  for rel,h in before['files'].items():
   if rel.startswith(('timemixer_dev/','timemixer_dev_v2/','timemixer_dev_v3/','timemixer_dev_v4/','configs/','tests/','utils/','models/','scripts/'))and(ROOT/rel).is_file():self.assertEqual(sha(ROOT/rel),h['sha256'],rel)
  for rel in ['scripts/ch3/start_timemixer_uv_v5.sh']:
   subprocess.run(['bash','-n',str(ROOT/rel)],check=True)
  subprocess.run(['bash','-n',str(PREPARATION/'commands.sh')],check=True)
  self.assertFalse((EVIDENCE.parent/'timemixer-uv-v5-development-runs/urbanev-f4-f1f4-uv-v5-first-batch-v1/urbanev-f4-f1f4-v5-first-r1').exists())
if __name__=='__main__':unittest.main(verbosity=2)
