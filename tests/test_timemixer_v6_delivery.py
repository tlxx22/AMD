"""v6 source/namespace/BN checkpoint and non-executable delivery checks."""
import copy,json,hashlib,subprocess,unittest,tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from tests import timemixer_v6_fixtures as f
from timemixer_dev_v6 import adapter,launcher
from timemixer_dev_v6.source import binding,verify_training_materials
from timemixer_dev_v6.worker import source_binding
from timemixer_dev_v6.training_state import read,validate as validate_state
from utils.ch3_onecycle import Schedule
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Delivery(unittest.TestCase):
 def test_first24_exact_v5_order_and_non_executable_v6_binding(self):
  P=f.EVIDENCE/'launch-preparation';a=json.loads((P/'launch-binding.template.json').read_text());ids=json.loads((P/'task-ids.json').read_text());old=json.loads((f.EVIDENCE.parent.parent/'M6-Timemixer-v5-implementation-r1/launch-authorized-r1/task-ids.json').read_text())
  self.assertEqual(ids,[i.replace('timemixer-uv-dev-v5','timemixer-uv-dev-v6',1)for i in old]);self.assertEqual(a['task_ids'],ids);self.assertEqual(a['source_binding'],source_binding());self.assertEqual(a['config_sha256'],sha(f.ROOT/'configs/timemixer_uv_round2_resolved_v6.json'));self.assertFalse(a['review_approved']);self.assertIsNone(a['commit']);self.assertFalse(a['resume_review_approved']);self.assertEqual(a['resource_plan']['waves'],launcher.batch(f.STORED,ids));self.assertEqual(len(ids),24);self.assertEqual(len(a['resource_plan']['waves']),6)
  for version in range(1,6):
   bad=copy.deepcopy(a);bad.update(review_approved=True,candidate_id='timemixer-uv-dev-v'+str(version))
   with self.assertRaises(PermissionError):launcher.validate_authorization(f.STORED,'unused',ids,bad)
  old=copy.deepcopy(a);old['review_approved']=True
  old['structural_revision']=dict(u='crossformer_full_TSA_receiver_conditioned_router_v6',v='moderntcn_Block_serial_nonlinear_local_dilated_v6',fusion='two_internal_finest_representation_residuals')
  with self.assertRaisesRegex(PermissionError,'candidate/revision'):
   launcher.validate_authorization(f.STORED,'unused',ids,old)
  with self.assertRaises(PermissionError):launcher.validate_authorization(f.STORED,'unused',ids,a)
  subprocess.run(['bash','-n',str(P/'commands.sh')],check=True)
 def test_fixed_sources_papers_and_old_registered_bytes_and_stopped_v5(self):
  b=binding();verify_training_materials();self.assertFalse(set(b)&{'SOFTS','APT','Pathformer'})
  for name,commit in [('TimeMixer','e24610583b36fdd8c76cc17a8df4e65759a5f460'),('Crossformer','c10c8eadb153d1dd9798250967747ca3ebb81383'),('ModernTCN','56a9a2c018385cd5acef015378cae7f084d1b11c')]:
   self.assertEqual(subprocess.check_output(['git','-C',b[name]['root'],'rev-parse','HEAD'],text=True).strip(),commit)
   for p,h in b[name]['files'].items():self.assertEqual(sha(p),h)
  snap=json.loads((f.EVIDENCE.parent/'before/snapshot.json').read_text())
  for p,v in snap['files'].items():
   if p.startswith('docs/'):continue
   self.assertEqual(sha(f.ROOT/p),v['sha256'],p)
  self.assertEqual(subprocess.check_output(['git','rev-parse','amd_reproduced_baseline_v1^{}'],cwd=f.ROOT,text=True).strip(),'fa9665627e6fcfb1d0c2bc22d943ca9666304fd6')
  rec=json.loads((f.EVIDENCE.parent/'stop-v5-r1/stop-receipt.json').read_text());self.assertEqual(rec['reason'],'USER_REQUESTED_STOP_FOR_DESIGN_REVISION');self.assertTrue(rec['owned_processes_exited'])
  F=Path(rec['attempt_path']);stats=json.loads((f.EVIDENCE.parent/'stop-v5-r1/post-stop-file-stat.json').read_text())
  current={str(p.relative_to(F)):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}for p in F.rglob('*')if p.is_file()};self.assertEqual(stats,current)
  self.assertFalse((f.EVIDENCE.parent.parent/'timemixer-uv-v6-development-runs/urbanev-f4-f1f4-uv-v6-first-batch-v1/urbanev-f4-f1f4-v6-first-r1').exists())
 def test_saved_BN_buffers_resume_schema_refuses_corruption_without_live_writes(self):
  tmp=tempfile.TemporaryDirectory(prefix='.uv-v6-runner-',dir=f.ROOT);self.addCleanup(tmp.cleanup);folder=Path(tmp.name);c,id,path=f.fixture(folder)
  from timemixer_dev_v6.worker import run_worker
  paused=run_worker(c,id,folder/'results','BN-current-source',path_override=path,pause_after_epoch=1)
  state=read(Path(paused['staging'])/'last.pt');ident=state['identity'];p=ident['profile'];m=adapter.build(p,use_u=True,use_v=True,task='MS',fold=ident['task']['fold']);tr=p['training'];o=torch.optim.Adam(m.parameters(),lr=tr['lr'],betas=tuple(tr['betas']),eps=tr['eps'],weight_decay=tr['weight_decay']);s=Schedule(o,tr['scheduler']);g=torch.Generator().manual_seed(33)
  before=f.live_state(m,o,s,g);validate_state(state,m,o,s,ident);f.nested_equal(self,before,f.live_state(m,o,s,g))
  counters=[k for k in state['model']if k.startswith('v.')and k.endswith('num_batches_tracked')];self.assertEqual(len(counters),2)
  for k in counters:self.assertEqual(state['model'][k].item(),state['steps'])
  for suffix,mutation in [('running_mean','nan'),('running_var','missing'),('num_batches_tracked','dtype')]:
   bad=copy.deepcopy(state);key=next(k for k in bad['model']if k.startswith('v.')and k.endswith(suffix))
   if mutation=='nan':bad['model'][key].fill_(float('nan'))
   elif mutation=='missing':bad['model'].pop(key)
   else:bad['model'][key]=bad['model'][key].float()
   with self.assertRaises(ValueError):validate_state(bad,m,o,s,ident)
   f.nested_equal(self,before,f.live_state(m,o,s,g))
if __name__=='__main__':unittest.main(verbosity=2)
