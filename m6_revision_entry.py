"""One TimeMixer fixed-LR revision. No next-model dispatch or automatic recovery."""
import argparse,json,sys
from pathlib import Path
from utils.ch3_contract import read_profiles,validate_manifest,digest,profile,task_by_id
from utils.ch3_revision import root,ids,waves,REVISION,readiness,execute
from utils.ch3_extension import controller_live,safe_stop

def cli(argv=None):
 p=argparse.ArgumentParser();p.add_argument('action',choices=['dry-run','preflight','start','status','logs','complete','safe-stop']);p.add_argument('--model',choices=['TimeMixer']);p.add_argument('--approval');p.add_argument('--probe-report');a=p.parse_args(argv);c=validate_manifest(read_profiles());out=root(c)
 if a.action=='safe-stop':return safe_stop(out)
 if a.action in ['status','logs','complete']:
  if a.action=='logs':print(json.dumps(dict(controller=str(out.parent/'controller.log'),worker_logs=[str(out/r/'worker.log')for r in ids()])));return
  if a.action=='status':print(json.dumps(dict(output=str(out),running=controller_live(out/'controller.json')if(out/'controller.json').exists()else False,completion_marker=(out/'complete.json').exists(),revision=REVISION)));return
  if not(out/'complete.json').exists():print(json.dumps(dict(complete=False,reason='revision has no completion')));return 2
  d=json.loads((out/'complete.json').read_text())
  if d.get('execution_revision')!=REVISION or d.get('attempt')!=2 or d.get('protocol_sha')!=digest(c)or d.get('task_ids')!=ids():raise ValueError('old/foreign revision completion')
  from utils.ch3_m6 import verify_result
  for r in ids():
   m=json.loads((out/r/'manifest.json').read_text());result=json.loads((out/r/'result.json').read_text())
   if m['identity'].get('execution_revision')!=REVISION or m['identity'].get('attempt')!=2:raise ValueError('revision manifest identity')
   verify_result(c,r,result)
  print(json.dumps(dict(complete=True,execution_revision=REVISION,result_review='Pending')));return
 approval=json.loads(Path(a.approval).read_text())if a.approval else None
 reasons=readiness(c,approval)
 if a.action in ['dry-run','preflight']:
  print(json.dumps(dict(revision=REVISION,attempt=2,output=str(out),task_ids=ids(),waves=waves(c),runs=16,max_run_epochs=200,coverage_runs=552,attempt_cap=568,attempt_epoch_cap=6440,profiles={r:profile(c,task_by_id(c,r))for r in ids()},blocked=reasons),ensure_ascii=False,indent=2));return 2 if a.action=='preflight'and reasons else 0
 if reasons:raise PermissionError('; '.join(reasons))
 return execute(c,approval)
if __name__=='__main__':sys.exit(cli())
