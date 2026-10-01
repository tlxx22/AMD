"""The one M waiting supervisor CLI; no model/optimizer/GPU imports."""
import argparse,json,os
from utils.ch3_contract import read_profiles,validate_manifest
from utils import ch3_m_handoff as handoff
from utils import ch3_m_launch as launch

def cli():
 p=argparse.ArgumentParser();p.add_argument('action',choices=('dry-run','preflight','prepare-launch','start','status','logs','complete','safe-stop'));p.add_argument('--wrapper-pid',type=int);a=p.parse_args();c=validate_manifest(read_profiles())
 if a.action in ('dry-run','preflight'):
  reasons=handoff.preflight(c);print(json.dumps(dict(scope='m-baselines-v1-handoff',states=handoff.STATES,poll_seconds=60,blocked=reasons,permit_generation=False),ensure_ascii=False,indent=2));return 2 if a.action=='preflight'and reasons else 0
 if a.action=='prepare-launch':print(launch.prepare(c,'handoff',a.wrapper_pid));return 0
 if a.action=='start':handoff.start(c,os.environ.get(launch.TOKEN_ENV));return 0
 if a.action=='safe-stop':print(json.dumps(handoff.safe_stop()));return 0
 if a.action=='complete':
  value=handoff.status()
  if not value['complete']:print(json.dumps(dict(complete=False,status=value),ensure_ascii=False));return 2
  value=json.loads((handoff.ROOT/'complete.json').read_text())
  if value['task_ids']!=[t['id']for t in c['tasks']]or value['technical_complete']is not True or value['result_review']!='pending'or handoff.status()['running']or(handoff.ROOT/'STOP').exists()or(handoff.ROOT/'failure.json').exists():raise ValueError('handoff completion identity/state')
  from m6_m_entry import verify_completion
  from utils import ch3_m_execution as execution
  from utils import ch3_m_tasks as scope
  for kind,probe in [('probe',True),('formal',False)]:
   receipt=value['receipts'][kind];permit=scope.bound(receipt['permit']);verify_completion(c,execution.PROBE_ROOT if probe else execution.CONTROL,permit,probe)
  print(json.dumps(dict(technical_complete=True,result_review='pending')));return 0
 value=handoff.status()
 if a.action=='logs':value['logs']=[str(launch.spec('handoff')['log']),str(launch.spec('probe')['log']),str(launch.spec('formal')['log'])]
 print(json.dumps(value,ensure_ascii=False,indent=2));return 0
if __name__=='__main__':raise SystemExit(cli())
