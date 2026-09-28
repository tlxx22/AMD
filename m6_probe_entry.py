"""Extension-only probe entry; dry-run is non-executable and never creates output."""
import argparse,json,sys
from utils.ch3_contract import read_profiles,validate_manifest,digest
from utils.ch3_extension_probe import specification,readiness,execute,ROOT,read
from utils.ch3_extension import safe_stop,controller_live

def cli(argv=None):
 p=argparse.ArgumentParser();p.add_argument('action',choices=['dry-run','preflight','start','status','logs','complete','safe-stop'],nargs='?',default='dry-run');p.add_argument('--approval');a=p.parse_args(argv)
 c=validate_manifest(read_profiles());spec=specification(c)
 if a.action=='safe-stop':return safe_stop(ROOT)
 if a.action in ('status','logs','complete'):
  if a.action=='logs':print(json.dumps(dict(controller=str(ROOT.parent/'extension-probe-controller.log'),workers=[str(f)for f in ROOT.glob('*/*/*/worker.log')] )));return 0
  if a.action=='complete':
   if not (ROOT/'complete.json').exists():print(json.dumps(dict(complete=False,reason='no complete resource report')));return 2
   from ch3_runner import validate_probe_report
   r=read(ROOT/'complete.json');validate_probe_report(c,r)
   if r.get('scope_sha')!=digest(spec):raise ValueError('foreign extension report')
   print(json.dumps(dict(complete=True,scientific_training=False)));return 0
  print(json.dumps(dict(root=str(ROOT),live=controller_live(ROOT/'controller.json'),complete=(ROOT/'complete.json').exists(),failure=(ROOT/'failure.json').exists())));return 0
 approval=read(a.approval)if a.approval else None;blocked=readiness(c,approval)
 if a.action in ('dry-run','preflight'):
  print(json.dumps(dict(scope=spec,scope_sha=digest(spec),blocked=blocked,license_generated=False),ensure_ascii=False,indent=2));return 2 if a.action=='preflight'and blocked else 0
 if blocked:raise PermissionError('; '.join(blocked))
 execute(c,approval);return 0
if __name__=='__main__':sys.exit(cli())
