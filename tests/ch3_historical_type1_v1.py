"""Read-only 9341 source snapshots for historical fixtures; no live old chain."""
import atexit,hashlib,json,subprocess,sys,tempfile,types
from pathlib import Path
from utils.ch3_contract import ROOT
COMMIT='9341e4eb44ed9225a3965b102b2504b1fecfe830'
P=ROOT.parent/'amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/baseline-type1-followup-v2'
temp=tempfile.TemporaryDirectory(prefix='historical-v1-fixture-',dir=P);atexit.register(temp.cleanup)
directory=Path(temp.name)
aliases={name:'tests._pinned_v1_'+name for name in ('tasks','chain','execution','native_tasks','native_execution','summary')}
paths={name:'utils/ch3_'+('type1_'+name if name in ('tasks','chain','execution','summary')else name)+'.py'for name in aliases}
modules={}
def load(name):
    raw=subprocess.check_output(['git','show',COMMIT+':'+paths[name]],cwd=ROOT)
    path=directory/(name+'.py');path.write_bytes(raw)
    code=raw.decode()
    for source,target in (('type1_tasks','tasks'),('type1_chain','chain'),('type1_execution','execution'),('native_tasks','native_tasks'),('native_execution','native_execution')):
        code=code.replace('from utils.ch3_'+source+' import','from '+aliases[target]+' import')
        code=code.replace('from utils import ch3_'+source+' as','import '+aliases[target]+' as')
    module=types.ModuleType(aliases[name]);module.__file__=str(path);module.source_commit=COMMIT;module.source_sha256=hashlib.sha256(raw).hexdigest()
    sys.modules[module.__name__]=module;modules[name]=module
    exec(compile(code,str(path),'exec'),module.__dict__)
    return module
s=load('tasks');q=load('chain');execution=load('execution');native_tasks=load('native_tasks');native=load('native_execution');summary=load('summary')
from utils import ch3_type1_upstream as u
profile=s.profile
# Test-created files belong to a temporary v1 fixture, never the retained v1 package.
original=s.PACKAGE;package=directory/s.PROTOCOL;package.mkdir()
for name in ('urban_subset-plan.json','epf_all-plan.json','start-approval.template.json'):(package/name).write_bytes((original/name).read_bytes())
s.PACKAGE=package;q.LOG=package/'followup-launcher.log'
