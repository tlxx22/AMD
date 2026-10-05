"""Version-pinned 147-run preparation audit; never executes the old queue."""
import hashlib,json,sys,types
from utils.ch3_contract import ROOT
from utils.ch3_native_recovery_records import bound as original_bound
P=ROOT.parent/'amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/baseline-type1-followup-v2'
A=P/'reviewed-147-candidate'
inventory=A/'workspace-change-inventory.json'
if hashlib.sha256(inventory.read_bytes()).hexdigest()!='7299ac8401859d152bda881fdad5bff51967f06dc4931e4a1af3b4ab5e847422':raise ValueError('147 reviewed inventory changed')
records=json.loads(inventory.read_text())

def bound(value):
    from pathlib import Path
    path=Path(value['path'])
    if path.is_relative_to(P):
        archived=A/path.relative_to(P)
        if archived.is_file()and hashlib.sha256(archived.read_bytes()).hexdigest()==value['sha256']:
            return original_bound(dict(path=str(archived),sha256=value['sha256']))
    return original_bound(value)

def load(name):
    path=A/'workspace-bytes/utils'/('ch3_type1_'+name+'.py');raw=path.read_bytes()
    rows=records['files']
    row=next(r for r in rows if r['path']=='utils/ch3_type1_'+name+'.py')
    if hashlib.sha256(raw).hexdigest()!=row['after_sha256']:raise ValueError('147 frozen test source changed')
    alias='tests._pinned_147_'+name;module=types.ModuleType(alias);module.__file__=str(path)
    sys.modules[alias]=module
    code=raw.decode().replace('from utils import ch3_type1_tasks as s','import tests._pinned_147_tasks as s')
    exec(compile(code,str(path),'exec'),module.__dict__);module.bound=bound
    return module

s=load('tasks');q=load('chain')
q.configs=lambda:{stage:s.validate(json.loads((A/'workspace-bytes'/s.file(stage).relative_to(ROOT)).read_text()))for stage in s.STAGES}
