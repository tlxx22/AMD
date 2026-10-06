"""C3-pinned 231-run tests; read Git objects without executing an old queue."""
import atexit,hashlib,subprocess,sys,tempfile,types
from pathlib import Path
from utils.ch3_contract import ROOT
COMMIT='5bd62dc93d611b3271467a46fd16335b622e0af1'
temp=tempfile.TemporaryDirectory(prefix='historical-231-');atexit.register(temp.cleanup)
aliases={n:'tests._pinned_231_'+n for n in ('tasks','chain','execution','native_tasks','native_execution','summary')}
def load(name):
    path='utils/ch3_'+('type1_'+name if name in ('tasks','chain','execution','summary')else name)+'.py'
    raw=subprocess.check_output(['git','show',COMMIT+':'+path],cwd=ROOT);file=Path(temp.name)/(name+'.py');file.write_bytes(raw);code=raw.decode()
    for source,target in (('type1_tasks','tasks'),('type1_chain','chain'),('type1_execution','execution'),('native_tasks','native_tasks'),('native_execution','native_execution')):
        code=code.replace('from utils.ch3_'+source+' import','from '+aliases[target]+' import').replace('from utils import ch3_'+source+' as','import '+aliases[target]+' as')
    m=types.ModuleType(aliases[name]);m.__file__=str(file);m.source_commit=COMMIT;m.source_sha256=hashlib.sha256(raw).hexdigest();sys.modules[m.__name__]=m;exec(compile(code,str(file),'exec'),m.__dict__);return m
s=load('tasks');q=load('chain');execution=load('execution');native_tasks=load('native_tasks');native=load('native_execution');summary=load('summary')
