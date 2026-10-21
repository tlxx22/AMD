"""Bound read-only author import; temporary package aliases are restored."""
import hashlib, importlib.util, json, subprocess, sys, threading, types
from pathlib import Path
_LOCK=threading.RLock()
_AUTHOR=None

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def binding():return json.loads(Path(__file__).with_name('source_binding.json').read_text())
def implementation_binding():return {p.name:sha(p) for p in sorted(Path(__file__).parent.iterdir()) if p.suffix in ('.py','.json','.txt')}
def verify_author():
    s=binding()['ModernTCN']
    if subprocess.check_output(['git','-C',s['repository'],'rev-parse','HEAD'],text=True).strip()!=s['commit']:raise RuntimeError('author commit changed')
    for p,h in s['files'].items():
        if sha(p)!=h:raise RuntimeError('author source changed: '+p)
    return s

def verify_training_materials():
    s=verify_author();return {'ModernTCN':{'commit':s['commit'],'files':s['files']}}

def load_author():
    global _AUTHOR
    with _LOCK:
        s=verify_author()
        if _AUTHOR is not None:return _AUTHOR
        names=lambda k:k in ('models','layers') or k.startswith(('models.','layers.'))
        saved={k:v for k,v in sys.modules.items() if names(k)}
        for k in saved:del sys.modules[k]
        for k in ('models','layers'):
            m=types.ModuleType(k);m.__path__=[str(Path(s['root'])/k)];sys.modules[k]=m
        name='_bound_moderntcn_uv_v1_author'
        spec=importlib.util.spec_from_file_location(name,s['entry']);module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        try:
            spec.loader.exec_module(module)
            for k,v in list(sys.modules.items()):
                if names(k) and getattr(v,'__file__',None):
                    p=str(Path(v.__file__).resolve())
                    if p not in s['files'] or sha(p)!=s['files'][p]:raise RuntimeError('unbound dependency '+p)
            _AUTHOR=module;return module
        finally:
            for k in list(sys.modules):
                if names(k):del sys.modules[k]
            sys.modules.update(saved)
