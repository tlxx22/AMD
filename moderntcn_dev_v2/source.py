"""Bind v2 code plus the exact read-only v1 leaves and native adapter used."""
from pathlib import Path
from moderntcn_dev_v1.source import load_author, sha, verify_author as _verify_author
import json
ROOT=Path(__file__).resolve().parents[1]
REUSED=('__init__.py','model.py','modules.py','source.py','source_binding.json','SOURCE-LICENSES.txt')
def reused_paths():return [ROOT/'moderntcn_dev_v1'/n for n in REUSED]
def binding():return json.loads(Path(__file__).with_name('source_binding.json').read_text())
def implementation_binding():
    own={p.name:sha(p) for p in sorted(Path(__file__).parent.iterdir()) if p.suffix in ('.py','.json','.txt')}
    own.update({'reused/'+str(p.relative_to(ROOT)):sha(p) for p in reused_paths()})
    return own
def verify_author():
    if binding()!=json.loads((ROOT/'moderntcn_dev_v1/source_binding.json').read_text()):raise ValueError('native source binding drift')
    return _verify_author()
def verify_training_materials():
    s=verify_author();return {'ModernTCN':{'commit':s['commit'],'files':s['files']}}
